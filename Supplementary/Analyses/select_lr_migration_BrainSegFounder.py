#!/usr/bin/env python3
"""Select BrainSegFounder LR across one audited selector-only code migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.BrainSegFounder.model import load_and_validate_manifest  # noqa: E402
from FineTuning.shared.cohort import DEFAULT_MANIFEST, sha256_file  # noqa: E402
from FineTuning.shared.protocol import (  # noqa: E402
    ADAPTED_LR_CONFIGS,
    _load_adapted_candidate,
    write_lr_selection_plot,
)
from FineTuning.shared.runtime import (  # noqa: E402
    atomic_json,
    atomic_tsv,
    ensure_new_output_dir,
)


MODEL = "BrainSegFounder"
VARIANT = "bottleneck_mean"
RUNTIME_SCRIPT_KEY = "FineTuning/BrainSegFounder/finetune.py"
LEGACY_RUNTIME_SHA256 = "f46bfe2a80c4cfd9106332c93e9bc57f9f816c4e4dbea688784c0a6ab5439ecf"
SELECTOR_RUNTIME_SHA256 = "c162123d7273effda333bc173df2ca336cc1e186f4d08441e8825b53f298f69f"
CURRENT_RUNTIME_SHA256 = "2e9a2d1857ee3e26f5ca774fd0894b5087619f6b9b6f279511d40d59ce70c36f"

LEGACY_SELECTOR_FIELDS = """            \"preprocessing_contract_sha256\",
            \"registered_shape\",
            \"model_input_shape\",
"""
CURRENT_SELECTOR_FIELDS = """            \"preprocessing_contract_sha256\",
            \"compatible_preprocessing_contracts\",
            \"model_input_shape\",
            \"bottleneck_shape\",
"""
TRACEABILITY_IMPORT = """from FineTuning.shared.internal_test import (  # noqa: E402
    validate_internal_test_runtime_code,
)
"""
LEGACY_RUNTIME_CHECK = """    if code != lock.get(\"code_sha256\"):
        raise ValueError(\"Runtime code differs from the selected checkpoint\")
"""
CURRENT_RUNTIME_CHECK = """    runtime_code_provenance = validate_internal_test_runtime_code(
        model_name=model.model_name,
        locked_code=lock.get(\"code_sha256\", {}),
        current_code=code,
    )
"""
RESULT_PROVENANCE_FIELD = (
    '        \"runtime_code_provenance\": runtime_code_provenance,\n'
)
RECEIPT_PROVENANCE_FIELD = (
    '            \"runtime_code_provenance\": runtime_code_provenance,\n'
)


def canonical_json(value: object) -> str:
    """Return a stable representation for exact nested provenance comparisons."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def recorded_path(path: Path, actual_root: Path, recorded_root: Path | None) -> str:
    """Rewrite a shared local mount path to its canonical cluster project root."""

    resolved = Path(path).resolve()
    if recorded_root is None:
        return str(resolved)
    try:
        relative = resolved.relative_to(actual_root.resolve())
    except ValueError as error:
        raise ValueError(f"Path is outside the candidate project root: {resolved}") from error
    return str(Path(recorded_root) / relative)


def verify_source_migration() -> dict[str, Any]:
    """Reconstruct both audited historical sources from the current evaluator."""

    runtime_path = PROJECT_ROOT / RUNTIME_SCRIPT_KEY
    source = runtime_path.read_text(encoding="utf-8")
    current_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if current_digest != CURRENT_RUNTIME_SHA256:
        raise ValueError(
            f"Unexpected current BrainSegFounder runtime hash: {current_digest}"
        )

    selector_source = source
    for current_fragment, historical_fragment in (
        (TRACEABILITY_IMPORT, ""),
        (CURRENT_RUNTIME_CHECK, LEGACY_RUNTIME_CHECK),
        (RECEIPT_PROVENANCE_FIELD, ""),
        (RESULT_PROVENANCE_FIELD, ""),
    ):
        if selector_source.count(current_fragment) != 1:
            raise ValueError(
                "Current traceability migration fragment is absent or ambiguous"
            )
        selector_source = selector_source.replace(
            current_fragment, historical_fragment, 1
        )
    selector_digest = hashlib.sha256(selector_source.encode("utf-8")).hexdigest()
    if selector_digest != SELECTOR_RUNTIME_SHA256:
        raise ValueError(
            "The corrected selector runtime cannot be reconstructed from the "
            "current traceability runtime"
        )

    if selector_source.count(CURRENT_SELECTOR_FIELDS) != 1:
        raise ValueError("Current selector field block is absent or ambiguous")
    legacy_source = selector_source.replace(
        CURRENT_SELECTOR_FIELDS, LEGACY_SELECTOR_FIELDS, 1
    )
    reconstructed_legacy_digest = hashlib.sha256(legacy_source.encode("utf-8")).hexdigest()
    if reconstructed_legacy_digest != LEGACY_RUNTIME_SHA256:
        raise ValueError(
            "The legacy runtime cannot be reconstructed by the approved selector-only change"
        )
    return {
        "runtime_script": RUNTIME_SCRIPT_KEY,
        "legacy_sha256": reconstructed_legacy_digest,
        "selector_corrected_sha256": selector_digest,
        "current_sha256": current_digest,
        "change_scope": (
            "LR-selection input-contract validation followed by post-training "
            "traceability guards; training loop unchanged"
        ),
    }


def verify_candidate_runtime_code(details: list[dict[str, Any]]) -> None:
    """Require identical runtime code after normalizing the one approved script hash."""

    normalized: list[str] = []
    observed_hashes: set[str] = set()
    for detail in details:
        code = dict(detail["lock"].get("code_sha256", {}))
        observed = code.get(RUNTIME_SCRIPT_KEY)
        if observed not in {LEGACY_RUNTIME_SHA256, SELECTOR_RUNTIME_SHA256}:
            raise ValueError(f"Unapproved BrainSegFounder runtime hash: {observed}")
        observed_hashes.add(str(observed))
        code[RUNTIME_SCRIPT_KEY] = SELECTOR_RUNTIME_SHA256
        normalized.append(canonical_json(code))
    if observed_hashes != {LEGACY_RUNTIME_SHA256, SELECTOR_RUNTIME_SHA256}:
        raise ValueError(
            "Migration selection must contain both legacy and corrected selector runtimes"
        )
    if len(set(normalized)) != 1:
        raise ValueError("LR candidates differ outside the approved selector-only migration")


def verify_high_reproduction(legacy_dir: Path, current_dir: Path) -> dict[str, Any]:
    """Require the repeated high run to reproduce every non-timing history value."""

    legacy_history = pd.read_csv(legacy_dir / "train_history.tsv", sep="\t")
    current_history = pd.read_csv(current_dir / "train_history.tsv", sep="\t")
    if list(legacy_history.columns) != list(current_history.columns):
        raise ValueError("Legacy and corrected high histories use different columns")
    compared_columns = [
        column for column in legacy_history.columns if column != "elapsed_seconds"
    ]
    if not legacy_history[compared_columns].equals(current_history[compared_columns]):
        raise ValueError(
            "The corrected high run does not exactly reproduce the legacy non-timing history"
        )

    prediction_files = (
        "best_validation_predictions.tsv",
        "locked_validation_predictions.tsv",
    )
    prediction_hashes: dict[str, str] = {}
    for filename in prediction_files:
        legacy_hash = sha256_file(legacy_dir / filename)
        current_hash = sha256_file(current_dir / filename)
        if legacy_hash != current_hash:
            raise ValueError(f"High-run prediction drift in {filename}")
        prediction_hashes[filename] = current_hash
    return {
        "legacy_high_run": str(legacy_dir.resolve()),
        "corrected_high_run": str(current_dir.resolve()),
        "non_timing_history_values": "bit-exact",
        "validation_predictions": "bit-exact",
        "prediction_artifacts_sha256": prediction_hashes,
    }


def command_select(args: argparse.Namespace) -> None:
    """Write the standard selection artifact plus an explicit migration receipt."""

    _, manifest = load_and_validate_manifest(args.manifest)
    manifest_sha256 = str(manifest["manifest_sha256"])
    paths = [Path(path).resolve() for path in args.run_dirs]
    if len(paths) != len(set(paths)):
        raise ValueError("Adapted LR candidate directories must be unique")
    candidates = [
        _load_adapted_candidate(
            path,
            expected_model=MODEL,
            expected_variant=VARIANT,
            manifest_sha256=manifest_sha256,
        )
        for path in paths
    ]
    rows = [row for row, _ in candidates]
    details = [detail for _, detail in candidates]
    if {row["lr_config"] for row in rows} != set(ADAPTED_LR_CONFIGS):
        raise ValueError("Selection requires conservative, intermediate, and high")
    if len(rows) != len(ADAPTED_LR_CONFIGS) or {row["seed"] for row in rows} != {0}:
        raise ValueError("Selection requires exactly three seed-0 LR candidates")
    if len({detail["release_key"] for detail in details}) != 1:
        raise ValueError("LR candidates use different base releases")
    if len({canonical_json(detail["lock"].get("adapter")) for detail in details}) != 1:
        raise ValueError("LR candidates use different BrainSegFounder adapters")

    source_migration = verify_source_migration()
    verify_candidate_runtime_code(details)
    current_high = paths[[row["lr_config"] for row in rows].index("high")]
    legacy_high = Path(args.legacy_high_run).resolve()
    actual_project_root = paths[0].parents[4]
    if any(path.parents[4] != actual_project_root for path in paths + [legacy_high]):
        raise ValueError("All migration runs must belong to one project root")
    recorded_project_root = (
        Path(args.recorded_project_root) if args.recorded_project_root is not None else None
    )
    legacy_row, legacy_detail = _load_adapted_candidate(
        legacy_high,
        expected_model=MODEL,
        expected_variant=VARIANT,
        manifest_sha256=manifest_sha256,
    )
    if legacy_row["lr_config"] != "high" or legacy_row["seed"] != 0:
        raise ValueError("The equivalence witness must be the legacy seed-0 high run")
    if legacy_detail["lock"]["code_sha256"].get(RUNTIME_SCRIPT_KEY) != LEGACY_RUNTIME_SHA256:
        raise ValueError("The equivalence witness does not use the allowlisted legacy runtime")
    high_reproduction = verify_high_reproduction(legacy_high, current_high)
    high_reproduction["legacy_high_run"] = recorded_path(
        legacy_high, actual_project_root, recorded_project_root
    )
    high_reproduction["corrected_high_run"] = recorded_path(
        current_high, actual_project_root, recorded_project_root
    )

    table = pd.DataFrame(rows)
    rank = table.sort_values(
        ["validation_mae_years", "lr_encoder"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected = rank.iloc[0].to_dict()
    if selected["lr_config"] != "high" or Path(selected["run_dir"]).resolve() != current_high:
        raise ValueError("The corrected high run is not the validation-selected candidate")

    output_dir = ensure_new_output_dir(args.output_dir)
    recorded_rows = []
    for row in rows:
        recorded = dict(row)
        recorded["run_dir"] = recorded_path(
            Path(row["run_dir"]), actual_project_root, recorded_project_root
        )
        recorded_rows.append(recorded)
    recorded_output_dir = recorded_path(
        output_dir, actual_project_root, recorded_project_root
    )
    migration = {
        "status": "APPROVED_SELECTOR_ONLY_RUNTIME_MIGRATION",
        "reason": (
            "The original selector requested a nonexistent registered_shape field; "
            "the corrected selector uses the released adapter fields."
        ),
        "source_migration": source_migration,
        "high_reproduction": high_reproduction,
        "migration_script": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
        "migration_script_sha256": sha256_file(SCRIPT_PATH),
    }
    atomic_json(migration, output_dir / "SELECTOR_RUNTIME_MIGRATION.json")
    atomic_tsv(
        pd.DataFrame(recorded_rows)
        .sort_values("lr_encoder", kind="stable")
        .reset_index(drop=True),
        output_dir / "adapted_lr_validation_grid.tsv",
    )
    payload = {
        "status": "LR_SELECTED_AFTER_TRAIN_VALIDATION",
        "model": MODEL,
        "variant": VARIANT,
        "selection_rule": "minimum uncorrected validation MAE; lower encoder LR breaks exact ties",
        "selection_seed": int(selected["seed"]),
        "selected_lr_config": str(selected["lr_config"]),
        "selected_run_dir": recorded_path(
            Path(selected["run_dir"]), actual_project_root, recorded_project_root
        ),
        "selected_checkpoint_sha256": str(selected["checkpoint_sha256"]),
        "selected_validation_mae_years": float(selected["validation_mae_years"]),
        "manifest_sha256": manifest_sha256,
        "candidates": recorded_rows,
        "selector_runtime_migration": str(
            Path(recorded_output_dir) / "SELECTOR_RUNTIME_MIGRATION.json"
        ),
        "internal_test_accessed": False,
        "on_harmony_accessed": False,
    }
    atomic_json(payload, output_dir / "SELECTED_LR.json")
    write_lr_selection_plot(table, output_dir, selected_config="high")
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dirs", type=Path, nargs=3, required=True)
    parser.add_argument("--legacy-high-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--recorded-project-root",
        type=Path,
        help="Canonical project root to record when operating through a shared local mount",
    )
    return parser.parse_args()


if __name__ == "__main__":
    command_select(parse_args())
