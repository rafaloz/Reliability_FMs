"""Shared, label-blind ON-Harmony prediction guards for adapted models."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from FineTuning.shared.cohort import PROJECT_ROOT, sha256_file
from FineTuning.shared.evaluation import SELECTION_NAME, validate_selected_evaluation_run
from FineTuning.shared.metrics import AgeStandardizer
from FineTuning.shared.runtime import (
    atomic_json,
    atomic_tsv,
    autocast_context,
    code_hashes,
    import_torch,
    package_versions,
    resolve_device,
    seed_torch,
)


EXPECTED_ACQUISITIONS = 165
EXPECTED_SUBJECTS = 20
DEFAULT_RAW_FILELIST = PROJECT_ROOT / "DATA" / "filelists" / "filelist_BrainIAC_raw.txt"
DEFAULT_METADATA = PROJECT_ROOT / "DATA" / "all_idps.csv"
CHECKPOINT_NAME = "selected_model.pt"
LOCK_NAME = "LOCKED.json"
METADATA_COLUMNS = (
    "session",
    "cohort",
    "scanner_code",
    "scanner_name",
    "vendor",
    "scanner_type",
    "scanner_typeloc",
    "scan_session_num",
    "scan_session",
    "subject",
)
FORBIDDEN_LABEL_COLUMNS = {
    "age",
    "age_true",
    "error_years",
    "absolute_error_years",
    "mae_years",
    "rmse_years",
}
ACQUISITION_PATTERN = re.compile(
    r"(?P<acquisition_id>sub-(?P<participant>[0-9]+)_ses-"
    r"(?P<scan_session>[A-Za-z0-9]+)_T1w)"
)


def canonical_sha256(payload: object) -> str:
    """Hash a JSON-compatible object with a stable serialization."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effective_precision(device: Any, amp: str) -> str:
    """Report the dtype autocast_context actually applies on this device."""

    if amp not in {"off", "float16", "bfloat16"}:
        raise ValueError(f"Unknown AMP mode: {amp}")
    if getattr(device, "type", None) == "cuda" and amp != "off":
        return amp
    return "float32"


def file_signature(paths: Iterable[Path]) -> tuple[tuple[str, int, int, int], ...]:
    """Capture cheap file identities around a potentially expensive hash pass."""

    result: list[tuple[str, int, int, int]] = []
    for path in paths:
        candidate = Path(path).expanduser().resolve()
        try:
            metadata = candidate.stat()
        except FileNotFoundError:
            raise FileNotFoundError(candidate) from None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise FileNotFoundError(f"Missing or empty input artifact: {candidate}")
        result.append(
            (
                str(candidate),
                int(metadata.st_size),
                int(metadata.st_mtime_ns),
                int(metadata.st_ctime_ns),
            )
        )
    return tuple(result)


def acquisition_from_path(path: Path | str) -> dict[str, str]:
    """Parse the stable acquisition identity embedded in every ON-Harmony T1 path."""

    candidate = Path(path)
    matches = list(ACQUISITION_PATTERN.finditer(candidate.name))
    if len(matches) != 1:
        raise ValueError(f"Cannot identify one ON-Harmony acquisition in {candidate.name}")
    return matches[0].groupdict()


def load_onharmony_manifest(
    filelist: Path | str = DEFAULT_RAW_FILELIST,
    metadata_path: Path | str = DEFAULT_METADATA,
    *,
    expected_acquisitions: int = EXPECTED_ACQUISITIONS,
    expected_subjects: int = EXPECTED_SUBJECTS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the canonical raw-T1 registry to acquisition metadata without labels."""

    filelist = Path(filelist).expanduser().resolve()
    metadata_path = Path(metadata_path).expanduser().resolve()
    if not filelist.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing ON-Harmony registry: {filelist}, {metadata_path}")
    paths = [
        Path(line.strip())
        for line in filelist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(paths) != expected_acquisitions:
        raise ValueError(
            f"Expected {expected_acquisitions} ON-Harmony inputs, found {len(paths)}"
        )
    if len({str(path) for path in paths}) != len(paths):
        raise ValueError("ON-Harmony filelist contains duplicate paths")

    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    missing = sorted(set(METADATA_COLUMNS) - set(metadata.columns))
    if missing:
        raise ValueError(f"ON-Harmony metadata lacks columns: {missing}")
    metadata = metadata[list(METADATA_COLUMNS)].copy()
    if len(metadata) != expected_acquisitions:
        raise ValueError(
            f"Expected {expected_acquisitions} metadata rows, found {len(metadata)}"
        )
    if metadata["session"].duplicated().any() or (metadata == "").any().any():
        raise ValueError("ON-Harmony metadata has duplicate sessions or empty fields")
    indexed = metadata.set_index("session", drop=False)

    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_absolute():
            raise ValueError(f"ON-Harmony raw path is not absolute: {path}")
        values = acquisition_from_path(path)
        if path.name != f"{values['acquisition_id']}.nii.gz":
            raise ValueError(f"Canonical registry contains a non-raw T1 path: {path}")
        metadata_key = f"{values['participant']}_{values['scan_session']}"
        if metadata_key not in indexed.index:
            raise ValueError(f"No metadata row for ON-Harmony acquisition {metadata_key}")
        source = indexed.loc[metadata_key]
        if isinstance(source, pd.DataFrame):
            raise ValueError(f"Ambiguous metadata rows for {metadata_key}")
        expected_subject = f"_{values['participant']}"
        if source["subject"] != expected_subject:
            raise ValueError(f"Subject metadata differs for {metadata_key}")
        if source["scan_session"] != values["scan_session"]:
            raise ValueError(f"Session metadata differs for {metadata_key}")
        rows.append(
            {
                "dataset": "ON-Harmony",
                "subject": f"sub-{values['participant']}",
                "session": values["scan_session"],
                "acquisition_id": values["acquisition_id"],
                "raw_t1w_path": str(path),
                "cohort": source["cohort"],
                "scanner_code": source["scanner_code"],
                "scanner_name": source["scanner_name"],
                "vendor": source["vendor"],
                "scanner_type": source["scanner_type"],
                "scanner_typeloc": source["scanner_typeloc"],
                "scan_session_num": source["scan_session_num"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame["acquisition_id"].duplicated().any():
        raise ValueError("Parsed ON-Harmony acquisition identifiers are not unique")
    if frame["subject"].nunique() != expected_subjects:
        raise ValueError(
            f"Expected {expected_subjects} subjects, found {frame['subject'].nunique()}"
        )
    if FORBIDDEN_LABEL_COLUMNS & set(frame.columns):
        raise RuntimeError("The ON-Harmony inference manifest contains outcome labels")

    summary: dict[str, Any] = {
        "status": "PASS",
        "label_blind": True,
        "filelist": str(filelist),
        "filelist_sha256": sha256_file(filelist),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "ordered_records_sha256": canonical_sha256(frame.to_dict(orient="records")),
        "n_acquisitions": int(len(frame)),
        "n_subjects": int(frame["subject"].nunique()),
        "cohort_counts": {
            str(key): int(value) for key, value in frame["cohort"].value_counts().items()
        },
        "scanner_counts": {
            str(key): int(value)
            for key, value in frame["scanner_code"].value_counts().items()
        },
    }
    summary["manifest_sha256"] = canonical_sha256(summary)
    return frame, summary


def select_audit_records(
    frame: pd.DataFrame,
    *,
    subject: str | None,
    session: str | None,
) -> pd.DataFrame:
    """Apply optional audit filters without changing canonical acquisition order."""

    selected = frame
    if subject is not None:
        normalized = subject if subject.startswith("sub-") else f"sub-{subject}"
        selected = selected.loc[selected["subject"] == normalized]
    if session is not None:
        selected = selected.loc[selected["session"] == session]
    selected = selected.reset_index(drop=True)
    if selected.empty:
        raise ValueError("No ON-Harmony acquisitions match the audit filters")
    return selected


def preprocessed_path(root: Path | str, record: Mapping[str, Any], suffix: str) -> Path:
    """Build a dataset-independent ON-Harmony preprocessing output path."""

    acquisition = str(record["acquisition_id"])
    return Path(root).expanduser().resolve() / acquisition / f"{acquisition}{suffix}"


ArtifactResolver = Callable[[Mapping[str, Any]], Mapping[str, Path]]
ArtifactValidator = Callable[[Mapping[str, Any], Mapping[str, Path]], Mapping[str, Any] | None]


class ArtifactRegistry:
    """Resolve, validate, and hash the exact external artifacts consumed by a model."""

    def __init__(
        self,
        *,
        contract_version: str,
        resolver: ArtifactResolver,
        validator: ArtifactValidator | None = None,
        contract: Mapping[str, Any] | None = None,
    ) -> None:
        self.contract_version = contract_version
        self.resolver = resolver
        self.validator = validator
        self.contract = dict(contract or {})
        self._cache: dict[
            str, tuple[tuple[tuple[str, int, int, int], ...], dict[str, Any]]
        ] = {}
        self._frozen_signatures: dict[
            str, tuple[tuple[str, int, int, int], ...]
        ] = {}

    def paths(self, record: Mapping[str, Any]) -> dict[str, Path]:
        paths = {
            str(label): Path(path).expanduser().resolve()
            for label, path in self.resolver(record).items()
        }
        if "input" not in paths or not paths:
            raise ValueError("Every external artifact registry requires an 'input' path")
        return paths

    def required_paths(self, record: Mapping[str, Any]) -> list[Path]:
        return list(self.paths(record).values())

    def input_path(self, record: Mapping[str, Any]) -> Path:
        return self.paths(record)["input"]

    def validate(self, record: Mapping[str, Any]) -> dict[str, Any]:
        paths = self.paths(record)
        signature = file_signature(paths.values())
        key = str(record["acquisition_id"])
        frozen_signature = self._frozen_signatures.get(key)
        if frozen_signature is not None and signature != frozen_signature:
            raise RuntimeError(
                f"ON-Harmony artifacts changed after being frozen: {key}"
            )
        cached = self._cache.get(key)
        if cached is not None and cached[0] == signature:
            return dict(cached[1])
        details = dict(self.validator(record, paths) or {}) if self.validator else {}
        provenance: dict[str, Any] = {
            "input_contract_version": self.contract_version,
            **details,
        }
        for label, path in paths.items():
            provenance[f"{label}_path"] = str(path)
            provenance[f"{label}_sha256"] = sha256_file(path)
        final_signature = file_signature(paths.values())
        if final_signature != signature:
            raise RuntimeError("ON-Harmony input artifacts changed during validation")
        if frozen_signature is not None and final_signature != frozen_signature:
            raise RuntimeError(
                f"ON-Harmony artifacts changed after being frozen: {key}"
            )
        self._cache[key] = (final_signature, provenance)
        return dict(provenance)

    def freeze(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Validate one acquisition and pin its artifact identities for this run."""

        provenance = self.validate(record)
        key = str(record["acquisition_id"])
        signature = self._cache[key][0]
        frozen_signature = self._frozen_signatures.get(key)
        if frozen_signature is not None and signature != frozen_signature:
            raise RuntimeError(
                f"ON-Harmony artifacts changed after being frozen: {key}"
            )
        self._frozen_signatures[key] = signature
        return provenance


@dataclass(frozen=True)
class RuntimeMigration:
    """One exact, audited transition from locked training code to inference code."""

    changed: Mapping[str, tuple[str, str]]
    added: Mapping[str, str]
    migration_id: str | None = None
    reason: str | None = None


_OLD_METRICS_SHA256 = (
    "d7ec28a8f8d4d921efc397da4662af7bec972bb567f7061aee7d1544cd0ebd3e"
)
_CURRENT_METRICS_SHA256 = (
    "7060bf26c257ebdd58802a8a260aae576cd4adb6525627a959922b52b3c4105b"
)
_EVALUATION_SHA256 = (
    "266e085f1020fea2b5c3af27a18f8189db8e2a9eb98e2ed1752342736990096e"
)


def _runtime_migration(
    migration_id: str,
    *,
    changed: Mapping[str, tuple[str, str]],
    added: Mapping[str, str] | None = None,
    reason: str,
) -> RuntimeMigration:
    """Build an immutable-by-convention, fully hash-pinned migration record."""

    return RuntimeMigration(
        changed=dict(changed),
        added=dict(added or {}),
        migration_id=migration_id,
        reason=reason,
    )


# These entries describe the retained validation-selected checkpoints only. A
# new code change must never be added here implicitly: it needs a new audited
# old->new transition and corresponding rejection tests.
_ADAPTED_RUNTIME_MIGRATIONS: dict[str, tuple[RuntimeMigration, ...]] = {
    "3D-Neuro-SimCLR": (
        _runtime_migration(
            "3dneurosimclr-post-review-v1",
            changed={
                "FineTuning/3DNeuroSimCLR/finetune.py": (
                    "045ff93d27b61b491c8b745b1af4e3845bebe399b2c7020b1056b594553dbd3a",
                    "eb8f6b81b2f91211e2a62ddad2979886d2534dd691d67b9bb419ac909d69a7ea",
                ),
                "FineTuning/shared/metrics.py": (
                    _OLD_METRICS_SHA256,
                    _CURRENT_METRICS_SHA256,
                ),
            },
            reason="post-training evaluation guards and metric validation were hardened",
        ),
    ),
    "AnatCL": (
        _runtime_migration(
            "anatcl-post-review-v1",
            changed={
                "FineTuning/AnatCL/finetune.py": (
                    "90eb9c3ee459b75f820fd2c70826ab8601905d25a77f03b417f46bbe5e9aec08",
                    "cabf8761b37a9f0763e76a63adc35df8af9363bc83222195c41cfd6d73cb77a5",
                ),
                "FineTuning/shared/metrics.py": (
                    _OLD_METRICS_SHA256,
                    _CURRENT_METRICS_SHA256,
                ),
            },
            added={"FineTuning/shared/evaluation.py": _EVALUATION_SHA256},
            reason="internal-test guards, provenance, and metric validation were hardened",
        ),
    ),
    "BrainIAC": tuple(
        _runtime_migration(
            f"brainiac-{label}-post-review-v1",
            changed={
                "FineTuning/BrainIAC/finetune.py": (legacy_hash, current_hash),
                "FineTuning/shared/metrics.py": (
                    _OLD_METRICS_SHA256,
                    _CURRENT_METRICS_SHA256,
                ),
            },
            reason="internal-test orchestration and metric validation were hardened",
        )
        for label, legacy_hash, current_hash in (
            (
                "selection-seed0",
                "bed265cdefaaf8465bbd294a1e16e8ce177bebdb0e445a3c33d63da909b7d328",
                "e513d57bcc26b01e18895401983e76d43e4fd04ff4785c34e8eaa6233bd150db",
            ),
            (
                "final-seeds12",
                "48f4f243140a1c40b8858d0dfaf35563445270edca07df8c4a4f7b5382ce480f",
                "e513d57bcc26b01e18895401983e76d43e4fd04ff4785c34e8eaa6233bd150db",
            ),
        )
    ),
    "BrainSegFounder": (
        _runtime_migration(
            "brainsegfounder-post-review-v1",
            changed={
                "FineTuning/BrainSegFounder/finetune.py": (
                    "c162123d7273effda333bc173df2ca336cc1e186f4d08441e8825b53f298f69f",
                    "2e9a2d1857ee3e26f5ca774fd0894b5087619f6b9b6f279511d40d59ce70c36f",
                ),
                "FineTuning/shared/metrics.py": (
                    _OLD_METRICS_SHA256,
                    _CURRENT_METRICS_SHA256,
                ),
            },
            reason="post-training evaluation guards and metric validation were hardened",
        ),
    ),
    "Curia-2": (
        _runtime_migration(
            "curia2-post-review-v1",
            changed={
                "FineTuning/Curia2/finetune.py": (
                    "df8ee990e880a16029b86362eb9f08100638b4033dae0de28e5f290f7baadabc",
                    "202809a4dae749c9ca3276b8475cce9a46b4884d0642e2604b9bf02eb8b1f925",
                ),
                "FineTuning/shared/metrics.py": (
                    _OLD_METRICS_SHA256,
                    _CURRENT_METRICS_SHA256,
                ),
            },
            reason="internal-test orchestration and metric validation were hardened",
        ),
    ),
    "NeuroVFM": (
        _runtime_migration(
            "neurovfm-post-review-v1",
            changed={
                "FineTuning/NeuroVFM/finetune.py": (
                    "f2a10474132aa5e8863de1555ea150511d4247ef9bde6085b7c9b7315d92be0d",
                    "e902cfe74d9128bda8a90bdfaa4caa7397679f9e557399fad668ccee3d15d599",
                ),
                "FineTuning/shared/metrics.py": (
                    _OLD_METRICS_SHA256,
                    _CURRENT_METRICS_SHA256,
                ),
                "Emb_extractors/extract_NeuroVFM.py": (
                    "6fc5fd23f08d00fcebca597cd7090a9238df7de5b3541e63177d41066d207a32",
                    "e8fd04a5ddf986589af08726e1785a090a87c8a8d31fb4dcdbe9ca49b1bf739a",
                ),
            },
            reason="checkpoint loading, extractor provenance, and metric validation were hardened",
        ),
    ),
    "y-Aware": (
        _runtime_migration(
            "yaware-post-review-v1",
            changed={
                "FineTuning/yAware/finetune.py": (
                    "ac4b7073ac85ca32a85b4694f3fde713e9c5fa7811b6c3a09b36e32e2c1df209",
                    "5a92db2fbbb9968f77571d3232a4cebc0d64b71e63a0a512f1e7c3b941d8b7e5",
                ),
                "FineTuning/shared/metrics.py": (
                    _OLD_METRICS_SHA256,
                    _CURRENT_METRICS_SHA256,
                ),
            },
            added={"FineTuning/shared/evaluation.py": _EVALUATION_SHA256},
            reason="internal-test guards, provenance, and metric validation were hardened",
        ),
    ),
}


_SELECTION_RUNTIME_MIGRATIONS = {
    "BrainIAC": RuntimeMigration(
        changed={
            "FineTuning/BrainIAC/finetune.py": (
                "bed265cdefaaf8465bbd294a1e16e8ce177bebdb0e445a3c33d63da909b7d328",
                "48f4f243140a1c40b8858d0dfaf35563445270edca07df8c4a4f7b5382ce480f",
            )
        },
        added={},
        migration_id="brainiac-selection-seed0-to-final-seeds12-v1",
        reason="the final seeds were trained after evaluation-only guards were added",
    )
}


def _current_migration_markers(migration: RuntimeMigration) -> dict[str, str]:
    markers = {key: new for key, (_, new) in migration.changed.items()}
    markers.update(migration.added)
    return markers


def _verify_current_migration_targets(migration: RuntimeMigration) -> None:
    """Fail if an approved migration target was edited after being pinned."""

    for key, expected_hash in _current_migration_markers(migration).items():
        path = PROJECT_ROOT / key
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Migration target changed after pinning: {key}; "
                f"expected={expected_hash}, observed={observed_hash}"
            )


def adapted_runtime_migration(
    model_name: str,
    locked_code: Mapping[str, str],
) -> RuntimeMigration | None:
    """Resolve one exact retained-checkpoint to current-runtime transition."""

    if not isinstance(locked_code, Mapping) or not locked_code:
        raise ValueError(f"Locked {model_name} runtime lacks a code-hash mapping")
    try:
        candidates = _ADAPTED_RUNTIME_MIGRATIONS[model_name]
    except KeyError as error:
        raise ValueError(f"No adapted runtime contract for {model_name!r}") from error

    current_markers = _current_migration_markers(candidates[0])
    if all(locked_code.get(key) == value for key, value in current_markers.items()):
        return None

    matches = [
        migration
        for migration in candidates
        if all(
            locked_code.get(key) == old_hash
            for key, (old_hash, _) in migration.changed.items()
        )
        and all(key not in locked_code for key in migration.added)
    ]
    if len(matches) != 1:
        observed = {
            key: locked_code.get(key)
            for key in sorted(
                {
                    candidate_key
                    for migration in candidates
                    for candidate_key in (
                        set(migration.changed) | set(migration.added)
                    )
                }
            )
        }
        raise ValueError(
            f"No unique approved {model_name} runtime migration for {observed}"
        )
    migration = matches[0]
    _verify_current_migration_targets(migration)
    return migration


def adapted_selection_runtime_migration(
    model_name: str,
) -> RuntimeMigration | None:
    """Return a separately pinned selection-anchor to final-run transition."""

    return _SELECTION_RUNTIME_MIGRATIONS.get(model_name)


def validate_prediction_runtime_code(
    locked: Mapping[str, str],
    current: Mapping[str, str],
    migration: RuntimeMigration | None = None,
) -> dict[str, Any]:
    """Accept exact code or a fully hash-pinned evaluation-only migration."""

    locked_dict = dict(locked)
    current_dict = dict(current)
    if current_dict == locked_dict:
        return {"mode": "exact_training_runtime", "code_sha256": current_dict}
    changed = sorted(
        key
        for key in set(locked_dict) & set(current_dict)
        if locked_dict[key] != current_dict[key]
    )
    added = sorted(set(current_dict) - set(locked_dict))
    removed = sorted(set(locked_dict) - set(current_dict))
    if migration is None:
        raise ValueError(
            "Runtime code differs from the adapted checkpoint: "
            f"changed={changed}, added={added}, removed={removed}"
        )
    if changed != sorted(migration.changed) or added != sorted(migration.added) or removed:
        raise ValueError(
            "Runtime drift exceeds the approved prediction migration: "
            f"changed={changed}, added={added}, removed={removed}"
        )
    for key, (old_hash, new_hash) in migration.changed.items():
        if locked_dict.get(key) != old_hash or current_dict.get(key) != new_hash:
            raise ValueError(f"Unapproved runtime hash transition for {key}")
    for key, expected_hash in migration.added.items():
        if current_dict.get(key) != expected_hash:
            raise ValueError(f"Unapproved added runtime dependency for {key}")
    return {
        "mode": "hash_pinned_evaluation_only_migration",
        "migration_id": migration.migration_id,
        "reason": migration.reason,
        "migration_registry_code": {
            str(Path(__file__).resolve().relative_to(PROJECT_ROOT)): sha256_file(
                Path(__file__).resolve()
            )
        },
        "changed_files": changed,
        "added_files": added,
        "locked_code_sha256": locked_dict,
        "prediction_code_sha256": current_dict,
    }


def validate_lock_header(
    run_dir: Path | str,
    *,
    expected_model: str,
    expected_variant: str,
    checkpoint_name: str = CHECKPOINT_NAME,
    lock_name: str = LOCK_NAME,
) -> tuple[dict[str, Any], Path, Path]:
    """Require an untouched validation-only adapted checkpoint and lock."""

    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint_path = run_dir / checkpoint_name
    lock_path = run_dir / lock_name
    if not checkpoint_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError(f"Expected {checkpoint_path} and {lock_path}")
    lock_sha256_before = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_sha256_after = sha256_file(lock_path)
    if lock_sha256_after != lock_sha256_before:
        raise RuntimeError("Adapted LOCKED.json changed while it was being read")
    expected = {
        "status": "LOCKED_AFTER_TRAIN_VALIDATION",
        "arm": "adapted",
        "model": expected_model,
        "variant": expected_variant,
        "internal_test_accessed": False,
        "on_harmony_accessed": False,
    }
    mismatches = {
        key: {"expected": value, "observed": lock.get(key)}
        for key, value in expected.items()
        if lock.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Invalid validation-only adapted lock: {mismatches}")
    observed_hash = sha256_file(checkpoint_path)
    if observed_hash != lock.get("checkpoint_sha256"):
        raise ValueError("Adapted checkpoint hash differs from LOCKED.json")
    return lock, checkpoint_path, lock_path


def capture_prediction_artifact_integrity(
    *,
    lock: Mapping[str, Any],
    lock_path: Path | str,
    checkpoint_path: Path | str,
    selection: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Capture one coherent lock, checkpoint, and LR-selection snapshot."""

    resolved_lock = Path(lock_path).expanduser().resolve()
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    selection_path = Path(str(selection.get("selection_artifact", ""))).expanduser().resolve()
    if not resolved_lock.is_file() or not resolved_checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing adapted integrity artifact: {resolved_lock}, {resolved_checkpoint}"
        )
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)

    lock_sha256_before = sha256_file(resolved_lock)
    observed_lock = json.loads(resolved_lock.read_text(encoding="utf-8"))
    lock_sha256_after = sha256_file(resolved_lock)
    if lock_sha256_after != lock_sha256_before or observed_lock != dict(lock):
        raise RuntimeError("Adapted LOCKED.json changed during run binding")

    checkpoint_sha256 = sha256_file(resolved_checkpoint)
    if checkpoint_sha256 != lock.get("checkpoint_sha256"):
        raise RuntimeError("Adapted checkpoint changed during run binding")

    selection_sha256_before = sha256_file(selection_path)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_sha256_after = sha256_file(selection_path)
    if selection_sha256_after != selection_sha256_before:
        raise RuntimeError("SELECTED_LR.json changed while it was being read")
    if selection_sha256_after != selection.get("selection_artifact_sha256"):
        raise RuntimeError("SELECTED_LR.json changed during run binding")
    expected_selection_fields = {
        "model": selection.get("selection_model"),
        "variant": selection.get("selection_variant"),
        "selection_seed": selection.get("selection_seed"),
        "selected_lr_config": selection.get("selected_lr_config"),
        "selected_checkpoint_sha256": selection.get("selection_checkpoint_sha256"),
    }
    mismatches = {
        key: {"expected": expected, "observed": selection_payload.get(key)}
        for key, expected in expected_selection_fields.items()
        if selection_payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"SELECTED_LR.json differs from the validated LR selection: {mismatches}"
        )

    return {
        "training_lock": {
            "path": str(resolved_lock),
            "sha256": lock_sha256_after,
        },
        "adapted_checkpoint": {
            "path": str(resolved_checkpoint),
            "sha256": checkpoint_sha256,
        },
        "lr_selection": {
            "path": str(selection_path),
            "sha256": selection_sha256_after,
        },
    }


def verify_prediction_artifact_integrity(
    integrity: Mapping[str, Mapping[str, str]],
) -> None:
    """Reject any artifact drift after binding and before result publication."""

    required = ("training_lock", "adapted_checkpoint", "lr_selection")
    missing = [label for label in required if label not in integrity]
    if missing:
        raise ValueError(f"Prediction binding lacks integrity artifacts: {missing}")
    for label in required:
        artifact = integrity[label]
        path = Path(str(artifact.get("path", ""))).expanduser().resolve()
        expected_sha256 = str(artifact.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"{label} changed during ON-Harmony prediction")


def locked_adapter(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Read the adapter contract from either generation of training lock."""

    adapter = lock.get("adapter")
    if adapter is None:
        adapter = lock.get("run_config", {}).get("model")
    if not isinstance(adapter, dict):
        raise ValueError("Adapted checkpoint lacks a model adapter contract")
    return dict(adapter)


def selection_compatible_lock(
    lock: Mapping[str, Any],
    *,
    selection_dir: Path | str,
    migration: RuntimeMigration | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize one explicitly pinned run/selection code transition for validation."""

    selection_path = Path(selection_dir).expanduser().resolve() / SELECTION_NAME
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_run = Path(str(selection.get("selected_run_dir", ""))).expanduser().resolve()
    selected_lock_path = selected_run / LOCK_NAME
    if not selected_lock_path.is_file():
        raise FileNotFoundError(selected_lock_path)
    selected_lock = json.loads(selected_lock_path.read_text(encoding="utf-8"))
    selected_code = dict(selected_lock.get("code_sha256", {}))
    evaluation_code = dict(lock.get("code_sha256", {}))
    if evaluation_code == selected_code:
        return dict(lock), {
            "mode": "exact_selection_anchor_runtime",
            "code_sha256": evaluation_code,
        }
    if migration is None:
        raise ValueError(
            "Evaluation run code differs from the LR-selection anchor without "
            "an approved migration"
        )
    provenance = validate_prediction_runtime_code(
        selected_code,
        evaluation_code,
        migration,
    )
    normalized = dict(lock)
    # validate_selected_evaluation_run still performs every checkpoint, LR,
    # manifest, release, and seed check; only this hash-pinned difference is
    # presented as equivalent to its seed-0 selection anchor.
    normalized["code_sha256"] = selected_code
    return normalized, provenance


def validate_run_binding(
    *,
    run_dir: Path | str,
    selection_dir: Path | str,
    model: Any,
    training_manifest_loader: Callable[[], tuple[Any, Mapping[str, Any]]],
    runtime_code_loader: Callable[[Any], Mapping[str, str]],
    expected_model: str,
    expected_variant: str,
    selection_variant: str | None = None,
    release_normalizer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    adapter_normalizer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    runtime_migration: RuntimeMigration | None = None,
    selection_runtime_migration: RuntimeMigration | None = None,
    requested_amp: str | None = None,
) -> dict[str, Any]:
    """Bind external inference to validation-only LR selection and training state."""

    lock, checkpoint_path, lock_path = validate_lock_header(
        run_dir,
        expected_model=expected_model,
        expected_variant=expected_variant,
    )
    _, training_manifest = training_manifest_loader()
    manifest_sha256 = str(training_manifest["manifest_sha256"])
    if lock.get("manifest", {}).get("manifest_sha256") != manifest_sha256:
        raise ValueError("Adapted checkpoint uses a different training manifest")
    selection_lock, selection_runtime_provenance = selection_compatible_lock(
        lock,
        selection_dir=selection_dir,
        migration=selection_runtime_migration,
    )
    selection = validate_selected_evaluation_run(
        selection_dir=Path(selection_dir),
        run_dir=Path(run_dir),
        lock=selection_lock,
        manifest_sha256=manifest_sha256,
        expected_model=expected_model,
        evaluation_variant=expected_variant,
        selection_variant=selection_variant,
        checkpoint_name=CHECKPOINT_NAME,
        lock_name=LOCK_NAME,
    )
    selection["evaluation_vs_selection_runtime"] = selection_runtime_provenance
    normalize_release = release_normalizer or (lambda value: dict(value))
    normalize_adapter = adapter_normalizer or (lambda value: dict(value))
    release = model.release_provenance()
    if normalize_release(release) != normalize_release(lock.get("release", {})):
        raise ValueError("Base release differs from the adapted checkpoint")
    current_code = dict(runtime_code_loader(model))
    runtime_provenance = validate_prediction_runtime_code(
        lock.get("code_sha256", {}), current_code, runtime_migration
    )
    if normalize_adapter(model.audit_description()) != normalize_adapter(locked_adapter(lock)):
        raise ValueError("Training adapter differs from the adapted checkpoint")
    arguments = lock.get("run_config", {}).get("arguments", {})
    locked_amp = str(arguments.get("amp", ""))
    if locked_amp not in {"off", "float16", "bfloat16"}:
        raise ValueError("Adapted checkpoint lacks a valid AMP setting")
    if requested_amp is not None and requested_amp != locked_amp:
        raise ValueError(
            f"Requested AMP {requested_amp!r} differs from locked {locked_amp!r}"
        )
    try:
        seed = int(arguments["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Adapted checkpoint lacks a valid training seed") from error
    artifact_integrity = capture_prediction_artifact_integrity(
        lock=lock,
        lock_path=lock_path,
        checkpoint_path=checkpoint_path,
        selection=selection,
    )
    return {
        "lock": lock,
        "checkpoint_path": checkpoint_path,
        "lock_path": lock_path,
        "selection": selection,
        "amp": locked_amp,
        "seed": seed,
        "runtime_code_provenance": runtime_provenance,
        "training_manifest": dict(training_manifest),
        "artifact_integrity": artifact_integrity,
    }


def load_adapted_model(
    *,
    model: Any,
    restore_checkpoint: Callable[[Any, Any, Any, Path, Any], Mapping[str, Any]],
    checkpoint_path: Path,
    device_name: str,
    amp: str,
    seed: int,
    device_validator: Callable[[Any, Any, str], None] | None = None,
) -> dict[str, Any]:
    """Load the public encoder and restore only the adapted boundary and age head."""

    torch = import_torch()
    device = resolve_device(torch, device_name)
    if device_validator is not None:
        device_validator(torch, device, amp)
    seed_torch(torch, seed)
    model.load(device)
    boundary = model.set_controlled_adaptation()
    head = torch.nn.Linear(model.embedding_dim, 1).to(device)
    checkpoint = dict(restore_checkpoint(torch, model, head, checkpoint_path, device))
    arguments = checkpoint.get("run_config", {}).get("arguments", {})
    if arguments.get("amp") != amp or int(arguments.get("seed")) != seed:
        raise ValueError("Checkpoint payload and LOCKED.json disagree on runtime settings")
    age_stats = AgeStandardizer(**checkpoint["age_standardizer"])
    if not math.isfinite(age_stats.mean) or not math.isfinite(age_stats.std) or age_stats.std <= 0:
        raise ValueError("Checkpoint contains an invalid age standardizer")
    model.controlled_train_mode()
    head.eval()
    return {
        "model": model,
        "torch": torch,
        "device": device,
        "head": head,
        "checkpoint": checkpoint,
        "boundary": boundary,
        "age_standardizer": age_stats,
    }


def validate_prediction_frame(predictions: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Require one finite, label-free prediction in canonical acquisition order."""

    if FORBIDDEN_LABEL_COLUMNS & set(predictions.columns):
        raise ValueError("ON-Harmony predictions contain labels or prediction errors")
    if len(predictions) != len(expected):
        raise ValueError("ON-Harmony prediction count differs from its manifest")
    if predictions["acquisition_id"].duplicated().any():
        raise ValueError("ON-Harmony predictions contain duplicate acquisitions")
    for column in ("subject", "session", "acquisition_id"):
        if predictions[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError(f"ON-Harmony prediction order differs in {column}")
    values = predictions["age_pred"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ON-Harmony predictions contain non-finite ages")


def validate_artifacts(registry: ArtifactRegistry, frame: pd.DataFrame) -> None:
    """Validate, hash, and freeze every artifact used by this model execution."""

    for record in frame.to_dict(orient="records"):
        registry.freeze(record)


def merge_prediction_provenance(
    prediction: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach provenance without allowing it to replace protected prediction fields."""

    collisions = sorted(set(prediction) & set(provenance))
    if collisions:
        raise ValueError(
            f"Input provenance cannot override prediction fields: {collisions}"
        )
    return {**dict(prediction), **dict(provenance)}


def predict_frame(
    *,
    runtime: Mapping[str, Any],
    model: Any,
    registry: ArtifactRegistry,
    frame: pd.DataFrame,
    amp: str,
    seed: int,
    checkpoint_sha256: str,
    lr_config: str,
) -> pd.DataFrame:
    """Generate chronological-age predictions without loading outcome labels."""

    torch = runtime["torch"]
    device = runtime["device"]
    head = runtime["head"]
    age_stats = runtime["age_standardizer"]
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    validate_artifacts(registry, frame)
    with torch.no_grad():
        for position, record in enumerate(frame.to_dict(orient="records"), start=1):
            provenance = registry.validate(record)
            with autocast_context(torch, device, amp):
                embedding = model.encode(record, training=False, epoch=0, seed=seed)
                prediction_z = head(embedding).reshape(())
            prediction_age = age_stats.inverse(float(prediction_z.detach().float().item()))
            if not math.isfinite(prediction_age):
                raise RuntimeError(f"Non-finite prediction for {record['acquisition_id']}")
            rows.append(
                merge_prediction_provenance(
                    {
                        "model": model.model_name,
                        "variant": model.variant,
                        "arm": "adapted",
                        "training_seed": seed,
                        "lr_config": lr_config,
                        "checkpoint_sha256": checkpoint_sha256,
                        "subject": record["subject"],
                        "session": record["session"],
                        "acquisition_id": record["acquisition_id"],
                        "cohort": record["cohort"],
                        "scanner_code": record["scanner_code"],
                        "scanner_name": record["scanner_name"],
                        "vendor": record["vendor"],
                        "scanner_type": record["scanner_type"],
                        "scanner_typeloc": record["scanner_typeloc"],
                        "scan_session_num": record["scan_session_num"],
                        "age_pred": prediction_age,
                    },
                    provenance,
                )
            )
            if position % 25 == 0 or position == len(frame):
                print(
                    f"  [ON-Harmony {position}/{len(frame)}] {record['acquisition_id']}",
                    flush=True,
                )
    result = pd.DataFrame(rows)
    result.attrs["elapsed_seconds"] = float(time.monotonic() - started)
    validate_artifacts(registry, frame)
    validate_prediction_frame(result, frame)
    return result


def prediction_code_hashes(script_path: Path) -> dict[str, str]:
    """Hash the model entrypoint and this shared external-inference implementation."""

    return code_hashes([Path(script_path), Path(__file__).resolve()])


def audit_payload(
    *,
    pipeline_version: str,
    model: Any,
    frame_summary: Mapping[str, Any],
    selected_count: int,
    input_contract: Mapping[str, Any],
    binding: Mapping[str, Any],
    script_path: Path,
    files_checked: bool,
) -> dict[str, Any]:
    """Build the common label-blind audit record before an optional smoke forward."""

    verify_prediction_artifact_integrity(binding["artifact_integrity"])
    lock = binding["lock"]
    integrity = binding["artifact_integrity"]
    return {
        "pipeline_version": pipeline_version,
        "status": "PASS",
        "label_blind": True,
        "outcome_labels_loaded": False,
        "model": model.model_name,
        "variant": model.variant,
        "on_harmony_manifest": dict(frame_summary),
        "selected_acquisitions": int(selected_count),
        "external_input_contract": dict(input_contract),
        "adapted_run": str(binding["checkpoint_path"].parent),
        "adapted_checkpoint": str(binding["checkpoint_path"]),
        "adapted_checkpoint_sha256": integrity["adapted_checkpoint"]["sha256"],
        "training_lock_sha256": integrity["training_lock"]["sha256"],
        "training_seed": binding["seed"],
        "amp": binding["amp"],
        "lr_selection": binding["selection"],
        "training_runtime_code": lock["code_sha256"],
        "runtime_code_provenance": binding["runtime_code_provenance"],
        "prediction_runtime_code": prediction_code_hashes(script_path),
        "files_checked": bool(files_checked),
    }


def attach_runtime_audit(payload: dict[str, Any], runtime: Mapping[str, Any]) -> None:
    """Add model-load, adaptation-boundary, software, and GPU details in place."""

    torch = runtime["torch"]
    device = runtime["device"]
    load_description = getattr(runtime["model"], "load_description", None)
    payload.update(
        {
            "software": package_versions(),
            "runtime": {
                "device": str(device),
                "gpu_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
                "effective_precision": effective_precision(device, str(payload["amp"])),
            },
            "controlled_adaptation": runtime["boundary"],
        }
    )
    if callable(load_description):
        payload["checkpoint_load"] = load_description()


def publish_predictions(
    *,
    output_dir: Path | str,
    pipeline_version: str,
    model: Any,
    predictions: pd.DataFrame,
    manifest: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    binding: Mapping[str, Any],
    runtime: Mapping[str, Any],
    script_path: Path,
    extra_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish predictions and their immutable provenance record."""

    integrity = binding["artifact_integrity"]
    verify_prediction_artifact_integrity(integrity)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output path already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    staging.mkdir(exist_ok=False)
    try:
        staged_predictions_path = staging / "predictions.tsv"
        final_predictions_path = destination / "predictions.tsv"
        atomic_tsv(predictions, staged_predictions_path)
        lock = binding["lock"]
        torch = runtime["torch"]
        device = runtime["device"]
        audit: dict[str, Any] = {
            "pipeline_version": pipeline_version,
            "status": "ON_HARMONY_PREDICTED",
            "arm": "adapted",
            "model": model.model_name,
            "variant": model.variant,
            "label_blind": True,
            "outcome_labels_loaded": False,
            "bias_correction": "none",
            "n_predictions": int(len(predictions)),
            "n_subjects": int(predictions["subject"].nunique()),
            "elapsed_seconds": float(predictions.attrs["elapsed_seconds"]),
            "training_seed": binding["seed"],
            "amp": binding["amp"],
            "adapted_run": str(binding["checkpoint_path"].parent),
            "adapted_checkpoint": str(binding["checkpoint_path"]),
            "adapted_checkpoint_sha256": integrity["adapted_checkpoint"]["sha256"],
            "training_lock_sha256": integrity["training_lock"]["sha256"],
            "lr_selection": binding["selection"],
            "runtime_code_provenance": binding["runtime_code_provenance"],
            "on_harmony_manifest": dict(manifest),
            "external_input_contract": dict(input_contract),
            "training_runtime_code": lock["code_sha256"],
            "prediction_runtime_code": prediction_code_hashes(script_path),
            "controlled_adaptation": runtime["boundary"],
            "software": package_versions(),
            "runtime": {
                "device": str(device),
                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "effective_precision": effective_precision(device, binding["amp"]),
            },
            "predictions_artifact": str(final_predictions_path),
            "predictions_artifact_sha256": sha256_file(staged_predictions_path),
            "internal_test_accessed": False,
            "on_harmony_accessed": True,
        }
        additions = dict(extra_audit or {})
        collisions = sorted(set(audit) & set(additions))
        if collisions:
            raise ValueError(
                f"extra_audit cannot override protected fields: {collisions}"
            )
        audit.update(additions)
        atomic_json(audit, staging / "prediction_audit.json")
        verify_prediction_artifact_integrity(integrity)
        if destination.exists():
            raise FileExistsError(
                f"Output path appeared during inference: {destination}"
            )
        os.replace(staging, destination)
        return audit
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_standard_audit(
    args: Any,
    *,
    context: Mapping[str, Any],
    pipeline_version: str,
    script_path: Path,
    runtime_factory: Callable[[], dict[str, Any]],
) -> None:
    """Execute the common audit/smoke workflow for one model-specific context."""

    selected = select_audit_records(
        context["frame"], subject=args.subject, session=args.session
    )
    if args.smoke_forward and len(selected) != 1:
        raise ValueError("--smoke-forward requires one selected acquisition")
    if args.check_input_files:
        validate_artifacts(context["registry"], selected)
    result = audit_payload(
        pipeline_version=pipeline_version,
        model=context["model"],
        frame_summary=context["manifest"],
        selected_count=len(selected),
        input_contract=context["input_contract"],
        binding=context["binding"],
        script_path=script_path,
        files_checked=args.check_input_files or args.smoke_forward,
    )
    if args.inspect_model or args.smoke_forward:
        runtime = runtime_factory()
        attach_runtime_audit(result, runtime)
        if args.smoke_forward:
            binding = context["binding"]
            smoke = predict_frame(
                runtime=runtime,
                model=context["model"],
                registry=context["registry"],
                frame=selected,
                amp=binding["amp"],
                seed=binding["seed"],
                checkpoint_sha256=binding["lock"]["checkpoint_sha256"],
                lr_config=binding["selection"]["selected_lr_config"],
            )
            result["smoke_forward"] = {
                "status": "PASS",
                "acquisition_id": str(smoke.iloc[0]["acquisition_id"]),
                "prediction_is_finite": True,
                "outcome_labels_loaded": False,
            }
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def run_standard_prediction(
    args: Any,
    *,
    context: Mapping[str, Any],
    pipeline_version: str,
    script_path: Path,
    runtime_factory: Callable[[], dict[str, Any]],
    extra_audit: (
        Mapping[str, Any]
        | Callable[[Mapping[str, Any], pd.DataFrame], Mapping[str, Any]]
        | None
    ) = None,
) -> None:
    """Execute and atomically publish one full 165-acquisition prediction run."""

    validate_artifacts(context["registry"], context["frame"])
    runtime = runtime_factory()
    binding = context["binding"]
    predictions = predict_frame(
        runtime=runtime,
        model=context["model"],
        registry=context["registry"],
        frame=context["frame"],
        amp=binding["amp"],
        seed=binding["seed"],
        checkpoint_sha256=binding["lock"]["checkpoint_sha256"],
        lr_config=binding["selection"]["selected_lr_config"],
    )
    resolved_extra_audit = (
        dict(extra_audit(runtime, predictions))
        if callable(extra_audit)
        else extra_audit
    )
    audit = publish_predictions(
        output_dir=args.output_dir,
        pipeline_version=pipeline_version,
        model=context["model"],
        predictions=predictions,
        manifest=context["manifest"],
        input_contract=context["input_contract"],
        binding=binding,
        runtime=runtime,
        script_path=script_path,
        extra_audit=resolved_extra_audit,
    )
    print(
        f"Saved {len(predictions)} label-free predictions: {audit['predictions_artifact']}\n"
        f"SHA-256: {audit['predictions_artifact_sha256']}",
        flush=True,
    )


def source_contract(
    *,
    contract_version: str,
    paths: Mapping[str, Path | str],
    expected_sha256: Mapping[str, str] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash preprocessing sources and optionally enforce their audited identities."""

    expected = dict(expected_sha256 or {})
    result: dict[str, Any] = {"contract_version": contract_version, **dict(details or {})}
    for label, raw_path in paths.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if label in expected and digest != expected[label]:
            raise ValueError(
                f"External preprocessing source drift for {label}: "
                f"expected {expected[label]}, observed {digest}"
            )
        result[label] = {"path": str(path), "sha256": digest}
    result["contract_sha256"] = canonical_sha256(result)
    return result
