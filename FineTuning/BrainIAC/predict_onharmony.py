#!/usr/bin/env python3
"""Run label-blind ON-Harmony inference with adapted BrainIAC checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.BrainIAC.finetune import restore_checkpoint, runtime_code_hashes  # noqa: E402
from FineTuning.BrainIAC.model import (  # noqa: E402
    BrainIACModel,
    BrainIACOptions,
    load_and_validate_manifest,
)
from FineTuning.shared.cohort import sha256_file  # noqa: E402
from FineTuning.shared.onharmony import (  # noqa: E402
    ArtifactRegistry,
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    adapted_runtime_migration,
    adapted_selection_runtime_migration,
    load_adapted_model,
    load_onharmony_manifest,
    preprocessed_path,
    run_standard_audit,
    run_standard_prediction,
    source_contract,
    validate_lock_header,
    validate_run_binding,
)


PIPELINE_VERSION = "brainiac_onharmony_prediction_v1"
INPUT_CONTRACT_VERSION = "onharmony_brainiac_release_aligned_v1"
PREPROCESSING_PIPELINE_VERSION = "brainiac-release-aligned-onharmony-v1"
PREPROCESSING_CONTRACT_SHA256 = (
    "d7315a8346a9fb900ba8392048d39f0ec99f77bb482eddb84025de247ce85dd2"
)
DEFAULT_PREPROC_ROOT = (
    PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "preproc_BrainIAC"
)
DEFAULT_LOG_ROOT = PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "logs_preprocs"
PREPROCESSOR = (
    PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "preprocessing_BrainIAC.py"
)
TEMPLATE = PROJECT_ROOT / "templates" / "temp_head.nii.gz"
EXPECTED_SOURCE_HASHES = {
    "preprocessor": "06124069330ae294271c74b7ca288f5f11b8e53cfc3dde20b2f583aaf580d1a0",
    "template": "d8eba10e9528a5212b78fb1be477d5e8229751bfec5e1e1a648dc81ff83a622e",
}
HISTORICAL_PREPROCESSOR_SHA256 = (
    "f298f8f0efab28e1b6a07746775c6b1ef5a37c16a4c9dbc3df707bdd52d8bdbb"
)
ALLOWED_GENERATOR_HASHES = {
    EXPECTED_SOURCE_HASHES["preprocessor"],
    HISTORICAL_PREPROCESSOR_SHA256,
}
def artifact_paths(
    record: Mapping[str, Any], preproc_root: Path, log_root: Path
) -> dict[str, Path]:
    acquisition = str(record["acquisition_id"])
    return {
        "input": preprocessed_path(preproc_root, record, "_brainiac_ready.nii.gz"),
        "brain": preprocessed_path(preproc_root, record, "_brain.nii.gz"),
        "qc": preprocessed_path(preproc_root, record, "_qc.json"),
        "qc_png": preprocessed_path(preproc_root, record, "_qc.png"),
        "preprocessing_log": log_root / f"brainiac_{acquisition}.log",
    }


def validate_artifact_set(
    record: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    qc = json.loads(paths["qc"].read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "raw_input": str(record["raw_t1w_path"]),
        "brainiac_shape": [96, 96, 96],
        "shape_ok": True,
        "norm_ok": True,
        "brain_ok": True,
    }
    mismatches = {
        key: {"expected": value, "observed": qc.get(key)}
        for key, value in expected.items()
        if qc.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Invalid ON-Harmony BrainIAC QC: {mismatches}")
    log = paths["preprocessing_log"].read_text(encoding="utf-8", errors="replace")
    generator_match = re.search(
        r"^script\s*:\s*([0-9a-f]{64})\s*$", log, flags=re.MULTILINE
    )
    generator_hash = generator_match.group(1) if generator_match else None
    if generator_hash not in ALLOWED_GENERATOR_HASHES:
        raise ValueError(
            "BrainIAC preprocessing log has an unrecognized generator SHA-256: "
            f"{generator_hash}"
        )
    required = (
        "[Preprocessing 4-5/5] Resample 96",
        "status=PASS",
        "All done.",
        "exit_code: 0",
    )
    missing = [fragment for fragment in required if fragment not in log]
    if missing:
        raise ValueError(
            f"BrainIAC preprocessing log is not a successful audited run: missing={missing}"
        )
    provenance_schema = "historical"
    if "pipeline_contract_sha256" in qc:
        provenance_schema = "current"
        current_expected = {
            "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
            "pipeline_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
            "script_sha256": EXPECTED_SOURCE_HASHES["preprocessor"],
        }
        current_mismatches = {
            key: {"expected": value, "observed": qc.get(key)}
            for key, value in current_expected.items()
            if qc.get(key) != value
        }
        if generator_hash != EXPECTED_SOURCE_HASHES["preprocessor"]:
            current_mismatches["log_generator"] = {
                "expected": EXPECTED_SOURCE_HASHES["preprocessor"],
                "observed": generator_hash,
            }
        raw_path = Path(str(record["raw_t1w_path"])).expanduser().resolve()
        raw_stat = raw_path.stat()
        raw_fingerprint = {
            "path": str(raw_path),
            "size_bytes": int(raw_stat.st_size),
            "mtime_ns": int(raw_stat.st_mtime_ns),
        }
        if qc.get("raw_fingerprint") != raw_fingerprint:
            current_mismatches["raw_fingerprint"] = {
                "expected": raw_fingerprint,
                "observed": qc.get("raw_fingerprint"),
            }
        output_sha256 = {
            "brain": sha256_file(paths["brain"]),
            "ready": sha256_file(paths["input"]),
            "qc_png": sha256_file(paths["qc_png"]),
        }
        if qc.get("output_sha256") != output_sha256:
            current_mismatches["output_sha256"] = {
                "expected": output_sha256,
                "observed": qc.get("output_sha256"),
            }
        if current_mismatches:
            raise ValueError(
                f"Invalid current ON-Harmony BrainIAC provenance: {current_mismatches}"
            )
    elif generator_hash != HISTORICAL_PREPROCESSOR_SHA256:
        raise ValueError("Legacy BrainIAC QC is not bound to the historical generator")
    return {
        "preprocessing_status": "PASS",
        "model_input_shape": "96x96x96",
        "preprocessing_script_sha256": str(generator_hash),
        "preprocessing_provenance_schema": provenance_schema,
    }


class OnHarmonyBrainIACModel(BrainIACModel):
    """Reuse the locked encoder while replacing only external input provenance."""

    def __init__(self, options: BrainIACOptions, registry: ArtifactRegistry) -> None:
        super().__init__(options)
        self.external_registry = registry

    def input_path(self, record: Any) -> Path:
        return self.external_registry.input_path(record)

    def required_input_paths(self, record: Any) -> list[Path]:
        return self.external_registry.required_paths(record)

    def validate_input_provenance(self, record: Any) -> dict[str, Any]:
        return self.external_registry.validate(record)


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    frame, manifest = load_onharmony_manifest(args.filelist, args.metadata)
    input_contract = source_contract(
        contract_version=INPUT_CONTRACT_VERSION,
        paths={"preprocessor": PREPROCESSOR, "template": TEMPLATE},
        expected_sha256=EXPECTED_SOURCE_HASHES,
        details={
            "model_input_shape": [96, 96, 96],
            "normalization": "nonzero z-score",
            "raw_filelist_sha256": manifest["filelist_sha256"],
            "accepted_historical_generator_sha256": HISTORICAL_PREPROCESSOR_SHA256,
        },
    )
    preproc_root = args.preproc_root.expanduser().resolve()
    log_root = args.preproc_log_root.expanduser().resolve()
    registry = ArtifactRegistry(
        contract_version=INPUT_CONTRACT_VERSION,
        resolver=lambda record: artifact_paths(record, preproc_root, log_root),
        validator=validate_artifact_set,
        contract=input_contract,
    )
    lock, _, _ = validate_lock_header(
        args.run_dir, expected_model="BrainIAC", expected_variant="canonical"
    )
    locked_args = lock.get("run_config", {}).get("arguments", {})
    model = OnHarmonyBrainIACModel(
        BrainIACOptions(
            checkpoint=Path(str(locked_args["checkpoint"])),
            preproc_root=Path(str(locked_args["preproc_root"])),
        ),
        registry,
    )
    binding = validate_run_binding(
        run_dir=args.run_dir,
        selection_dir=args.selection_dir,
        model=model,
        training_manifest_loader=load_and_validate_manifest,
        runtime_code_loader=runtime_code_hashes,
        expected_model="BrainIAC",
        expected_variant="canonical",
        runtime_migration=adapted_runtime_migration(
            model.model_name, lock.get("code_sha256", {})
        ),
        selection_runtime_migration=adapted_selection_runtime_migration(
            model.model_name
        ),
        requested_amp=args.amp,
    )
    return {
        "frame": frame,
        "manifest": manifest,
        "registry": registry,
        "model": model,
        "binding": binding,
        "input_contract": input_contract,
    }


def runtime_for(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    binding = context["binding"]
    return load_adapted_model(
        model=context["model"],
        restore_checkpoint=restore_checkpoint,
        checkpoint_path=binding["checkpoint_path"],
        device_name=args.device,
        amp=binding["amp"],
        seed=binding["seed"],
    )


def command_audit(args: argparse.Namespace) -> None:
    context = build_context(args)
    run_standard_audit(
        args,
        context=context,
        pipeline_version=PIPELINE_VERSION,
        script_path=SCRIPT_PATH,
        runtime_factory=lambda: runtime_for(args, context),
    )


def command_predict(args: argparse.Namespace) -> None:
    context = build_context(args)
    run_standard_prediction(
        args,
        context=context,
        pipeline_version=PIPELINE_VERSION,
        script_path=SCRIPT_PATH,
        runtime_factory=lambda: runtime_for(args, context),
    )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--filelist", type=Path, default=DEFAULT_RAW_FILELIST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--preproc-root", type=Path, default=DEFAULT_PREPROC_ROOT)
    parser.add_argument("--preproc-log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "float16", "bfloat16"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    add_common(audit)
    audit.add_argument("--subject")
    audit.add_argument("--session")
    audit.add_argument("--check-input-files", action="store_true")
    audit.add_argument("--inspect-model", action="store_true")
    audit.add_argument("--smoke-forward", action="store_true")
    audit.set_defaults(func=command_audit)
    predict = commands.add_parser("predict")
    add_common(predict)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.set_defaults(func=command_predict)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
