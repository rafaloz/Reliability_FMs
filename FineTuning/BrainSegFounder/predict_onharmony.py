#!/usr/bin/env python3
"""Run label-blind ON-Harmony inference with adapted BrainSegFounder checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.BrainSegFounder.finetune import (  # noqa: E402
    restore_checkpoint,
    runtime_code_hashes,
)
from FineTuning.BrainSegFounder.model import (  # noqa: E402
    BrainSegFounderModel,
    BrainSegFounderOptions,
    load_and_validate_manifest,
)
from FineTuning.shared.cohort import sha256_file  # noqa: E402
from FineTuning.shared.onharmony import (  # noqa: E402
    ArtifactRegistry,
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    adapted_runtime_migration,
    load_adapted_model,
    load_onharmony_manifest,
    preprocessed_path,
    run_standard_audit,
    run_standard_prediction,
    source_contract,
    validate_lock_header,
    validate_run_binding,
)


PIPELINE_VERSION = "brainsegfounder_onharmony_prediction_v1"
INPUT_CONTRACT_VERSION = "brainsegfounder-ukb-gdcoff-onharmony-v4"
INPUT_CONTRACT_SHA256 = "08f983cc00900d7c57015eb8482eb763f90bd2eeb9bd960aa06563020dc7e30b"
DEFAULT_PREPROC_ROOT = (
    PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "preproc_BrainSegFounder"
)
PREPROCESSOR = (
    PROJECT_ROOT
    / "preprocessings"
    / "OnHarmonyDataset"
    / "preprocessing_BrainSegFounder.py"
)
UKB_ROOT = PROJECT_ROOT / "templates" / "UKBiobank"
EXPECTED_SOURCE_HASHES = {
    "preprocessor": "419c0cf9c5822f8da6cfa3086f18ed08f17ad3b4b742801e42ec9a5844474629",
    "brain_mask": "4e599d7d0bd843251bb24391b0ddc636b73216f12a9ccc017a34cb0d17e287e4",
    "dilated_mask": "1365d6747d6878501f97551ce44f143f1662b9dce3ab3f362243e5a456c5725d",
    "fnirt_config": "41e2f1b068e4841c8f6d7934375651fbe2cf2e66ac40da3d21f3fbb9b9576e37",
}
# Preserve the exact generator identities of the 165 existing v4 outputs.
# The current source adds only integrated run/validate orchestration; all three
# identities execute the same scientific contract and are kept distinguishable.
ALLOWED_GENERATOR_HASHES = {
    EXPECTED_SOURCE_HASHES["preprocessor"],
    "9981ad301a771074adbb512610f5a2d4cfb55f333cae581a6ccc9900c8f69966",
    "8759774c74f4a7012d515ea6a34441dba37440aef3c39a34a0cf88c6a73dd46b",
}


def artifact_paths(record: Mapping[str, Any], preproc_root: Path) -> dict[str, Path]:
    directory = preproc_root / str(record["acquisition_id"])
    return {
        "input": preprocessed_path(
            preproc_root, record, "_brainsegfounder_ready.nii.gz"
        ),
        "qc": directory / "preprocessing_qc.json",
        "preprocessing_log": directory / "preprocessing.log",
    }


def validate_artifact_set(
    record: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    qc = json.loads(paths["qc"].read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "pipeline_version": INPUT_CONTRACT_VERSION,
        "pipeline_contract_sha256": INPUT_CONTRACT_SHA256,
    }
    mismatches = {
        key: {"expected": value, "observed": qc.get(key)}
        for key, value in expected.items()
        if qc.get(key) != value
    }
    generator_hash = qc.get("script_sha256")
    if generator_hash not in ALLOWED_GENERATOR_HASHES:
        mismatches["script_sha256"] = {
            "expected_one_of": sorted(ALLOWED_GENERATOR_HASHES),
            "observed": generator_hash,
        }
    raw_path = Path(str(record["raw_t1w_path"])).expanduser().resolve()
    raw_stat = raw_path.stat()
    expected_raw_source = {
        "path": str(raw_path),
        "size_bytes": int(raw_stat.st_size),
        "mtime_ns": int(raw_stat.st_mtime_ns),
    }
    raw_source = qc.get("raw_input")
    if raw_source != expected_raw_source:
        mismatches["raw_input"] = {
            "expected": expected_raw_source,
            "observed": raw_source,
        }
    observed_hash = sha256_file(paths["input"])
    if qc.get("output_sha256", {}).get("ready") != observed_hash:
        mismatches["output_sha256.ready"] = {
            "expected": observed_hash,
            "observed": qc.get("output_sha256", {}).get("ready"),
        }
    model_view = qc.get("model_view", {})
    if model_view.get("shape") != [96, 96, 96]:
        mismatches["model_view.shape"] = {
            "expected": [96, 96, 96],
            "observed": model_view.get("shape"),
        }
    if mismatches:
        raise ValueError(f"Invalid ON-Harmony BrainSegFounder QC: {mismatches}")
    log = paths["preprocessing_log"].read_text(encoding="utf-8", errors="replace")
    required = (
        f"pipeline_version={INPUT_CONTRACT_VERSION}",
        f"contract={INPUT_CONTRACT_SHA256}",
        "/bin/fnirt ",
        "/bin/convertwarp ",
        "/bin/applywarp ",
    )
    missing = [fragment for fragment in required if fragment not in log]
    if missing:
        raise ValueError(
            "BrainSegFounder preprocessing log is not a successful v4 run: "
            f"missing={missing}"
        )
    return {
        "preprocessing_status": "PASS",
        "preprocessing_pipeline_version": INPUT_CONTRACT_VERSION,
        "preprocessing_contract_sha256": INPUT_CONTRACT_SHA256,
        "preprocessing_script_sha256": str(generator_hash),
        "registration_recovery_used": bool(
            qc.get("registration_recovery", {}).get("used", False)
        ),
    }


class OnHarmonyBrainSegFounderModel(BrainSegFounderModel):
    """Reuse the locked Swin encoder while replacing external input provenance."""

    def __init__(
        self, options: BrainSegFounderOptions, registry: ArtifactRegistry
    ) -> None:
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
        paths={
            "preprocessor": PREPROCESSOR,
            "brain_mask": UKB_ROOT / "MNI152_T1_1mm_brain_mask.nii.gz",
            "dilated_mask": UKB_ROOT / "MNI152_T1_1mm_brain_mask_dil_GD7.nii.gz",
            "fnirt_config": UKB_ROOT / "bb_fnirt.cnf",
        },
        expected_sha256=EXPECTED_SOURCE_HASHES,
        details={
            "pipeline_contract_sha256": INPUT_CONTRACT_SHA256,
            "gradient_distortion_correction": "OFF; scanner coefficients unavailable",
            "model_input_shape": [96, 96, 96],
            "raw_filelist_sha256": manifest["filelist_sha256"],
            "orchestration": "integrated run/validate commands in preprocessor",
        },
    )
    preproc_root = args.preproc_root.expanduser().resolve()
    registry = ArtifactRegistry(
        contract_version=INPUT_CONTRACT_VERSION,
        resolver=lambda record: artifact_paths(record, preproc_root),
        validator=validate_artifact_set,
        contract=input_contract,
    )
    lock, _, _ = validate_lock_header(
        args.run_dir,
        expected_model="BrainSegFounder",
        expected_variant="bottleneck_mean",
    )
    locked_args = lock.get("run_config", {}).get("arguments", {})
    model = OnHarmonyBrainSegFounderModel(
        BrainSegFounderOptions(
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
        expected_model="BrainSegFounder",
        expected_variant="bottleneck_mean",
        runtime_migration=adapted_runtime_migration(
            model.model_name, lock.get("code_sha256", {})
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
    recoveries = sum(
        bool(context["registry"].validate(record).get("registration_recovery_used"))
        for record in context["frame"].to_dict(orient="records")
    )
    run_standard_prediction(
        args,
        context=context,
        pipeline_version=PIPELINE_VERSION,
        script_path=SCRIPT_PATH,
        runtime_factory=lambda: runtime_for(args, context),
        extra_audit={"registration_recovery_acquisitions": int(recoveries)},
    )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--filelist", type=Path, default=DEFAULT_RAW_FILELIST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--preproc-root", type=Path, default=DEFAULT_PREPROC_ROOT)
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
