#!/usr/bin/env python3
"""Run label-blind ON-Harmony inference with adapted y-Aware checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from FineTuning.yAware.finetune import restore_checkpoint, runtime_code_hashes  # noqa: E402
from FineTuning.yAware.model import (  # noqa: E402
    YAwareModel,
    YAwareOptions,
    load_and_validate_manifest,
)


PIPELINE_VERSION = "yaware_onharmony_prediction_v1"
INPUT_CONTRACT_VERSION = "onharmony_yaware_cat12_ixi555_v1"
DEFAULT_PREPROC_ROOT = (
    PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "preproc_yAware"
)
DEFAULT_LOG_ROOT = PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "logs_preprocs"
PREPROCESSOR = PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "preprocessing_yAware.py"
EXPECTED_PREPROCESSOR_SHA256 = (
    "6180bd9db4905804c89d59d5b4a0e6571dea32dbbf8b79ab1bb51a65df82c33f"
)


def artifact_paths(
    record: Mapping[str, Any], preproc_root: Path, log_root: Path
) -> dict[str, Path]:
    acquisition = str(record["acquisition_id"])
    return {
        "input": preprocessed_path(preproc_root, record, "_yaware_ready.npy"),
        "preprocessing_log": log_root / f"yaware_corrected_{acquisition}.log",
    }


def validate_artifact_set(
    record: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    log = paths["preprocessing_log"].read_text(encoding="utf-8", errors="replace")
    expected_done_suffix = (
        f"/{record['acquisition_id']}/{paths['input'].name}"
    )
    done_paths = [
        line.strip()
        .removeprefix("Done:")
        .split("  shape=", maxsplit=1)[0]
        .strip()
        for line in log.splitlines()
        if line.strip().startswith("Done:")
    ]
    required = (
        "y-Aware preprocessing (IXI Dartel, vox=1.5)",
        "Z-score normalize + save as .npy",
        "shape=(121, 145, 121)",
        "shape=(1, 121, 145, 121)",
        "All done.",
    )
    missing = [fragment for fragment in required if fragment not in log]
    if not any(value.endswith(expected_done_suffix) for value in done_paths):
        missing.append(f"Done: *{expected_done_suffix}")
    if missing:
        raise ValueError(
            f"y-Aware preprocessing log is not a successful corrected run: missing={missing}"
        )
    return {
        "preprocessing_status": "PASS",
        "model_input_shape": "1x121x145x121",
        "normalization": "whole-volume z-score",
    }


class OnHarmonyYAwareModel(YAwareModel):
    """Reuse the locked encoder while replacing only external CAT12 provenance."""

    def __init__(self, options: YAwareOptions, registry: ArtifactRegistry) -> None:
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
        paths={"preprocessor": PREPROCESSOR},
        expected_sha256={"preprocessor": EXPECTED_PREPROCESSOR_SHA256},
        details={
            "tissue": "modulated normalized grey matter (mwp1)",
            "template": "IXI555 MNI152 Dartel",
            "model_input_shape": [1, 121, 145, 121],
            "normalization": "whole-volume z-score",
            "raw_filelist_sha256": manifest["filelist_sha256"],
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
        args.run_dir, expected_model="y-Aware", expected_variant="canonical"
    )
    locked_args = lock.get("run_config", {}).get("arguments", {})
    model = OnHarmonyYAwareModel(
        YAwareOptions(
            checkpoint=Path(str(locked_args["checkpoint"])),
            preproc_root=Path(str(locked_args["preproc_root"])),
            qc_root=Path(str(locked_args["qc_root"])),
        ),
        registry,
    )
    binding = validate_run_binding(
        run_dir=args.run_dir,
        selection_dir=args.selection_dir,
        model=model,
        training_manifest_loader=load_and_validate_manifest,
        runtime_code_loader=runtime_code_hashes,
        expected_model="y-Aware",
        expected_variant="canonical",
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
