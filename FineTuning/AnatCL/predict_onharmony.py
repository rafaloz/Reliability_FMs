#!/usr/bin/env python3
"""Run label-blind ON-Harmony inference with adapted AnatCL checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.AnatCL.finetune import restore_checkpoint, runtime_code_hashes  # noqa: E402
from FineTuning.AnatCL.model import (  # noqa: E402
    AnatCLModel,
    AnatCLOptions,
    load_and_validate_manifest,
    semantic_adapter_contract,
    semantic_release_contract,
)
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


PIPELINE_VERSION = "anatcl_onharmony_prediction_v2"
INPUT_CONTRACT_VERSION = "onharmony_anatcl_cat12_ixi555_v2"
DEFAULT_PREPROC_ROOT = (
    PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "preproc_AnatCL"
)
DEFAULT_LOG_ROOT = PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "logs_preprocs"
PREPROCESSOR = PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "preprocessing_AnatCL.py"
TEMPLATE = PROJECT_ROOT / "templates" / "Template_1_IXI555_MNI152.nii"
EXPECTED_PREPROCESSOR_SHA256 = (
    "1816e7385d67b637445d54e343b6311cd7bc9979c092a809839616048259a035"
)
EXPECTED_TEMPLATE_SHA256 = (
    "d681851db70c3e8619eb923b8a34a110962cf2daf6f8657fb5ba5d5469c6af9a"
)
REQUIRED_BATCH_FRAGMENTS = (
    "spm.tools.cat.estwrite.opts.affreg = 'mni';",
    "spm.tools.cat.estwrite.extopts.registration.dartel.darteltpm",
    "spm.tools.cat.estwrite.extopts.vox = 1.5;",
    "spm.tools.cat.estwrite.output.GM.mod = 1;",
)


def prediction_release_contract(details: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the pre-load release while preserving strict source identity."""

    normalized = semantic_release_contract(details)
    installed = normalized.pop("installed_source_sha256", None)
    expected = normalized.get("expected_source_sha256")
    if installed is not None and installed != expected:
        raise ValueError(
            "Locked AnatCL installed-source hashes differ from its expected release"
        )
    return normalized


def artifact_paths(
    record: Mapping[str, Any], preproc_root: Path, log_root: Path
) -> dict[str, Path]:
    acquisition = str(record["acquisition_id"])
    subject_dir = preproc_root / acquisition
    return {
        "input": preprocessed_path(preproc_root, record, "_anatcl_ready.nii.gz"),
        "preprocessing_log": log_root / f"anatcl_corrected_{acquisition}.log",
        "cat12_batch": subject_dir / "cat12_work" / "cat12_batch.m",
        "cat12_runtime_log": (
            subject_dir / "cat12_work" / "report" / f"catlog_{acquisition}.txt"
        ),
    }


def validate_artifact_set(
    record: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    log = paths["preprocessing_log"].read_text(encoding="utf-8", errors="replace")
    batch = paths["cat12_batch"].read_text(encoding="utf-8", errors="replace")
    catlog = paths["cat12_runtime_log"].read_text(
        encoding="utf-8", errors="replace"
    )
    expected_done_suffix = (
        f"/{record['acquisition_id']}/{paths['input'].name}"
    )
    done_paths = [
        line.strip().removeprefix("Done:").strip()
        for line in log.splitlines()
        if line.strip().startswith("Done:")
    ]
    required = (
        "AnatCL preprocessing (IXI Dartel, vox=1.5)",
        "Saving CAT12 output (native shape, no resample)",
        "Shape: (121, 145, 121)",
        "Zooms: (1.5, 1.5, 1.5)",
        "All done.",
    )
    missing = [fragment for fragment in required if fragment not in log]
    missing.extend(
        f"CAT12 batch: {fragment}"
        for fragment in REQUIRED_BATCH_FRAGMENTS
        if fragment not in batch
    )
    if "dartel registration" not in catlog.lower():
        missing.append("CAT12 runtime log: Dartel registration")
    if "CAT12." not in catlog:
        missing.append("CAT12 runtime log: CAT12 version")
    if not any(value.endswith(expected_done_suffix) for value in done_paths):
        missing.append(f"Done: *{expected_done_suffix}")
    if missing:
        raise ValueError(
            f"AnatCL preprocessing log is not a successful corrected run: missing={missing}"
        )
    return {
        "preprocessing_status": "PASS",
        "shared_input_shape": "121x145x121",
        "runtime_crop_y": "8:136",
        "historical_execution_evidence": "cat12_batch.m + catlog + wrapper log",
        "historical_python_generator_available": False,
    }


class OnHarmonyAnatCLModel(AnatCLModel):
    """Reuse one locked fold while replacing only external CAT12 provenance."""

    def __init__(self, options: AnatCLOptions, registry: ArtifactRegistry) -> None:
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
            "reference_preprocessor": PREPROCESSOR,
            "template_reference": TEMPLATE,
        },
        expected_sha256={
            "reference_preprocessor": EXPECTED_PREPROCESSOR_SHA256,
            "template_reference": EXPECTED_TEMPLATE_SHA256,
        },
        details={
            "tissue": "modulated normalized grey matter (mwp1)",
            "template": "IXI555 MNI152 Dartel",
            "shared_shape": [121, 145, 121],
            "spacing_mm": [1.5, 1.5, 1.5],
            "runtime_crop_y": [8, 136],
            "raw_filelist_sha256": manifest["filelist_sha256"],
            "reference_implementation_is_historical_generator": False,
            "historical_generator_source_available": False,
            "material_execution_evidence": [
                "cat12_batch",
                "cat12_runtime_log",
                "preprocessing_log",
                "input",
            ],
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
    provisional_lock = Path(args.run_dir).expanduser().resolve() / "LOCKED.json"
    if not provisional_lock.is_file():
        raise FileNotFoundError(provisional_lock)
    import json

    lock_data = json.loads(provisional_lock.read_text(encoding="utf-8"))
    variant = str(lock_data.get("variant", ""))
    if not variant.startswith("fold") or not variant[4:].isdigit():
        raise ValueError(f"Invalid AnatCL fold variant in lock: {variant!r}")
    fold = int(variant[4:])
    lock, _, _ = validate_lock_header(
        args.run_dir, expected_model="AnatCL", expected_variant=variant
    )
    locked_args = lock.get("run_config", {}).get("arguments", {})
    if int(locked_args.get("fold", -1)) != fold:
        raise ValueError("AnatCL run directory and locked fold disagree")
    model = OnHarmonyAnatCLModel(
        AnatCLOptions(
            fold=fold,
            checkpoint_dir=Path(str(locked_args["checkpoint_dir"])),
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
        expected_model="AnatCL",
        expected_variant=variant,
        selection_variant="fold0",
        release_normalizer=prediction_release_contract,
        adapter_normalizer=semantic_adapter_contract,
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
        extra_audit={"anatcl_fold": context["model"].fold},
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
