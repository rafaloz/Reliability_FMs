#!/usr/bin/env python3
"""Run label-blind ON-Harmony inference with adapted 3D-Neuro-SimCLR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
MODEL_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
for import_root in (PROJECT_ROOT, MODEL_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from finetune import restore_checkpoint, runtime_code_hashes  # noqa: E402
from model import (  # noqa: E402
    NeuroSimCLRModel,
    NeuroSimCLROptions,
    load_and_validate_manifest,
)
from FineTuning.shared.onharmony import (  # noqa: E402
    ArtifactRegistry,
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    adapted_runtime_migration,
    file_signature,
    load_adapted_model,
    load_onharmony_manifest,
    preprocessed_path,
    run_standard_audit,
    run_standard_prediction,
    source_contract,
    validate_lock_header,
    validate_run_binding,
)


PIPELINE_VERSION = "3d_neuro_simclr_onharmony_prediction_v3"
INPUT_CONTRACT_VERSION = "onharmony_3dneurosimclr_icbm2009c_v3"
EXPECTED_TEMPLATE_SHA256 = (
    "93359ab97c1c027376397612a9b6c30e95406c15bf8695bd4a8efcb2064eaa34"
)
EXPECTED_PREPROCESSOR_SHA256 = (
    "053f0a8a526fe14d678f169574ef2657b8a2f93a1d96d5c3da09650d7ddbb445"
)
DEFAULT_PREPROC_ROOT = (
    PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "preproc_SimCLR"
)
DEFAULT_LOG_ROOT = (
    PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "logs_preprocs"
)
PREPROCESSOR = (
    PROJECT_ROOT
    / "preprocessings"
    / "OnHarmonyDataset"
    / "preprocessing_3DNeuroSimCLR.py"
)
TEMPLATE = PROJECT_ROOT / "templates" / "mni_icbm152_t1_tal_nlin_sym_09c.nii"


def artifact_paths(
    record: Mapping[str, Any], preproc_root: Path, log_root: Path
) -> dict[str, Path]:
    """Resolve the corrected image, mask, and acquisition-level preprocessing log."""

    acquisition = str(record["acquisition_id"])
    return {
        "input": preprocessed_path(preproc_root, record, "_simclr_ready.nii.gz"),
        "mask": preprocessed_path(preproc_root, record, "_mask_mni.nii.gz"),
        "preprocessing_log": log_root / f"simclr_{acquisition}.log",
    }


def validate_artifact_set(
    record: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Require one successful corrected TurboPrep-equivalent preprocessing run."""

    log_path = paths["preprocessing_log"]
    log = log_path.read_text(encoding="utf-8", errors="replace")
    expected_done_suffix = f"/{record['acquisition_id']}/{paths['input'].name}"
    done_paths = [
        line.strip().removeprefix("Done:").strip()
        for line in log.splitlines()
        if line.startswith("  Done:") or line.startswith("Done:")
    ]
    exit_lines = [
        line.strip() for line in log.splitlines() if line.strip().startswith("exit_code:")
    ]
    required = (
        "3D-Neuro-SimCLR preprocessing (TurboPrep-equivalent)",
        f"Template SHA-256: {EXPECTED_TEMPLATE_SHA256}",
        "[Preprocessing 5/5] WhiteStripe normalization",
        "All done.",
    )
    missing = [fragment for fragment in required if fragment not in log]
    if exit_lines != ["exit_code: 0"]:
        missing.append("exactly one terminal 'exit_code: 0'")
    if not any(value.endswith(expected_done_suffix) for value in done_paths):
        missing.append(f"Done: *{expected_done_suffix}")
    if str(record["raw_t1w_path"]) not in log:
        missing.append("the canonical raw T1w path")
    if missing or "WhiteStripe failed" in log:
        raise ValueError(
            "3D-Neuro-SimCLR preprocessing log is not a successful corrected run: "
            f"{log_path}; missing={missing}"
        )
    return {
        "preprocessing_status": "PASS",
        "template_sha256": EXPECTED_TEMPLATE_SHA256,
        "registered_shape": "193x229x193",
    }


def build_input_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind external inputs to the audited implementation and complete ICBM atlas."""

    import nibabel as nib

    template_image = nib.load(str(TEMPLATE))
    shape = tuple(int(value) for value in template_image.shape)
    spacing = tuple(float(value) for value in template_image.header.get_zooms()[:3])
    if shape != (193, 229, 193):
        raise ValueError(f"Unexpected ON-Harmony template shape: {shape}")
    if not np.allclose(spacing, (1.0, 1.0, 1.0), rtol=0.0, atol=1e-6):
        raise ValueError(f"Unexpected ON-Harmony template spacing: {spacing}")
    return source_contract(
        contract_version=INPUT_CONTRACT_VERSION,
        paths={"preprocessor": PREPROCESSOR, "template": TEMPLATE},
        expected_sha256={
            "preprocessor": EXPECTED_PREPROCESSOR_SHA256,
            "template": EXPECTED_TEMPLATE_SHA256,
        },
        details={
            "method": "TurboPrep-equivalent rigid registration and WhiteStripe",
            "raw_filelist": manifest["filelist"],
            "raw_filelist_sha256": manifest["filelist_sha256"],
            "registered_shape": list(shape),
            "template_spacing": list(spacing),
            "template_affine": np.asarray(template_image.affine, dtype=float).tolist(),
        },
    )


class OnHarmonyNeuroSimCLRModel(NeuroSimCLRModel):
    """Reuse the locked encoder while redirecting only external input provenance."""

    def __init__(
        self,
        options: NeuroSimCLROptions,
        registry: ArtifactRegistry,
        input_contract: Mapping[str, Any],
    ) -> None:
        super().__init__(options)
        self.external_registry = registry
        self.external_input_contract = dict(input_contract)

    def input_path(self, record: Any) -> Path:
        return self.external_registry.input_path(record)

    def mask_path(self, record: Any) -> Path:
        return self.external_registry.paths(record)["mask"]

    def required_input_paths(self, record: Any) -> list[Path]:
        return self.external_registry.required_paths(record)

    def validate_input_provenance(self, record: Any) -> dict[str, Any]:
        return self.external_registry.validate(record)

    def load_input(self, record: Any) -> Any:
        """Enforce the exact template grid across the shared provenance checks."""

        import nibabel as nib

        paths = tuple(self.required_input_paths(record))
        before = file_signature(paths)
        tensor = super().load_input(record)
        image = nib.load(str(self.input_path(record)))
        expected_affine = np.asarray(
            self.external_input_contract["template_affine"], dtype=float
        )
        expected_spacing = tuple(
            float(value) for value in self.external_input_contract["template_spacing"]
        )
        observed_spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if not np.allclose(image.affine, expected_affine, rtol=0.0, atol=1e-5):
            raise ValueError("ON-Harmony input affine differs from the locked template")
        if not np.allclose(observed_spacing, expected_spacing, rtol=0.0, atol=1e-6):
            raise ValueError("ON-Harmony input spacing differs from the locked template")
        if file_signature(paths) != before:
            raise RuntimeError("ON-Harmony inputs changed while the model loaded them")
        provenance = self.external_registry.validate(record)
        if provenance != self.last_input_provenance:
            raise RuntimeError("ON-Harmony input hashes changed while the model loaded them")
        self.last_input_provenance = provenance
        return tensor


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    """Create the model-specific adapter consumed by the common inference engine."""

    frame, manifest = load_onharmony_manifest(args.filelist, args.metadata)
    input_contract = build_input_contract(manifest)
    preproc_root = args.preproc_root.expanduser().resolve()
    log_root = args.preproc_log_root.expanduser().resolve()
    registry = ArtifactRegistry(
        contract_version=INPUT_CONTRACT_VERSION,
        resolver=lambda record: artifact_paths(record, preproc_root, log_root),
        validator=validate_artifact_set,
        contract=input_contract,
    )
    lock, _, _ = validate_lock_header(
        args.run_dir,
        expected_model="3D-Neuro-SimCLR",
        expected_variant="canonical",
    )
    locked_args = lock.get("run_config", {}).get("arguments", {})
    model = OnHarmonyNeuroSimCLRModel(
        NeuroSimCLROptions(
            checkpoint=(
                Path(str(locked_args["checkpoint"]))
                if locked_args.get("checkpoint") is not None
                else None
            ),
            preproc_root=(
                Path(str(locked_args["preproc_root"]))
                if locked_args.get("preproc_root") is not None
                else None
            ),
        ),
        registry,
        input_contract,
    )
    binding = validate_run_binding(
        run_dir=args.run_dir,
        selection_dir=args.selection_dir,
        model=model,
        training_manifest_loader=load_and_validate_manifest,
        runtime_code_loader=runtime_code_hashes,
        expected_model="3D-Neuro-SimCLR",
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
