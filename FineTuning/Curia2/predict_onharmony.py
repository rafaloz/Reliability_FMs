#!/usr/bin/env python3
"""Run label-blind ON-Harmony inference with adapted Curia-2 checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.Curia2.finetune import restore_checkpoint, runtime_code_hashes  # noqa: E402
from FineTuning.Curia2.model import (  # noqa: E402
    Curia2Model,
    Curia2Options,
    load_and_validate_manifest,
)
from FineTuning.shared.onharmony import (  # noqa: E402
    ArtifactRegistry,
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    adapted_runtime_migration,
    load_adapted_model,
    load_onharmony_manifest,
    run_standard_audit,
    run_standard_prediction,
    validate_lock_header,
    validate_run_binding,
)


PIPELINE_VERSION = "curia2_onharmony_prediction_v1"
INPUT_CONTRACT_VERSION = "onharmony_raw_t1w_curia2_v1"


class OnHarmonyCuria2Model(Curia2Model):
    """Keep the locked release/adapter while replacing only its input registry."""

    def __init__(self, options: Curia2Options, registry: ArtifactRegistry) -> None:
        super().__init__(options)
        self.external_registry = registry

    def input_path(self, record: Any) -> Path:
        return self.external_registry.input_path(record)

    def required_input_paths(self, record: Any) -> list[Path]:
        return self.external_registry.required_paths(record)


def artifact_paths(record: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve the canonical raw T1w consumed by the locked Curia-2 processor."""

    return {"input": Path(str(record["raw_t1w_path"]))}


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    frame, manifest = load_onharmony_manifest(args.filelist, args.metadata)
    registry = ArtifactRegistry(
        contract_version=INPUT_CONTRACT_VERSION,
        resolver=artifact_paths,
        contract={"source": "canonical ON-Harmony raw T1w registry"},
    )
    lock, _, _ = validate_lock_header(
        args.run_dir, expected_model="Curia-2", expected_variant="canonical"
    )
    locked_args = lock.get("run_config", {}).get("arguments", {})
    model = OnHarmonyCuria2Model(
        Curia2Options(
            model_dir=Path(str(locked_args["model_dir"])),
            slice_batch_size=int(locked_args["slice_batch_size"]),
        ),
        registry,
    )
    binding = validate_run_binding(
        run_dir=args.run_dir,
        selection_dir=args.selection_dir,
        model=model,
        training_manifest_loader=load_and_validate_manifest,
        runtime_code_loader=runtime_code_hashes,
        expected_model="Curia-2",
        expected_variant="canonical",
        runtime_migration=adapted_runtime_migration(
            model.model_name, lock.get("code_sha256", {})
        ),
        requested_amp=args.amp,
    )
    input_contract = {
        "contract_version": INPUT_CONTRACT_VERSION,
        "raw_filelist": manifest["filelist"],
        "raw_filelist_sha256": manifest["filelist_sha256"],
        "processor": "pinned Curia-2 release processor from the locked adapter",
        "slice_policy": "all axial slices; mean CLS readout",
    }
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
