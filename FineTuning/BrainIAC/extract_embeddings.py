#!/usr/bin/env python3
"""Audit or extract frozen BrainIAC embeddings for one cohort split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
MODEL_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
for import_root in (PROJECT_ROOT, MODEL_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from model import (  # noqa: E402
    BrainIACModel,
    BrainIACOptions,
    load_and_validate_manifest,
)
from FineTuning.shared.cohort import (  # noqa: E402
    ALLOWED_SPLITS,
    DEFAULT_MANIFEST,
    frame_for_splits,
    sha256_file,
)
from FineTuning.shared.embedding_bundle import (  # noqa: E402
    EMBEDDING_BUNDLE_SCHEMA_VERSION,
    ensure_new_embedding_bundle,
    publish_embedding_bundle,
)
from FineTuning.shared.metrics import (  # noqa: E402
    AgeStandardizer,
    validate_embedding_frame,
)
from FineTuning.shared.runtime import (  # noqa: E402
    autocast_context,
    code_hashes,
    import_torch,
    package_versions,
    resolve_device,
    seed_torch,
)
from FineTuning.shared.test_receipt import (  # noqa: E402
    complete_internal_test_embedding_receipt,
    internal_test_embedding_receipt_path,
    reserve_internal_test_embedding_receipt,
)


PIPELINE_VERSION = "brainiac_brain_age_v1"


def model_from_args(args: argparse.Namespace) -> BrainIACModel:
    """Construct the model contract without loading PyTorch or the checkpoint."""

    return BrainIACModel(
        BrainIACOptions(
            checkpoint=args.checkpoint,
            preproc_root=args.preproc_root,
        )
    )


def runtime_code_hashes(model: BrainIACModel) -> dict[str, str]:
    """Hash every local source file that determines frozen extraction."""

    return code_hashes(
        [
            SCRIPT_PATH,
            PROJECT_ROOT / "FineTuning" / "shared" / "cohort.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "metrics.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "runtime.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "embedding_bundle.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "test_receipt.py",
            *model.code_paths(),
        ]
    )


def missing_inputs(model: BrainIACModel, frame: pd.DataFrame) -> list[str]:
    """Return missing model-ready image or preprocessing-QC paths."""

    missing: list[str] = []
    for record in frame.to_dict(orient="records"):
        missing.extend(
            str(path) for path in model.required_input_paths(record) if not path.is_file()
        )
    return missing


def embedding_sidecar_path(path: Path | str) -> Path:
    """Place the immutable audit sidecar next to its embedding table."""

    return Path(str(Path(path).resolve()) + ".audit.json")


def command_audit(args: argparse.Namespace) -> None:
    """Validate cohort, release, controlled boundary, and optional gradients."""

    if args.smoke_forward and args.split == "internal_test":
        raise ValueError("Smoke-forward is forbidden on the locked internal-test split")
    model = model_from_args(args)
    frame, manifest = load_and_validate_manifest(args.manifest)
    selected = frame if args.split is None else frame_for_splits(frame, [args.split])
    missing = missing_inputs(model, selected) if args.check_input_files else []
    payload: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "model": model.audit_description(),
        "manifest": manifest,
        "selected_split": args.split or "all",
        "selected_subjects": int(len(selected)),
        "input_files_checked": bool(args.check_input_files),
        "missing_model_inputs": len(missing),
        "missing_model_input_examples": missing[:10],
        "release": model.release_provenance(),
        "software": package_versions(),
    }
    if args.check_input_files and missing:
        payload["status"] = "FAIL"
        print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
        raise FileNotFoundError(
            f"BrainIAC has {len(missing)} missing inputs; first 10: {missing[:10]}"
        )
    payload["status"] = "PASS"
    if args.inspect_model or args.smoke_forward:
        torch = import_torch()
        device = resolve_device(torch, args.device)
        seed_torch(torch, args.seed)
        model.load(device)
        payload["checkpoint_load"] = model.load_description()
        payload["controlled_adaptation"] = model.set_controlled_adaptation()
        payload["code_sha256"] = runtime_code_hashes(model)
        if args.smoke_forward:
            smoke_selected = frame_for_splits(frame, [args.split or "train"])
            record = next(
                (
                    item
                    for item in smoke_selected.to_dict(orient="records")
                    if all(path.is_file() for path in model.required_input_paths(item))
                ),
                None,
            )
            if record is None:
                raise FileNotFoundError(
                    "No complete BrainIAC input is available for smoke-forward"
                )
            age_stats = AgeStandardizer.fit(frame_for_splits(frame, ["train"])["age"])
            head = torch.nn.Linear(model.embedding_dim, 1).to(device)
            model.controlled_train_mode()
            head.train()
            with autocast_context(torch, device, args.amp):
                embedding = model.encode(record, training=True, epoch=1, seed=args.seed)
                prediction = head(embedding).reshape(())
                target = torch.tensor(
                    age_stats.transform(float(record["age"])),
                    dtype=torch.float32,
                    device=device,
                )
                loss = (prediction.float() - target).square()
            loss.backward()
            named = model.trainable_named_parameters()
            missing_gradient = [name for name, value in named if value.grad is None]
            nonfinite_gradient = [
                name
                for name, value in named
                if value.grad is not None and not torch.isfinite(value.grad).all()
            ]
            nonzero_gradient = [
                name
                for name, value in named
                if value.grad is not None and bool(torch.count_nonzero(value.grad).item())
            ]
            head_missing = [name for name, value in head.named_parameters() if value.grad is None]
            if missing_gradient or nonfinite_gradient or not nonzero_gradient or head_missing:
                raise RuntimeError(
                    "Differentiability smoke test failed: "
                    f"missing={missing_gradient[:10]}, nonfinite={nonfinite_gradient[:10]}, "
                    f"nonzero={len(nonzero_gradient)}, head_missing={head_missing}"
                )
            payload["differentiability_smoke_test"] = {
                "status": "PASS",
                "dataset": str(record["dataset"]),
                "subject": str(record["subject"]),
                "split": str(record["split"]),
                "input_path": str(model.input_path(record)),
                "embedding_shape": list(embedding.shape),
                "loss": float(loss.detach().cpu().item()),
                "trainable_encoder_parameter_tensors": len(named),
                "encoder_parameter_tensors_with_gradient": len(named)
                - len(missing_gradient),
                "encoder_parameter_tensors_with_nonzero_gradient": len(nonzero_gradient),
                "head_parameter_tensors_with_gradient": len(list(head.parameters()))
                - len(head_missing),
                "optimizer_step_performed": False,
                "input_provenance": model.last_input_provenance,
            }
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


def command_extract(args: argparse.Namespace) -> None:
    """Extract deterministic float32 embeddings for one immutable cohort split."""

    if args.split == "internal_test" and not args.acknowledge_locked_test:
        raise ValueError("Pass --acknowledge-locked-test for internal-test extraction")
    model = model_from_args(args)
    output = ensure_new_embedding_bundle(
        args.output, embedding_sidecar_path(args.output)
    )
    receipt = internal_test_embedding_receipt_path(model.model_name, model.variant)
    frame, manifest = load_and_validate_manifest(args.manifest)
    selected = frame_for_splits(frame, [args.split])
    missing = missing_inputs(model, selected)
    if missing:
        raise FileNotFoundError(
            f"BrainIAC has {len(missing)} missing inputs; first 10: {missing[:10]}"
        )

    torch = import_torch()
    device = resolve_device(torch, args.device)
    seed_torch(torch, args.seed)
    model.load(device)
    frozen_boundary = model.set_frozen()
    release = model.release_provenance()
    code = runtime_code_hashes(model)
    started_receipt = None
    if args.split == "internal_test":
        started_receipt = reserve_internal_test_embedding_receipt(
            receipt,
            {
                "pipeline_version": PIPELINE_VERSION,
                "model": model.model_name,
                "variant": model.variant,
                "readout": model.readout,
                "embedding_dim": model.embedding_dim,
                "split": "internal_test",
                "manifest_artifact": str(manifest["manifest"]),
                "manifest_sha256": manifest["manifest_sha256"],
                "requested_artifact": str(output),
                "requested_sidecar": str(embedding_sidecar_path(output)),
                "requested_n_subjects": int(len(selected)),
                "on_harmony_accessed": False,
            },
        )

    print(
        f"Frozen BrainIAC extraction: split={args.split}; "
        f"n={len(selected)}; device={device}; amp={args.amp}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    model.controlled_train_mode()
    with torch.no_grad():
        for position, record in enumerate(selected.to_dict(orient="records"), start=1):
            with autocast_context(torch, device, args.amp):
                embedding = model.encode(record, training=False, epoch=0, seed=args.seed)
            values = embedding.detach().float().cpu().numpy()
            provenance = model.last_input_provenance
            if provenance is None:
                raise RuntimeError("Input provenance was not captured during extraction")
            row = {
                "dataset": str(record["dataset"]),
                "subject": str(record["subject"]),
                "split": str(record["split"]),
                "age": float(record["age"]),
                "sex": str(record["sex"]),
                "input_path": provenance["image_path"],
                "input_sha256": provenance["image_sha256"],
                "preprocessing_qc_path": provenance["qc_path"],
                "preprocessing_qc_sha256": provenance["qc_sha256"],
                "preprocessing_pipeline_version": provenance["pipeline_version"],
                "preprocessing_contract_sha256": provenance[
                    "pipeline_contract_sha256"
                ],
                "model": model.model_name,
                "variant": model.variant,
            }
            row.update({f"emb_{index}": float(value) for index, value in enumerate(values)})
            rows.append(row)
            if position % 25 == 0 or position == len(selected):
                print(
                    f"  [{position}/{len(selected)}] "
                    f"{record['dataset']}/{record['subject']}",
                    flush=True,
                )

    embeddings = pd.DataFrame(rows)
    columns = validate_embedding_frame(
        embeddings,
        expected_manifest=selected,
        expected_model=model.model_name,
        expected_variant=model.variant,
    )
    if len(columns) != model.embedding_dim:
        raise RuntimeError(
            f"Expected {model.embedding_dim} embedding dimensions, observed {len(columns)}"
        )
    sidecar = {
        "manifest_schema_version": EMBEDDING_BUNDLE_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "arm": "frozen_embedding_extraction",
        "model": model.model_name,
        "variant": model.variant,
        "readout": model.readout,
        "split": args.split,
        "n_subjects": int(len(embeddings)),
        "embedding_dim": model.embedding_dim,
        "artifact": str(output),
        "elapsed_seconds": float(time.monotonic() - started),
        "adapter": model.audit_description(),
        "checkpoint_load": model.load_description(),
        "frozen_boundary": frozen_boundary,
        "manifest": manifest,
        "release": release,
        "code_sha256": code,
        "software": package_versions(),
        "runtime": {
            "device": str(device),
            "amp": args.amp,
            "seed": args.seed,
            "input_hashes_verified": True,
        },
        "internal_test_accessed": args.split == "internal_test",
        "internal_test_receipt": None
        if started_receipt is None
        else {"path": str(receipt), "attempt_id": started_receipt["attempt_id"]},
        "on_harmony_accessed": False,
    }
    sidecar = publish_embedding_bundle(
        embeddings, output, embedding_sidecar_path(output), sidecar
    )
    if args.split == "internal_test":
        if started_receipt is None:
            raise RuntimeError("Internal-test extraction lacks its exclusive reservation")
        complete_internal_test_embedding_receipt(
            receipt,
            started_receipt,
            {
                "artifact": str(output),
                "artifact_sha256": sidecar["artifact_sha256"],
                "sidecar": str(embedding_sidecar_path(output)),
                "sidecar_sha256": sha256_file(embedding_sidecar_path(output)),
                "n_subjects": int(len(embeddings)),
            },
        )
    print(f"Saved: {output}", flush=True)
    print(f"Audit: {embedding_sidecar_path(output)}", flush=True)


def add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preproc-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    add_model_options(audit)
    audit.add_argument("--split", choices=ALLOWED_SPLITS)
    audit.add_argument("--check-input-files", action="store_true")
    audit.add_argument("--inspect-model", action="store_true")
    audit.add_argument("--smoke-forward", action="store_true")
    audit.set_defaults(func=command_audit)

    extract = commands.add_parser("extract")
    add_model_options(extract)
    extract.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--acknowledge-locked-test", action="store_true")
    extract.set_defaults(func=command_extract)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
