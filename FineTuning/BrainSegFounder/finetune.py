#!/usr/bin/env python3
"""Train, select, or evaluate controlled final-block BrainSegFounder adaptation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
MODEL_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
for import_root in (PROJECT_ROOT, MODEL_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from model import (  # noqa: E402
    BrainSegFounderModel,
    BrainSegFounderOptions,
    load_and_validate_manifest,
)
from FineTuning.shared.cohort import (  # noqa: E402
    DEFAULT_MANIFEST,
    frame_for_splits,
    sha256_file,
)
from FineTuning.shared.metrics import (  # noqa: E402
    AgeStandardizer,
    regression_metrics,
)
from FineTuning.shared.evaluation import validate_selected_evaluation_run  # noqa: E402
from FineTuning.shared.internal_test import (  # noqa: E402
    validate_internal_test_runtime_code,
)
from FineTuning.shared.protocol import (  # noqa: E402
    ADAPTED_FIXED_HYPERPARAMETERS,
    ADAPTED_LR_CONFIGS,
    adapted_learning_rates,
    select_adapted_learning_rate,
    write_training_curves,
)
from FineTuning.shared.runtime import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
    atomic_tsv,
    autocast_context,
    capture_trainable_state,
    code_hashes,
    ensure_new_output_dir,
    import_torch,
    package_versions,
    resolve_device,
    restore_trainable_state,
    seed_torch,
)


PIPELINE_VERSION = "brainsegfounder_brain_age_v1"
CHECKPOINT_NAME = "selected_model.pt"
LOCK_NAME = "LOCKED.json"
INTERNAL_TEST_RECEIPT = "INTERNAL_TEST_EVALUATED.json"


def model_from_args(args: argparse.Namespace) -> BrainSegFounderModel:
    return BrainSegFounderModel(
        BrainSegFounderOptions(
            checkpoint=args.checkpoint,
            preproc_root=args.preproc_root,
        )
    )


def runtime_code_hashes(model: BrainSegFounderModel) -> dict[str, str]:
    """Hash every source file that can change controlled BrainSegFounder adaptation."""

    return code_hashes(
        [
            SCRIPT_PATH,
            PROJECT_ROOT / "FineTuning" / "shared" / "cohort.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "evaluation.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "metrics.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "protocol.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "runtime.py",
            *model.code_paths(),
        ]
    )


def json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "func"
    }


def require_inputs(model: BrainSegFounderModel, frame: pd.DataFrame) -> None:
    missing: list[str] = []
    for record in frame.to_dict(orient="records"):
        missing.extend(
            str(path)
            for path in model.required_input_paths(record)
            if not path.is_file()
        )
    if missing:
        raise FileNotFoundError(
            f"BrainSegFounder has {len(missing)} missing inputs; "
            f"first 10: {missing[:10]}"
        )


def evaluate_frame(
    *,
    torch: Any,
    model: BrainSegFounderModel,
    head: Any,
    frame: pd.DataFrame,
    age_stats: AgeStandardizer,
    device: Any,
    amp: str,
    seed: int,
    progress_label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate the deterministic encoder view without augmentations."""

    rows: list[dict[str, Any]] = []
    model.controlled_train_mode()
    head.eval()
    started = time.monotonic()
    with torch.no_grad():
        for position, record in enumerate(frame.to_dict(orient="records"), start=1):
            with autocast_context(torch, device, amp):
                embedding = model.encode(record, training=False, epoch=0, seed=seed)
                prediction_z = head(embedding).reshape(())
            prediction_age = age_stats.inverse(float(prediction_z.detach().float().item()))
            true_age = float(record["age"])
            rows.append(
                {
                    "dataset": str(record["dataset"]),
                    "subject": str(record["subject"]),
                    "split": str(record["split"]),
                    "age_true": true_age,
                    "age_pred": prediction_age,
                    "error_years": prediction_age - true_age,
                    "absolute_error_years": abs(prediction_age - true_age),
                    "input_path": str(model.input_path(record)),
                }
            )
            if position % 25 == 0 or position == len(frame):
                print(
                    f"  [{progress_label} {position}/{len(frame)}] "
                    f"{record['dataset']}/{record['subject']}",
                    flush=True,
                )
    predictions = pd.DataFrame(rows)
    metrics = regression_metrics(predictions["age_true"], predictions["age_pred"])
    metrics.update(
        {
            "split": str(frame["split"].iloc[0]),
            "elapsed_seconds": float(time.monotonic() - started),
            "bias_correction": "none",
        }
    )
    return metrics, predictions


def checkpoint_payload(
    *,
    model: BrainSegFounderModel,
    head: Any,
    age_stats: AgeStandardizer,
    epoch: int,
    validation_metrics: dict[str, Any],
    run_config: dict[str, Any],
    manifest: dict[str, Any],
    release: dict[str, Any],
    code: dict[str, str],
) -> dict[str, Any]:
    """Store only adapted weights plus the immutable reconstruction contract."""

    return {
        "pipeline_version": PIPELINE_VERSION,
        "arm": "adapted",
        "model": model.model_name,
        "variant": model.variant,
        "epoch": int(epoch),
        "controlled_encoder_state": capture_trainable_state(model),
        "head_state_dict": {
            name: value.detach().cpu().clone() for name, value in head.state_dict().items()
        },
        "age_standardizer": {"mean": age_stats.mean, "std": age_stats.std},
        "validation_metrics": validation_metrics,
        "run_config": run_config,
        "manifest": manifest,
        "release": release,
        "code_sha256": code,
    }


def restore_checkpoint(
    torch: Any, model: BrainSegFounderModel, head: Any, checkpoint_path: Path, device: Any
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected = (PIPELINE_VERSION, "adapted", model.model_name, model.variant)
    observed = (
        payload.get("pipeline_version"),
        payload.get("arm"),
        payload.get("model"),
        payload.get("variant"),
    )
    if observed != expected:
        raise ValueError(f"Unsupported BrainSegFounder checkpoint contract: {observed}")
    restore_trainable_state(model, payload["controlled_encoder_state"])
    head.load_state_dict(payload["head_state_dict"], strict=True)
    return payload


def command_train(args: argparse.Namespace) -> None:
    """Train the declared BrainSegFounder boundary with train/validation only."""

    args.lr_encoder, args.lr_head = adapted_learning_rates(args.lr_config)
    positive = {
        "epochs": args.epochs,
        "patience": args.patience,
        "lr_encoder": args.lr_encoder,
        "lr_head": args.lr_head,
        "accum_subjects": args.accum_subjects,
    }
    bad = {name: value for name, value in positive.items() if value <= 0}
    if bad:
        raise ValueError(f"Controlled-adaptation values must be positive: {bad}")
    if args.weight_decay < 0 or args.min_delta < 0:
        raise ValueError("weight-decay and min-delta cannot be negative")

    model = model_from_args(args)
    frame, manifest = load_and_validate_manifest(args.manifest)
    train = frame_for_splits(frame, ["train"])
    validation = frame_for_splits(frame, ["validation"])
    require_inputs(model, frame_for_splits(frame, ["train", "validation"]))
    output_dir = ensure_new_output_dir(args.output_dir)

    torch = import_torch()
    device = resolve_device(torch, args.device)
    seed_torch(torch, args.seed)
    model.load(device)
    boundary = model.set_controlled_adaptation()
    release = model.release_provenance()
    code = runtime_code_hashes(model)
    head = torch.nn.Linear(model.embedding_dim, 1).to(device)
    age_stats = AgeStandardizer.fit(train["age"])

    encoder_parameters = [value for _, value in model.trainable_named_parameters()]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.lr_encoder},
            {"params": head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.MSELoss()
    run_config = {
        "pipeline_version": PIPELINE_VERSION,
        "arm": "adapted",
        "arguments": json_safe_args(args),
        "model": model.audit_description(),
        "controlled_adaptation": boundary,
        "head": {
            "type": "torch.nn.Linear",
            "input_dimensions": model.embedding_dim,
            "output_dimensions": 1,
            "parameters": int(sum(value.numel() for value in head.parameters())),
        },
        "loss": "MSE on train-standardized chronological age",
        "selection": "minimum uncorrected validation MAE in years",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "augmentation": "none",
        "age_standardizer": {"mean": age_stats.mean, "std": age_stats.std},
        "manifest": manifest,
        "release": release,
        "code_sha256": code,
        "software": package_versions(),
        "internal_test_used_for_training_or_selection": False,
        "on_harmony_accessed": False,
    }
    atomic_json(run_config, output_dir / "run_config.json")

    print(f"Controlled adaptation: BrainSegFounder/{model.variant}; device={device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU              : {torch.cuda.get_device_name(device)}", flush=True)
    print(f"  train/validation : {len(train)}/{len(validation)}", flush=True)
    print("  internal test    : NOT EVALUATED BY THIS COMMAND", flush=True)
    print(
        f"  trainable encoder: {boundary['encoder_trainable_parameters']:,}/"
        f"{boundary['encoder_total_parameters']:,}",
        flush=True,
    )
    print(f"  trainable head   : {sum(value.numel() for value in head.parameters()):,}", flush=True)

    history: list[dict[str, Any]] = []
    best_mae = math.inf
    best_epoch = 0
    bad_epochs = 0
    checkpoint_path = output_dir / CHECKPOINT_NAME
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        epoch_lr_encoder = float(optimizer.param_groups[0]["lr"])
        epoch_lr_head = float(optimizer.param_groups[1]["lr"])
        model.controlled_train_mode()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        loss_sum = 0.0
        absolute_error_sum = 0.0

        for step, row_index in enumerate(order, start=1):
            record = train.iloc[int(row_index)].to_dict()
            target_z = torch.tensor(
                age_stats.transform(float(record["age"])),
                dtype=torch.float32,
                device=device,
            )
            try:
                with autocast_context(torch, device, args.amp):
                    embedding = model.encode(record, training=True, epoch=epoch, seed=args.seed)
                    prediction_z = head(embedding).reshape(())
                    loss = criterion(prediction_z.float(), target_z)
                group_start = ((step - 1) // args.accum_subjects) * args.accum_subjects
                group_size = min(args.accum_subjects, len(order) - group_start)
                (loss / group_size).backward()
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    raise RuntimeError(
                        "CUDA OOM in BrainSegFounder. This protocol already processes "
                        "one 3D volume "
                        "at a time; use a larger GPU or disable mixed-precision drift only "
                        "after a new protocol is declared."
                    ) from error
                raise

            if step % args.accum_subjects == 0 or step == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            prediction_age = age_stats.inverse(float(prediction_z.detach().float().item()))
            loss_sum += float(loss.detach().item())
            absolute_error_sum += abs(prediction_age - float(record["age"]))
            if step % 50 == 0 or step == len(order):
                print(
                    f"  [epoch {epoch:02d} train {step}/{len(order)}] "
                    f"loss={loss_sum / step:.4f} MAE={absolute_error_sum / step:.2f}y",
                    flush=True,
                )

        scheduler.step()
        validation_metrics, validation_predictions = evaluate_frame(
            torch=torch,
            model=model,
            head=head,
            frame=validation,
            age_stats=age_stats,
            device=device,
            amp=args.amp,
            seed=args.seed,
            progress_label=f"epoch {epoch:02d} validation",
        )
        current_mae = float(validation_metrics["mae_years"])
        improved = current_mae < best_mae - args.min_delta
        if improved:
            best_mae = current_mae
            best_epoch = epoch
            bad_epochs = 0
            atomic_torch_save(
                torch,
                checkpoint_payload(
                    model=model,
                    head=head,
                    age_stats=age_stats,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                    run_config=run_config,
                    manifest=manifest,
                    release=release,
                    code=code,
                ),
                checkpoint_path,
            )
            atomic_tsv(validation_predictions, output_dir / "best_validation_predictions.tsv")
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_mse_z": loss_sum / len(order),
                "train_mae_years_online": absolute_error_sum / len(order),
                "validation_mse_z": (float(validation_metrics["rmse_years"]) / age_stats.std) ** 2,
                "validation_mae_years": current_mae,
                "validation_rmse_years": validation_metrics["rmse_years"],
                "validation_r2": validation_metrics["r2"],
                "lr_encoder": epoch_lr_encoder,
                "lr_head": epoch_lr_head,
                "elapsed_seconds": time.monotonic() - epoch_started,
                "selected": improved,
            }
        )
        atomic_tsv(pd.DataFrame(history), output_dir / "train_history.tsv")
        print(
            f"Epoch {epoch}: validation MAE={current_mae:.3f}y; "
            f"best={best_mae:.3f}y at epoch {best_epoch}" + (" *" if improved else ""),
            flush=True,
        )
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} non-improving epochs", flush=True)
            break

    if not checkpoint_path.is_file():
        raise RuntimeError("Training ended without a selected checkpoint")
    checkpoint = restore_checkpoint(torch, model, head, checkpoint_path, device)
    locked_metrics, locked_predictions = evaluate_frame(
        torch=torch,
        model=model,
        head=head,
        frame=validation,
        age_stats=AgeStandardizer(**checkpoint["age_standardizer"]),
        device=device,
        amp=args.amp,
        seed=args.seed,
        progress_label="locked validation",
    )
    atomic_tsv(locked_predictions, output_dir / "locked_validation_predictions.tsv")
    atomic_json(locked_metrics, output_dir / "locked_validation_metrics.json")
    runtime_resources: dict[str, Any] = {
        "device": str(device),
        "measurement_scope": "training through locked-validation completion",
    }
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        runtime_resources.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "peak_gpu_memory_allocated_bytes": peak_allocated,
                "peak_gpu_memory_allocated_gib": float(peak_allocated / 1024**3),
                "peak_gpu_memory_reserved_bytes": peak_reserved,
                "peak_gpu_memory_reserved_gib": float(peak_reserved / 1024**3),
            }
        )
    lock = {
        "status": "LOCKED_AFTER_TRAIN_VALIDATION",
        "pipeline_version": PIPELINE_VERSION,
        "arm": "adapted",
        "model": model.model_name,
        "variant": model.variant,
        "checkpoint": CHECKPOINT_NAME,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_epoch": int(best_epoch),
        "selection_metric": "uncorrected validation MAE in years",
        "selection_value": float(best_mae),
        "locked_validation_metrics": locked_metrics,
        "manifest": manifest,
        "release": release,
        "adapter": model.audit_description(),
        "code_sha256": code,
        "run_config": run_config,
        "training_elapsed_seconds": float(time.monotonic() - training_started),
        "runtime_resources": runtime_resources,
        "internal_test_accessed": False,
        "on_harmony_accessed": False,
    }
    atomic_json(lock, output_dir / LOCK_NAME)
    write_training_curves(
        pd.DataFrame(history),
        output_dir,
        title=f"BrainSegFounder {model.variant} | {args.lr_config}",
        selected_epoch=best_epoch,
    )
    print(f"Locked adapted checkpoint: {checkpoint_path}", flush=True)
    print(f"SHA-256: {lock['checkpoint_sha256']}", flush=True)
    if device.type == "cuda":
        print(
            "Peak GPU memory: "
            f"allocated={runtime_resources['peak_gpu_memory_allocated_gib']:.2f} GiB; "
            f"reserved={runtime_resources['peak_gpu_memory_reserved_gib']:.2f} GiB",
            flush=True,
        )


def validate_selection_candidates(run_dirs: list[Path]) -> None:
    """Require the declared seed-0 selection and one resolved BrainSegFounder input contract."""

    resolved_contracts: list[str] = []
    for run_dir in run_dirs:
        lock_path = Path(run_dir).resolve() / LOCK_NAME
        if not lock_path.is_file():
            raise FileNotFoundError(lock_path)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        arguments = lock.get("run_config", {}).get("arguments", {})
        if arguments.get("seed") != 0:
            raise ValueError(f"BrainSegFounder LR selection requires seed 0: {run_dir}")
        adapter = lock.get("adapter", {})
        keys = (
            "checkpoint",
            "input_root",
            "preprocessing_pipeline_version",
            "preprocessing_contract_sha256",
            "compatible_preprocessing_contracts",
            "model_input_shape",
            "bottleneck_shape",
        )
        missing = [key for key in keys if adapter.get(key) is None]
        if missing:
            raise ValueError(f"BrainSegFounder candidate lacks adapter fields {missing}: {run_dir}")
        resolved_contracts.append(
            json.dumps(
                {key: adapter[key] for key in keys},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    if len(set(resolved_contracts)) != 1:
        raise ValueError("BrainSegFounder LR candidates use different resolved input contracts")


def command_select(args: argparse.Namespace) -> None:
    """Select one of the three seed-0 LR configurations using validation only."""

    _, manifest = load_and_validate_manifest(args.manifest)
    validate_selection_candidates(args.run_dirs)
    output_dir = ensure_new_output_dir(args.output_dir)
    result = select_adapted_learning_rate(
        args.run_dirs,
        output_dir,
        expected_model="BrainSegFounder",
        expected_variant="bottleneck_mean",
        manifest_sha256=str(manifest["manifest_sha256"]),
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def validate_evaluation_options(args: argparse.Namespace, checkpoint: dict[str, Any]) -> None:
    selected = checkpoint.get("run_config", {}).get("arguments", {})
    for key in ("amp", "seed"):
        requested = getattr(args, key)
        if selected.get(key) != requested:
            raise ValueError(
                f"--{key.replace('_', '-')} differs from locked protocol: "
                f"selected={selected.get(key)!r}, requested={requested!r}"
            )


def command_evaluate(args: argparse.Namespace) -> None:
    """Perform the explicitly unlocked one-time internal-test evaluation."""

    if not args.acknowledge_locked_test:
        raise ValueError("Pass --acknowledge-locked-test for internal-test evaluation")
    model = model_from_args(args)
    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / CHECKPOINT_NAME
    lock_path = run_dir / LOCK_NAME
    receipt = run_dir / INTERNAL_TEST_RECEIPT
    if receipt.exists():
        raise FileExistsError(f"This model already has an internal-test receipt: {receipt}")
    if not checkpoint_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError(f"Expected {checkpoint_path} and {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("arm"), lock.get("model"), lock.get("variant")) != (
        "adapted", model.model_name, model.variant
    ):
        raise ValueError("LOCKED.json is not the expected adapted BrainSegFounder run")
    if lock.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Adapted checkpoint hash differs from LOCKED.json")
    if (
        lock.get("status") != "LOCKED_AFTER_TRAIN_VALIDATION"
        or lock.get("internal_test_accessed") is not False
        or lock.get("on_harmony_accessed") is not False
    ):
        raise ValueError("Evaluation requires a validation-only locked run")

    frame, manifest = load_and_validate_manifest(args.manifest)
    if lock.get("manifest", {}).get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Checkpoint uses a different subject manifest")
    selection_provenance = validate_selected_evaluation_run(
        selection_dir=args.selection_dir,
        run_dir=run_dir,
        lock=lock,
        manifest_sha256=str(manifest["manifest_sha256"]),
        expected_model=model.model_name,
        evaluation_variant=model.variant,
        checkpoint_name=CHECKPOINT_NAME,
        lock_name=LOCK_NAME,
    )

    torch = import_torch()
    device = resolve_device(torch, args.device)
    seed_torch(torch, args.seed)
    model.load(device)
    boundary = model.set_controlled_adaptation()
    release = model.release_provenance()
    code = runtime_code_hashes(model)
    if release != lock.get("release"):
        raise ValueError("Base release differs from the selected checkpoint")
    runtime_code_provenance = validate_internal_test_runtime_code(
        model_name=model.model_name,
        locked_code=lock.get("code_sha256", {}),
        current_code=code,
    )
    if model.audit_description() != lock.get("adapter"):
        raise ValueError("BrainSegFounder adapter/input root differs from the selected checkpoint")
    head = torch.nn.Linear(model.embedding_dim, 1).to(device)
    checkpoint = restore_checkpoint(torch, model, head, checkpoint_path, device)
    validate_evaluation_options(args, checkpoint)
    age_stats = AgeStandardizer(**checkpoint["age_standardizer"])
    output_dir = ensure_new_output_dir(args.output_dir)
    test = frame_for_splits(frame, ["internal_test"])
    require_inputs(model, test)

    metrics, predictions = evaluate_frame(
        torch=torch,
        model=model,
        head=head,
        frame=test,
        age_stats=age_stats,
        device=device,
        amp=args.amp,
        seed=args.seed,
        progress_label="internal_test",
    )
    predictions_path = output_dir / "predictions_internal_test.tsv"
    atomic_tsv(predictions, predictions_path)
    result = {
        "pipeline_version": PIPELINE_VERSION,
        "arm": "adapted",
        "model": model.model_name,
        "variant": model.variant,
        "metrics": metrics,
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "predictions_artifact": str(predictions_path),
        "predictions_artifact_sha256": sha256_file(predictions_path),
        "manifest": manifest,
        "internal_test_subjects": int(len(test)),
        "release": release,
        "lr_selection": selection_provenance,
        "runtime_code_provenance": runtime_code_provenance,
        "controlled_adaptation": boundary,
        "bias_correction": "none",
        "internal_test_accessed": True,
        "on_harmony_accessed": False,
    }
    atomic_json(result, output_dir / "metrics_internal_test.json")
    atomic_json(
        {
            "status": "INTERNAL_TEST_EVALUATED",
            "arm": "adapted",
            "model": model.model_name,
            "variant": model.variant,
            "internal_test_accessed": True,
            "lr_selection": selection_provenance,
            "runtime_code_provenance": runtime_code_provenance,
            "predictions_artifact": str(predictions_path),
            "predictions_artifact_sha256": sha256_file(predictions_path),
            "result": str(output_dir / "metrics_internal_test.json"),
            "result_sha256": sha256_file(output_dir / "metrics_internal_test.json"),
        },
        receipt,
    )
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)


def add_common(parser: argparse.ArgumentParser, *, training: bool = False) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preproc-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        choices=("off", "float16", "bfloat16"),
        required=training,
        default=None if training else "bfloat16",
    )
    parser.add_argument("--seed", type=int, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    add_common(train, training=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--lr-config", choices=tuple(ADAPTED_LR_CONFIGS), required=True)
    train.set_defaults(**ADAPTED_FIXED_HYPERPARAMETERS)
    train.set_defaults(func=command_train)

    select = commands.add_parser("select")
    select.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    select.add_argument("--run-dirs", type=Path, nargs=3, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.set_defaults(func=command_select)

    evaluate = commands.add_parser("evaluate")
    add_common(evaluate)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--selection-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--acknowledge-locked-test", action="store_true")
    evaluate.set_defaults(func=command_evaluate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
