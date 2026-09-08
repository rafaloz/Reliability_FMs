#!/usr/bin/env python3
"""Train, select, or evaluate controlled last-block Curia-2 adaptation."""

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
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.Curia2.model import (  # noqa: E402
    Curia2Model,
    Curia2Options,
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
from FineTuning.shared.internal_test import (  # noqa: E402
    internal_test_orchestrator_hashes,
    internal_test_runtime_migration,
    internal_test_selection_runtime_migration,
    load_internal_test_model,
    validate_internal_test_binding,
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


PIPELINE_VERSION = "curia2_brain_age_v2_all_axial_slices"
CHECKPOINT_NAME = "selected_model.pt"
LOCK_NAME = "LOCKED.json"
INTERNAL_TEST_RECEIPT = "INTERNAL_TEST_EVALUATED.json"
INTERNAL_TEST_SELECTION_RUNTIME_MIGRATION = internal_test_selection_runtime_migration(
    "Curia-2"
)
CODE_MIGRATION_RECEIPT = "AUDITED_CODE_MIGRATION.json"
LOADER_MIGRATION_ID = "curia2-strict-3d-loader-v1"
LEGACY_FINETUNE_SHA256 = "753f52f7e7d77668c2dcc47a9d9fd50743c2c73869396e6864b3bb5a03435b12"
LEGACY_MODEL_SHA256 = "36b07399c2a3253eb844bf44ae1cced31a9d9a40ac134900a17fa77934de5f1c"
STRICT_3D_MODEL_SHA256 = "a1bfbc84f6b8ac0208e32ee1948aa9404e0e2bd5b743fa9f11de940461562c78"
FINETUNE_CODE_KEY = "FineTuning/Curia2/finetune.py"
MODEL_CODE_KEY = "FineTuning/Curia2/model.py"
LOADER_MIGRATION_CODE_KEYS = {FINETUNE_CODE_KEY, MODEL_CODE_KEY}


def model_from_args(args: argparse.Namespace) -> Curia2Model:
    return Curia2Model(
        Curia2Options(
            model_dir=args.model_dir,
            slice_batch_size=args.slice_batch_size,
        )
    )


def runtime_code_hashes(model: Curia2Model) -> dict[str, str]:
    """Hash every source file that can change controlled Curia-2 adaptation."""

    return code_hashes(
        [
            SCRIPT_PATH,
            PROJECT_ROOT / "FineTuning" / "shared" / "cohort.py",
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


def require_inputs(model: Curia2Model, frame: pd.DataFrame) -> None:
    missing: list[str] = []
    for record in frame.to_dict(orient="records"):
        missing.extend(str(path) for path in model.required_input_paths(record) if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Curia-2 has {len(missing)} missing inputs; first 10: {missing[:10]}")


def audit_nifti_dimensions(model: Curia2Model, frame: pd.DataFrame) -> dict[str, Any]:
    """Read every NIfTI header and prove that the legacy 4D branch was unreachable."""

    import nibabel as nib

    dimensionality: dict[str, int] = {}
    non_3d: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for record in frame.to_dict(orient="records"):
        path = model.input_path(record)
        try:
            shape = tuple(int(value) for value in nib.load(str(path)).shape)
        except Exception as error:  # noqa: BLE001 - retain the failing file in the audit
            errors.append(
                {
                    "dataset": str(record["dataset"]),
                    "subject": str(record["subject"]),
                    "path": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        key = str(len(shape))
        dimensionality[key] = dimensionality.get(key, 0) + 1
        if len(shape) != 3:
            non_3d.append(
                {
                    "dataset": str(record["dataset"]),
                    "subject": str(record["subject"]),
                    "path": str(path),
                    "shape": list(shape),
                }
            )
    summary = {
        "n_manifest_rows": int(len(frame)),
        "n_headers_read": int(sum(dimensionality.values())),
        "dimensionality_counts": dict(sorted(dimensionality.items())),
        "n_non_3d": int(len(non_3d)),
        "n_header_errors": int(len(errors)),
    }
    if non_3d or errors:
        raise ValueError(
            "Curia-2 loader migration requires every manifest input to be readable and "
            f"exactly 3D; summary={summary}, non_3d={non_3d[:5]}, errors={errors[:5]}"
        )
    if summary["n_headers_read"] != len(frame) or dimensionality != {"3": len(frame)}:
        raise ValueError(f"Incomplete Curia-2 dimensionality audit: {summary}")
    return summary


def differing_code_keys(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """List source files added, removed, or changed between two runtime manifests."""

    return sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))


def validate_loader_migration_receipt(
    receipt: dict[str, Any],
    *,
    lock: dict[str, Any],
    lock_sha256: str,
    current_code: dict[str, str],
    manifest_sha256: str,
    checkpoint_sha256: str,
    dimension_audit: dict[str, Any],
) -> dict[str, Any]:
    """Validate the single audited no-op migration accepted for legacy seed 0."""

    old_code = lock.get("code_sha256")
    if not isinstance(old_code, dict):
        raise ValueError("Legacy Curia-2 lock has no runtime code manifest")
    mismatches = differing_code_keys(old_code, current_code)
    expected = {
        "status": "AUDITED_RUNTIME_CODE_MIGRATION",
        "migration_id": LOADER_MIGRATION_ID,
        "source_lock_sha256": lock_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "old_code_sha256": old_code,
        "new_code_sha256": current_code,
        "differing_code_keys": mismatches,
        "nifti_dimensionality_audit": dimension_audit,
    }
    bad = {key: (receipt.get(key), value) for key, value in expected.items() if receipt.get(key) != value}
    if bad:
        raise ValueError(f"Curia-2 code-migration receipt is inconsistent: {bad}")
    if set(mismatches) != LOADER_MIGRATION_CODE_KEYS:
        raise ValueError(f"Unaudited Curia-2 runtime changes are present: {mismatches}")
    if old_code.get(FINETUNE_CODE_KEY) != LEGACY_FINETUNE_SHA256:
        raise ValueError("Migration source is not the audited legacy Curia-2 finetune script")
    if old_code.get(MODEL_CODE_KEY) != LEGACY_MODEL_SHA256:
        raise ValueError("Migration source is not the audited legacy Curia-2 loader")
    if current_code.get(MODEL_CODE_KEY) != STRICT_3D_MODEL_SHA256:
        raise ValueError("Runtime is not the audited strict-3D Curia-2 loader")
    if dimension_audit != {
        "n_manifest_rows": 1411,
        "n_headers_read": 1411,
        "dimensionality_counts": {"3": 1411},
        "n_non_3d": 0,
        "n_header_errors": 0,
    }:
        raise ValueError(f"Unexpected Curia-2 cohort dimensionality audit: {dimension_audit}")
    return {
        "mode": "audited_runtime_code_migration",
        "migration_id": LOADER_MIGRATION_ID,
        "differing_code_keys": mismatches,
    }


def evaluate_frame(
    *,
    torch: Any,
    model: Curia2Model,
    head: Any,
    frame: pd.DataFrame,
    age_stats: AgeStandardizer,
    device: Any,
    amp: str,
    seed: int,
    progress_label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate with deterministic all-slice or explicitly configured inference."""

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
    model: Curia2Model,
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
    torch: Any, model: Curia2Model, head: Any, checkpoint_path: Path, device: Any
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
        raise ValueError(f"Unsupported Curia-2 checkpoint contract: {observed}")
    restore_trainable_state(model, payload["controlled_encoder_state"])
    head.load_state_dict(payload["head_state_dict"], strict=True)
    return payload


def command_train(args: argparse.Namespace) -> None:
    """Train the declared Curia-2 boundary with train/validation only."""

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

    print(f"Controlled adaptation: Curia-2; device={device}", flush=True)
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
                        "CUDA OOM in Curia-2. Reduce --slice-batch-size; all axial "
                        "slices remain required by the protocol."
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
        "code_sha256": code,
        "run_config": run_config,
        "training_elapsed_seconds": float(time.monotonic() - training_started),
        "internal_test_accessed": False,
        "on_harmony_accessed": False,
    }
    atomic_json(lock, output_dir / LOCK_NAME)
    write_training_curves(
        pd.DataFrame(history),
        output_dir,
        title=f"Curia-2 | {args.lr_config}",
        selected_epoch=best_epoch,
    )
    print(f"Locked adapted checkpoint: {checkpoint_path}", flush=True)
    print(f"SHA-256: {lock['checkpoint_sha256']}", flush=True)


def validate_selection_candidates(run_dirs: list[Path]) -> None:
    """Reject LR-selection candidates that were not trained with seed 0."""

    for run_dir in run_dirs:
        lock_path = Path(run_dir).resolve() / LOCK_NAME
        if not lock_path.is_file():
            raise FileNotFoundError(lock_path)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        seed = lock.get("run_config", {}).get("arguments", {}).get("seed")
        if seed != 0:
            raise ValueError(f"Curia-2 LR selection requires seed 0: {run_dir}")


def command_select(args: argparse.Namespace) -> None:
    """Select one of the three seed-0 LR configurations using validation only."""

    _, manifest = load_and_validate_manifest(args.manifest)
    validate_selection_candidates(args.run_dirs)
    output_dir = ensure_new_output_dir(args.output_dir)
    result = select_adapted_learning_rate(
        args.run_dirs,
        output_dir,
        expected_model="Curia-2",
        expected_variant="canonical",
        manifest_sha256=str(manifest["manifest_sha256"]),
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def validate_evaluation_options(args: argparse.Namespace, checkpoint: dict[str, Any]) -> None:
    selected = checkpoint.get("run_config", {}).get("arguments", {})
    for key in ("amp", "seed", "slice_batch_size"):
        requested = getattr(args, key)
        if selected.get(key) != requested:
            raise ValueError(
                f"--{key.replace('_', '-')} differs from locked protocol: "
                f"selected={selected.get(key)!r}, requested={requested!r}"
            )


def command_attest_loader_migration(args: argparse.Namespace) -> None:
    """Attest that the strict 3D guard is a no-op for the locked seed-0 cohort."""

    model = model_from_args(args)
    run_dir = args.run_dir.resolve()
    lock_path = run_dir / LOCK_NAME
    checkpoint_path = run_dir / CHECKPOINT_NAME
    receipt_path = run_dir / CODE_MIGRATION_RECEIPT
    if receipt_path.exists():
        raise FileExistsError(f"Curia-2 migration receipt already exists: {receipt_path}")
    if not lock_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Expected {lock_path} and {checkpoint_path}")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("arm"), lock.get("model"), lock.get("variant")) != (
        "adapted", model.model_name, model.variant
    ):
        raise ValueError("LOCKED.json is not the expected adapted Curia-2 run")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if lock.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Adapted checkpoint differs from the legacy LOCKED.json")

    frame, manifest = load_and_validate_manifest(args.manifest)
    manifest_sha256 = str(manifest["manifest_sha256"])
    if lock.get("manifest", {}).get("manifest_sha256") != manifest_sha256:
        raise ValueError("Legacy checkpoint uses a different subject manifest")
    if model.release_provenance() != lock.get("release"):
        raise ValueError("Base release differs from the legacy checkpoint")

    current_code = runtime_code_hashes(model)
    old_code = lock.get("code_sha256")
    if not isinstance(old_code, dict):
        raise ValueError("Legacy Curia-2 lock has no runtime code manifest")
    mismatches = differing_code_keys(old_code, current_code)
    if set(mismatches) != LOADER_MIGRATION_CODE_KEYS:
        raise ValueError(f"Migration contains runtime changes outside its scope: {mismatches}")
    if old_code.get(FINETUNE_CODE_KEY) != LEGACY_FINETUNE_SHA256:
        raise ValueError("Legacy finetune source hash is not the audited version")
    if old_code.get(MODEL_CODE_KEY) != LEGACY_MODEL_SHA256:
        raise ValueError("Legacy loader source hash is not the audited version")
    if current_code.get(MODEL_CODE_KEY) != STRICT_3D_MODEL_SHA256:
        raise ValueError("Current loader source hash is not the audited strict-3D version")

    dimension_audit = audit_nifti_dimensions(model, frame)
    receipt = {
        "status": "AUDITED_RUNTIME_CODE_MIGRATION",
        "migration_id": LOADER_MIGRATION_ID,
        "reason": (
            "The legacy loader selected volume[0] only for 4D NIfTI inputs. All 1411 "
            "manifest inputs are exactly 3D, so the strict-3D guard leaves every training "
            "and validation input unchanged."
        ),
        "source_lock": str(lock_path),
        "source_lock_sha256": sha256_file(lock_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_sha256,
        "old_code_sha256": old_code,
        "new_code_sha256": current_code,
        "differing_code_keys": mismatches,
        "nifti_dimensionality_audit": dimension_audit,
        "training_or_validation_repeated": False,
        "internal_test_accessed": False,
        "on_harmony_accessed": False,
    }
    validate_loader_migration_receipt(
        receipt,
        lock=lock,
        lock_sha256=sha256_file(lock_path),
        current_code=current_code,
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        dimension_audit=dimension_audit,
    )
    atomic_json(receipt, receipt_path)
    print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


def validate_runtime_code(
    *,
    model: Curia2Model,
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    run_dir: Path,
    lock_path: Path,
    checkpoint_path: Path,
    lock: dict[str, Any],
    current_code: dict[str, str],
) -> dict[str, Any]:
    """Require an exact code match or the narrowly scoped audited loader migration."""

    if current_code == lock.get("code_sha256"):
        return {"mode": "exact_runtime_code_match"}
    receipt_path = run_dir / CODE_MIGRATION_RECEIPT
    if not receipt_path.is_file():
        raise ValueError(
            "Runtime code differs from the selected checkpoint. Run the explicit "
            "attest-loader-migration command only for the audited legacy Curia-2 seed 0."
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    dimension_audit = audit_nifti_dimensions(model, frame)
    provenance = validate_loader_migration_receipt(
        receipt,
        lock=lock,
        lock_sha256=sha256_file(lock_path),
        current_code=current_code,
        manifest_sha256=str(manifest["manifest_sha256"]),
        checkpoint_sha256=sha256_file(checkpoint_path),
        dimension_audit=dimension_audit,
    )
    provenance["receipt"] = str(receipt_path)
    provenance["receipt_sha256"] = sha256_file(receipt_path)
    return provenance


def command_evaluate(args: argparse.Namespace) -> None:
    """Open the locked internal test after validating selection and provenance."""

    if not args.acknowledge_locked_test:
        raise ValueError("Pass --acknowledge-locked-test for internal-test evaluation")
    run_dir = args.run_dir.expanduser().resolve()
    receipt_path = run_dir / INTERNAL_TEST_RECEIPT
    if receipt_path.exists():
        raise FileExistsError(
            f"This model already has an internal-test receipt: {receipt_path}"
        )

    locked_code = json.loads((run_dir / LOCK_NAME).read_text(encoding="utf-8")).get(
        "code_sha256", {}
    )
    runtime_migration = internal_test_runtime_migration("Curia-2", locked_code)
    model = model_from_args(args)
    binding = validate_internal_test_binding(
        run_dir=run_dir,
        selection_dir=args.selection_dir,
        model=model,
        training_manifest_loader=lambda: load_and_validate_manifest(args.manifest),
        runtime_code_loader=runtime_code_hashes,
        expected_model="Curia-2",
        expected_variant="canonical",
        runtime_migration=runtime_migration,
        selection_runtime_migration=INTERNAL_TEST_SELECTION_RUNTIME_MIGRATION,
        requested_amp=args.amp,
    )
    if args.seed != binding["seed"]:
        raise ValueError(
            f"--seed differs from locked protocol: "
            f"selected={binding['seed']!r}, requested={args.seed!r}"
        )

    frame, manifest = load_and_validate_manifest(args.manifest)
    test = frame_for_splits(frame, ["internal_test"])
    require_inputs(model, test)
    runtime = load_internal_test_model(
        model=model,
        restore_checkpoint=restore_checkpoint,
        checkpoint_path=binding["checkpoint_path"],
        device_name=args.device,
        amp=binding["amp"],
        seed=binding["seed"],
    )
    validate_evaluation_options(args, runtime["checkpoint"])

    output_dir = ensure_new_output_dir(args.output_dir)
    metrics, predictions = evaluate_frame(
        torch=runtime["torch"],
        model=model,
        head=runtime["head"],
        frame=test,
        age_stats=runtime["age_standardizer"],
        device=runtime["device"],
        amp=binding["amp"],
        seed=binding["seed"],
        progress_label="internal_test",
    )
    predictions_path = output_dir / "predictions_internal_test.tsv"
    metrics_path = output_dir / "metrics_internal_test.json"
    atomic_tsv(predictions, predictions_path)
    evaluator_code = internal_test_orchestrator_hashes(SCRIPT_PATH)
    result: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "arm": "adapted",
        "model": model.model_name,
        "variant": model.variant,
        "metrics": metrics,
        "checkpoint_sha256": binding["lock"]["checkpoint_sha256"],
        "predictions_artifact": str(predictions_path),
        "predictions_artifact_sha256": sha256_file(predictions_path),
        "manifest": manifest,
        "internal_test_subjects": int(len(test)),
        "release": model.release_provenance(),
        "lr_selection": binding["selection"],
        "runtime_code_provenance": binding["runtime_code_provenance"],
        "evaluation_orchestrator_code": evaluator_code,
        "controlled_adaptation": runtime["boundary"],
        "bias_correction": "none",
        "internal_test_accessed": True,
        "on_harmony_accessed": False,
    }
    atomic_json(result, metrics_path)
    atomic_json(
        {
            "status": "INTERNAL_TEST_EVALUATED",
            "arm": "adapted",
            "model": model.model_name,
            "variant": model.variant,
            "internal_test_accessed": True,
            "lr_selection": binding["selection"],
            "runtime_code_provenance": binding["runtime_code_provenance"],
            "evaluation_orchestrator_code": evaluator_code,
            "predictions_artifact": str(predictions_path),
            "predictions_artifact_sha256": sha256_file(predictions_path),
            "result": str(metrics_path),
            "result_sha256": sha256_file(metrics_path),
        },
        receipt_path,
    )
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)


def add_common(
    parser: argparse.ArgumentParser,
    *,
    training: bool = False,
    seed_required: bool = False,
) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        choices=("off", "float16", "bfloat16"),
        required=training,
        default=None if training else "bfloat16",
    )
    require_seed = training or seed_required
    parser.add_argument("--seed", type=int, required=require_seed, default=None if require_seed else 42)
    parser.add_argument("--slice-batch-size", type=int, default=8)


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

    attest = commands.add_parser("attest-loader-migration")
    attest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    attest.add_argument("--model-dir", type=Path)
    attest.add_argument("--slice-batch-size", type=int, default=8)
    attest.add_argument("--run-dir", type=Path, required=True)
    attest.set_defaults(func=command_attest_loader_migration)

    evaluate = commands.add_parser("evaluate")
    add_common(evaluate, seed_required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--selection-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--acknowledge-locked-test", action="store_true")
    evaluate.set_defaults(func=command_evaluate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"train", "evaluate", "attest-loader-migration"}:
        model_from_args(args).options.validate()
    args.func(args)


if __name__ == "__main__":
    main()
