"""Frozen and controlled-adaptation hyperparameter protocol and plots."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from FineTuning.shared.cohort import sha256_file
from FineTuning.shared.runtime import atomic_json, atomic_tsv


FROZEN_RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

ADAPTED_LR_CONFIGS: dict[str, dict[str, float]] = {
    "conservative": {"lr_encoder": 1e-6, "lr_head": 1e-4},
    "intermediate": {"lr_encoder": 3e-6, "lr_head": 3e-4},
    "high": {"lr_encoder": 1e-5, "lr_head": 1e-3},
}

ADAPTED_FIXED_HYPERPARAMETERS: dict[str, int | float] = {
    "epochs": 30,
    "patience": 7,
    "min_delta": 0.01,
    "weight_decay": 0.01,
    "accum_subjects": 8,
}


def ridge_alpha_argument() -> str:
    """Return the frozen Ridge grid in the CLI's stable comma-separated form."""

    return ",".join(f"{value:g}" for value in FROZEN_RIDGE_ALPHAS)


def adapted_learning_rates(config: str) -> tuple[float, float]:
    """Resolve one named and predeclared encoder/head learning-rate pair."""

    if config not in ADAPTED_LR_CONFIGS:
        raise ValueError(
            f"Unknown adapted LR configuration {config!r}; "
            f"expected one of {tuple(ADAPTED_LR_CONFIGS)}"
        )
    values = ADAPTED_LR_CONFIGS[config]
    return float(values["lr_encoder"]), float(values["lr_head"])


def _pyplot() -> Any:
    """Load matplotlib lazily with a non-interactive backend for cluster jobs."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Training-curve generation requires matplotlib") from error
    return plt


def _atomic_save_figure(figure: Any, destination: Path, *, fmt: str, dpi: int | None) -> None:
    """Write a figure atomically so interrupted jobs do not leave partial plots."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    options: dict[str, Any] = {
        "format": fmt,
        "bbox_inches": "tight",
        "facecolor": "white",
    }
    if dpi is not None:
        options["dpi"] = dpi
    figure.savefig(temporary, **options)
    os.replace(temporary, destination)


def _save_png_pdf(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    """Save one plot in review-friendly PNG and vector PDF formats."""

    outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    _atomic_save_figure(figure, outputs[0], fmt="png", dpi=220)
    _atomic_save_figure(figure, outputs[1], fmt="pdf", dpi=None)
    return outputs


def write_ridge_selection_plot(candidates: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot validation MAE across the fixed Ridge regularization grid."""

    required = {"alpha", "mae_years"}
    if not required.issubset(candidates.columns) or candidates.empty:
        raise ValueError(f"Ridge selection table must contain {sorted(required)}")
    table = candidates.sort_values("alpha", kind="stable").reset_index(drop=True)
    values = table[["alpha", "mae_years"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 0] <= 0).any():
        raise ValueError("Ridge selection plot received invalid alpha/MAE values")
    selected = candidates.sort_values(
        ["mae_years", "alpha"], ascending=[True, False], kind="stable"
    ).iloc[0]

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    axis.plot(table["alpha"], table["mae_years"], color="#355C7D", marker="o", linewidth=1.8)
    axis.scatter(
        [selected["alpha"]], [selected["mae_years"]],
        marker="*", s=150, color="#A23B3B", zorder=3, label="Selected",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Ridge alpha")
    axis.set_ylabel("Validation MAE (years)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    outputs = _save_png_pdf(figure, Path(output_dir), "ridge_validation_curve")
    plt.close(figure)
    return outputs


def write_training_curves(
    history: pd.DataFrame,
    output_dir: Path,
    *,
    title: str,
    selected_epoch: int,
) -> list[Path]:
    """Plot comparable standardized loss and age MAE for one adapted run."""

    required = {
        "epoch",
        "train_mse_z",
        "validation_mse_z",
        "train_mae_years_online",
        "validation_mae_years",
    }
    if not required.issubset(history.columns) or history.empty:
        raise ValueError(f"Training history must contain {sorted(required)}")
    table = history.sort_values("epoch", kind="stable").reset_index(drop=True)
    values = table[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training history contains non-finite values")
    if selected_epoch not in set(table["epoch"].astype(int)):
        raise ValueError(f"Selected epoch {selected_epoch} is absent from training history")

    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    epoch = table["epoch"]
    axes[0].plot(epoch, table["train_mse_z"], color="#355C7D", label="Train (online)")
    axes[0].plot(epoch, table["validation_mse_z"], color="#A23B3B", label="Validation")
    axes[0].set_ylabel("MSE (standardized age)")
    axes[1].plot(
        epoch, table["train_mae_years_online"], color="#355C7D", label="Train (online)"
    )
    axes[1].plot(epoch, table["validation_mae_years"], color="#A23B3B", label="Validation")
    axes[1].set_ylabel("MAE (years)")
    for axis in axes:
        axis.axvline(selected_epoch, color="#333333", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.suptitle(f"{title} | selected epoch {selected_epoch}", fontsize=11)
    outputs = _save_png_pdf(figure, Path(output_dir), "training_curves")
    plt.close(figure)
    return outputs


def _canonical_json(value: object) -> str:
    """Create a stable representation for equality checks on nested provenance."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_adapted_candidate(
    run_dir: Path,
    *,
    expected_model: str,
    expected_variant: str,
    manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one locked validation-only run before hyperparameter selection."""

    run_dir = run_dir.resolve()
    lock_path = run_dir / "LOCKED.json"
    history_path = run_dir / "train_history.tsv"
    if not lock_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Selection requires {lock_path} and {history_path}")
    if (run_dir / "INTERNAL_TEST_EVALUATED.json").exists():
        raise ValueError(f"Candidate already accessed internal_test: {run_dir}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("arm") != "adapted" or lock.get("status") != "LOCKED_AFTER_TRAIN_VALIDATION":
        raise ValueError(f"Candidate is not a locked adapted run: {run_dir}")
    if (lock.get("model"), lock.get("variant")) != (expected_model, expected_variant):
        raise ValueError(f"Candidate model/variant differs: {run_dir}")
    if lock.get("internal_test_accessed") is not False or lock.get("on_harmony_accessed") is not False:
        raise ValueError(f"Candidate violates the validation-only firewall: {run_dir}")
    if lock.get("manifest", {}).get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Candidate uses a different subject manifest: {run_dir}")
    checkpoint = run_dir / str(lock.get("checkpoint", "selected_model.pt"))
    if not checkpoint.is_file() or sha256_file(checkpoint) != lock.get("checkpoint_sha256"):
        raise ValueError(f"Candidate checkpoint hash differs: {run_dir}")

    arguments = lock.get("run_config", {}).get("arguments", {})
    lr_config = str(arguments.get("lr_config", ""))
    expected_encoder, expected_head = adapted_learning_rates(lr_config)
    if not math.isclose(float(arguments.get("lr_encoder", math.nan)), expected_encoder):
        raise ValueError(f"Encoder LR does not match {lr_config}: {run_dir}")
    if not math.isclose(float(arguments.get("lr_head", math.nan)), expected_head):
        raise ValueError(f"Head LR does not match {lr_config}: {run_dir}")
    for key, expected in ADAPTED_FIXED_HYPERPARAMETERS.items():
        observed = arguments.get(key)
        if isinstance(expected, float):
            valid = math.isclose(float(observed), expected)
        else:
            valid = observed == expected
        if not valid:
            raise ValueError(
                f"Candidate {key}={observed!r}, expected fixed value {expected!r}: {run_dir}"
            )
    metric = float(lock.get("selection_value", math.nan))
    locked_mae = float(lock.get("locked_validation_metrics", {}).get("mae_years", math.nan))
    if not math.isfinite(metric) or not math.isclose(
        metric, locked_mae, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"Candidate validation metric is invalid or inconsistent: {run_dir}")

    row = {
        "lr_config": lr_config,
        "lr_encoder": expected_encoder,
        "lr_head": expected_head,
        "seed": int(arguments["seed"]),
        "selected_epoch": int(lock["selected_epoch"]),
        "validation_mae_years": locked_mae,
        "validation_rmse_years": float(lock["locked_validation_metrics"]["rmse_years"]),
        "validation_r2": float(lock["locked_validation_metrics"]["r2"]),
        "training_elapsed_seconds": float(lock["training_elapsed_seconds"]),
        "run_dir": str(run_dir),
        "checkpoint_sha256": str(lock["checkpoint_sha256"]),
    }
    details = {
        "lock": lock,
        "arguments": arguments,
        "release_key": _canonical_json(lock.get("release")),
        "code_key": _canonical_json(lock.get("code_sha256")),
    }
    return row, details


def select_adapted_learning_rate(
    run_dirs: Iterable[Path],
    output_dir: Path,
    *,
    expected_model: str,
    expected_variant: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Select one of the three validation-only LR runs and write an audit artifact."""

    paths = [Path(path).resolve() for path in run_dirs]
    if len(paths) != len(set(paths)):
        raise ValueError("Adapted LR candidate directories must be unique")
    candidates = [
        _load_adapted_candidate(
            path,
            expected_model=expected_model,
            expected_variant=expected_variant,
            manifest_sha256=manifest_sha256,
        )
        for path in paths
    ]
    rows = [row for row, _ in candidates]
    details = [detail for _, detail in candidates]
    observed_configs = {row["lr_config"] for row in rows}
    if observed_configs != set(ADAPTED_LR_CONFIGS) or len(rows) != len(ADAPTED_LR_CONFIGS):
        raise ValueError(
            "Selection requires exactly one run for each LR configuration: "
            f"observed={sorted(observed_configs)}"
        )
    if len({row["seed"] for row in rows}) != 1:
        raise ValueError("LR configurations must use the same selection seed")
    if len({detail["release_key"] for detail in details}) != 1:
        raise ValueError("LR candidates use different base releases")
    if len({detail["code_key"] for detail in details}) != 1:
        raise ValueError("LR candidates use different runtime code")
    shared_keys = (
        "amp", "slice_batch_size",
        "model", "fold", "input_root", "checkpoint", "model_dir", "model_source",
        "model_revision",
    )
    for key in shared_keys:
        if len({_canonical_json(detail["arguments"].get(key)) for detail in details}) != 1:
            raise ValueError(f"LR candidates differ in non-LR argument {key}")

    table = pd.DataFrame(rows)
    rank = table.sort_values(
        ["validation_mae_years", "lr_encoder"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected = rank.iloc[0].to_dict()
    output_dir = Path(output_dir)
    atomic_tsv(
        table.sort_values("lr_encoder", kind="stable").reset_index(drop=True),
        output_dir / "adapted_lr_validation_grid.tsv",
    )
    payload = {
        "status": "LR_SELECTED_AFTER_TRAIN_VALIDATION",
        "model": expected_model,
        "variant": expected_variant,
        "selection_rule": "minimum uncorrected validation MAE; lower encoder LR breaks exact ties",
        "selection_seed": int(selected["seed"]),
        "selected_lr_config": str(selected["lr_config"]),
        "selected_run_dir": str(selected["run_dir"]),
        "selected_checkpoint_sha256": str(selected["checkpoint_sha256"]),
        "selected_validation_mae_years": float(selected["validation_mae_years"]),
        "manifest_sha256": manifest_sha256,
        "candidates": rows,
        "internal_test_accessed": False,
        "on_harmony_accessed": False,
    }
    atomic_json(payload, output_dir / "SELECTED_LR.json")
    write_lr_selection_plot(table, output_dir, selected_config=str(selected["lr_config"]))
    return payload


def write_lr_selection_plot(
    candidates: pd.DataFrame,
    output_dir: Path,
    *,
    selected_config: str,
) -> list[Path]:
    """Plot the three validation MAEs used for adapted LR selection."""

    by_config = candidates.set_index("lr_config")
    order = list(ADAPTED_LR_CONFIGS)
    if set(by_config.index) != set(order):
        raise ValueError("LR selection plot requires the three declared configurations")
    values = np.asarray([by_config.loc[name, "validation_mae_years"] for name in order], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("LR selection plot received non-finite validation MAE")
    labels = [
        f"{name}\n{ADAPTED_LR_CONFIGS[name]['lr_encoder']:g} / "
        f"{ADAPTED_LR_CONFIGS[name]['lr_head']:g}"
        for name in order
    ]
    colors = ["#A23B3B" if name == selected_config else "#6F8294" for name in order]

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    positions = np.arange(len(order))
    axis.bar(positions, values, width=0.55, color=colors)
    axis.set_xticks(positions, labels)
    axis.set_xlabel("LR configuration (encoder / head)")
    axis.set_ylabel("Validation MAE (years)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    outputs = _save_png_pdf(figure, Path(output_dir), "adapted_lr_validation_comparison")
    plt.close(figure)
    return outputs
