#!/usr/bin/env python3
"""Render the post hoc BrainIAC learning-rate sensitivity curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-figureS7-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "Results" / "FigureS7" / "brainiac_lr_sensitivity_inputs"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Paper" / "ImagingNeuroscience" / "Figures"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "Results" / "FigureS7" / "brainiac_lr_sensitivity_manifest.json"
)

TRAIN_COLOR = "#545963"
VALIDATION_COLOR = "#FF7F11"
SELECTED_COLOR = "#3E8F27"

CONDITIONS = (
    {
        "key": "common_high",
        "title": r"Common grid: $10^{-5}$",
        "expected_lr_encoder": 1e-5,
        "expected_lr_head": 1e-3,
        "expected_mae": 11.88184457331376,
    },
    {
        "key": "encoder_1e4",
        "title": r"Post hoc: $10^{-4}$",
        "expected_lr_encoder": 1e-4,
        "expected_lr_head": 1e-3,
        "expected_mae": 10.08586079649433,
    },
    {
        "key": "encoder_3e4",
        "title": r"Post hoc: $3\times10^{-4}$",
        "expected_lr_encoder": 3e-4,
        "expected_lr_head": 1e-3,
        "expected_mae": 9.282639591895393,
    },
)

REQUIRED_COLUMNS = {
    "epoch",
    "train_mse_z",
    "validation_mse_z",
    "validation_mae_years",
    "selected",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["ArialMT", "Arial", "Arimo", "Liberation Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "brainiac-lr-sensitivity-v1",
            "savefig.facecolor": "white",
        }
    )


def read_condition(input_dir: Path, condition: dict) -> dict:
    key = condition["key"]
    history_path = input_dir / f"{key}_train_history.tsv"
    metrics_path = input_dir / f"{key}_locked_validation_metrics.json"
    config_path = input_dir / f"{key}_run_config.json"
    for path in (history_path, metrics_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    history = pd.read_csv(history_path, sep="\t")
    missing = REQUIRED_COLUMNS.difference(history.columns)
    if missing:
        raise ValueError(f"{history_path} lacks columns: {sorted(missing)}")
    if history.empty or not np.array_equal(
        history["epoch"].to_numpy(dtype=int), np.arange(1, len(history) + 1)
    ):
        raise ValueError(f"{history_path} does not contain consecutive epochs")
    numeric = history[["train_mse_z", "validation_mse_z", "validation_mae_years"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{history_path} contains non-finite metrics")

    selected = history.loc[history["selected"].astype(str).str.lower() == "true"]
    if selected.empty:
        raise ValueError(f"{history_path} contains no selected checkpoint")
    selected_row = selected.iloc[-1]

    metrics = json.loads(metrics_path.read_text())
    config = json.loads(config_path.read_text())
    arguments = config["arguments"]
    observed_lr_encoder = float(arguments["lr_encoder"])
    observed_lr_head = float(arguments["lr_head"])
    observed_mae = float(metrics["mae_years"])
    if int(arguments["seed"]) != 0 or str(arguments["amp"]) != "bfloat16":
        raise ValueError(f"Unexpected BrainIAC run contract in {config_path}")
    if not np.isclose(
        observed_lr_encoder, condition["expected_lr_encoder"], rtol=0, atol=1e-15
    ):
        raise ValueError(f"Unexpected encoder LR in {config_path}")
    if not np.isclose(
        observed_lr_head, condition["expected_lr_head"], rtol=0, atol=1e-15
    ):
        raise ValueError(f"Unexpected head LR in {config_path}")
    if not np.isclose(observed_mae, condition["expected_mae"], rtol=0, atol=1e-12):
        raise ValueError(f"Unexpected locked validation MAE in {metrics_path}")
    if not np.isclose(
        float(selected_row["validation_mae_years"]), observed_mae, rtol=0, atol=1e-12
    ):
        raise ValueError(f"History and locked validation MAE differ for {key}")

    return {
        **condition,
        "history": history,
        "selected_epoch": int(selected_row["epoch"]),
        "validation_mae_years": observed_mae,
        "source_run_dir": str(arguments["output_dir"]),
        "paths": (history_path, metrics_path, config_path),
    }


def render(conditions: list[dict], output_dir: Path) -> list[Path]:
    configure_matplotlib()
    plotted_conditions = [
        condition for condition in conditions if condition["key"] != "common_high"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.75), sharex=True, sharey=True)
    for panel_index, (axis, condition) in enumerate(
        zip(axes, plotted_conditions, strict=True)
    ):
        history = condition["history"]
        epochs = history["epoch"].to_numpy(dtype=int)
        axis.plot(
            epochs,
            history["train_mse_z"],
            color=TRAIN_COLOR,
            linewidth=2.1,
            label="Train (online)",
        )
        axis.plot(
            epochs,
            history["validation_mse_z"],
            color=VALIDATION_COLOR,
            linewidth=2.1,
            label="Validation",
        )
        selected_epoch = condition["selected_epoch"]
        selected_row = history.loc[history["epoch"] == selected_epoch].iloc[0]
        axis.axvline(
            selected_epoch,
            color=SELECTED_COLOR,
            linestyle="--",
            linewidth=1.35,
            alpha=0.9,
            label="Selected epoch" if panel_index == 0 else None,
        )
        axis.scatter(
            [selected_epoch],
            [float(selected_row["validation_mse_z"])],
            s=34,
            color=SELECTED_COLOR,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        axis.set_title(condition["title"], pad=8)
        axis.set_xlabel("Epoch")
        axis.set_xlim(1, 30)
        axis.set_ylim(0.08, 0.82)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, zorder=0)
        axis.text(
            -0.11,
            1.08,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    axes[0].set_ylabel("MSE (standardised age)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.23,
        top=0.86,
        wspace=0.15,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "FigureS7_brainiac_lr_sensitivity.png",
        output_dir / "FigureS7_brainiac_lr_sensitivity.svg",
    ]
    with tempfile.TemporaryDirectory(prefix="figureS7-", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        temporary_outputs = [temporary_dir / output.name for output in outputs]
        figure.savefig(temporary_outputs[0], dpi=300, bbox_inches="tight")
        figure.savefig(temporary_outputs[1], bbox_inches="tight")
        for temporary_output, output in zip(temporary_outputs, outputs, strict=True):
            os.replace(temporary_output, output)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    conditions = [read_condition(input_dir, condition) for condition in CONDITIONS]
    outputs = render(conditions, output_dir)
    payload = {
        "schema_version": "brainiac_lr_sensitivity_figure_v1",
        "analysis_role": "post_hoc_validation_only_sensitivity",
        "internal_test_evaluated": False,
        "on_harmony_evaluated": False,
        "conditions": [
            {
                "key": condition["key"],
                "lr_encoder": condition["expected_lr_encoder"],
                "lr_head": condition["expected_lr_head"],
                "selected_epoch": condition["selected_epoch"],
                "validation_mae_years": condition["validation_mae_years"],
                "source_run_dir": condition["source_run_dir"],
                "inputs_sha256": {
                    path.name: sha256_file(path) for path in condition["paths"]
                },
            }
            for condition in conditions
        ],
        "generator": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
        "generator_sha256": sha256_file(SCRIPT_PATH),
        "outputs_sha256": {output.name: sha256_file(output) for output in outputs},
    }
    manifest = args.manifest.expanduser().resolve()
    atomic_json(manifest, payload)
    for output in outputs:
        print(output)
    print(manifest)


if __name__ == "__main__":
    main()
