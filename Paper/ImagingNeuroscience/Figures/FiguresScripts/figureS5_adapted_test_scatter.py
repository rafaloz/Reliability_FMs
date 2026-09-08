#!/usr/bin/env python3
"""Render locked internal-test scatterplots for the seven adapted models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-supp-adapted-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Analyses.figure3_brain_age import (  # noqa: E402
    EXPECTED_INTERNAL_TEST_ROWS,
    internal_test_table,
)
from Analyses.figure4_adaptation import (  # noqa: E402
    DEFAULT_ADAPTED_AUDIT,
    DEFAULT_BRAIN_AGE_ROOT,
    MODEL_SPECS,
    validate_adapted_internal_test,
    validate_global_adapted_audit,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Paper" / "ImagingNeuroscience" / "Figures"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "Results" / "FigureS5" / "adapted_scatter_manifest.json"
)

DATASET_ORDER = ("CamCAN", "NeuroCog", "WAND", "MR-ART", "NIMH-RV")
DATASET_COLORS = {
    "CamCAN": "#0072B2",
    "NeuroCog": "#CC79A7",
    "WAND": "#E69F00",
    "MR-ART": "#D55E00",
    "NIMH-RV": "#009E73",
}
PANEL_LETTERS = "abcdefg"
AGE_LIMITS = (0.0, 110.0)
AGE_TICKS = (0, 20, 40, 60, 80, 100)


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
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "supp-adapted-scatter-v1",
            "savefig.facecolor": "white",
        }
    )


def recompute_metrics(frame) -> dict[str, float]:
    truth = frame["age_true"].to_numpy(dtype=np.float64)
    prediction = frame["age_pred"].to_numpy(dtype=np.float64)
    error = prediction - truth
    denominator = np.square(truth - truth.mean()).sum()
    if denominator <= 0:
        raise ValueError("Internal-test ages have zero variance")
    return {
        "mae_years": float(np.abs(error).mean()),
        "rmse_years": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / denominator),
        "mean_error_years": float(error.mean()),
    }


def validate_reported_metrics(observed: dict, reported: dict, label: str) -> None:
    for key, value in observed.items():
        expected = float(reported[key])
        if not np.isclose(value, expected, rtol=0.0, atol=1e-10):
            raise ValueError(
                f"Reported/recomputed {key} mismatch for {label}: "
                f"{expected} != {value}"
            )


def load_conditions(
    brain_age_root: Path, adapted_audit: Path
) -> tuple[list[dict], list[dict], dict]:
    global_audit, source_map = validate_global_adapted_audit(adapted_audit)
    conditions = []
    sources = []
    reference_identity = None
    for spec in MODEL_SPECS:
        frames = []
        metrics = []
        for run in spec.primary_runs:
            target = f"{spec.model_dir}/{run.run}"
            internal_dir = (
                brain_age_root / "internal_test" / "adapted" / spec.model_dir / run.run
            )
            predictions_path = internal_dir / "predictions_internal_test.tsv"
            metrics_path = internal_dir / "metrics_internal_test.json"
            frame, provenance = validate_adapted_internal_test(
                metrics_path,
                predictions_path,
                spec=spec,
                run=run,
                source=source_map[target],
            )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            observed = recompute_metrics(frame)
            validate_reported_metrics(observed, payload["metrics"], target)
            identity = internal_test_table(frame)
            if reference_identity is None:
                reference_identity = identity
            elif not identity.equals(reference_identity):
                raise ValueError(f"Internal-test cohort mismatch for {target}")
            frames.append(frame)
            metrics.append(observed)
            sources.append(
                {
                    "model": spec.label,
                    "run": run.run,
                    "seed": run.seed,
                    "fold": run.fold,
                    "predictions": str(predictions_path.resolve()),
                    "predictions_sha256": provenance["predictions_sha256"],
                    "metrics": str(metrics_path.resolve()),
                    "metrics_sha256": provenance["metrics_sha256"],
                    "checkpoint_sha256": provenance["checkpoint_sha256"],
                }
            )
        conditions.append(
            {
                "condition": spec.condition,
                "label": spec.label,
                "frames": frames,
                "metrics": metrics,
                "run_definition": spec.primary_definition,
            }
        )
    if reference_identity is None or len(reference_identity) != EXPECTED_INTERNAL_TEST_ROWS:
        raise RuntimeError("The locked 140-participant internal test was not loaded")
    observed_datasets = set(reference_identity["dataset"].astype(str))
    if observed_datasets != set(DATASET_ORDER):
        raise ValueError(f"Unexpected internal-test datasets: {sorted(observed_datasets)}")
    return conditions, sources, global_audit


def metric_text(metrics: list[dict]) -> str:
    mae = np.asarray([record["mae_years"] for record in metrics], dtype=float)
    r2 = np.asarray([record["r2"] for record in metrics], dtype=float)
    return (
        f"MAE {mae.mean():.2f} $\\pm$ {mae.std(ddof=1):.2f} y\n"
        f"$R^2$ {r2.mean():.3f} $\\pm$ {r2.std(ddof=1):.3f}"
    )


def render(conditions: list[dict], output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib()
    figure = plt.figure(figsize=(8.15, 8.55), constrained_layout=False)
    grid = figure.add_gridspec(
        3,
        3,
        left=0.105,
        right=0.975,
        bottom=0.075,
        top=0.925,
        wspace=0.20,
        hspace=0.31,
    )
    panel_positions = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0))
    panel_axes = []
    for row, column in panel_positions:
        shared = panel_axes[0] if panel_axes else None
        panel_axes.append(figure.add_subplot(grid[row, column], sharex=shared, sharey=shared))

    for index, (axis, condition) in enumerate(zip(panel_axes, conditions)):
        for frame in condition["frames"]:
            for dataset in DATASET_ORDER:
                subset = frame.loc[frame["dataset"].astype(str) == dataset]
                axis.scatter(
                    subset["age_true"],
                    subset["age_pred"],
                    s=12,
                    c=DATASET_COLORS[dataset],
                    alpha=0.25,
                    edgecolors="none",
                    rasterized=False,
                    zorder=2,
                )
        axis.plot(
            AGE_LIMITS,
            AGE_LIMITS,
            color="#555D63",
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
        axis.set_title(
            f"{PANEL_LETTERS[index]}   {condition['label']}",
            loc="left",
            fontweight="bold",
            pad=7,
        )
        axis.text(
            0.04,
            0.95,
            metric_text(condition["metrics"]),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
            zorder=4,
        )
        axis.set_xlim(AGE_LIMITS)
        axis.set_ylim(AGE_LIMITS)
        axis.set_xticks(AGE_TICKS)
        axis.set_yticks(AGE_TICKS)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#E1E5E8", linewidth=0.6, zorder=0)

    legend_axis = figure.add_subplot(grid[2, 1:])
    legend_axis.axis("off")
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=DATASET_COLORS[dataset],
            markeredgecolor="none",
            label=dataset,
        )
        for dataset in DATASET_ORDER
    ]
    legend_axis.legend(
        handles=handles,
        title="Source dataset",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.96),
        ncol=2,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.3,
        labelspacing=0.65,
    )
    figure.supxlabel("Chronological age (years)", y=0.018, fontsize=10.5)
    figure.supylabel("Predicted age (years)", x=0.025, fontsize=10.5)
    figure.suptitle(
        "Locked internal-test predictions: controlled encoder adaptation",
        x=0.105,
        y=0.982,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "FigureS5_adapted_test_scatter.png",
        output_dir / "FigureS5_adapted_test_scatter.svg",
    ]
    for destination in outputs:
        temporary = destination.with_name(
            f".{destination.stem}.{os.getpid()}{destination.suffix}"
        )
        figure.savefig(
            temporary,
            dpi=dpi if destination.suffix == ".png" else None,
            format=destination.suffix.lstrip("."),
            bbox_inches="tight",
        )
        os.replace(temporary, destination)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-age-root", default=DEFAULT_BRAIN_AGE_ROOT)
    parser.add_argument("--adapted-audit", default=DEFAULT_ADAPTED_AUDIT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    adapted_audit = Path(args.adapted_audit).expanduser().resolve()
    conditions, sources, global_audit = load_conditions(
        Path(args.brain_age_root).expanduser().resolve(), adapted_audit
    )
    outputs = render(
        conditions, Path(args.output_dir).expanduser().resolve(), args.dpi
    )
    manifest_path = Path(args.manifest).expanduser().resolve()
    atomic_json(
        manifest_path,
        {
            "figure": "Supplementary Figure S8: adapted internal-test predictions",
            "cohort": {"split": "internal_test", "subjects": 140},
            "display": (
                "one point per locked subject-run prediction; metrics averaged across "
                "runs; predictions not ensembled"
            ),
            "global_adapted_audit": str(adapted_audit),
            "global_adapted_audit_sha256": sha256_file(adapted_audit),
            "global_audit_status": global_audit["status"],
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "sources": sources,
            "outputs": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in outputs
            ],
        },
    )
    for path in outputs:
        print(f"Saved: {path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
