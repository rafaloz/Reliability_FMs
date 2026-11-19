#!/usr/bin/env python3
"""Render Figure 3 from the locked frozen-Ridge brain-age analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-figure3-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in SCRIPT_PATH.parents if (parent / 'Analyses').is_dir())
DEFAULT_RESULTS = PROJECT_ROOT / "Results" / "Figure3" / "figure3_brain_age_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figures"
DEFAULT_MANIFEST = PROJECT_ROOT / "Results" / "Figure3" / "figure3_figure_manifest.json"

DISPLAY_ORDER = (
    "yaware",
    "anatcl",
    "brainsegfounder",
    "3dneurosimclr",
    "neurovfm",
    "curia2",
    "brainiac",
)

COLORS = {
    "turquoise": "#3E8F27",
    "orange": "#FF7F11",
    "embedding": "#545963",
    "accuracy": "#545963",
    "connector": "#B8BEC4",
    "grid": "#D9DEE2",
    "zero": "#545963",
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
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "figure3-brain-age-v2",
            "savefig.facecolor": "white",
        }
    )


def finite_triplet(record: dict, name: str) -> tuple[float, float, float]:
    values = np.asarray(
        [record.get("estimate"), record.get("ci_lower"), record.get("ci_upper")],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {name}")
    estimate, lower, upper = values.tolist()
    if lower > estimate or estimate > upper:
        raise ValueError(f"Invalid confidence interval for {name}")
    return estimate, lower, upper


def validate_payload(payload: dict) -> dict:
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Figure 3 result schema")
    conditions = payload.get("conditions", {})
    if set(conditions) != set(DISPLAY_ORDER):
        raise ValueError(
            "Figure 3 conditions differ from the seven prespecified models: "
            f"{sorted(conditions)}"
        )
    if payload.get("cohorts", {}).get("internal_test", {}).get("subjects") != 140:
        raise ValueError("Figure 3 internal-test cohort is not locked to 140 subjects")
    onharmony = payload.get("cohorts", {}).get("onharmony", {})
    expected_onharmony = {
        "sessions": 165,
        "participants": 20,
        "participant_scanner_cells": 120,
        "scanners": 8,
        "scanners_per_participant": 6,
    }
    for key, expected in expected_onharmony.items():
        if onharmony.get(key) != expected:
            raise ValueError(f"Unexpected ON-Harmony {key}: {onharmony.get(key)}")

    for condition in DISPLAY_ORDER:
        record = conditions[condition]
        finite_triplet(record.get("internal_test_mae", {}), f"{condition} MAE")
        finite_triplet(record.get("embedding_dbicc", {}), f"{condition} embedding dbICC")
        finite_triplet(record.get("prediction_dbicc", {}), f"{condition} prediction dbICC")
        finite_triplet(
            record.get("cross_scanner_absolute_difference", {}),
            f"{condition} absolute scanner difference",
        )
    analysis_script = Path(payload.get("code", {}).get("analysis_script", ""))
    expected_hash = payload.get("code", {}).get("analysis_script_sha256")
    if not analysis_script.is_file():
        raise FileNotFoundError(
            f"Figure 3 analysis script recorded in the results is absent: "
            f"{analysis_script}"
        )
    if sha256_file(analysis_script) != expected_hash:
        raise ValueError("Figure 3 analysis script hash no longer matches its results")
    return conditions


def draw_interval(
    axis: mpl.axes.Axes,
    y: float,
    record: dict,
    *,
    color: str,
    marker: str = "o",
    marker_face: str | None = None,
    marker_edge: str | None = None,
    marker_size: float = 5.8,
    zorder: int = 4,
) -> None:
    estimate, lower, upper = finite_triplet(record, "plotted interval")
    axis.errorbar(
        estimate,
        y,
        xerr=np.asarray([[estimate - lower], [upper - estimate]]),
        linestyle="none",
        marker=marker,
        markerfacecolor=marker_face if marker_face is not None else color,
        markeredgecolor=marker_edge if marker_edge is not None else color,
        markeredgewidth=1.15,
        markersize=marker_size,
        color=color,
        ecolor=color,
        elinewidth=1.25,
        capsize=2.5,
        capthick=0.9,
        zorder=zorder,
    )


def style_forest_axis(axis: mpl.axes.Axes, count: int) -> None:
    axis.set_ylim(count - 0.48, -0.52)
    axis.set_yticks(np.arange(count))
    axis.tick_params(axis="y", length=0)
    axis.xaxis.grid(True, color=COLORS["grid"], linewidth=0.65, zorder=0)
    axis.set_axisbelow(True)
    for row in np.arange(count - 1) + 0.5:
        axis.axhline(row, color="#EEF0F2", linewidth=0.55, zorder=0)


def render(payload: dict, *, output_dir: Path, dpi: int) -> list[Path]:
    conditions = validate_payload(payload)
    configure_matplotlib()

    count = len(DISPLAY_ORDER)
    labels = [conditions[condition]["label"] for condition in DISPLAY_ORDER]

    figure = plt.figure(figsize=(13.6, 4.8), constrained_layout=False)
    figure.suptitle(
        "Brain-age prediction from frozen embeddings",
        x=0.53, y=0.99, fontsize=20, fontweight="bold",
    )
    top = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.16, 1.0),
        wspace=0.20,
        left=0.19,
        right=0.965,
        bottom=0.14,
        top=0.69,
    )
    axis_mae = figure.add_subplot(top[0])
    axis_icc = figure.add_subplot(top[1], sharey=axis_mae)
    axis_difference = figure.add_subplot(top[2], sharey=axis_mae)

    for axis in (axis_mae, axis_icc, axis_difference):
        style_forest_axis(axis, count)
        axis.tick_params(axis="x", labelsize=12)
    axis_mae.set_yticklabels(labels)
    axis_mae.tick_params(axis="y", pad=7, labelsize=13)
    for axis in (axis_icc, axis_difference):
        axis.tick_params(axis="y", labelleft=False)

    for row, condition in enumerate(DISPLAY_ORDER):
        record = conditions[condition]
        mae = record["internal_test_mae"]
        mae_value, mae_lower, mae_upper = finite_triplet(
            mae, "internal-test MAE"
        )
        axis_mae.barh(
            row,
            mae_value,
            height=0.18,
            color=COLORS["turquoise"],
            edgecolor="none",
            zorder=2,
        )
        axis_mae.errorbar(
            mae_value,
            row,
            xerr=np.asarray(
                [[mae_value - mae_lower], [mae_upper - mae_value]]
            ),
            linestyle="none",
            marker="none",
            color=COLORS["accuracy"],
            ecolor=COLORS["accuracy"],
            elinewidth=1.05,
            capsize=2.4,
            capthick=0.85,
            zorder=3,
        )
        embedding = record["embedding_dbicc"]
        ridge = record["prediction_dbicc"]
        embedding_value = float(embedding["estimate"])
        ridge_value = float(ridge["estimate"])
        axis_icc.plot(
            [embedding_value, ridge_value],
            [row - 0.10, row + 0.10],
            color=COLORS["connector"],
            linewidth=1.1,
            zorder=1,
        )
        draw_interval(
            axis_icc,
            row - 0.10,
            embedding,
            color=COLORS["embedding"],
            marker_face="white",
            marker_size=5.3,
            zorder=3,
        )
        draw_interval(
            axis_icc,
            row + 0.10,
            ridge,
            color=COLORS["orange"],
            marker_size=5.8,
            zorder=4,
        )

        difference = record["cross_scanner_absolute_difference"]
        draw_interval(
            axis_difference,
            row,
            difference,
            color=COLORS["accuracy"],
            marker_face=COLORS["accuracy"],
            marker_edge=COLORS["accuracy"],
        )

    axis_mae.set_xlim(0, 13.35)
    axis_mae.set_xticks((0, 4, 8, 12))
    axis_mae.set_xlabel("Internal-test MAE (years)")
    axis_mae.set_title("a   Internal test accuracy", loc="left", fontweight="bold")

    axis_icc.axvline(
        0, color=COLORS["zero"], linestyle="--", linewidth=0.8, zorder=0
    )
    axis_icc.set_xlim(-0.12, 1.04)
    axis_icc.set_xticks(np.arange(0, 1.01, 0.2))
    axis_icc.set_xlabel("Cross-scanner dbICC")
    axis_icc.set_title(
        "b   Embedding vs brain age\nprediction reliability",
        loc="left",
        fontweight="bold",
    )

    legend_handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["embedding"],
            label="Full embedding",
        ),
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["orange"],
            markeredgecolor=COLORS["orange"],
            label="Frozen Ridge brain age",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.91),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
    )

    axis_difference.set_xlim(0, 12.25)
    axis_difference.set_xticks((0, 4, 8, 12))
    axis_difference.set_xlabel("Mean absolute difference (years)")
    axis_difference.set_title(
        "c   Cross-scanner differences", loc="left", fontweight="bold"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "Figure3_brain_age.png",
        output_dir / "Figure3_brain_age.svg",
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
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    result_path = Path(args.results).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    outputs = render(payload, output_dir=output_dir, dpi=args.dpi)
    manifest = {
        "panels": ["internal_test_accuracy", "cross_scanner_reliability", "cross_scanner_absolute_difference"],
        "figure": "Figure 3: frozen-Ridge brain-age utility and cross-acquisition robustness",
        "analysis": str(result_path),
        "analysis_sha256": sha256_file(result_path),
        "analysis_script": payload["code"]["analysis_script"],
        "analysis_script_sha256": payload["code"]["analysis_script_sha256"],
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "outputs": [
            {
                "path": str(output),
                "sha256": sha256_file(output),
                "bytes": output.stat().st_size,
            }
            for output in outputs
        ],
    }
    atomic_json(manifest_path, manifest)
    for output in outputs:
        print(f"Saved: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
