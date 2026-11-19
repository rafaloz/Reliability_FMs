#!/usr/bin/env python3
"""Render manuscript Figure 4 from the frozen and fine-tuned analysis results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import figure3_brain_age as frozen_plot
import figure4_adaptation as adapted_plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).resolve()
ORDER = frozen_plot.DISPLAY_ORDER
COLORS = {"embedding": "#545963", "ridge": "#3E8F27", "adapted": "#FF7F11"}
MARKERS = {"embedding": "o", "ridge": "o", "adapted": "D"}


def validate_inputs(frozen, adapted):
    frozen_conditions = frozen_plot.validate_payload(frozen)
    adapted_conditions = adapted_plot.validate_payload(adapted)
    metrics = {
        "mae": "internal_test_mae",
        "dbicc": "prediction_dbicc",
        "scanner_difference": "cross_scanner_absolute_difference",
    }
    for name in ORDER:
        for metric, frozen_key in metrics.items():
            reference = frozen_conditions[name][frozen_key]
            repeated = adapted_conditions[name]["absolute"]["frozen"][metric]
            for key in ("estimate", "ci_lower", "ci_upper", "bootstrap_values"):
                np.testing.assert_allclose(reference[key], repeated[key], rtol=0, atol=1e-12)
    return frozen_conditions, adapted_conditions


def draw_state(axis, row, interval, state, offset):
    frozen_plot.draw_interval(
        axis, row + offset, interval, color=COLORS[state], marker=MARKERS[state],
        marker_face="white" if state == "embedding" else COLORS[state],
        marker_size=5.2 if state == "adapted" else 5.6, zorder=4,
    )


def draw_pair(axis, row, record, metric):
    frozen = record["absolute"]["frozen"][metric]
    adapted = record["absolute"]["adapted"][metric]
    axis.plot(
        [frozen["estimate"], adapted["estimate"]], [row - 0.12, row + 0.12],
        color="#9EA5AC", linewidth=1.2, zorder=2,
    )
    draw_state(axis, row, frozen, "ridge", -0.12)
    draw_state(axis, row, adapted, "adapted", 0.12)


def render(frozen, adapted, output_dir, dpi=400):
    frozen_conditions, adapted_conditions = validate_inputs(frozen, adapted)
    adapted_plot.configure_matplotlib()
    fig = plt.figure(figsize=(14.8, 5.4))
    grid = fig.add_gridspec(
        1, 3, width_ratios=(1.0, 1.45, 1.0), wspace=0.22,
        left=0.17, right=0.985, bottom=0.13, top=0.81,
    )
    axes = [fig.add_subplot(grid[i]) for i in range(3)]
    axes[1].sharey(axes[0])
    axes[2].sharey(axes[0])
    labels = [frozen_conditions[name]["label"] for name in ORDER]
    for axis, panel in zip(axes, adapted_plot.PANELS):
        adapted_plot.style_axis(axis, panel, len(ORDER))
    axes[0].set_yticklabels(labels)
    axes[0].tick_params(axis="y", pad=7)
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    axes[0].set_title("a   Internal-test accuracy", loc="left", fontweight="bold", pad=11)
    axes[0].set_xlabel("MAE (years)")
    axes[1].set_xlim(-0.12, 1.04)
    axes[1].set_xlabel("Cross-scanner dbICC")
    axes[1].axvline(0, color="#ADB2B8", linewidth=0.8, linestyle="--", zorder=0)
    axes[2].set_xlim(0, 13)
    axes[2].set_xticks((0, 2, 4, 6, 8, 10, 12))
    axes[2].set_xlabel("Mean absolute difference (years)")

    plotted = {}
    for row, name in enumerate(ORDER):
        record = adapted_conditions[name]
        draw_pair(axes[0], row, record, "mae")
        draw_pair(axes[2], row, record, "scanner_difference")
        states = (
            ("embedding", frozen_conditions[name]["embedding_dbicc"], -0.23),
            ("ridge", record["absolute"]["frozen"]["dbicc"], 0),
            ("adapted", record["absolute"]["adapted"]["dbicc"], 0.23),
        )
        axes[1].plot(
            [state[1]["estimate"] for state in states[:2]],
            [row + state[2] for state in states[:2]],
            color="#CDD1D5", linewidth=0.8, zorder=2,
        )
        axes[1].plot(
            [state[1]["estimate"] for state in states[1:]],
            [row + state[2] for state in states[1:]],
            color="#9EA5AC", linewidth=1.2, zorder=2,
        )
        for state, interval, offset in states:
            draw_state(axes[1], row, interval, state, offset)
        plotted[name] = {
            "embedding_dbicc": {k: states[0][1][k] for k in ("estimate", "ci_lower", "ci_upper")},
            "prediction_metrics": {
                arm: {
                    metric: {k: record["absolute"][arm][metric][k] for k in ("estimate", "ci_lower", "ci_upper")}
                    for metric in ("mae", "dbicc", "scanner_difference")
                } for arm in ("frozen", "adapted")
            },
        }

    fig.legend(
        handles=[
            mpl.lines.Line2D([], [], linestyle="none", marker=MARKERS[key], markersize=6,
                markerfacecolor="white" if key == "embedding" else COLORS[key],
                markeredgecolor=COLORS[key], label=label)
            for key, label in [
                ("embedding", "Full embedding"),
                ("ridge", "Frozen Ridge"),
                ("adapted", "Fine-tuned"),
            ]
        ],
        loc="upper center", bbox_to_anchor=(0.55, 0.985), ncol=3,
        frameon=False, handletextpad=0.45, columnspacing=2.0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg", "pdf"):
        destination = output_dir / ("Figure_combined_brain_age." + suffix)
        temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.{suffix}")
        fig.savefig(temporary, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        os.replace(temporary, destination)
        outputs.append(destination)
    preview = output_dir / "Figure_combined_brain_age_preview.png"
    fig.savefig(preview, dpi=120, bbox_inches="tight")
    outputs.append(preview)
    plt.close(fig)
    return outputs, plotted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-results", type=Path, required=True)
    parser.add_argument("--adapted-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_results.read_text())
    adapted = json.loads(args.adapted_results.read_text())
    outputs, plotted = render(frozen, adapted, args.output_dir, args.dpi)
    sources = [args.frozen_results, args.adapted_results, SCRIPT, Path(frozen_plot.__file__), Path(adapted_plot.__file__)]
    manifest = {
        "status": "PASS",
        "manuscript_figure": 4,
        "variant": "paired points, distinct markers, wider dbICC panel",
        "inputs": {str(p.resolve()): frozen_plot.sha256_file(p) for p in sources},
        "outputs": {str(p.resolve()): frozen_plot.sha256_file(p) for p in outputs},
        "intervals": "95% participant-bootstrap CIs for each absolute metric, not paired-gain intervals",
        "plotted_values": plotted,
    }
    frozen_plot.atomic_json(args.output_dir / "figure_combined_brain_age.manifest.json", manifest)
    print("PASS: frozen estimates, intervals and all bootstrap samples match across inputs.")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
