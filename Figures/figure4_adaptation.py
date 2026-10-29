#!/usr/bin/env python3
"""Render Figure 4 as paired frozen-Ridge versus adapted comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-figure4-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in SCRIPT_PATH.parents if (parent / 'Analyses').is_dir())
DEFAULT_RESULTS = PROJECT_ROOT / "Results" / "Figure4" / "figure4_adaptation_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figures"
DEFAULT_MANIFEST = PROJECT_ROOT / "Results" / "Figure4" / "figure4_figure_manifest.json"

DISPLAY_ORDER = (
    "yaware",
    "anatcl",
    "brainsegfounder",
    "3dneurosimclr",
    "neurovfm",
    "curia2",
    "brainiac",
)

SCANNER_ORDER = (
    "750",
    "Prem21",
    "Prem48",
    "Ach",
    "Ing",
    "Pri32",
    "Pri64",
    "Trio",
)

SCANNER_LABELS = (
    "MR750",
    "Premier 21",
    "Premier 48",
    "Achieva",
    "Ingenia",
    "Prisma 32",
    "Prisma 64",
    "Trio",
)
SCANNER_REFERENCE = "Ach"

VENDOR_GROUPS = (
    (0, 2, "GE Healthcare"),
    (3, 4, "Philips"),
    (5, 7, "Siemens"),
)

SCANNER_COLOR_LIMIT_YEARS = 16.0

EXPECTED_PRIMARY_COUNTS = {
    "yaware": 3,
    "anatcl": 3,
    "brainsegfounder": 3,
    "3dneurosimclr": 3,
    "neurovfm": 3,
    "curia2": 3,
    "brainiac": 3,
}

PANELS = (
    {
        "endpoint": "accuracy_gain",
        "absolute_key": "mae",
        "run_key": "mae_years",
        "title": "a   Internal test accuracy",
        "subtitle": "",
        "xlabel": "MAE (years)",
        "formula": "frozen_minus_adapted",
        "xlim": (0.0, 14.8),
        "ticks": (0, 4, 8, 12),
    },
    {
        "endpoint": "reliability_gain",
        "absolute_key": "dbicc",
        "run_key": "prediction_dbicc",
        "title": "b   Cross-scanner reliability",
        "subtitle": "",
        "xlabel": "Predicted-age dbICC",
        "formula": "adapted_minus_frozen",
        "xlim": (0.0, 1.05),
        "ticks": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    },
    {
        "endpoint": "stability_gain",
        "absolute_key": "scanner_difference",
        "run_key": "scanner_difference_years",
        "title": "c   Cross-scanner differences",
        "subtitle": "",
        "xlabel": "Mean absolute difference (years)",
        "formula": "frozen_minus_adapted",
        "xlim": (1.2, 12.2),
        "ticks": (2, 4, 6, 8, 10, 12),
    },
)

COLORS = {
    "turquoise": "#3E8F27",
    "orange": "#FF7F11",
    "frozen": "#545963",
    "grid": "#DCE1E4",
    "row": "#F4F5F6",
    "divider": "#B8BEC3",
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
            "svg.hashsalt": "figure4-adaptation-v3",
            "savefig.facecolor": "white",
        }
    )


def finite_triplet(record: dict, name: str) -> tuple[float, float, float]:
    values = np.asarray(
        [record.get("estimate"), record.get("ci_lower"), record.get("ci_upper")],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {name}")
    estimate, lower, upper = values.tolist()
    if lower > estimate or estimate > upper:
        raise ValueError(f"Invalid confidence interval for {name}")
    return estimate, lower, upper


def adapted_interval_from_paired_gain(
    frozen: float,
    gain: dict,
    *,
    formula: str,
) -> tuple[float, float, float]:
    estimate, lower, upper = finite_triplet(gain, "paired gain")
    if formula == "frozen_minus_adapted":
        adapted = frozen - estimate
        adapted_lower = frozen - upper
        adapted_upper = frozen - lower
    elif formula == "adapted_minus_frozen":
        adapted = frozen + estimate
        adapted_lower = frozen + lower
        adapted_upper = frozen + upper
    else:
        raise ValueError(f"Unknown gain formula: {formula}")
    return adapted, adapted_lower, adapted_upper


def validate_payload(payload: dict) -> dict:
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Figure 4 result schema")
    if payload.get("gain_definitions", {}).get("orientation") != (
        "positive values always indicate improvement after adaptation"
    ):
        raise ValueError("Figure 4 gain orientation is not locked")
    conditions = payload.get("conditions", {})
    if set(conditions) != set(DISPLAY_ORDER):
        raise ValueError("Figure 4 does not contain the seven prespecified models")
    internal = payload.get("cohorts", {}).get("internal_test", {})
    onharmony = payload.get("cohorts", {}).get("onharmony", {})
    if internal.get("subjects") != 140 or internal.get("bootstrap_n") != 1000:
        raise ValueError("Figure 4 internal-test contract has drifted")
    for key, expected in {
        "sessions": 165,
        "participants": 20,
        "scanners": 8,
        "bootstrap_n": 1000,
    }.items():
        if onharmony.get(key) != expected:
            raise ValueError(f"Unexpected Figure 4 ON-Harmony {key}")
    hierarchy = payload.get("training_hierarchy", {})
    if "predictions are never averaged" not in hierarchy.get("aggregation", ""):
        raise ValueError("Figure 4 no-ensemble rule is absent")
    if tuple(payload.get("scanner_order", ())) != SCANNER_ORDER:
        raise ValueError("Figure 4 scanner order has drifted")

    for condition in DISPLAY_ORDER:
        record = conditions[condition]
        primary_runs = record.get("primary_runs", [])
        if len(primary_runs) != EXPECTED_PRIMARY_COUNTS[condition]:
            raise ValueError(f"Wrong primary-run count for {condition}")
        run_results = record.get("run_results", {})
        if not set(primary_runs).issubset(run_results):
            raise ValueError(f"Primary runs absent for {condition}")
        for panel in PANELS:
            gain = record.get("primary_gains", {}).get(panel["endpoint"], {})
            finite_triplet(gain, f"{condition} {panel['endpoint']}")
            if gain.get("n_primary_runs") != EXPECTED_PRIMARY_COUNTS[condition]:
                raise ValueError(f"Wrong aggregated-run count for {condition}")
            bootstrap = np.asarray(gain.get("bootstrap_values", []), dtype=float)
            if bootstrap.shape != (1000,) or not np.isfinite(bootstrap).all():
                raise ValueError(f"Invalid bootstrap for {condition}")
            frozen = record.get("absolute", {}).get("frozen", {}).get(
                panel["absolute_key"], {}
            )
            adapted = record.get("absolute", {}).get("adapted", {}).get(
                panel["absolute_key"], {}
            )
            frozen_value = finite_triplet(frozen, f"{condition} frozen")[0]
            adapted_value = finite_triplet(adapted, f"{condition} adapted")[0]
            implied, _, _ = adapted_interval_from_paired_gain(
                frozen_value,
                gain,
                formula=panel["formula"],
            )
            if not np.isclose(implied, adapted_value, atol=1e-10):
                raise ValueError(f"Absolute and contrast estimates disagree for {condition}")
        for run_name, run in run_results.items():
            if run.get("run") != run_name:
                raise ValueError(f"Run identity mismatch for {condition}/{run_name}")
            values = []
            for panel in PANELS:
                values.extend(
                    (
                        run["absolute"]["frozen"][panel["run_key"]],
                        run["absolute"]["adapted"][panel["run_key"]],
                        run["gains"][panel["endpoint"]],
                    )
                )
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                raise ValueError(f"Invalid run values for {condition}/{run_name}")

        scanner_record = record.get("adapted_scanner_effects", {})
        if scanner_record.get("reference_scanner") != SCANNER_REFERENCE:
            raise ValueError(f"Adapted scanner effects are not relative to NOT1 for {condition}")
        scanner_effects = scanner_record.get("effects", [])
        if tuple(effect.get("scanner") for effect in scanner_effects) != SCANNER_ORDER:
            raise ValueError(f"Wrong adapted scanner order for {condition}")
        estimates = []
        bootstrap_columns = []
        for effect in scanner_effects:
            estimate, _, _ = finite_triplet(
                {
                    "estimate": effect.get("effect_years"),
                    "ci_lower": effect.get("ci_lower"),
                    "ci_upper": effect.get("ci_upper"),
                },
                f"{condition} adapted scanner effect",
            )
            estimates.append(estimate)
            bootstrap = np.asarray(effect.get("bootstrap_values", []), dtype=float)
            if bootstrap.shape != (1000,) or not np.isfinite(bootstrap).all():
                raise ValueError(f"Invalid adapted scanner bootstrap for {condition}")
            bootstrap_columns.append(bootstrap)
        reference_index = SCANNER_ORDER.index(SCANNER_REFERENCE)
        if not np.isclose(estimates[reference_index], 0.0, atol=1e-8):
            raise ValueError(f"Adapted reference scanner is not zero for {condition}")
        bootstrap_matrix = np.column_stack(bootstrap_columns)
        if not np.allclose(bootstrap_matrix[:, reference_index], 0.0, atol=1e-8):
            raise ValueError(
                f"Adapted reference scanner bootstrap is not zero for {condition}"
            )

    analysis_script = Path(payload.get("code", {}).get("analysis_script", ""))
    expected_hash = payload.get("code", {}).get("analysis_script_sha256")
    if not analysis_script.is_file():
        raise FileNotFoundError(analysis_script)
    if sha256_file(analysis_script) != expected_hash:
        raise ValueError("Figure 4 analysis script hash no longer matches its results")
    return conditions


def add_vendor_groups(axis: mpl.axes.Axes) -> None:
    transform = axis.get_xaxis_transform()
    for start, end, label in VENDOR_GROUPS:
        axis.plot(
            [start - 0.42, end + 0.42],
            [1.045, 1.045],
            transform=transform,
            color=COLORS["frozen"],
            linewidth=0.8,
            clip_on=False,
        )
        axis.text(
            (start + end) / 2,
            1.065,
            label,
            transform=transform,
            ha="center",
            va="bottom",
            color=COLORS["frozen"],
            fontsize=11.5,
            clip_on=False,
        )


def style_axis(axis: mpl.axes.Axes, panel: dict, count: int) -> None:
    axis.set_xlim(*panel["xlim"])
    axis.set_ylim(count - 0.48, -0.52)
    axis.set_xticks(panel["ticks"])
    axis.set_yticks(np.arange(count))
    axis.tick_params(axis="y", length=0)
    axis.xaxis.grid(True, color=COLORS["grid"], linewidth=0.65, zorder=-2)
    axis.set_axisbelow(True)
    for row in range(count):
        if row % 2:
            axis.axhspan(
                row - 0.5,
                row + 0.5,
                color=COLORS["row"],
                alpha=0.65,
                zorder=-3,
            )
    axis.set_xlabel(panel["xlabel"])
    axis.set_title(panel["title"], loc="left", fontweight="bold", pad=11)
    if panel["subtitle"]:
        axis.text(
            0.0,
            1.035,
            panel["subtitle"],
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.1,
            color="#596168",
        )


def draw_run_pairs(
    axis: mpl.axes.Axes,
    row: float,
    record: dict,
    panel: dict,
) -> None:
    names = record["primary_runs"]
    offsets = np.linspace(-0.18, 0.18, len(names)) if len(names) > 1 else np.zeros(1)
    for run_name, offset in zip(names, offsets):
        run = record["run_results"][run_name]
        frozen = float(run["absolute"]["frozen"][panel["run_key"]])
        adapted = float(run["absolute"]["adapted"][panel["run_key"]])
        y = row + float(offset)
        axis.plot(
            [frozen, adapted],
            [y, y],
            color=mpl.colors.to_rgba(COLORS["divider"], 0.72),
            linewidth=0.8,
            zorder=2,
        )
        axis.scatter(
            frozen,
            y,
            s=12,
            facecolors="white",
            edgecolors=mpl.colors.to_rgba(COLORS["frozen"], 0.55),
            linewidths=0.7,
            zorder=3,
        )
        axis.scatter(
            adapted,
            y,
            s=13,
            facecolors=mpl.colors.to_rgba(COLORS["orange"], 0.45),
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )


def draw_primary_pair(
    axis: mpl.axes.Axes,
    row: float,
    record: dict,
    panel: dict,
) -> None:
    gain = record["primary_gains"][panel["endpoint"]]
    frozen = float(record["absolute"]["frozen"][panel["absolute_key"]]["estimate"])
    adapted_reported = float(
        record["absolute"]["adapted"][panel["absolute_key"]]["estimate"]
    )
    adapted, lower, upper = adapted_interval_from_paired_gain(
        frozen,
        gain,
        formula=panel["formula"],
    )
    if not np.isclose(adapted, adapted_reported, atol=1e-10):
        raise ValueError("Adapted point is inconsistent with the paired gain")

    axis.plot(
        [frozen, adapted],
        [row, row],
        color=COLORS["divider"],
        linewidth=1.6,
        solid_capstyle="round",
        zorder=4,
    )
    axis.scatter(
        frozen,
        row,
        s=55,
        marker="o",
        facecolors="white",
        edgecolors=COLORS["frozen"],
        linewidths=1.35,
        zorder=5,
    )
    axis.errorbar(
        adapted,
        row,
        xerr=np.asarray([[adapted - lower], [upper - adapted]]),
        linestyle="none",
        marker="D",
        markerfacecolor=COLORS["orange"],
        markeredgecolor="white",
        markeredgewidth=0.75,
        markersize=6.6,
        color=COLORS["orange"],
        ecolor=COLORS["orange"],
        elinewidth=1.35,
        capsize=2.8,
        capthick=0.9,
        zorder=6,
    )


def draw_primary_accuracy_bars(
    axis: mpl.axes.Axes,
    row: float,
    record: dict,
    panel: dict,
) -> None:
    gain = record["primary_gains"][panel["endpoint"]]
    frozen, frozen_lower, frozen_upper = finite_triplet(
        record["absolute"]["frozen"][panel["absolute_key"]],
        "frozen accuracy",
    )
    adapted_reported, adapted_lower, adapted_upper = finite_triplet(
        record["absolute"]["adapted"][panel["absolute_key"]],
        "adapted accuracy",
    )
    adapted, _, _ = adapted_interval_from_paired_gain(
        frozen,
        gain,
        formula=panel["formula"],
    )
    if not np.isclose(adapted, adapted_reported, atol=1e-10):
        raise ValueError("Adapted point is inconsistent with the paired gain")

    axis.barh(
        row - 0.09,
        frozen,
        height=0.18,
        color=COLORS["turquoise"],
        edgecolor="none",
        zorder=4,
    )
    axis.barh(
        row + 0.09,
        adapted,
        height=0.18,
        color=COLORS["orange"],
        edgecolor="none",
        zorder=5,
    )
    axis.errorbar(
        frozen,
        row - 0.09,
        xerr=np.asarray(
            [[frozen - frozen_lower], [frozen_upper - frozen]]
        ),
        linestyle="none",
        marker="none",
        color="#202427",
        ecolor="#202427",
        elinewidth=1.0,
        capsize=2.2,
        capthick=0.8,
        zorder=6,
    )
    axis.errorbar(
        adapted,
        row + 0.09,
        xerr=np.asarray(
            [[adapted - adapted_lower], [adapted_upper - adapted]]
        ),
        linestyle="none",
        marker="none",
        color="#202427",
        ecolor="#202427",
        elinewidth=1.0,
        capsize=2.2,
        capthick=0.8,
        zorder=7,
    )


def draw_primary_absolute_pair(
    axis: mpl.axes.Axes,
    row: float,
    record: dict,
    panel: dict,
) -> None:
    gain = record["primary_gains"][panel["endpoint"]]
    frozen, frozen_lower, frozen_upper = finite_triplet(
        record["absolute"]["frozen"][panel["absolute_key"]],
        "frozen endpoint",
    )
    adapted, adapted_lower, adapted_upper = finite_triplet(
        record["absolute"]["adapted"][panel["absolute_key"]],
        "adapted endpoint",
    )
    implied, _, _ = adapted_interval_from_paired_gain(
        frozen,
        gain,
        formula=panel["formula"],
    )
    if not np.isclose(implied, adapted, atol=1e-10):
        raise ValueError("Adapted point is inconsistent with the paired gain")

    axis.plot(
        [frozen, adapted],
        [row - 0.10, row + 0.10],
        color=COLORS["divider"],
        linewidth=1.1,
        zorder=2,
    )
    axis.errorbar(
        frozen,
        row - 0.10,
        xerr=np.asarray([[frozen - frozen_lower], [frozen_upper - frozen]]),
        linestyle="none",
        marker="o",
        markerfacecolor=COLORS["turquoise"],
        markeredgecolor=COLORS["turquoise"],
        markeredgewidth=1.15,
        markersize=5.3,
        color=COLORS["turquoise"],
        ecolor=COLORS["turquoise"],
        elinewidth=1.25,
        capsize=2.5,
        capthick=0.9,
        zorder=3,
    )
    axis.errorbar(
        adapted,
        row + 0.10,
        xerr=np.asarray([[adapted - adapted_lower], [adapted_upper - adapted]]),
        linestyle="none",
        marker="o",
        markerfacecolor=COLORS["orange"],
        markeredgecolor=COLORS["orange"],
        markeredgewidth=1.15,
        markersize=5.8,
        color=COLORS["orange"],
        ecolor=COLORS["orange"],
        elinewidth=1.25,
        capsize=2.5,
        capthick=0.9,
        zorder=4,
    )


def render(payload: dict, *, output_dir: Path, dpi: int) -> list[Path]:
    conditions = validate_payload(payload)
    configure_matplotlib()
    count = len(DISPLAY_ORDER)

    figure = plt.figure(figsize=(14.2, 6.8), constrained_layout=False)
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 0.78),
        hspace=0.74,
        left=0.18,
        right=0.975,
        bottom=0.09,
        top=0.87,
    )
    top = outer[0].subgridspec(
        1,
        3,
        width_ratios=(1.12, 1.0, 1.08),
        wspace=0.20,
    )
    axes = [
        figure.add_subplot(top[0]),
        figure.add_subplot(top[1]),
        figure.add_subplot(top[2]),
    ]
    axes[1].sharey(axes[0])
    axes[2].sharey(axes[0])
    bottom = outer[1].subgridspec(1, 2, width_ratios=(1.0, 0.028), wspace=0.045)
    axis_heat = figure.add_subplot(bottom[0])
    axis_colorbar = figure.add_subplot(bottom[1])
    labels = [conditions[condition]["label"] for condition in DISPLAY_ORDER]
    for axis, panel in zip(axes, PANELS):
        style_axis(axis, panel, count)
        for row, condition in enumerate(DISPLAY_ORDER):
            record = conditions[condition]
            if panel["endpoint"] == "accuracy_gain":
                draw_primary_accuracy_bars(axis, row, record, panel)
            elif panel["endpoint"] in {"reliability_gain", "stability_gain"}:
                draw_primary_absolute_pair(axis, row, record, panel)
            else:
                draw_run_pairs(axis, row, record, panel)
                draw_primary_pair(axis, row, record, panel)

    axes[0].set_yticklabels(labels)
    axes[0].tick_params(axis="y", pad=7)
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)

    figure.legend(
        handles=[
            mpl.lines.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=COLORS["turquoise"],
                markeredgecolor=COLORS["turquoise"],
                label="Frozen Ridge",
            ),
            mpl.lines.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=COLORS["orange"],
                markeredgecolor=COLORS["orange"],
                label="Fine-tuned",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
    )

    effects = np.asarray(
        [
            [
                effect["effect_years"]
                for effect in conditions[condition]["adapted_scanner_effects"][
                    "effects"
                ]
            ]
            for condition in DISPLAY_ORDER
        ],
        dtype=float,
    )
    if np.max(np.abs(effects)) > SCANNER_COLOR_LIMIT_YEARS:
        raise ValueError(
            "Fine-tuned scanner effects exceed the shared Figure 3/4 color scale"
        )
    diverging = LinearSegmentedColormap.from_list(
        "scanner_shift", (COLORS["turquoise"], "#F7F7F5", COLORS["orange"])
    )
    normalization = TwoSlopeNorm(
        vmin=-SCANNER_COLOR_LIMIT_YEARS,
        vcenter=0.0,
        vmax=SCANNER_COLOR_LIMIT_YEARS,
    )
    image = axis_heat.imshow(
        effects,
        aspect="auto",
        cmap=diverging,
        norm=normalization,
        interpolation="nearest",
    )
    labels = [conditions[condition]["label"] for condition in DISPLAY_ORDER]
    axis_heat.set_yticks(np.arange(count), labels)
    axis_heat.set_xticks(np.arange(len(SCANNER_LABELS)), SCANNER_LABELS)
    axis_heat.tick_params(axis="y", length=0, pad=7, labelsize=13)
    axis_heat.tick_params(axis="x", length=0, pad=7, labelsize=11.8)
    axis_heat.set_xlabel("Scanner (reference: NOT1 / Achieva)")
    axis_heat.set_title(
        "d   Fine-tuned participant-adjusted scanner shifts",
        loc="left",
        fontweight="bold",
        pad=48,
    )
    for boundary in (2.5, 4.5):
        axis_heat.axvline(boundary, color="white", linewidth=2.4, zorder=3)
        axis_heat.axvline(boundary, color="#AAB0B5", linewidth=0.65, zorder=4)
    for boundary in np.arange(count - 1) + 0.5:
        axis_heat.axhline(boundary, color="white", linewidth=1.0, zorder=3)
    add_vendor_groups(axis_heat)

    for row in range(effects.shape[0]):
        for column in range(effects.shape[1]):
            value = float(effects[row, column])
            color = (
                "white"
                if value <= -0.53 * SCANNER_COLOR_LIMIT_YEARS
                else "#25292D"
            )
            axis_heat.text(
                column,
                row,
                "ref." if SCANNER_ORDER[column] == SCANNER_REFERENCE else f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=11.2,
                color=color,
                fontweight=(
                    "bold"
                    if abs(value) >= 0.70 * SCANNER_COLOR_LIMIT_YEARS
                    else "normal"
                ),
                zorder=5,
            )

    colorbar = figure.colorbar(image, cax=axis_colorbar)
    colorbar.set_label(
        "Difference from NOT1 (years)",
        rotation=270,
        labelpad=26,
        fontsize=13.5,
    )
    colorbar.outline.set_linewidth(0.7)
    colorbar.ax.tick_params(labelsize=11.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "Figure4_adaptation.png",
        output_dir / "Figure4_adaptation.svg",
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
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--results", default=DEFAULT_RESULTS)
    argument_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    argument_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    argument_parser.add_argument("--dpi", type=int, default=400)
    args = argument_parser.parse_args()

    result_path = Path(args.results).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    outputs = render(payload, output_dir=output_dir, dpi=args.dpi)
    manifest = {
        "figure": "Figure 4: paired frozen-Ridge versus controlled-adaptation comparison",
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
