#!/usr/bin/env python3
"""Render Figure 2 from the locked cross-acquisition reliability analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-figure2-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(parent for parent in SCRIPT_PATH.parents if (parent / 'Analyses').is_dir())
DEFAULT_RESULTS = PROJECT_ROOT / "Results" / "Figure2" / "figure2_reliability_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figures"
DEFAULT_MANIFEST = PROJECT_ROOT / "Results" / "Figure2" / "figure2_figure_manifest.json"

DISPLAY_ORDER = (
    "freesurfer",
    "matched_simclr",
    "yaware",
    "anatcl",
    "brainsegfounder",
    "3dneurosimclr",
    "neurovfm",
    "curia2",
    "brainiac",
)

COLORS = {
    "subject": "#3E8F27",
    "scanner": "#FF7F11",
    "residual": "#545963",
    "primary": "#202428",
    "reference": "#6F767D",
    "grid": "#D9DEE2",
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
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "figure2-reliability-v3",
            "savefig.facecolor": "white",
        }
    )


def validate_payload(payload: dict) -> dict:
    conditions = payload.get("conditions", {})
    missing = [condition for condition in DISPLAY_ORDER if condition not in conditions]
    if missing:
        raise ValueError(f"Figure 2 result lacks conditions: {missing}")
    for condition in DISPLAY_ORDER:
        record = conditions[condition]
        dbicc = record.get("dbicc")
        variance = record.get("variance")
        if not dbicc or variance is None:
            raise ValueError(
                f"Incomplete Figure 2 result for {condition}; run analysis with REML"
            )
        numeric = [
            dbicc.get("estimate"),
            dbicc.get("ci_lower"),
            dbicc.get("ci_upper"),
            variance.get("proportion_subject"),
            variance.get("proportion_scanner"),
            variance.get("proportion_residual"),
        ]
        if not np.isfinite(np.asarray(numeric, dtype=float)).all():
            raise ValueError(f"Non-finite Figure 2 result for {condition}")
        proportions = np.asarray(numeric[-3:], dtype=float)
        if not np.isclose(proportions.sum(), 1.0, atol=1e-8):
            raise ValueError(f"Variance proportions do not sum to one for {condition}")
    return conditions


def point_style(role: str) -> dict:
    if role == "reference":
        return {
            "marker": "D",
            "markerfacecolor": COLORS["reference"],
            "markeredgecolor": COLORS["reference"],
        }
    return {
        "marker": "o",
        "markerfacecolor": COLORS["primary"],
        "markeredgecolor": COLORS["primary"],
    }


def render(payload: dict, *, output_dir: Path, dpi: int) -> list[Path]:
    conditions = validate_payload(payload)
    configure_matplotlib()

    count = len(DISPLAY_ORDER)
    y = np.arange(count, dtype=float)
    labels = [
        "AnatCL-SimCLR"
        if condition == "matched_simclr"
        else conditions[condition]["label"]
        for condition in DISPLAY_ORDER
    ]

    figure, (axis_dbicc, axis_variance) = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.13},
        constrained_layout=False,
    )

    for row, condition in enumerate(DISPLAY_ORDER):
        record = conditions[condition]
        dbicc = record["dbicc"]
        style = point_style(record["role"])
        estimate = float(dbicc["estimate"])
        lower = float(dbicc["ci_lower"])
        upper = float(dbicc["ci_upper"])
        axis_dbicc.errorbar(
            estimate,
            row,
            xerr=np.asarray([[estimate - lower], [upper - estimate]]),
            linestyle="none",
            color=style["markeredgecolor"],
            ecolor=style["markeredgecolor"],
            elinewidth=2.0,
            capsize=3.5,
            capthick=1.5,
            markersize=8.5,
            markeredgewidth=1.5,
            zorder=4,
            **style,
        )
    axis_dbicc.axvline(0, color="#8B9298", linestyle="--", linewidth=1.1, zorder=1)
    axis_dbicc.set_xlim(-0.06, 1.10)
    axis_dbicc.set_xticks(np.arange(0, 1.01, 0.2))
    axis_dbicc.set_xlabel("Full-vector cross-scanner dbICC")
    axis_dbicc.set_title("a   Cross-acquisition reliability", loc="left", fontweight="bold")
    axis_dbicc.set_yticks(y, labels)
    axis_dbicc.invert_yaxis()
    axis_dbicc.tick_params(axis="y", length=0, pad=7)
    axis_dbicc.xaxis.grid(True, color=COLORS["grid"], linewidth=0.9, zorder=0)
    axis_dbicc.set_axisbelow(True)

    legend_handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["primary"],
            markeredgecolor=COLORS["primary"],
            label="Foundation model",
        ),
        mpl.lines.Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor=COLORS["reference"],
            markeredgecolor=COLORS["reference"],
            label="FreeSurfer reference",
        ),
    ]
    axis_dbicc.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.18),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.45,
        borderaxespad=0,
    )

    left = np.zeros(count, dtype=float)
    bar_height = 0.68
    components = (
        ("subject", "Participant"),
        ("scanner", "Scanner"),
        ("residual", "Residual"),
    )
    for component, component_label in components:
        values = np.asarray(
            [
                conditions[condition]["variance"][f"proportion_{component}"]
                for condition in DISPLAY_ORDER
            ],
            dtype=float,
        )
        axis_variance.barh(
            y,
            values,
            left=left,
            height=bar_height,
            color=COLORS[component],
            edgecolor="white",
            linewidth=0.55,
            label=component_label,
            zorder=2,
        )
        for row, (start, value) in enumerate(zip(left, values)):
            if value >= 0.085:
                text_color = "white"
                axis_variance.text(
                    start + value / 2,
                    row,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=13.5,
                    color=text_color,
                    fontweight="bold" if value >= 0.16 else "normal",
                    zorder=4,
                )
        left += values

    axis_variance.set_xlim(0, 1)
    axis_variance.set_xticks(np.linspace(0, 1, 6))
    axis_variance.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    axis_variance.set_xlabel("Proportion of total representation variance")
    axis_variance.set_title("b   Variance decomposition", loc="left", fontweight="bold")
    axis_variance.tick_params(axis="y", length=0)
    axis_variance.xaxis.grid(True, color=COLORS["grid"], linewidth=0.9, zorder=0)
    axis_variance.set_axisbelow(True)
    axis_variance.legend(
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.18),
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.7,
        borderaxespad=0,
    )

    figure.subplots_adjust(left=0.19, right=0.99, bottom=0.16, top=0.78)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "Figure2_reliability.png",
        output_dir / "Figure2_reliability.svg",
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
        "figure": "Figure 2: cross-acquisition representation reliability",
        "analysis": str(result_path),
        "analysis_sha256": sha256_file(result_path),
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
