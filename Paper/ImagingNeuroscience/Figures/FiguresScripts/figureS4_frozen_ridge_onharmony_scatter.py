#!/usr/bin/env python3
"""Render ON-Harmony scatterplots for the seven frozen-Ridge models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-supp-ridge-oh-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Analyses.figure2_reliability import (  # noqa: E402
    bootstrap_indices,
    compute_dbicc,
    select_primary_acquisition_per_cell,
)
from Analyses.figure3_brain_age import (  # noqa: E402
    DEFAULT_ANATCL_FROZEN_INTERNAL_ROOT,
    DEFAULT_ANATCL_FROZEN_ONHARMONY_ROOT,
    DEFAULT_BRAIN_AGE_ROOT,
    DEFAULT_PARTICIPANTS,
    EXPECTED_ONHARMONY_ROWS,
    EXPECTED_ONHARMONY_SUBJECTS,
    MODEL_SPECS,
    acquisition_table,
    frozen_prediction_dirs,
    load_onharmony_predictions,
    load_participants,
    pairwise_absolute_difference_by_subject,
    participant_scanner_cells,
    validate_prediction_audit,
)
from FineTuning.shared.metrics import regression_metrics  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Paper" / "ImagingNeuroscience" / "Figures"
DEFAULT_FIGURE3_RESULTS = (
    PROJECT_ROOT / "Results" / "Figure3" / "figure3_brain_age_results.json"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "Results" / "FigureS4" / "frozen_ridge_onharmony_manifest.json"
)

SCANNER_ORDER = ("750", "Prem21", "Prem48", "Ach", "Ing", "Pri32", "Pri64", "Trio")
SCANNER_LABELS = {
    "750": "MR750 (GE)",
    "Prem21": "Premier 21 (GE)",
    "Prem48": "Premier 48 (GE)",
    "Ach": "Achieva (Philips)",
    "Ing": "Ingenia (Philips)",
    "Pri32": "Prisma 32 (Siemens)",
    "Pri64": "Prisma 64 (Siemens)",
    "Trio": "Trio (Siemens)",
}
SCANNER_COLORS = {
    "750": "#235789",
    "Prem21": "#4D8FC3",
    "Prem48": "#9BC4E2",
    "Ach": "#D55E00",
    "Ing": "#F2A65A",
    "Pri32": "#188977",
    "Pri64": "#67B99A",
    "Trio": "#8C6BB1",
}
PANEL_LETTERS = "abcdefg"
AGE_LIMITS = (-15.0, 75.0)
AGE_TICKS = (-10, 10, 30, 50, 70)
FORBIDDEN_OUTCOMES = {"age", "age_true", "bag", "error", "error_years"}


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
            "svg.hashsalt": "supp-frozen-ridge-onharmony-v1",
            "savefig.facecolor": "white",
        }
    )


def load_locked_results(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_conditions = {spec.condition for spec in MODEL_SPECS}
    if set(payload.get("conditions", {})) != expected_conditions:
        raise ValueError("Figure 3 results do not contain the seven expected models")
    code = payload.get("code", {})
    analysis_script = Path(code.get("analysis_script", ""))
    if not analysis_script.is_file():
        raise FileNotFoundError(analysis_script)
    if sha256_file(analysis_script) != code.get("analysis_script_sha256"):
        raise ValueError("Figure 3 analysis script has drifted")
    return payload


def attach_age(frame: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    forbidden = FORBIDDEN_OUTCOMES.intersection(frame.columns)
    if forbidden:
        raise ValueError(f"Prediction artifact contains outcomes: {sorted(forbidden)}")
    result = frame.merge(
        participants[["participant_id", "age"]],
        left_on="subject_id",
        right_on="participant_id",
        how="left",
        validate="many_to_one",
    )
    if result["age"].isna().any():
        raise ValueError("One or more prediction rows could not be joined to age")
    result = result.rename(columns={"age": "age_true"})
    values = result[["age_true", "age_pred"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite chronological or predicted age")
    return result


def point_metrics(frames: list[pd.DataFrame]) -> tuple[list[float], list[float]]:
    samples = bootstrap_indices(EXPECTED_ONHARMONY_SUBJECTS, 10, 1701)
    dbicc = []
    scanner_difference = []
    for frame in frames:
        cells = participant_scanner_cells(frame)
        metric = compute_dbicc(cells, ["age_pred"], samples)
        dbicc.append(float(metric["estimate"]))
        scanner_difference.append(
            float(pairwise_absolute_difference_by_subject(cells).mean())
        )
    return dbicc, scanner_difference


def validate_metric_vector(
    observed: list[float], locked: dict, *, label: str
) -> None:
    expected = locked.get("fold_estimates", [locked["estimate"]])
    if len(observed) != len(expected) or not np.allclose(
        observed, np.asarray(expected, dtype=float), rtol=0.0, atol=1e-10
    ):
        raise ValueError(f"Recomputed/locked metric mismatch for {label}")


def load_conditions(
    brain_age_root: Path,
    anatcl_internal_root: Path,
    anatcl_onharmony_root: Path,
    participants_path: Path,
    figure3_results_path: Path,
) -> tuple[list[dict], list[dict], dict]:
    locked = load_locked_results(figure3_results_path)
    participants = load_participants(participants_path)
    participant_contract = locked["cohorts"]["onharmony"]
    if sha256_file(participants_path) != participant_contract["participants_sha256"]:
        raise ValueError("Participant table differs from the Figure 3 contract")

    conditions = []
    sources = []
    reference_acquisitions = None
    for spec in MODEL_SPECS:
        frames = []
        recorded_sources = locked["conditions"][spec.condition]["inputs"]["onharmony"]
        if len(recorded_sources) != len(spec.variants):
            raise ValueError(f"Wrong locked source count for {spec.label}")
        for index, variant in enumerate(spec.variants):
            _, onharmony_dir = frozen_prediction_dirs(
                brain_age_root,
                spec=spec,
                variant=variant,
                anatcl_internal_root=anatcl_internal_root,
                anatcl_onharmony_root=anatcl_onharmony_root,
            )
            predictions_path = onharmony_dir / "predictions.tsv"
            audit_path = onharmony_dir / "prediction_audit.json"
            provenance = validate_prediction_audit(
                audit_path, predictions_path, spec=spec, variant=variant
            )
            recorded = recorded_sources[index]
            for key in ("predictions_sha256", "audit_sha256"):
                if provenance[key] != recorded[key]:
                    raise ValueError(f"Figure 3 source drift for {spec.label}/{variant}")
            frame = load_onharmony_predictions(
                predictions_path, spec=spec, variant=variant
            )
            identity = acquisition_table(frame)
            if reference_acquisitions is None:
                reference_acquisitions = identity
            elif not identity.equals(reference_acquisitions):
                raise ValueError(f"Acquisition mismatch for {spec.label}/{variant}")
            primary_frame = select_primary_acquisition_per_cell(frame)
            frames.append(attach_age(primary_frame, participants))
            sources.append(
                {
                    "model": spec.label,
                    "variant": variant,
                    "predictions": str(predictions_path.resolve()),
                    "predictions_sha256": provenance["predictions_sha256"],
                    "audit": str(audit_path.resolve()),
                    "audit_sha256": provenance["audit_sha256"],
                }
            )

        dbicc, scanner_difference = point_metrics(frames)
        condition_results = locked["conditions"][spec.condition]
        validate_metric_vector(
            dbicc, condition_results["prediction_dbicc"], label=f"{spec.label} dbICC"
        )
        validate_metric_vector(
            scanner_difference,
            condition_results["cross_scanner_absolute_difference"],
            label=f"{spec.label} scanner difference",
        )
        conditions.append(
            {
                "condition": spec.condition,
                "label": spec.label,
                "frames": frames,
                "accuracy": [
                    regression_metrics(frame["age_true"], frame["age_pred"])
                    for frame in frames
                ],
                "dbicc": condition_results["prediction_dbicc"],
                "scanner_difference": condition_results[
                    "cross_scanner_absolute_difference"
                ],
            }
        )

    if reference_acquisitions is None or len(reference_acquisitions) != EXPECTED_ONHARMONY_ROWS:
        raise RuntimeError("The complete ON-Harmony acquisition set was not loaded")
    return conditions, sources, locked


def metric_text(condition: dict) -> str:
    metrics = condition["accuracy"]
    mae = np.asarray([record["mae_years"] for record in metrics], dtype=float)
    r2 = np.asarray([record["r2"] for record in metrics], dtype=float)
    if len(metrics) == 1:
        return f"MAE {mae[0]:.2f} y\n$R^2$ {r2[0]:.3f}"
    return (
        f"MAE {mae.mean():.2f} $\\pm$ {mae.std(ddof=1):.2f} y\n"
        f"$R^2$ {r2.mean():.3f} $\\pm$ {r2.std(ddof=1):.3f}"
    )


def render(conditions: list[dict], output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib()
    figure = plt.figure(figsize=(8.15, 8.55), constrained_layout=False)
    grid = figure.add_gridspec(
        3, 3, left=0.105, right=0.975, bottom=0.075, top=0.925,
        wspace=0.20, hspace=0.31,
    )
    positions = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0))
    axes = []
    for row, column in positions:
        shared = axes[0] if axes else None
        axes.append(figure.add_subplot(grid[row, column], sharex=shared, sharey=shared))

    for index, (axis, condition) in enumerate(zip(axes, conditions)):
        frames = condition["frames"]
        point_alpha = 0.48 if len(frames) == 1 else 0.18
        point_size = 14 if len(frames) == 1 else 10
        for frame in frames:
            for scanner in SCANNER_ORDER:
                subset = frame.loc[frame["scanner"].astype(str) == scanner]
                axis.scatter(
                    subset["age_true"], subset["age_pred"], s=point_size,
                    c=SCANNER_COLORS[scanner], alpha=point_alpha, edgecolors="none",
                    rasterized=False, zorder=2,
                )
        axis.plot(
            AGE_LIMITS, AGE_LIMITS, color="#555D63", linewidth=1.0,
            linestyle=(0, (4, 3)), zorder=1,
        )
        axis.set_title(
            f"{PANEL_LETTERS[index]}   {condition['label']}", loc="left",
            fontweight="bold", pad=7,
        )
        axis.text(
            0.04, 0.95, metric_text(condition), transform=axis.transAxes,
            ha="left", va="top", fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.0},
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
            [], [], marker="o", linestyle="none", markersize=5.5,
            markerfacecolor=SCANNER_COLORS[scanner], markeredgecolor="none",
            label=SCANNER_LABELS[scanner],
        )
        for scanner in SCANNER_ORDER
    ]
    legend_axis.legend(
        handles=handles, title="Scanner", loc="upper left", bbox_to_anchor=(0.00, 0.98),
        ncol=2, frameon=False, handletextpad=0.5, columnspacing=0.9,
        labelspacing=0.50, fontsize=7.8, title_fontsize=8.5,
    )
    figure.supxlabel("Chronological age (years)", y=0.018, fontsize=10.5)
    figure.supylabel("Predicted age (years)", x=0.025, fontsize=10.5)
    figure.suptitle(
        "ON-Harmony predictions: frozen embeddings with Ridge regression",
        x=0.105, y=0.982, ha="left", fontsize=12, fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "FigureS4_frozen_ridge_onharmony_scatter.png",
        output_dir / "FigureS4_frozen_ridge_onharmony_scatter.svg",
    ]
    for destination in outputs:
        temporary = destination.with_name(
            f".{destination.stem}.{os.getpid()}{destination.suffix}"
        )
        figure.savefig(
            temporary, dpi=dpi if destination.suffix == ".png" else None,
            format=destination.suffix.lstrip("."), bbox_inches="tight",
        )
        os.replace(temporary, destination)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-age-root", default=DEFAULT_BRAIN_AGE_ROOT)
    parser.add_argument(
        "--anatcl-internal-test-root", default=DEFAULT_ANATCL_FROZEN_INTERNAL_ROOT
    )
    parser.add_argument(
        "--anatcl-onharmony-root", default=DEFAULT_ANATCL_FROZEN_ONHARMONY_ROOT
    )
    parser.add_argument("--participants", default=DEFAULT_PARTICIPANTS)
    parser.add_argument("--figure3-results", default=DEFAULT_FIGURE3_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    participants_path = Path(args.participants).expanduser().resolve()
    results_path = Path(args.figure3_results).expanduser().resolve()
    conditions, sources, locked = load_conditions(
        Path(args.brain_age_root).expanduser().resolve(),
        Path(args.anatcl_internal_test_root).expanduser().resolve(),
        Path(args.anatcl_onharmony_root).expanduser().resolve(),
        participants_path,
        results_path,
    )
    outputs = render(conditions, Path(args.output_dir).expanduser().resolve(), args.dpi)
    manifest_path = Path(args.manifest).expanduser().resolve()
    atomic_json(
        manifest_path,
        {
            "figure": "Supplementary Figure S9: frozen-Ridge ON-Harmony predictions",
            "cohort": {
                "source_acquisitions": 165,
                "primary_acquisitions": 120,
                "participants": 20,
                "scanners": 8,
                "participant_scanner_cells": 120,
            },
            "display": (
                "one acquisition 001 point per participant-scanner cell and run; "
                "chronological age joined after label-blind inference; metrics "
                "calculated per run and then averaged; predictions not ensembled"
            ),
            "participants": str(participants_path),
            "participants_sha256": sha256_file(participants_path),
            "figure3_results": str(results_path),
            "figure3_results_sha256": sha256_file(results_path),
            "figure3_analysis_script_sha256": locked["code"]["analysis_script_sha256"],
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "sources": sources,
            "displayed_accuracy": {
                condition["condition"]: condition["accuracy"]
                for condition in conditions
            },
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in outputs
            ],
        },
    )
    for path in outputs:
        print(f"Saved: {path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
