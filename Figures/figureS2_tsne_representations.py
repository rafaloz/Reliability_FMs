#!/usr/bin/env python3
"""Render supplementary t-SNE views of the definitive frozen representations.

The figure updates the descriptive t-SNE figure from the legacy manuscript.
It uses the exact analysis inputs bound to Figure 2, retains acquisition 001
from each participant-scanner cell, and excludes the post hoc AnatCL-SimCLR
control so that the panels contain FreeSurfer and the seven primary FMs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-figureS2-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "Results" / "Figure2" / "inputs"
DEFAULT_FIGURE2_RESULTS = (
    PROJECT_ROOT / "Results" / "Figure2" / "figure2_reliability_results.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figures"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "Results" / "FigureS2" / "figureS2_tsne_manifest.json"
)

PRIMARY_ACQUISITION_SUFFIX = "001"
EXPECTED_ROWS = 165
EXPECTED_SELECTED_ROWS = 120
EXPECTED_SUBJECTS = 20
EXPECTED_SCANNERS = 8

MODEL_SPECS = (
    ("freesurfer", "FreeSurfer", "freesurfer__fold0.csv", 201),
    ("yaware", "y-Aware", "yaware__fold0.csv", 1024),
    ("anatcl", "AnatCL", "anatcl__fold0.csv", 512),
    (
        "brainsegfounder",
        "BrainSegFounder",
        "brainsegfounder__fold0.csv",
        768,
    ),
    (
        "3dneurosimclr",
        "3D-Neuro-SimCLR",
        "3dneurosimclr__fold0.csv",
        512,
    ),
    ("neurovfm", "NeuroVFM", "neurovfm__fold0.csv", 768),
    ("curia2", "Curia-2", "curia2__fold0.csv", 768),
    ("brainiac", "BrainIAC", "brainiac__fold0.csv", 768),
)

VENDOR_BY_SCANNER = {
    "OXF1PRI": "Siemens",
    "OXF2PRI": "Siemens",
    "OXF3TRI": "Siemens",
    "NOT1ACH": "Philips",
    "NOT2ING": "Philips",
    "NOT3GEM": "GE",
    "NOT4GEP": "GE",
    "OXF4GEP": "GE",
}
VENDOR_COLORS = {
    "Siemens": "#3E8F27",
    "Philips": "#FF7F11",
    "GE": "#545963",
}
PANEL_LETTERS = "abcdefgh"


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


def numbered_feature_columns(frame: pd.DataFrame) -> list[str]:
    indexed = []
    for column in frame.columns:
        if not column.startswith("emb_"):
            continue
        suffix = column.removeprefix("emb_")
        if suffix.isdigit():
            indexed.append((int(suffix), column))
    indexed.sort()
    if [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError("Embedding columns are not contiguous from emb_0")
    return [column for _, column in indexed]


def expected_input_hashes(figure2_results: Path) -> dict[str, str]:
    payload = json.loads(figure2_results.read_text(encoding="utf-8"))
    cohort = payload.get("cohort", {})
    expected_cohort = {
        "sessions": EXPECTED_ROWS,
        "primary_acquisitions": EXPECTED_SELECTED_ROWS,
        "participants": EXPECTED_SUBJECTS,
        "scanners": EXPECTED_SCANNERS,
    }
    for key, expected in expected_cohort.items():
        if cohort.get(key) != expected:
            raise ValueError(
                f"Figure 2 cohort mismatch for {key}: {cohort.get(key)} != {expected}"
            )

    hashes = {}
    for condition, _, filename, _ in MODEL_SPECS:
        records = payload.get("conditions", {}).get(condition, {}).get("inputs", [])
        matches = [
            record
            for record in records
            if Path(record.get("analysis_input", "")).name == filename
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Figure 2 does not bind exactly one input named {filename}"
            )
        hashes[filename] = matches[0]["analysis_input_sha256"]
    return hashes


def load_primary_representations(
    input_root: Path,
    figure2_results: Path,
) -> tuple[list[dict], list[dict]]:
    expected_hashes = expected_input_hashes(figure2_results)
    conditions = []
    sources = []
    reference_keys = None

    for condition, label, filename, expected_dimensions in MODEL_SPECS:
        path = input_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256_file(path)
        if observed_hash != expected_hashes[filename]:
            raise ValueError(
                f"Figure 2 input hash mismatch for {path}: "
                f"{observed_hash} != {expected_hashes[filename]}"
            )

        frame = pd.read_csv(path)
        required = {"subject_id", "session_id", "scanner"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks metadata columns: {sorted(missing)}")
        if len(frame) != EXPECTED_ROWS:
            raise ValueError(f"{path} has {len(frame)} rows; expected {EXPECTED_ROWS}")
        features = numbered_feature_columns(frame)
        if len(features) != expected_dimensions:
            raise ValueError(
                f"{path} has {len(features)} features; expected {expected_dimensions}"
            )

        selected = frame.loc[
            frame["session_id"].astype(str).str.endswith(
                PRIMARY_ACQUISITION_SUFFIX
            )
        ].copy()
        selected = selected.sort_values(
            ["subject_id", "scanner"], kind="stable"
        ).reset_index(drop=True)
        if len(selected) != EXPECTED_SELECTED_ROWS:
            raise ValueError(
                f"{path} yields {len(selected)} primary acquisitions; "
                f"expected {EXPECTED_SELECTED_ROWS}"
            )
        if selected.duplicated(["subject_id", "scanner"]).any():
            raise ValueError(f"{path} contains duplicate primary participant-scanner cells")
        if selected["subject_id"].nunique() != EXPECTED_SUBJECTS:
            raise ValueError(f"{path} does not contain {EXPECTED_SUBJECTS} participants")
        if selected["scanner"].nunique() != EXPECTED_SCANNERS:
            raise ValueError(f"{path} does not contain {EXPECTED_SCANNERS} scanners")
        per_subject = selected.groupby("subject_id", sort=True).size()
        if not per_subject.eq(6).all():
            raise ValueError(f"{path} does not contain six acquisitions per participant")

        keys = selected[["subject_id", "scanner", "session_id"]].astype(str)
        if reference_keys is None:
            reference_keys = keys.reset_index(drop=True)
        elif not keys.reset_index(drop=True).equals(reference_keys):
            raise ValueError(f"Acquisition keys differ for {path}")

        values = selected[features].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{path} contains non-finite representation values")
        scale = values.std(axis=0, ddof=0)
        nonconstant = scale > 0
        if not nonconstant.any():
            raise ValueError(f"{path} contains no variable representation dimensions")
        values = values[:, nonconstant]
        standardised = StandardScaler().fit_transform(values)
        if not np.isfinite(standardised).all():
            raise ValueError(f"Standardisation produced non-finite values for {path}")

        conditions.append(
            {
                "condition": condition,
                "label": label,
                "values": standardised,
                "subjects": selected["subject_id"].astype(str).to_numpy(),
                "scanners": selected["scanner"].astype(str).to_numpy(),
                "dimensions_input": len(features),
                "dimensions_retained": int(nonconstant.sum()),
            }
        )
        sources.append(
            {
                "condition": condition,
                "path": str(path.resolve()),
                "sha256": observed_hash,
                "rows_input": len(frame),
                "rows_selected": len(selected),
                "dimensions_input": len(features),
                "dimensions_retained_after_constant_filter": int(nonconstant.sum()),
            }
        )
    return conditions, sources


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["ArialMT", "Arial", "Arimo", "Liberation Sans"],
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "supp-tsne-representations-v1",
            "savefig.facecolor": "white",
        }
    )


def render(
    conditions: list[dict],
    *,
    output_dir: Path,
    perplexity: float,
    seed: int,
    dpi: int,
) -> tuple[list[Path], list[dict]]:
    configure_matplotlib()
    if len(conditions) != 8:
        raise ValueError(f"Expected eight representations, found {len(conditions)}")

    subjects = sorted(set(conditions[0]["subjects"]))
    palette = mpl.colormaps["tab20"].resampled(len(subjects))
    subject_colors = {subject: palette(index) for index, subject in enumerate(subjects)}

    figure, axes = plt.subplots(4, 4, figsize=(13.2, 9.0))
    projections = []
    for index, record in enumerate(conditions):
        projection = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(record["values"])
        if not np.isfinite(projection).all():
            raise RuntimeError(f"t-SNE produced non-finite coordinates for {record['label']}")

        group = index // 4
        column = index % 4
        subject_axis = axes[group * 2, column]
        vendor_axis = axes[group * 2 + 1, column]
        title = f"{PANEL_LETTERS[index]}   {record['label']}"
        subject_axis.set_title(title, loc="left", fontweight="bold", pad=5)

        for subject in subjects:
            mask = record["subjects"] == subject
            subject_axis.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=34,
                color=subject_colors[subject],
                alpha=0.90,
                edgecolor="white",
                linewidth=0.35,
            )

        vendors = np.asarray(
            [VENDOR_BY_SCANNER.get(scanner) for scanner in record["scanners"]]
        )
        if any(vendor is None for vendor in vendors):
            unknown = sorted(
                set(
                    scanner
                    for scanner, vendor in zip(record["scanners"], vendors)
                    if vendor is None
                )
            )
            raise ValueError(f"Unknown scanner/vendor mappings: {unknown}")
        for vendor in ("Siemens", "Philips", "GE"):
            mask = vendors == vendor
            vendor_axis.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=34,
                color=VENDOR_COLORS[vendor],
                alpha=0.92,
                edgecolor="white",
                linewidth=0.35,
                label=vendor,
            )

        for axis in (subject_axis, vendor_axis):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines["left"].set_color("#7A8085")
            axis.spines["bottom"].set_color("#7A8085")
            axis.set_box_aspect(0.70)
        if column == 0:
            subject_axis.set_ylabel("Participant", fontweight="bold", labelpad=7)
            vendor_axis.set_ylabel("Manufacturer", fontweight="bold", labelpad=7)

        projections.append(
            {
                "condition": record["condition"],
                "label": record["label"],
                "coordinate_sha256": hashlib.sha256(
                    np.asarray(projection, dtype=np.float32).tobytes()
                ).hexdigest(),
            }
        )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=VENDOR_COLORS[vendor],
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=vendor,
        )
        for vendor in ("Siemens", "Philips", "GE")
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.45,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.965,
        bottom=0.07,
        hspace=0.20,
        wspace=0.03,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "FigureS2_tsne_representations.png",
        output_dir / "FigureS2_tsne_representations.svg",
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
            pad_inches=0.06,
        )
        os.replace(temporary, destination)
    plt.close(figure)
    return outputs, projections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--figure2-results", default=DEFAULT_FIGURE2_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    if not 0 < args.perplexity < EXPECTED_SELECTED_ROWS:
        parser.error(
            f"--perplexity must be between 0 and {EXPECTED_SELECTED_ROWS}"
        )
    input_root = Path(args.input_root).expanduser().resolve()
    figure2_results = Path(args.figure2_results).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not figure2_results.is_file():
        raise FileNotFoundError(figure2_results)

    conditions, sources = load_primary_representations(input_root, figure2_results)
    outputs, projections = render(
        conditions,
        output_dir=output_dir,
        perplexity=args.perplexity,
        seed=args.seed,
        dpi=args.dpi,
    )
    payload = {
        "schema_version": 1,
        "figure": "Supplementary Figure S6: t-SNE of frozen representations",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_figure": "Figure 3 of the legacy manuscript, regenerated",
        "cohort": {
            "sessions_available": EXPECTED_ROWS,
            "primary_acquisitions": EXPECTED_SELECTED_ROWS,
            "participants": EXPECTED_SUBJECTS,
            "scanners": EXPECTED_SCANNERS,
            "selection": "session_id suffix 001 in every participant-scanner cell",
        },
        "method": {
            "feature_preprocessing": (
                "remove constant dimensions, then coordinate-wise StandardScaler"
            ),
            "algorithm": "sklearn.manifold.TSNE",
            "n_components": 2,
            "perplexity": args.perplexity,
            "initialization": "PCA",
            "learning_rate": "auto",
            "random_seed": args.seed,
            "interpretation": "descriptive visualization; not used for inference",
            "anatcl_variant": "fold0 only; coordinates are not aligned across folds",
        },
        "figure2_results": {
            "path": str(figure2_results),
            "sha256": sha256_file(figure2_results),
        },
        "sources": sources,
        "projections": projections,
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "script": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "outputs": [
            {
                "path": str(output),
                "sha256": sha256_file(output),
                "bytes": output.stat().st_size,
            }
            for output in outputs
        ],
    }
    atomic_json(manifest_path, payload)
    for output in outputs:
        print(f"Saved: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
