#!/usr/bin/env python3
"""Build the auditable analysis inputs and results underlying Figure 2.

The primary endpoint is full-vector cross-scanner dbICC. Acquisition 001 is
retained in every participant-scanner cell so that each cell contributes one
scan with the same precision. Uncertainty is obtained from a nonparametric
participant bootstrap, using the same ordered resamples for every condition
and fold.

The companion R script fits crossed participant/scanner random-intercept
models coordinate by coordinate. Native-scale variance components are summed
before conversion to proportions, yielding a trace decomposition aligned with
the geometry of the complete representation rather than an average of
coordinate-wise percentages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_EMBEDDING_ROOT = PROJECT_ROOT / "DATA" / "embeddings"
DEFAULT_ANATCL_EMBEDDING_ROOT = PROJECT_ROOT / "DATA" / "embeddings_bfloat16_v2"
DEFAULT_FREESURFER_ROOT = (
    PROJECT_ROOT / "Results" / "freesurfer_thalamus_merge_201" / "lme4_inputs"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Results" / "Figure2"
R_SCRIPT = SCRIPT_PATH.with_name("figure2_variance_components.R")
EXPECTED_ROWS = 165
EXPECTED_SUBJECTS = 20
EXPECTED_SCANNERS = 8
EXPECTED_SUBJECT_SCANNER_CELLS = 120
METADATA_COLUMNS = ("subject_id", "session_id", "scanner")
DBICC_REFERENCE = "Xu, Reiss, and Cribben (2021), doi:10.1111/biom.13287"
SIMCLR_PIPELINE_VERSION = "anatcl_matched_simclr_onharmony_embedding_v1"
PRIMARY_ACQUISITION_SUFFIX = "001"


@dataclass(frozen=True)
class InputSpec:
    condition: str
    label: str
    files: tuple[str, ...]
    feature_prefix: str
    expected_dimensions: int
    role: str
    manifest_files: tuple[str, ...] = ()
    primary_folds: tuple[int, ...] = (0,)


INPUT_SPECS = (
    InputSpec(
        "freesurfer",
        "FreeSurfer",
        ("FreeSurfer.csv",),
        "emb_",
        201,
        "reference",
        ("manifest.json",),
    ),
    InputSpec(
        "freesurfer_combat",
        "FreeSurfer + ComBat",
        ("FreeSurfer_harmonized.csv",),
        "emb_",
        201,
        "harmonization_sensitivity",
        ("manifest.json",),
    ),
    InputSpec(
        "matched_simclr",
        "Matched SimCLR",
        tuple(f"embeddings_AnatCL_SimCLR_fold{fold}.csv" for fold in range(5)),
        "emb_",
        512,
        "targeted_control",
        tuple(
            f"embeddings_AnatCL_SimCLR_fold{fold}_manifest.json"
            for fold in range(5)
        ),
    ),
    InputSpec(
        "yaware",
        "y-Aware",
        ("embeddings_yAware.csv",),
        "emb_",
        1024,
        "primary_model",
        ("embeddings_yAware_manifest.json",),
    ),
    InputSpec(
        "anatcl",
        "AnatCL",
        tuple(f"embeddings_AnatCL_fold{fold}.csv" for fold in range(5)),
        "emb_",
        512,
        "primary_model",
        tuple(
            f"embeddings_AnatCL_fold{fold}_manifest.json" for fold in range(5)
        ),
    ),
    InputSpec(
        "brainsegfounder",
        "BrainSegFounder",
        ("embeddings_BrainSegFounder.csv",),
        "emb_",
        768,
        "primary_model",
        ("embeddings_BrainSegFounder_manifest.json",),
    ),
    InputSpec(
        "3dneurosimclr",
        "3D-Neuro-SimCLR",
        ("embeddings_3DNeuroSimCLR.csv",),
        "emb_",
        512,
        "primary_model",
        ("embeddings_3DNeuroSimCLR_manifest.json",),
    ),
    InputSpec(
        "neurovfm",
        "NeuroVFM",
        ("embeddings_NeuroVFM.csv",),
        "emb_",
        768,
        "primary_model",
        ("embeddings_NeuroVFM_manifest.json",),
    ),
    InputSpec(
        "curia2",
        "Curia-2",
        ("embeddings_Curia2.csv",),
        "cls_all_",
        768,
        "primary_model",
        ("embeddings_Curia2_manifest.json",),
    ),
    InputSpec(
        "brainiac",
        "BrainIAC",
        ("embeddings_BrainIAC.csv",),
        "emb_",
        768,
        "primary_model",
        ("embeddings_BrainIAC_manifest.json",),
    ),
)


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


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def numbered_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}([0-9]+)$")
    indexed = []
    for column in frame.columns:
        match = pattern.fullmatch(column)
        if match:
            indexed.append((int(match.group(1)), column))
    indexed.sort()
    observed = [index for index, _ in indexed]
    expected = list(range(len(indexed)))
    if observed != expected:
        raise ValueError(
            f"Feature columns with prefix {prefix!r} are not contiguous from zero"
        )
    return [column for _, column in indexed]


def parse_bids_metadata(source: pd.Series) -> pd.DataFrame:
    values = source.astype(str).map(os.path.basename)
    subjects = values.str.extract(r"(sub-[0-9]+)", expand=False)
    sessions = values.str.extract(r"(ses-[A-Za-z0-9]+)", expand=False)
    scanners = sessions.str.replace(r"[0-9]{3}$", "", regex=True).str.replace(
        "ses-", "", regex=False
    )
    parsed = pd.DataFrame(
        {"subject_id": subjects, "session_id": sessions, "scanner": scanners}
    )
    if parsed.isna().any().any() or parsed.eq("").any().any():
        bad = parsed[parsed.isna().any(axis=1) | parsed.eq("").any(axis=1)]
        raise ValueError(f"Could not parse BIDS metadata for rows {bad.index.tolist()[:10]}")
    return parsed


def normalize_metadata(frame: pd.DataFrame, *, condition: str) -> pd.DataFrame:
    result = frame.copy()
    if condition.startswith("freesurfer"):
        result["subject_id"] = (
            result["subject_id"].astype(str).str.replace(r"^_", "sub-", regex=True)
        )
        result["session_id"] = (
            "ses-"
            + result["session_id"].astype(str).str.extract(
                r"_([A-Za-z0-9]+)$", expand=False
            )
        )
    elif not set(METADATA_COLUMNS).issubset(result.columns):
        source_column = "stem" if "stem" in result else "file_path"
        if source_column not in result:
            raise ValueError(f"{condition} lacks metadata and a parseable source column")
        parsed = parse_bids_metadata(result[source_column])
        for column in METADATA_COLUMNS:
            result[column] = parsed[column]

    if result[list(METADATA_COLUMNS)].isna().any().any():
        raise ValueError(f"{condition} contains missing acquisition metadata")
    for column in METADATA_COLUMNS:
        result[column] = result[column].astype(str)
    if result[list(METADATA_COLUMNS)].eq("").any().any():
        raise ValueError(f"{condition} contains incomplete acquisition metadata")
    return result


def canonical_keys(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[list(METADATA_COLUMNS)]
        .astype(str)
        .sort_values(list(METADATA_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )


def validate_manifest(
    manifest_path: Path,
    data_path: Path,
    *,
    spec: InputSpec,
    fold_index: int,
) -> dict:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing manifest for {spec.label} fold {fold_index}: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if spec.condition == "matched_simclr":
        if payload.get("manifest_schema_version") != "onharmony_embedding_manifest_v1":
            raise ValueError(f"Stale SimCLR manifest schema: {manifest_path}")
        if payload.get("pipeline_version") != SIMCLR_PIPELINE_VERSION:
            raise ValueError(f"Stale SimCLR pipeline contract: {manifest_path}")
        if payload.get("variant") != f"fold{fold_index}":
            raise ValueError(f"Wrong SimCLR fold manifest: {manifest_path}")
    expected_hash = payload.get("output_sha256")
    if expected_hash is None and spec.condition.startswith("freesurfer"):
        output = payload.get("outputs", {}).get(
            "FreeSurfer_harmonized"
            if spec.condition == "freesurfer_combat"
            else "FreeSurfer",
            {},
        )
        expected_hash = output.get("sha256")
    actual_hash = sha256_file(data_path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"Manifest/data SHA-256 mismatch for {data_path}: "
            f"manifest={expected_hash}, actual={actual_hash}"
        )
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "data_sha256": actual_hash,
    }


def prepare_input(
    path: Path,
    *,
    spec: InputSpec,
    destination: Path,
) -> tuple[pd.DataFrame, list[str]]:
    frame = normalize_metadata(pd.read_csv(path), condition=spec.condition)
    features = numbered_columns(frame, spec.feature_prefix)
    if len(features) != spec.expected_dimensions:
        raise ValueError(
            f"{path} has {len(features)} {spec.feature_prefix} dimensions; "
            f"expected {spec.expected_dimensions}"
        )
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"{path} has {len(frame)} rows; expected {EXPECTED_ROWS}")
    if frame[list(METADATA_COLUMNS)].duplicated().any():
        raise ValueError(f"Duplicate acquisition metadata in {path}")
    values = frame[features].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite representation values in {path}")
    renamed = frame[list(METADATA_COLUMNS) + features].copy()
    renamed = renamed.rename(
        columns={feature: f"emb_{index}" for index, feature in enumerate(features)}
    )
    renamed = renamed.sort_values(list(METADATA_COLUMNS), kind="stable").reset_index(
        drop=True
    )
    atomic_csv(destination, renamed)
    return renamed, [f"emb_{index}" for index in range(len(features))]


def select_primary_acquisition_per_cell(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain acquisition 001 when participant-scanner cells contain repeats."""
    required = ["subject_id", "scanner"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Primary-acquisition input is missing columns: {missing}")
    cell_columns = ["subject_id", "scanner"]
    n_cells = frame[cell_columns].drop_duplicates().shape[0]
    if not frame.duplicated(cell_columns).any():
        return frame.sort_values(cell_columns, kind="stable").reset_index(drop=True)

    session_column = next(
        (column for column in ("session_id", "session") if column in frame), None
    )
    if session_column is None:
        raise ValueError(
            "Repeated participant-scanner cells require session_id or session"
        )
    sessions = frame[session_column].astype(str)
    selected = frame.loc[sessions.str.endswith(PRIMARY_ACQUISITION_SUFFIX)].copy()
    if len(selected) != n_cells or selected.duplicated(cell_columns).any():
        raise ValueError(
            "Every participant-scanner cell must contain exactly one acquisition 001"
        )
    return selected.sort_values(cell_columns, kind="stable").reset_index(drop=True)


def aggregate_subject_scanner_cells(
    frame: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    required = ["subject_id", "scanner", *feature_columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"dbICC input is missing columns: {missing}")
    values = frame[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("dbICC input contains non-finite values")
    selected = select_primary_acquisition_per_cell(frame)
    cells = selected[["subject_id", "scanner", *feature_columns]].copy()
    counts = cells.groupby("subject_id", sort=True).size()
    if len(counts) != EXPECTED_SUBJECTS or (counts < 2).any():
        raise ValueError(
            "Cross-scanner dbICC requires the expected participants with at least "
            "two scanner cells each"
        )
    return cells


def squared_euclidean(values: np.ndarray) -> np.ndarray:
    norms = np.einsum("ij,ij->i", values, values)
    distances = norms[:, None] + norms[None, :] - 2.0 * values.dot(values.T)
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    return distances


@dataclass(frozen=True)
class DistanceBlocks:
    subjects: tuple[str, ...]
    within_sums: np.ndarray
    within_counts: np.ndarray
    between_sums: np.ndarray
    between_counts: np.ndarray
    n_cells: int
    n_features: int


def distance_blocks(cells: pd.DataFrame, feature_columns: list[str]) -> DistanceBlocks:
    values = cells[feature_columns].to_numpy(dtype=np.float64)
    distances = squared_euclidean(values)
    labels = cells["subject_id"].astype(str).to_numpy()
    subjects = tuple(sorted(np.unique(labels).tolist()))
    indices = [np.flatnonzero(labels == subject) for subject in subjects]
    count = len(subjects)
    within_sums = np.zeros(count, dtype=np.float64)
    within_counts = np.zeros(count, dtype=np.int64)
    between_sums = np.zeros((count, count), dtype=np.float64)
    between_counts = np.zeros((count, count), dtype=np.int64)
    for left, index in enumerate(indices):
        upper = np.triu_indices(len(index), k=1)
        block = distances[np.ix_(index, index)]
        within_sums[left] = block[upper].sum()
        within_counts[left] = len(upper[0])
        for right in range(left + 1, count):
            cross = distances[np.ix_(index, indices[right])]
            between_sums[left, right] = between_sums[right, left] = cross.sum()
            between_counts[left, right] = between_counts[right, left] = cross.size
    return DistanceBlocks(
        subjects,
        within_sums,
        within_counts,
        between_sums,
        between_counts,
        len(cells),
        len(feature_columns),
    )


def dbicc_from_totals(
    within_sum: float,
    within_count: int,
    between_sum: float,
    between_count: int,
) -> tuple[float, float, float]:
    if within_count <= 0 or between_count <= 0:
        raise ValueError("dbICC requires within- and between-participant pairs")
    within = within_sum / within_count
    between = between_sum / between_count
    if not np.isfinite(between) or between <= 0:
        raise ValueError("Between-participant mean squared distance is not positive")
    return float(1.0 - within / between), float(within), float(between)


def point_dbicc(blocks: DistanceBlocks) -> tuple[float, float, float]:
    upper = np.triu_indices(len(blocks.subjects), k=1)
    return dbicc_from_totals(
        float(blocks.within_sums.sum()),
        int(blocks.within_counts.sum()),
        float(blocks.between_sums[upper].sum()),
        int(blocks.between_counts[upper].sum()),
    )


def bootstrap_indices(n_subjects: int, n_bootstrap: int, seed: int) -> np.ndarray:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    return np.random.RandomState(seed).randint(
        0, n_subjects, size=(n_bootstrap, n_subjects)
    )


def bootstrap_dbicc(blocks: DistanceBlocks, samples: np.ndarray) -> np.ndarray:
    n_subjects = len(blocks.subjects)
    if samples.ndim != 2 or samples.shape[1] != n_subjects:
        raise ValueError("Bootstrap sample matrix does not match participant count")
    pair_left, pair_right = np.triu_indices(n_subjects, k=1)
    estimates = np.full(len(samples), np.nan, dtype=np.float64)
    for index, sampled in enumerate(samples):
        within_sum = float(blocks.within_sums[sampled].sum())
        within_count = int(blocks.within_counts[sampled].sum())
        source_left = sampled[pair_left]
        source_right = sampled[pair_right]
        distinct = source_left != source_right
        if not distinct.any():
            continue
        source_left = source_left[distinct]
        source_right = source_right[distinct]
        estimates[index] = dbicc_from_totals(
            within_sum,
            within_count,
            float(blocks.between_sums[source_left, source_right].sum()),
            int(blocks.between_counts[source_left, source_right].sum()),
        )[0]
    if not np.isfinite(estimates).all():
        raise RuntimeError("One or more dbICC bootstrap replicates failed")
    return estimates


def compute_dbicc(
    frame: pd.DataFrame,
    feature_columns: list[str],
    samples: np.ndarray,
) -> dict:
    cells = aggregate_subject_scanner_cells(frame, feature_columns)
    blocks = distance_blocks(cells, feature_columns)
    estimate, msd_within, msd_between = point_dbicc(blocks)
    bootstrap = bootstrap_dbicc(blocks, samples)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_values": bootstrap.tolist(),
        "msd_within_subject": msd_within,
        "msd_between_subject": msd_between,
        "n_sessions_input": len(frame),
        "n_subject_scanner_cells": len(cells),
        "n_subjects": len(blocks.subjects),
        "n_scanners": int(cells["scanner"].nunique()),
        "n_dimensions": len(feature_columns),
        "distance": "squared Euclidean distance on native full vectors",
        "repeat_handling": "acquisition 001 retained in each participant-scanner cell",
        "formula": "1 - MSD_within_participant / MSD_between_participant",
        "bootstrap": {
            "method": "nonparametric participant bootstrap",
            "duplicate_correction": (
                "duplicate copies of one source participant excluded from the "
                "between-participant distance pool"
            ),
            "n": len(samples),
        },
        "reference": DBICC_REFERENCE,
    }


def aggregate_fold_results(results: list[dict]) -> dict:
    if len(results) < 2:
        return results[0]
    bootstrap = np.asarray(
        [result["bootstrap_values"] for result in results], dtype=np.float64
    )
    aggregate_bootstrap = bootstrap.mean(axis=0)
    lower, upper = np.percentile(aggregate_bootstrap, [2.5, 97.5])
    return {
        "estimate": float(np.mean([result["estimate"] for result in results])),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_values": aggregate_bootstrap.tolist(),
        "fold_estimates": [float(result["estimate"]) for result in results],
        "fold_standard_deviation": float(
            np.std([result["estimate"] for result in results], ddof=1)
        ),
        "n_folds": len(results),
        "fold_aggregation": (
            "arithmetic mean of selected fold dbICC estimates; confidence interval "
            "from the mean across folds within each shared participant bootstrap"
        ),
        "n_sessions_input": results[0]["n_sessions_input"],
        "n_subject_scanner_cells": results[0]["n_subject_scanner_cells"],
        "n_subjects": results[0]["n_subjects"],
        "n_scanners": results[0]["n_scanners"],
        "n_dimensions_per_fold": results[0]["n_dimensions"],
        "distance": results[0]["distance"],
        "repeat_handling": results[0]["repeat_handling"],
        "formula": results[0]["formula"],
        "bootstrap": results[0]["bootstrap"],
        "reference": results[0]["reference"],
    }


def paired_bootstrap_contrast(left: dict, right: dict, *, label: str) -> dict:
    left_values = np.asarray(left["bootstrap_values"], dtype=np.float64)
    right_values = np.asarray(right["bootstrap_values"], dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError(f"Bootstrap shape mismatch for {label}")
    differences = left_values - right_values
    lower, upper = np.percentile(differences, [2.5, 97.5])
    nonpositive = int(np.count_nonzero(differences <= 0))
    nonnegative = int(np.count_nonzero(differences >= 0))
    p_value = min(
        1.0,
        2.0 * (min(nonpositive, nonnegative) + 1) / (len(differences) + 1),
    )
    return {
        "contrast": label,
        "estimate": float(left["estimate"] - right["estimate"]),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value_two_sided": float(p_value),
        "bootstrap_values": differences.tolist(),
        "multiplicity": "targeted descriptive contrast; no family-wise adjustment",
    }


def run_reml(
    input_path: Path,
    output_path: Path,
    *,
    condition: str,
    workers: int,
    rscript: str,
    reuse_existing: bool = False,
) -> dict:
    input_sha256 = sha256_file(input_path)
    command = [
        rscript,
        str(R_SCRIPT),
        "--input",
        str(input_path),
        "--input-sha256",
        input_sha256,
        "--output",
        str(output_path),
        "--condition",
        condition,
        "--workers",
        str(workers),
    ]
    if not reuse_existing:
        subprocess.run(command, check=True)
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    expected_input = str(input_path.resolve())
    expected_dimensions = len(numbered_columns(pd.read_csv(input_path, nrows=0), "emb_"))
    expected = {
        "schema_version": 1,
        "condition": condition,
        "input_csv": expected_input,
        "n_sessions_input": EXPECTED_ROWS,
        "n_subject_scanner_cells": EXPECTED_SUBJECT_SCANNER_CELLS,
        "n_subjects": EXPECTED_SUBJECTS,
        "n_scanners": EXPECTED_SCANNERS,
        "n_dimensions": expected_dimensions,
        "input_csv_sha256": input_sha256,
    }
    mismatch = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Invalid REML result {output_path}: {mismatch}")
    payload["result_json"] = str(output_path.resolve())
    payload["result_json_sha256"] = sha256_file(output_path)
    payload["execution"] = "reused validated result" if reuse_existing else "executed now"
    return payload


def aggregate_trace_results(results: list[dict]) -> dict:
    traces = {
        component: float(sum(result[f"trace_{component}"] for result in results))
        for component in ("subject", "scanner", "residual")
    }
    total = sum(traces.values())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid fold-aggregated variance trace")
    return {
        "trace_subject": traces["subject"],
        "trace_scanner": traces["scanner"],
        "trace_residual": traces["residual"],
        "trace_total": total,
        "proportion_subject": traces["subject"] / total,
        "proportion_scanner": traces["scanner"] / total,
        "proportion_residual": traces["residual"] / total,
        "n_folds": len(results),
        "n_dimensions_valid": int(
            sum(result["n_dimensions_valid"] for result in results)
        ),
        "n_dimensions_failed_or_constant": int(
            sum(result["n_dimensions_failed_or_constant"] for result in results)
        ),
        "n_singular": int(sum(result["n_singular"] for result in results)),
        "estimator": results[0]["estimator"],
        "formula": results[0]["formula"],
        "repeat_handling": results[0]["repeat_handling"],
        "response_scaling": results[0]["response_scaling"],
        "pooling": (
            "component traces summed across all coordinates and folds before "
            "conversion to proportions"
        ),
    }


def paths_for_spec(
    spec: InputSpec,
    embedding_root: Path,
    freesurfer_root: Path,
    anatcl_embedding_root: Path,
) -> tuple[list[Path], list[Path]]:
    if spec.condition.startswith("freesurfer"):
        root = freesurfer_root
    elif spec.condition == "anatcl":
        root = anatcl_embedding_root
    else:
        root = embedding_root
    return (
        [root / name for name in spec.files],
        [root / name for name in spec.manifest_files],
    )


def assert_shared_acquisition_table(
    prepared: dict[str, list[pd.DataFrame]],
) -> None:
    reference = canonical_keys(prepared["3dneurosimclr"][0])
    if len(reference) != EXPECTED_ROWS:
        raise ValueError("Reference acquisition table is incomplete")
    for condition, frames in prepared.items():
        for fold, frame in enumerate(frames):
            observed = canonical_keys(frame)
            if not observed.equals(reference):
                merged = reference.merge(
                    observed,
                    on=list(METADATA_COLUMNS),
                    how="outer",
                    indicator=True,
                )
                mismatch = merged[merged["_merge"] != "both"].head(10)
                raise ValueError(
                    f"{condition} fold {fold} does not share the canonical "
                    f"acquisition table: {mismatch.to_dict(orient='records')}"
                )


def build_analysis(args: argparse.Namespace) -> dict:
    embedding_root = Path(args.embedding_root).expanduser().resolve()
    anatcl_embedding_root = Path(args.anatcl_embedding_root).expanduser().resolve()
    freesurfer_root = Path(args.freesurfer_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    input_root = output_root / "inputs"
    reml_root = output_root / "reml"
    output_root.mkdir(parents=True, exist_ok=True)

    samples = bootstrap_indices(EXPECTED_SUBJECTS, args.n_bootstrap, args.seed)
    prepared: dict[str, list[pd.DataFrame]] = {}
    features_by_condition: dict[str, list[list[str]]] = {}
    provenance: dict[str, list[dict]] = {}

    for spec in INPUT_SPECS:
        data_paths, manifest_paths = paths_for_spec(
            spec, embedding_root, freesurfer_root, anatcl_embedding_root
        )
        if len(manifest_paths) not in (1, len(data_paths)):
            raise ValueError(f"Ambiguous manifest registry for {spec.condition}")
        prepared[spec.condition] = []
        features_by_condition[spec.condition] = []
        provenance[spec.condition] = []
        for fold, data_path in enumerate(data_paths):
            if not data_path.is_file():
                raise FileNotFoundError(data_path)
            manifest_path = manifest_paths[0] if len(manifest_paths) == 1 else manifest_paths[fold]
            manifest_record = validate_manifest(
                manifest_path, data_path, spec=spec, fold_index=fold
            )
            destination = input_root / f"{spec.condition}__fold{fold}.csv"
            frame, feature_columns = prepare_input(
                data_path, spec=spec, destination=destination
            )
            prepared[spec.condition].append(frame)
            features_by_condition[spec.condition].append(feature_columns)
            provenance[spec.condition].append(
                {
                    "source": str(data_path.resolve()),
                    "source_sha256": sha256_file(data_path),
                    "manifest": manifest_record,
                    "analysis_input": str(destination.resolve()),
                    "analysis_input_sha256": sha256_file(destination),
                    "rows": len(frame),
                    "dimensions": len(feature_columns),
                }
            )

    assert_shared_acquisition_table(prepared)
    conditions = {}
    for spec in INPUT_SPECS:
        fold_dbicc = [
            compute_dbicc(frame, feature_columns, samples)
            for frame, feature_columns in zip(
                prepared[spec.condition], features_by_condition[spec.condition]
            )
        ]
        if any(
            result["n_subject_scanner_cells"] != EXPECTED_SUBJECT_SCANNER_CELLS
            for result in fold_dbicc
        ):
            raise ValueError(
                f"{spec.label} does not contain exactly "
                f"{EXPECTED_SUBJECT_SCANNER_CELLS} primary acquisitions"
            )
        if not spec.primary_folds or any(
            fold < 0 or fold >= len(fold_dbicc) for fold in spec.primary_folds
        ):
            raise ValueError(f"Invalid primary fold selection for {spec.condition}")
        dbicc = aggregate_fold_results([fold_dbicc[fold] for fold in spec.primary_folds])
        fold_reml = []
        if not args.skip_reml:
            for fold in range(len(fold_dbicc)):
                input_path = Path(
                    provenance[spec.condition][fold]["analysis_input"]
                )
                output_path = reml_root / f"{spec.condition}__fold{fold}.json"
                fold_reml.append(
                    run_reml(
                        input_path,
                        output_path,
                        condition=f"{spec.condition}__fold{fold}",
                        workers=args.workers,
                        rscript=args.rscript,
                        reuse_existing=args.reuse_reml,
                    )
                )
            variance = aggregate_trace_results([fold_reml[fold] for fold in spec.primary_folds])
        else:
            variance = None
        conditions[spec.condition] = {
            "label": spec.label,
            "role": spec.role,
            "primary_folds": list(spec.primary_folds),
            "dbicc": dbicc,
            "variance": variance,
            "fold_dbicc": fold_dbicc,
            "fold_variance": fold_reml,
            "inputs": provenance[spec.condition],
            "fold_sensitivity": (
                {
                    "definition": "all released folds evaluated separately",
                    "dbicc": aggregate_fold_results(fold_dbicc),
                    "variance": aggregate_trace_results(fold_reml) if fold_reml else None,
                }
                if len(fold_dbicc) > 1 else None
            ),
        }

    contrasts = {
        "anatcl_minus_matched_simclr": paired_bootstrap_contrast(
            conditions["anatcl"]["dbicc"],
            conditions["matched_simclr"]["dbicc"],
            label="AnatCL fold 0 minus AnatCL-SimCLR fold 0",
        ),
        "anatcl_minus_matched_simclr_all_folds": paired_bootstrap_contrast(
            conditions["anatcl"]["fold_sensitivity"]["dbicc"],
            conditions["matched_simclr"]["fold_sensitivity"]["dbicc"],
            label="AnatCL minus AnatCL-SimCLR: five matched folds sensitivity",
        ),
        "combat_minus_raw_freesurfer": paired_bootstrap_contrast(
            conditions["freesurfer_combat"]["dbicc"],
            conditions["freesurfer"]["dbicc"],
            label="FreeSurfer + ComBat minus raw FreeSurfer",
        ),
    }

    payload = {
        "schema_version": 1,
        "analysis": "Figure 2: cross-acquisition representation reliability",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "sessions": EXPECTED_ROWS,
            "primary_acquisitions": EXPECTED_SUBJECT_SCANNER_CELLS,
            "participants": EXPECTED_SUBJECTS,
            "scanners": EXPECTED_SCANNERS,
            "acquisition_table": "exactly matched across all conditions and folds",
        },
        "dbicc": {
            "primary_endpoint": "full-vector cross-scanner dbICC",
            "distance": "squared Euclidean distance on native representations",
            "repeat_handling": (
                "acquisition 001 retained in each participant-scanner cell"
            ),
            "bootstrap_seed": args.seed,
            "bootstrap_n": args.n_bootstrap,
            "bootstrap_samples_sha256": hashlib.sha256(
                samples.astype(np.int64).tobytes()
            ).hexdigest(),
            "reference": DBICC_REFERENCE,
        },
        "variance_decomposition": {
            "estimator": None if args.skip_reml else "REML via lme4::lmer",
            "execution": (
                None
                if args.skip_reml
                else "validated existing per-condition results"
                if args.reuse_reml
                else "executed during this command"
            ),
            "formula": "value ~ 1 + (1 | subject_id) + (1 | scanner)",
            "repeat_handling": (
                "acquisition 001 retained in each participant-scanner cell"
            ),
            "pooling": "native-scale component traces",
        },
        "conditions": conditions,
        "targeted_contrasts": contrasts,
        "code": {
            "analysis_script": str(SCRIPT_PATH),
            "analysis_script_sha256": sha256_file(SCRIPT_PATH),
            "reml_script": str(R_SCRIPT),
            "reml_script_sha256": sha256_file(R_SCRIPT),
        },
    }
    result_path = output_root / "figure2_reliability_results.json"
    atomic_json(result_path, payload)

    rows = []
    for spec in INPUT_SPECS:
        record = conditions[spec.condition]
        variance = record["variance"] or {}
        rows.append(
            {
                "condition": spec.condition,
                "label": spec.label,
                "role": spec.role,
                "dbicc": record["dbicc"]["estimate"],
                "dbicc_ci_lower": record["dbicc"]["ci_lower"],
                "dbicc_ci_upper": record["dbicc"]["ci_upper"],
                "subject_variance_proportion": variance.get("proportion_subject"),
                "scanner_variance_proportion": variance.get("proportion_scanner"),
                "residual_variance_proportion": variance.get("proportion_residual"),
            }
        )
    atomic_csv(output_root / "figure2_reliability_summary.csv", pd.DataFrame(rows))
    return payload


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--embedding-root", default=DEFAULT_EMBEDDING_ROOT)
    argument_parser.add_argument(
        "--anatcl-embedding-root",
        default=DEFAULT_ANATCL_EMBEDDING_ROOT,
        help="Authoritative BF16 ON-Harmony embeddings for AnatCL only",
    )
    argument_parser.add_argument("--freesurfer-root", default=DEFAULT_FREESURFER_ROOT)
    argument_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    argument_parser.add_argument("--n-bootstrap", type=int, default=1000)
    argument_parser.add_argument("--seed", type=int, default=42)
    argument_parser.add_argument("--workers", type=int, default=8)
    argument_parser.add_argument("--rscript", default="Rscript")
    argument_parser.add_argument(
        "--skip-reml",
        action="store_true",
        help="Prepare inputs and compute dbICC without fitting variance components",
    )
    argument_parser.add_argument(
        "--reuse-reml",
        action="store_true",
        help=(
            "Reuse per-condition REML JSON only after validating its identity, "
            "cohort, dimensions, and input path; current input/result hashes are recorded"
        ),
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    result = build_analysis(args)
    print("Figure 2 analysis complete")
    for spec in INPUT_SPECS:
        dbicc = result["conditions"][spec.condition]["dbicc"]
        print(
            f"  {spec.label:24s} {dbicc['estimate']:.3f} "
            f"[{dbicc['ci_lower']:.3f}, {dbicc['ci_upper']:.3f}]"
        )


if __name__ == "__main__":
    main()
