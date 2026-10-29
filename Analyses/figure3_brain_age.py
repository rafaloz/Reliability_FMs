#!/usr/bin/env python3
"""Build the auditable analysis results underlying Figure 3.

Figure 3 asks whether a frozen Ridge readout can recover a useful and
cross-acquisition-robust brain-age signal from each complete representation.
It separates four quantities that should not be conflated:

* held-out brain-age accuracy (MAE on the locked internal test),
* relative reliability (cross-scanner dbICC of predicted age),
* absolute instability (expected absolute change between two scanners), and
* systematic scanner shifts after controlling for participant.

Acquisition 001 is retained in each participant-scanner cell.
All uncertainty is obtained by resampling participants, with shared resamples
across models and folds. Scanner-effect bootstraps are stratified by ON-Harmony
phase so that the partially crossed scanner design remains connected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Analyses.figure2_reliability import (  # noqa: E402
    aggregate_fold_results,
    bootstrap_indices,
    compute_dbicc,
    paired_bootstrap_contrast,
    select_primary_acquisition_per_cell,
)


DEFAULT_BRAIN_AGE_ROOT = PROJECT_ROOT / "Results" / "brain_age"
DEFAULT_ANATCL_FROZEN_INTERNAL_ROOT = (
    DEFAULT_BRAIN_AGE_ROOT / "internal_test" / "frozen_bfloat16_v3" / "AnatCL"
)
DEFAULT_ANATCL_FROZEN_ONHARMONY_ROOT = (
    DEFAULT_BRAIN_AGE_ROOT / "onharmony" / "frozen_bfloat16_v3" / "AnatCL"
)
DEFAULT_FIGURE2_RESULTS = (
    PROJECT_ROOT / "Results" / "Figure2" / "figure2_reliability_results.json"
)
DEFAULT_PARTICIPANTS = PROJECT_ROOT / "DATA" / "participants.tsv"
DEFAULT_COHORT_MANIFEST = (
    PROJECT_ROOT
    / "preprocessings"
    / "FineTunningDataset"
    / "brainage_splits"
    / "brainage_definitive_subject_manifest.tsv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Results" / "Figure3"

EXPECTED_ONHARMONY_ROWS = 165
EXPECTED_ONHARMONY_SUBJECTS = 20
EXPECTED_SCANNERS = 8
EXPECTED_SCANNERS_PER_SUBJECT = 6
EXPECTED_SUBJECT_SCANNER_CELLS = 120
EXPECTED_INTERNAL_TEST_ROWS = 140

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
SCANNER_REFERENCE = "Ach"
SCANNER_REFERENCE_LABEL = "NOT1 (Achieva)"


@dataclass(frozen=True)
class ModelSpec:
    condition: str
    label: str
    figure2_condition: str
    variants: tuple[str, ...]
    expected_model: str


MODEL_SPECS = (
    ModelSpec(
        "yaware",
        "y-Aware",
        "yaware",
        ("yAware/canonical",),
        "y-Aware",
    ),
    ModelSpec(
        "anatcl",
        "AnatCL",
        "anatcl",
        ("AnatCL/fold0",),
        "AnatCL",
    ),
    ModelSpec(
        "brainsegfounder",
        "BrainSegFounder",
        "brainsegfounder",
        ("BrainSegFounder/bottleneck_mean",),
        "BrainSegFounder",
    ),
    ModelSpec(
        "3dneurosimclr",
        "3D-Neuro-SimCLR",
        "3dneurosimclr",
        ("3DNeuroSimCLR/canonical",),
        "3D-Neuro-SimCLR",
    ),
    ModelSpec(
        "neurovfm",
        "NeuroVFM",
        "neurovfm",
        ("NeuroVFM/canonical",),
        "NeuroVFM",
    ),
    ModelSpec(
        "curia2",
        "Curia-2",
        "curia2",
        ("Curia2/canonical",),
        "Curia-2",
    ),
    ModelSpec(
        "brainiac",
        "BrainIAC",
        "brainiac",
        ("BrainIAC/canonical",),
        "BrainIAC",
    ),
)


def frozen_prediction_dirs(
    brain_age_root: Path,
    *,
    spec: ModelSpec,
    variant: str,
    anatcl_internal_root: Path,
    anatcl_onharmony_root: Path,
) -> tuple[Path, Path]:
    """Resolve frozen result roots without changing any non-AnatCL model."""
    if spec.condition == "anatcl":
        fold = variant_name(variant)
        return anatcl_internal_root / fold, anatcl_onharmony_root / fold
    return (
        brain_age_root / "internal_test" / "frozen" / variant,
        brain_age_root / "onharmony" / "frozen" / variant,
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
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def finite_percentile(values: np.ndarray, limits: tuple[float, float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Bootstrap distribution must be a finite vector")
    lower, upper = np.percentile(values, limits)
    return float(lower), float(upper)


def variant_name(variant_path: str) -> str:
    return Path(variant_path).name


def validate_prediction_audit(
    audit_path: Path,
    predictions_path: Path,
    *,
    spec: ModelSpec,
    variant: str,
) -> dict:
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "status": "ON_HARMONY_PREDICTED",
        "arm": "frozen",
        "model": spec.expected_model,
        "variant": variant_name(variant),
        "label_blind": True,
        "outcome_labels_loaded": False,
        "bias_correction": "none",
        "n_predictions": EXPECTED_ONHARMONY_ROWS,
        "n_subjects": EXPECTED_ONHARMONY_SUBJECTS,
        "internal_test_accessed": False,
        "on_harmony_accessed": True,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Invalid ON-Harmony audit {audit_path}: {mismatches}")
    actual_hash = sha256_file(predictions_path)
    if payload.get("predictions_artifact_sha256") != actual_hash:
        raise ValueError(f"ON-Harmony prediction hash mismatch: {predictions_path}")
    return {
        "audit": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": actual_hash,
        "ridge_model_sha256": payload.get("ridge_model_sha256"),
        "ridge_alpha": payload.get("ridge_alpha"),
        "training_lock_sha256": payload.get("training_lock_sha256"),
    }


def load_onharmony_predictions(
    predictions_path: Path,
    *,
    spec: ModelSpec,
    variant: str,
) -> pd.DataFrame:
    frame = pd.read_csv(predictions_path, sep="\t")
    required = {
        "model",
        "variant",
        "arm",
        "subject",
        "session",
        "acquisition_id",
        "cohort",
        "scanner_code",
        "scanner_name",
        "vendor",
        "age_pred",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{predictions_path} lacks columns: {missing}")
    if len(frame) != EXPECTED_ONHARMONY_ROWS:
        raise ValueError(f"{predictions_path} has {len(frame)} rows")
    if frame["acquisition_id"].duplicated().any():
        raise ValueError(f"Duplicate ON-Harmony acquisition: {predictions_path}")
    expected_values = {
        "model": spec.expected_model,
        "variant": variant_name(variant),
        "arm": "frozen",
    }
    for column, expected in expected_values.items():
        observed = set(frame[column].astype(str).unique())
        if observed != {expected}:
            raise ValueError(
                f"Unexpected {column} in {predictions_path}: {sorted(observed)}"
            )
    values = frame["age_pred"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite ON-Harmony predictions: {predictions_path}")
    if frame["subject"].nunique() != EXPECTED_ONHARMONY_SUBJECTS:
        raise ValueError(f"Wrong ON-Harmony participant count: {predictions_path}")
    if frame["scanner_code"].nunique() != EXPECTED_SCANNERS:
        raise ValueError(f"Wrong ON-Harmony scanner count: {predictions_path}")
    result = frame.copy()
    result["subject_id"] = result["subject"].astype(str)
    result["scanner"] = result["scanner_code"].astype(str)
    return result


def validate_internal_test_metrics(
    metrics_path: Path,
    predictions_path: Path,
    *,
    spec: ModelSpec,
    variant: str,
) -> dict:
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = {
        "arm": "frozen",
        "model": spec.expected_model,
        "variant": variant_name(variant),
        "n_predictions": EXPECTED_INTERNAL_TEST_ROWS,
        "n_subjects": EXPECTED_INTERNAL_TEST_ROWS,
        "bias_correction": "none",
        "internal_test_accessed": True,
        "on_harmony_accessed": False,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Invalid internal-test metrics {metrics_path}: {mismatches}")
    actual_hash = sha256_file(predictions_path)
    if payload.get("predictions_artifact_sha256") != actual_hash:
        raise ValueError(f"Internal-test prediction hash mismatch: {predictions_path}")
    return {
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": actual_hash,
        "training_lock_sha256": payload.get("training_lock_sha256"),
        "model_artifact_sha256": payload.get("model_artifact_sha256"),
        "reported_metrics": payload.get("metrics"),
    }


def load_internal_test_predictions(
    predictions_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(predictions_path, sep="\t")
    required = {
        "dataset",
        "subject",
        "split",
        "age_true",
        "age_pred",
        "error_years",
        "absolute_error_years",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{predictions_path} lacks columns: {missing}")
    if len(frame) != EXPECTED_INTERNAL_TEST_ROWS:
        raise ValueError(f"{predictions_path} has {len(frame)} rows")
    if frame[["dataset", "subject"]].duplicated().any():
        raise ValueError(f"Duplicate internal-test subject: {predictions_path}")
    if set(frame["split"].astype(str)) != {"internal_test"}:
        raise ValueError(f"Wrong split in {predictions_path}")
    numeric = frame[
        ["age_true", "age_pred", "error_years", "absolute_error_years"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite internal-test values: {predictions_path}")
    error = frame["age_pred"].to_numpy(float) - frame["age_true"].to_numpy(float)
    if not np.allclose(error, frame["error_years"].to_numpy(float), atol=1e-10):
        raise ValueError(f"Inconsistent errors in {predictions_path}")
    if not np.allclose(
        np.abs(error), frame["absolute_error_years"].to_numpy(float), atol=1e-10
    ):
        raise ValueError(f"Inconsistent absolute errors in {predictions_path}")
    return frame.sort_values(["dataset", "subject"], kind="stable").reset_index(
        drop=True
    )


def acquisition_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "subject",
        "session",
        "acquisition_id",
        "cohort",
        "scanner_code",
        "scanner_name",
        "vendor",
    ]
    return frame[columns].astype(str).sort_values(columns, kind="stable").reset_index(
        drop=True
    )


def internal_test_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["dataset", "subject", "age_true"]].sort_values(
        ["dataset", "subject"], kind="stable"
    ).reset_index(drop=True)


def participant_scanner_cells(frame: pd.DataFrame) -> pd.DataFrame:
    selected = select_primary_acquisition_per_cell(frame)
    cells = selected[["subject_id", "scanner", "cohort", "age_pred"]].copy()
    cells = cells.sort_values(["subject_id", "scanner"], kind="stable").reset_index(
        drop=True
    )
    if len(cells) != EXPECTED_SUBJECT_SCANNER_CELLS:
        raise ValueError(f"Expected 120 participant-scanner cells, got {len(cells)}")
    per_subject = cells.groupby("subject_id").size()
    if not (per_subject == EXPECTED_SCANNERS_PER_SUBJECT).all():
        raise ValueError("Each participant must contribute exactly six scanner cells")
    return cells


def pairwise_absolute_difference_by_subject(cells: pd.DataFrame) -> pd.Series:
    values: dict[str, float] = {}
    for subject, group in cells.groupby("subject_id", sort=True):
        predictions = group["age_pred"].to_numpy(dtype=np.float64)
        if len(predictions) < 2:
            raise ValueError(f"Participant {subject} has fewer than two scanners")
        differences = [
            abs(left - right) for left, right in combinations(predictions.tolist(), 2)
        ]
        values[str(subject)] = float(np.mean(differences))
    result = pd.Series(values, dtype=np.float64).sort_index()
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Non-finite pairwise scanner difference")
    return result


def aggregate_scalar_bootstrap(
    fold_vectors: list[np.ndarray],
    samples: np.ndarray,
    *,
    metric: str,
) -> dict:
    values = np.asarray(fold_vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{metric} fold values must form a matrix")
    if samples.ndim != 2 or samples.shape[1] != values.shape[1]:
        raise ValueError(f"{metric} bootstrap samples do not match observations")
    if not np.isfinite(values).all():
        raise ValueError(f"{metric} contains non-finite values")
    fold_estimates = values.mean(axis=1)
    bootstrap = values[:, samples].mean(axis=2).mean(axis=0)
    lower, upper = finite_percentile(bootstrap, (2.5, 97.5))
    result = {
        "estimate": float(fold_estimates.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_values": bootstrap.tolist(),
        "fold_estimates": fold_estimates.tolist(),
        "n_folds": int(values.shape[0]),
        "fold_aggregation": "arithmetic mean across folds within each shared participant bootstrap",
        "participant_weighting": "macro average; every participant has equal weight",
    }
    if values.shape[0] > 1:
        result["fold_standard_deviation"] = float(np.std(fold_estimates, ddof=1))
    return result


def reference_scanner_effects(
    effects: np.ndarray,
    scanner_order: tuple[str, ...],
    reference_scanner: str,
) -> np.ndarray:
    """Contrast each scanner with the reference, including within bootstrap draws."""
    values = np.asarray(effects, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != len(scanner_order):
        raise ValueError("Scanner effects do not match the scanner order")
    if not np.isfinite(values).all():
        raise ValueError("Scanner effects contain non-finite values")
    if reference_scanner not in scanner_order:
        raise ValueError(f"Unknown reference scanner: {reference_scanner}")
    index = scanner_order.index(reference_scanner)
    return values - values[..., index : index + 1]


def scanner_effects(
    cells: pd.DataFrame,
    scanner_order: tuple[str, ...],
    *,
    reference_scanner: str | None = None,
) -> np.ndarray:
    """Fit participant-adjusted shifts, optionally relative to a named scanner."""
    required = {"subject_id", "scanner", "age_pred"}
    if not required.issubset(cells.columns):
        raise ValueError(f"Scanner model lacks columns: {sorted(required - set(cells))}")
    scanners = tuple(scanner_order)
    observed = set(cells["scanner"].astype(str))
    if observed != set(scanners):
        raise ValueError(
            f"Scanner design mismatch: expected={sorted(scanners)}, observed={sorted(observed)}"
        )
    subjects = tuple(sorted(cells["subject_id"].astype(str).unique()))
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    scanner_index = {scanner: index for index, scanner in enumerate(scanners)}
    n_rows = len(cells)
    subject_design = np.zeros((n_rows, len(subjects)), dtype=np.float64)
    scanner_design = np.zeros((n_rows, len(scanners) - 1), dtype=np.float64)
    for row, record in enumerate(cells.itertuples(index=False)):
        subject_design[row, subject_index[str(record.subject_id)]] = 1.0
        index = scanner_index[str(record.scanner)]
        if index < len(scanners) - 1:
            scanner_design[row, index] = 1.0
        else:
            scanner_design[row, :] = -1.0
    design = np.column_stack([subject_design, scanner_design])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Participant/scanner design is not identifiable")
    coefficients = np.linalg.lstsq(
        design, cells["age_pred"].to_numpy(dtype=np.float64), rcond=None
    )[0]
    first = coefficients[len(subjects) :]
    effects = np.concatenate([first, [-float(first.sum())]])
    if not np.isclose(effects.sum(), 0.0, atol=1e-10):
        raise RuntimeError("Scanner effects do not satisfy the sum-to-zero constraint")
    if reference_scanner is not None:
        return reference_scanner_effects(effects, scanners, reference_scanner)
    return effects


def stratified_phase_bootstrap(
    participants: pd.DataFrame, n_bootstrap: int, seed: int
) -> list[list[str]]:
    required = {"participant_id", "phase"}
    if not required.issubset(participants.columns):
        raise ValueError("Participant table lacks participant_id/phase")
    grouped = {
        str(phase): sorted(group["participant_id"].astype(str).tolist())
        for phase, group in participants.groupby("phase", sort=True)
    }
    if set(grouped) != {"A", "B"} or any(len(values) != 10 for values in grouped.values()):
        raise ValueError("Expected ten ON-Harmony participants in each phase")
    generator = np.random.RandomState(seed)
    samples: list[list[str]] = []
    for _ in range(n_bootstrap):
        sample: list[str] = []
        for phase in ("A", "B"):
            values = grouped[phase]
            indices = generator.randint(0, len(values), size=len(values))
            sample.extend(values[index] for index in indices)
        samples.append(sample)
    return samples


def resample_cells(cells: pd.DataFrame, sample: list[str]) -> pd.DataFrame:
    pieces = []
    for copy_index, subject in enumerate(sample):
        piece = cells.loc[cells["subject_id"] == subject].copy()
        if piece.empty:
            raise ValueError(f"Bootstrap participant absent from cells: {subject}")
        piece["subject_id"] = f"boot_{copy_index:03d}"
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def compute_scanner_effect_bootstrap(
    fold_cells: list[pd.DataFrame],
    samples: list[list[str]],
    scanner_order: tuple[str, ...],
) -> dict:
    fold_estimates = np.asarray(
        [scanner_effects(cells, scanner_order, reference_scanner=SCANNER_REFERENCE)
         for cells in fold_cells],
        dtype=np.float64,
    )
    estimate = fold_estimates.mean(axis=0)
    bootstrap = np.empty((len(samples), len(scanner_order)), dtype=np.float64)
    for index, sample in enumerate(samples):
        effects = [
            scanner_effects(
                resample_cells(cells, sample), scanner_order,
                reference_scanner=SCANNER_REFERENCE,
            )
            for cells in fold_cells
        ]
        bootstrap[index] = np.mean(effects, axis=0)
    if not np.isfinite(bootstrap).all():
        raise ValueError("Non-finite scanner-effect bootstrap")
    lower = np.percentile(bootstrap, 2.5, axis=0)
    upper = np.percentile(bootstrap, 97.5, axis=0)
    records = []
    for scanner_index, scanner in enumerate(scanner_order):
        records.append(
            {
                "scanner": scanner,
                "effect_years": float(estimate[scanner_index]),
                "ci_lower": float(lower[scanner_index]),
                "ci_upper": float(upper[scanner_index]),
                "bootstrap_values": bootstrap[:, scanner_index].tolist(),
            }
        )
    return {
        "effects": records,
        "rms_effect_years": float(np.sqrt(np.mean(estimate**2))),
        "maximum_absolute_effect_years": float(np.max(np.abs(estimate))),
        "model": "age_prediction ~ participant_fixed_intercept + scanner_fixed_effect",
        "constraint": "reference scanner effect fixed to zero",
        "reference_scanner": SCANNER_REFERENCE,
        "reference_scanner_label": SCANNER_REFERENCE_LABEL,
        "repeat_handling": (
            "acquisition 001 retained in each participant-scanner cell"
        ),
        "fold_aggregation": "arithmetic mean of scanner effects across folds within each bootstrap",
        "bootstrap": "participant bootstrap stratified 10/10 by ON-Harmony phase",
        "confidence_intervals": "pointwise descriptive 95% percentile intervals; no multiplicity adjustment",
    }


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    labels = list(p_values)
    values = np.asarray([p_values[label] for label in labels], dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid p-values for BH correction")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return {label: float(result[index]) for index, label in enumerate(labels)}


def load_participants(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"participant_id", "age", "phase"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Participant table lacks: {sorted(required - set(frame))}")
    if len(frame) != EXPECTED_ONHARMONY_SUBJECTS or frame["participant_id"].duplicated().any():
        raise ValueError("Invalid ON-Harmony participant table")
    if not np.isfinite(frame["age"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite ON-Harmony ages")
    return frame


def build_analysis(args: argparse.Namespace) -> dict:
    brain_age_root = Path(args.brain_age_root).expanduser().resolve()
    anatcl_internal_root = Path(args.anatcl_internal_test_root).expanduser().resolve()
    anatcl_onharmony_root = Path(args.anatcl_onharmony_root).expanduser().resolve()
    figure2_path = Path(args.figure2_results).expanduser().resolve()
    participants_path = Path(args.participants).expanduser().resolve()
    cohort_manifest_path = Path(args.cohort_manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    figure2 = json.loads(figure2_path.read_text(encoding="utf-8"))
    figure2_code = figure2.get("code", {})
    figure2_script = Path(figure2_code.get("analysis_script", "")).expanduser()
    expected_figure2_script_hash = figure2_code.get("analysis_script_sha256")
    if not figure2_script.is_file():
        raise FileNotFoundError(
            f"Figure 2 analysis script recorded in the results is absent: "
            f"{figure2_script}"
        )
    if sha256_file(figure2_script) != expected_figure2_script_hash:
        raise ValueError("Figure 2 analysis script hash no longer matches its results")
    figure2_bootstrap = figure2.get("dbicc", {})
    n_bootstrap = int(figure2_bootstrap.get("bootstrap_n", 0))
    bootstrap_seed = int(figure2_bootstrap.get("bootstrap_seed", -1))
    if n_bootstrap < 1 or bootstrap_seed < 0:
        raise ValueError("Figure 2 does not contain a locked bootstrap contract")
    onharmony_samples = bootstrap_indices(
        EXPECTED_ONHARMONY_SUBJECTS, n_bootstrap, bootstrap_seed
    )
    samples_hash = hashlib.sha256(
        onharmony_samples.astype(np.int64).tobytes()
    ).hexdigest()
    if samples_hash != figure2_bootstrap.get("bootstrap_samples_sha256"):
        raise ValueError("Figure 2 participant bootstrap cannot be reproduced")

    participants = load_participants(participants_path)
    participant_subjects = set(participants["participant_id"].astype(str))
    internal_samples = bootstrap_indices(
        EXPECTED_INTERNAL_TEST_ROWS, n_bootstrap, args.internal_test_seed
    )
    scanner_samples = stratified_phase_bootstrap(
        participants, n_bootstrap, args.scanner_effect_seed
    )

    prepared: dict[str, dict] = {}
    reference_acquisitions = None
    reference_internal_test = None

    for spec in MODEL_SPECS:
        onharmony_frames = []
        internal_frames = []
        onharmony_provenance = []
        internal_provenance = []
        for variant in spec.variants:
            internal_dir, onharmony_dir = frozen_prediction_dirs(
                brain_age_root,
                spec=spec,
                variant=variant,
                anatcl_internal_root=anatcl_internal_root,
                anatcl_onharmony_root=anatcl_onharmony_root,
            )
            prediction_path = onharmony_dir / "predictions.tsv"
            audit_path = onharmony_dir / "prediction_audit.json"
            onharmony_provenance.append(
                validate_prediction_audit(
                    audit_path,
                    prediction_path,
                    spec=spec,
                    variant=variant,
                )
            )
            onharmony = load_onharmony_predictions(
                prediction_path, spec=spec, variant=variant
            )
            if set(onharmony["subject_id"]) != participant_subjects:
                raise ValueError(f"Participant table mismatch for {variant}")
            table = acquisition_table(onharmony)
            if reference_acquisitions is None:
                reference_acquisitions = table
            elif not table.equals(reference_acquisitions):
                raise ValueError(f"ON-Harmony acquisition table mismatch for {variant}")
            onharmony_frames.append(onharmony)

            internal_prediction_path = internal_dir / "predictions_internal_test.tsv"
            metrics_path = internal_dir / "metrics_internal_test.json"
            provenance_record = validate_internal_test_metrics(
                metrics_path,
                internal_prediction_path,
                spec=spec,
                variant=variant,
            )
            internal = load_internal_test_predictions(internal_prediction_path)
            reported_mae = float(provenance_record["reported_metrics"]["mae_years"])
            observed_mae = float(internal["absolute_error_years"].mean())
            if not np.isclose(reported_mae, observed_mae, atol=1e-10):
                raise ValueError(f"Reported/recomputed MAE mismatch for {variant}")
            internal_table = internal_test_table(internal)
            if reference_internal_test is None:
                reference_internal_test = internal_table
            elif not internal_table.equals(reference_internal_test):
                raise ValueError(f"Internal-test cohort mismatch for {variant}")
            internal_frames.append(internal)
            internal_provenance.append(provenance_record)

        prepared[spec.condition] = {
            "onharmony": onharmony_frames,
            "internal_test": internal_frames,
            "onharmony_provenance": onharmony_provenance,
            "internal_test_provenance": internal_provenance,
        }

    if reference_acquisitions is None or reference_internal_test is None:
        raise RuntimeError("No Figure 3 inputs were loaded")

    conditions = {}
    contrasts = {}
    scanner_rows = []
    for spec in MODEL_SPECS:
        record = prepared[spec.condition]
        fold_cells = [participant_scanner_cells(frame) for frame in record["onharmony"]]
        fold_dbicc = []
        fold_pairwise = []
        for frame, cells in zip(record["onharmony"], fold_cells):
            fold_dbicc.append(
                compute_dbicc(cells, ["age_pred"], onharmony_samples)
            )
            pairwise = pairwise_absolute_difference_by_subject(cells)
            expected_subjects = sorted(participant_subjects)
            if pairwise.index.tolist() != expected_subjects:
                raise ValueError(f"Pairwise participant order mismatch for {spec.label}")
            fold_pairwise.append(pairwise.to_numpy(dtype=np.float64))

        prediction_dbicc = aggregate_fold_results(fold_dbicc)
        pairwise_difference = aggregate_scalar_bootstrap(
            fold_pairwise,
            onharmony_samples,
            metric=f"{spec.label} pairwise scanner difference",
        )
        pairwise_difference.update(
            {
                "unit": "years",
                "definition": (
                    "participant-macro mean of all absolute differences between "
                    "the six participant-scanner cell predictions"
                ),
                "pairs_per_participant": 15,
                "equivalence": (
                    "identical for predicted age and BAG because chronological "
                    "age is constant within participant"
                ),
            }
        )

        internal_errors = [
            frame["absolute_error_years"].to_numpy(dtype=np.float64)
            for frame in record["internal_test"]
        ]
        internal_mae = aggregate_scalar_bootstrap(
            internal_errors,
            internal_samples,
            metric=f"{spec.label} internal-test MAE",
        )
        internal_mae.update(
            {
                "unit": "years",
                "split": "internal_test",
                "n_subjects": EXPECTED_INTERNAL_TEST_ROWS,
                "bias_correction": "none",
            }
        )

        scanner = compute_scanner_effect_bootstrap(
            fold_cells, scanner_samples, SCANNER_ORDER
        )
        for effect in scanner["effects"]:
            scanner_rows.append(
                {
                    "condition": spec.condition,
                    "label": spec.label,
                    "scanner": effect["scanner"],
                    "effect_years": effect["effect_years"],
                    "ci_lower": effect["ci_lower"],
                    "ci_upper": effect["ci_upper"],
                }
            )

        figure2_condition = figure2.get("conditions", {}).get(spec.figure2_condition)
        if not figure2_condition:
            raise ValueError(f"Figure 2 lacks {spec.figure2_condition}")
        embedding_dbicc = figure2_condition["dbicc"]
        change = paired_bootstrap_contrast(
            prediction_dbicc,
            embedding_dbicc,
            label=f"{spec.label}: predicted-age dbICC minus full-embedding dbICC",
        )
        change["multiplicity"] = "seven prespecified model-wise contrasts; BH-FDR reported"
        contrasts[spec.condition] = change

        conditions[spec.condition] = {
            "label": spec.label,
            "role": "primary_model",
            "variants": list(spec.variants),
            "internal_test_mae": internal_mae,
            "embedding_dbicc": embedding_dbicc,
            "prediction_dbicc": prediction_dbicc,
            "dbicc_change": change,
            "cross_scanner_absolute_difference": pairwise_difference,
            "scanner_effects": scanner,
            "inputs": {
                "onharmony": record["onharmony_provenance"],
                "internal_test": record["internal_test_provenance"],
            },
        }

    adjusted = benjamini_hochberg(
        {
            condition: float(contrast["p_value_two_sided"])
            for condition, contrast in contrasts.items()
        }
    )
    for condition, q_value in adjusted.items():
        contrasts[condition]["q_value_bh"] = q_value
        conditions[condition]["dbicc_change"]["q_value_bh"] = q_value

    payload = {
        "schema_version": 1,
        "analysis": "Figure 3: frozen-Ridge brain-age utility and cross-acquisition robustness",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohorts": {
            "internal_test": {
                "subjects": EXPECTED_INTERNAL_TEST_ROWS,
                "manifest": str(cohort_manifest_path.resolve()),
                "manifest_sha256": sha256_file(cohort_manifest_path),
                "bootstrap_n": n_bootstrap,
                "bootstrap_seed": args.internal_test_seed,
                "bootstrap_samples_sha256": hashlib.sha256(
                    internal_samples.astype(np.int64).tobytes()
                ).hexdigest(),
            },
            "onharmony": {
                "sessions": EXPECTED_ONHARMONY_ROWS,
                "primary_acquisitions": EXPECTED_SUBJECT_SCANNER_CELLS,
                "participants": EXPECTED_ONHARMONY_SUBJECTS,
                "participant_scanner_cells": EXPECTED_SUBJECT_SCANNER_CELLS,
                "scanners": EXPECTED_SCANNERS,
                "scanners_per_participant": EXPECTED_SCANNERS_PER_SUBJECT,
                "participants_file": str(participants_path.resolve()),
                "participants_sha256": sha256_file(participants_path),
                "acquisition_table": "exactly matched across all models and folds",
                "repeat_handling": (
                    "acquisition 001 retained in each participant-scanner cell"
                ),
                "dbicc_bootstrap_n": n_bootstrap,
                "dbicc_bootstrap_seed": bootstrap_seed,
                "dbicc_bootstrap_samples_sha256": samples_hash,
                "scanner_effect_bootstrap_seed": args.scanner_effect_seed,
                "scanner_effect_bootstrap": "stratified 10/10 by phase",
            },
        },
        "endpoints": {
            "accuracy": "uncorrected MAE on the locked 140-subject internal test",
            "relative_reliability": "cross-scanner dbICC of scalar predicted age",
            "absolute_instability": (
                "participant-macro mean absolute pairwise difference across six "
                "scanner-cell predictions"
            ),
            "scanner_bias": (
                "participant-adjusted scanner shifts relative to NOT1 (Achieva)"
            ),
            "bag_policy": (
                "BAG ICC is not used; within-participant absolute differences and "
                "participant-adjusted scanner effects are algebraically identical "
                "for predicted age and BAG"
            ),
        },
        "scanner_order": list(SCANNER_ORDER),
        "conditions": conditions,
        "dbicc_changes": contrasts,
        "code": {
            "analysis_script": str(SCRIPT_PATH),
            "analysis_script_sha256": sha256_file(SCRIPT_PATH),
            "figure2_analysis_script": figure2["code"]["analysis_script"],
            "figure2_analysis_script_sha256": figure2["code"][
                "analysis_script_sha256"
            ],
            "figure2_results": str(figure2_path),
            "figure2_results_sha256": sha256_file(figure2_path),
        },
    }
    result_path = output_root / "figure3_brain_age_results.json"
    atomic_json(result_path, payload)

    summary_rows = []
    for spec in MODEL_SPECS:
        record = conditions[spec.condition]
        summary_rows.append(
            {
                "condition": spec.condition,
                "label": spec.label,
                "internal_test_mae_years": record["internal_test_mae"]["estimate"],
                "internal_test_mae_ci_lower": record["internal_test_mae"]["ci_lower"],
                "internal_test_mae_ci_upper": record["internal_test_mae"]["ci_upper"],
                "embedding_dbicc": record["embedding_dbicc"]["estimate"],
                "prediction_dbicc": record["prediction_dbicc"]["estimate"],
                "prediction_dbicc_ci_lower": record["prediction_dbicc"]["ci_lower"],
                "prediction_dbicc_ci_upper": record["prediction_dbicc"]["ci_upper"],
                "dbicc_change": record["dbicc_change"]["estimate"],
                "dbicc_change_ci_lower": record["dbicc_change"]["ci_lower"],
                "dbicc_change_ci_upper": record["dbicc_change"]["ci_upper"],
                "dbicc_change_p": record["dbicc_change"]["p_value_two_sided"],
                "dbicc_change_q_bh": record["dbicc_change"]["q_value_bh"],
                "cross_scanner_absolute_difference_years": record[
                    "cross_scanner_absolute_difference"
                ]["estimate"],
                "cross_scanner_absolute_difference_ci_lower": record[
                    "cross_scanner_absolute_difference"
                ]["ci_lower"],
                "cross_scanner_absolute_difference_ci_upper": record[
                    "cross_scanner_absolute_difference"
                ]["ci_upper"],
                "scanner_effect_rms_years": record["scanner_effects"][
                    "rms_effect_years"
                ],
                "scanner_effect_maximum_absolute_years": record[
                    "scanner_effects"
                ]["maximum_absolute_effect_years"],
            }
        )
    atomic_csv(output_root / "figure3_brain_age_summary.csv", pd.DataFrame(summary_rows))
    atomic_csv(output_root / "figure3_scanner_effects.csv", pd.DataFrame(scanner_rows))
    return payload


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--brain-age-root", default=DEFAULT_BRAIN_AGE_ROOT)
    argument_parser.add_argument(
        "--anatcl-internal-test-root",
        default=DEFAULT_ANATCL_FROZEN_INTERNAL_ROOT,
        help="Authoritative BF16 v3 frozen internal-test results for AnatCL",
    )
    argument_parser.add_argument(
        "--anatcl-onharmony-root",
        default=DEFAULT_ANATCL_FROZEN_ONHARMONY_ROOT,
        help="Authoritative BF16 v3 frozen ON-Harmony results for AnatCL",
    )
    argument_parser.add_argument("--figure2-results", default=DEFAULT_FIGURE2_RESULTS)
    argument_parser.add_argument("--participants", default=DEFAULT_PARTICIPANTS)
    argument_parser.add_argument("--cohort-manifest", default=DEFAULT_COHORT_MANIFEST)
    argument_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    argument_parser.add_argument("--internal-test-seed", type=int, default=31415)
    argument_parser.add_argument("--scanner-effect-seed", type=int, default=2718)
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    result = build_analysis(args)
    print("Figure 3 analysis complete")
    for spec in MODEL_SPECS:
        record = result["conditions"][spec.condition]
        print(
            f"  {spec.label:22s} "
            f"MAE={record['internal_test_mae']['estimate']:.2f} y; "
            f"prediction dbICC={record['prediction_dbicc']['estimate']:.3f}; "
            f"cross-scanner |change|="
            f"{record['cross_scanner_absolute_difference']['estimate']:.2f} y"
        )


if __name__ == "__main__":
    main()
