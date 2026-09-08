#!/usr/bin/env python3
"""Build the auditable paired analysis underlying Figure 4.

Figure 4 asks whether controlled task adaptation improves held-out brain-age
accuracy without sacrificing cross-acquisition robustness. Frozen and adapted
predictions are compared within the same internal-test subjects and the same
ON-Harmony participants. Positive gains always indicate improvement:

* frozen MAE minus adapted MAE,
* adapted predicted-age dbICC minus frozen predicted-age dbICC, and
* frozen minus adapted expected absolute cross-scanner change.

Metrics are calculated separately for every retained training run and only then
averaged inside each shared participant bootstrap. Predictions are never
averaged into an ensemble. Six model families use the three retained seeds.
AnatCL's primary estimator uses folds 0-4 at seed 0; fold-0 seeds 0-2 remain a
separate initialization-sensitivity analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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

from Analyses.figure2_reliability import bootstrap_indices, compute_dbicc  # noqa: E402
from Analyses.figure3_brain_age import (  # noqa: E402
    EXPECTED_INTERNAL_TEST_ROWS,
    EXPECTED_ONHARMONY_ROWS,
    EXPECTED_ONHARMONY_SUBJECTS,
    EXPECTED_SCANNERS,
    SCANNER_ORDER,
    acquisition_table,
    benjamini_hochberg,
    internal_test_table,
    load_participants,
    load_internal_test_predictions,
    pairwise_absolute_difference_by_subject,
    participant_scanner_cells,
    sha256_file,
    stratified_phase_bootstrap,
)


DEFAULT_BRAIN_AGE_ROOT = PROJECT_ROOT / "Results" / "brain_age"
DEFAULT_FIGURE3_RESULTS = (
    PROJECT_ROOT / "Results" / "Figure3" / "figure3_brain_age_results.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Results" / "Figure4"
DEFAULT_ANATCL_FROZEN_INTERNAL_ROOT = (
    DEFAULT_BRAIN_AGE_ROOT / "internal_test" / "frozen_bfloat16_v3" / "AnatCL"
)
DEFAULT_ANATCL_FROZEN_ONHARMONY_ROOT = (
    DEFAULT_BRAIN_AGE_ROOT / "onharmony" / "frozen_bfloat16_v3" / "AnatCL"
)
DEFAULT_ADAPTED_AUDIT = (
    DEFAULT_BRAIN_AGE_ROOT
    / "internal_test"
    / "adapted"
    / "internal_test_metrics_all_runs.audit.json"
)


@dataclass(frozen=True)
class RunSpec:
    run: str
    seed: int
    adapted_variant: str
    frozen_variant: str
    fold: int | None = None
    primary: bool = True


@dataclass(frozen=True)
class ModelSpec:
    condition: str
    label: str
    model_dir: str
    expected_model: str
    runs: tuple[RunSpec, ...]
    primary_definition: str

    @property
    def primary_runs(self) -> tuple[RunSpec, ...]:
        return tuple(run for run in self.runs if run.primary)


def seeded_runs(lr: str, variant: str) -> tuple[RunSpec, ...]:
    return (
        RunSpec(f"selection_seed0_{lr}", 0, variant, variant),
        RunSpec(f"final_seed1_{lr}", 1, variant, variant),
        RunSpec(f"final_seed2_{lr}", 2, variant, variant),
    )


MODEL_SPECS = (
    ModelSpec(
        "yaware",
        "y-Aware",
        "yAware",
        "y-Aware",
        seeded_runs("high", "canonical"),
        "arithmetic mean of the selected-LR seeds 0, 1, and 2",
    ),
    ModelSpec(
        "anatcl",
        "AnatCL",
        "AnatCL",
        "AnatCL",
        (
            RunSpec("fold0_seed0_high", 0, "fold0", "fold0", fold=0),
            RunSpec("fold1_seed0_high", 0, "fold1", "fold1", fold=1),
            RunSpec("fold2_seed0_high", 0, "fold2", "fold2", fold=2),
            RunSpec("fold3_seed0_high", 0, "fold3", "fold3", fold=3),
            RunSpec("fold4_seed0_high", 0, "fold4", "fold4", fold=4),
            RunSpec(
                "fold0_seed1_high",
                1,
                "fold0",
                "fold0",
                fold=0,
                primary=False,
            ),
            RunSpec(
                "fold0_seed2_high",
                2,
                "fold0",
                "fold0",
                fold=0,
                primary=False,
            ),
        ),
        "arithmetic mean of matched folds 0-4 at seed 0",
    ),
    ModelSpec(
        "brainsegfounder",
        "BrainSegFounder",
        "BrainSegFounder",
        "BrainSegFounder",
        seeded_runs("high", "bottleneck_mean"),
        "arithmetic mean of the selected-LR seeds 0, 1, and 2",
    ),
    ModelSpec(
        "3dneurosimclr",
        "3D-Neuro-SimCLR",
        "3DNeuroSimCLR",
        "3D-Neuro-SimCLR",
        seeded_runs("intermediate", "canonical"),
        "arithmetic mean of the selected-LR seeds 0, 1, and 2",
    ),
    ModelSpec(
        "neurovfm",
        "NeuroVFM",
        "NeuroVFM",
        "NeuroVFM",
        seeded_runs("high", "canonical"),
        "arithmetic mean of the selected-LR seeds 0, 1, and 2",
    ),
    ModelSpec(
        "curia2",
        "Curia-2",
        "Curia2",
        "Curia-2",
        seeded_runs("high", "canonical"),
        "arithmetic mean of the selected-LR seeds 0, 1, and 2",
    ),
    ModelSpec(
        "brainiac",
        "BrainIAC",
        "BrainIAC",
        "BrainIAC",
        seeded_runs("high", "canonical"),
        "arithmetic mean of the selected-LR seeds 0, 1, and 2",
    ),
)


ENDPOINTS = (
    "accuracy_gain",
    "reliability_gain",
    "stability_gain",
)


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


def all_target_names() -> tuple[str, ...]:
    return tuple(
        f"{spec.model_dir}/{run.run}" for spec in MODEL_SPECS for run in spec.runs
    )


def empirical_two_sided_p(values: np.ndarray, null: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Empirical test requires a finite bootstrap vector")
    lower = (np.count_nonzero(values <= null) + 1) / (len(values) + 1)
    upper = (np.count_nonzero(values >= null) + 1) / (len(values) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def metric_record(estimate: float, bootstrap_values: np.ndarray, *, name: str) -> dict:
    values = np.asarray(bootstrap_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(estimate):
        raise ValueError(f"Invalid metric distribution for {name}")
    lower, upper = np.percentile(values, (2.5, 97.5))
    return {
        "estimate": float(estimate),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_values": values.tolist(),
        "bootstrap_n": int(len(values)),
    }


def scalar_metric(values: np.ndarray, samples: np.ndarray, *, name: str) -> dict:
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or not np.isfinite(observations).all():
        raise ValueError(f"{name} observations must be a finite vector")
    if samples.ndim != 2 or samples.shape[1] != len(observations):
        raise ValueError(f"{name} bootstrap does not match observations")
    bootstrap = observations[samples].mean(axis=1)
    return metric_record(float(observations.mean()), bootstrap, name=name)


def contrast_metric(
    frozen: dict,
    adapted: dict,
    *,
    adapted_minus_frozen: bool,
    name: str,
) -> dict:
    frozen_bootstrap = np.asarray(frozen["bootstrap_values"], dtype=np.float64)
    adapted_bootstrap = np.asarray(adapted["bootstrap_values"], dtype=np.float64)
    if frozen_bootstrap.shape != adapted_bootstrap.shape:
        raise ValueError(f"Paired bootstrap mismatch for {name}")
    if adapted_minus_frozen:
        estimate = float(adapted["estimate"] - frozen["estimate"])
        bootstrap = adapted_bootstrap - frozen_bootstrap
        formula = "adapted minus frozen"
    else:
        estimate = float(frozen["estimate"] - adapted["estimate"])
        bootstrap = frozen_bootstrap - adapted_bootstrap
        formula = "frozen minus adapted"
    result = metric_record(estimate, bootstrap, name=name)
    result.update(
        {
            "p_value_two_sided": empirical_two_sided_p(bootstrap),
            "formula": formula,
            "direction": "positive values indicate improvement after adaptation",
        }
    )
    return result


def aggregate_metric_records(records: list[dict], *, name: str) -> dict:
    if not records:
        raise ValueError(f"No records to aggregate for {name}")
    estimates = np.asarray([record["estimate"] for record in records], dtype=np.float64)
    bootstrap = np.asarray(
        [record["bootstrap_values"] for record in records], dtype=np.float64
    )
    if bootstrap.ndim != 2 or not np.isfinite(bootstrap).all():
        raise ValueError(f"Invalid run bootstrap matrix for {name}")
    result = metric_record(
        float(estimates.mean()), bootstrap.mean(axis=0), name=name
    )
    result.update(
        {
            "run_estimates": estimates.tolist(),
            "n_primary_runs": int(len(records)),
            "run_standard_deviation": (
                float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
            ),
            "run_range": float(np.ptp(estimates)),
            "aggregation": (
                "arithmetic mean of per-run metrics within every shared participant "
                "bootstrap; predictions are not ensembled"
            ),
        }
    )
    return result


def aggregate_contrasts(records: list[dict], *, name: str) -> dict:
    result = aggregate_metric_records(records, name=name)
    bootstrap = np.asarray(result["bootstrap_values"], dtype=np.float64)
    result.update(
        {
            "p_value_two_sided": empirical_two_sided_p(bootstrap),
            "formula": records[0]["formula"],
            "direction": "positive values indicate improvement after adaptation",
        }
    )
    return result


def scanner_effects_batched(
    cells: pd.DataFrame,
    predictions: np.ndarray,
    scanner_order: tuple[str, ...],
) -> np.ndarray:
    """Fit one fixed-effects design to multiple prediction vectors."""
    required = {"subject_id", "scanner"}
    if not required.issubset(cells.columns):
        raise ValueError(
            f"Scanner model lacks columns: {sorted(required - set(cells))}"
        )
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(cells):
        raise ValueError("Batched scanner predictions do not match the cell table")
    if not np.isfinite(values).all():
        raise ValueError("Batched scanner predictions contain non-finite values")

    scanners = tuple(scanner_order)
    observed = set(cells["scanner"].astype(str))
    if observed != set(scanners):
        raise ValueError(
            f"Scanner design mismatch: expected={sorted(scanners)}, "
            f"observed={sorted(observed)}"
        )
    subjects = tuple(sorted(cells["subject_id"].astype(str).unique()))
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    scanner_index = {scanner: index for index, scanner in enumerate(scanners)}
    subject_design = np.zeros((len(cells), len(subjects)), dtype=np.float64)
    scanner_design = np.zeros((len(cells), len(scanners) - 1), dtype=np.float64)
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
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    first = coefficients[len(subjects) :, :]
    effects = np.vstack([first, -first.sum(axis=0)]).T
    if not np.allclose(effects.sum(axis=1), 0.0, atol=1e-10):
        raise RuntimeError("Scanner effects do not satisfy the sum-to-zero constraint")
    return effects


def compute_scanner_effect_bootstraps_batched(
    cells_by_condition: dict[str, list[pd.DataFrame]],
    samples: list[list[str]],
    scanner_order: tuple[str, ...],
) -> dict[str, dict]:
    """Bootstrap all retained runs with one shared design per replicate."""
    if not cells_by_condition:
        raise ValueError("No adapted scanner cells were supplied")
    reference_cells = None
    reference_keys = None
    prediction_columns = []
    column_indices: dict[str, list[int]] = {}
    for condition, run_cells in cells_by_condition.items():
        if not run_cells:
            raise ValueError(f"No primary adapted runs for {condition}")
        column_indices[condition] = []
        for cells in run_cells:
            keys = cells[["subject_id", "scanner", "cohort"]].reset_index(drop=True)
            if reference_keys is None:
                reference_cells = cells.copy().reset_index(drop=True)
                reference_keys = keys
            elif not keys.equals(reference_keys):
                raise ValueError(
                    f"Adapted participant-scanner cells differ for {condition}"
                )
            column_indices[condition].append(len(prediction_columns))
            prediction_columns.append(cells["age_pred"].to_numpy(dtype=np.float64))
    if reference_cells is None:
        raise RuntimeError("No adapted participant-scanner cells were loaded")

    prediction_matrix = np.column_stack(prediction_columns)
    run_estimates = scanner_effects_batched(
        reference_cells, prediction_matrix, scanner_order
    )
    bootstrap_by_condition = {
        condition: np.empty((len(samples), len(scanner_order)), dtype=np.float64)
        for condition in cells_by_condition
    }
    subject_values = reference_cells["subject_id"].astype(str).to_numpy()
    indices_by_subject = {
        subject: np.flatnonzero(subject_values == subject)
        for subject in np.unique(subject_values)
    }
    for bootstrap_index, sample in enumerate(samples):
        indices = []
        copied_subjects = []
        for copy_index, subject in enumerate(sample):
            if subject not in indices_by_subject:
                raise ValueError(f"Bootstrap participant absent from cells: {subject}")
            subject_indices = indices_by_subject[subject]
            indices.extend(subject_indices.tolist())
            copied_subjects.extend(
                [f"boot_{copy_index:03d}"] * len(subject_indices)
            )
        sampled_cells = reference_cells.iloc[indices].copy().reset_index(drop=True)
        sampled_cells["subject_id"] = copied_subjects
        sampled_predictions = prediction_matrix[np.asarray(indices, dtype=np.int64)]
        sampled_run_effects = scanner_effects_batched(
            sampled_cells, sampled_predictions, scanner_order
        )
        for condition, columns in column_indices.items():
            bootstrap_by_condition[condition][bootstrap_index] = (
                sampled_run_effects[columns].mean(axis=0)
            )

    results = {}
    for condition, columns in column_indices.items():
        estimate = run_estimates[columns].mean(axis=0)
        bootstrap = bootstrap_by_condition[condition]
        if not np.isfinite(bootstrap).all():
            raise ValueError(f"Non-finite scanner-effect bootstrap for {condition}")
        lower = np.percentile(bootstrap, 2.5, axis=0)
        upper = np.percentile(bootstrap, 97.5, axis=0)
        effects = []
        for scanner_index, scanner in enumerate(scanner_order):
            effects.append(
                {
                    "scanner": scanner,
                    "effect_years": float(estimate[scanner_index]),
                    "ci_lower": float(lower[scanner_index]),
                    "ci_upper": float(upper[scanner_index]),
                    "bootstrap_values": bootstrap[:, scanner_index].tolist(),
                }
            )
        results[condition] = {
            "effects": effects,
            "rms_effect_years": float(np.sqrt(np.mean(estimate**2))),
            "maximum_absolute_effect_years": float(np.max(np.abs(estimate))),
            "model": (
                "age_prediction ~ participant_fixed_intercept + scanner_fixed_effect"
            ),
            "constraint": (
                "scanner effects sum to zero with equal weight across eight scanners"
            ),
            "repeat_handling": (
                "acquisition 001 retained in each participant-scanner cell"
            ),
            "bootstrap": "participant bootstrap stratified 10/10 by ON-Harmony phase",
            "confidence_intervals": (
                "pointwise descriptive 95% percentile intervals; "
                "no multiplicity adjustment"
            ),
            "batched_solver": (
                "one identical fixed-effects design solves all retained runs "
                "within each bootstrap replicate"
            ),
        }
    return results


def resolve_project_artifact(relative_path: str, *, label: str) -> Path:
    candidate = (PROJECT_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} escapes the project root: {relative_path}") from error
    return candidate


def validate_global_adapted_audit(path: Path) -> tuple[dict, dict[str, dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "adapted_internal_test_summary_audit_v1",
        "status": "VERIFIED_COMPLETE",
        "arm": "adapted",
        "n_targets": 25,
        "internal_test_subjects_per_target": EXPECTED_INTERNAL_TEST_ROWS,
    }
    mismatch = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Invalid adapted internal-test audit: {mismatch}")
    expected_targets = set(all_target_names())
    if set(payload.get("expected_targets", [])) != expected_targets:
        raise ValueError("Adapted audit target matrix differs from the 25 locked runs")

    summary = payload.get("summary_artifact", {})
    summary_path = resolve_project_artifact(summary.get("path", ""), label="summary")
    if not summary_path.is_file() or sha256_file(summary_path) != summary.get("sha256"):
        raise ValueError("Adapted summary artifact is absent or has drifted")
    summary_frame = pd.read_csv(summary_path, sep="\t")
    if len(summary_frame) != 25 or summary.get("rows") != 25:
        raise ValueError("Adapted summary does not contain exactly 25 runs")

    launcher = payload.get("launcher", {})
    launcher_path = resolve_project_artifact(launcher.get("path", ""), label="launcher")
    if not launcher_path.is_file() or sha256_file(launcher_path) != launcher.get("sha256"):
        raise ValueError("Adapted internal-test launcher has drifted")

    sources = payload.get("sources", [])
    if len(sources) != 25:
        raise ValueError("Adapted audit must contain exactly 25 source records")
    source_map = {}
    for source in sources:
        target = source.get("target")
        if target in source_map or target not in expected_targets:
            raise ValueError(f"Unknown or duplicate adapted source target: {target}")
        for field in ("metrics_artifact", "predictions_artifact"):
            artifact = resolve_project_artifact(source.get(field, ""), label=field)
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            if sha256_file(artifact) != source.get(f"{field}_sha256"):
                raise ValueError(f"Adapted source hash mismatch: {target}/{field}")
        source_map[target] = source
    return payload, source_map


def validate_adapted_internal_test(
    metrics_path: Path,
    predictions_path: Path,
    *,
    spec: ModelSpec,
    run: RunSpec,
    source: dict,
) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    expected = {
        "arm": "adapted",
        "model": spec.expected_model,
        "variant": run.adapted_variant,
        "internal_test_subjects": EXPECTED_INTERNAL_TEST_ROWS,
        "bias_correction": "none",
        "internal_test_accessed": True,
        "on_harmony_accessed": False,
    }
    mismatch = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    metric_expected = {
        "n": EXPECTED_INTERNAL_TEST_ROWS,
        "split": "internal_test",
        "bias_correction": "none",
    }
    mismatch.update(
        {
            f"metrics.{key}": {"expected": value, "observed": metrics.get(key)}
            for key, value in metric_expected.items()
            if metrics.get(key) != value
        }
    )
    if mismatch:
        raise ValueError(f"Invalid adapted internal-test metrics for {run.run}: {mismatch}")
    if sha256_file(metrics_path) != source.get("metrics_artifact_sha256"):
        raise ValueError(f"Global audit metrics hash mismatch for {run.run}")
    prediction_hash = sha256_file(predictions_path)
    if prediction_hash != source.get("predictions_artifact_sha256"):
        raise ValueError(f"Global audit prediction hash mismatch for {run.run}")
    if prediction_hash != payload.get("predictions_artifact_sha256"):
        raise ValueError(f"Per-run prediction hash mismatch for {run.run}")
    if payload.get("checkpoint_sha256") != source.get("checkpoint_sha256"):
        raise ValueError(f"Checkpoint hash mismatch for {run.run}")
    frame = load_internal_test_predictions(predictions_path)
    observed_mae = float(frame["absolute_error_years"].mean())
    if not np.isclose(observed_mae, float(metrics.get("mae_years")), atol=1e-10):
        raise ValueError(f"Reported/recomputed MAE mismatch for {run.run}")
    return frame, {
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": prediction_hash,
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "training_lock_sha256": payload.get("training_lock_sha256"),
    }


def validate_adapted_onharmony(
    audit_path: Path,
    predictions_path: Path,
    *,
    spec: ModelSpec,
    run: RunSpec,
    checkpoint_sha256: str,
) -> tuple[pd.DataFrame, dict]:
    if not audit_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(audit_path if not audit_path.is_file() else predictions_path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "status": "ON_HARMONY_PREDICTED",
        "arm": "adapted",
        "model": spec.expected_model,
        "variant": run.adapted_variant,
        "label_blind": True,
        "outcome_labels_loaded": False,
        "bias_correction": "none",
        "n_predictions": EXPECTED_ONHARMONY_ROWS,
        "n_subjects": EXPECTED_ONHARMONY_SUBJECTS,
        "training_seed": run.seed,
        "internal_test_accessed": False,
        "on_harmony_accessed": True,
        "adapted_checkpoint_sha256": checkpoint_sha256,
    }
    mismatch = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Invalid adapted ON-Harmony audit for {run.run}: {mismatch}")
    prediction_hash = sha256_file(predictions_path)
    if payload.get("predictions_artifact_sha256") != prediction_hash:
        raise ValueError(f"Adapted ON-Harmony prediction hash mismatch for {run.run}")
    frame = pd.read_csv(predictions_path, sep="\t")
    required = {
        "model",
        "variant",
        "arm",
        "training_seed",
        "checkpoint_sha256",
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
    if len(frame) != EXPECTED_ONHARMONY_ROWS or frame["acquisition_id"].duplicated().any():
        raise ValueError(f"Invalid adapted ON-Harmony row identity for {run.run}")
    expected_columns = {
        "model": spec.expected_model,
        "variant": run.adapted_variant,
        "arm": "adapted",
        "training_seed": run.seed,
        "checkpoint_sha256": checkpoint_sha256,
    }
    for column, expected_value in expected_columns.items():
        observed = set(frame[column].unique())
        if observed != {expected_value}:
            raise ValueError(f"Unexpected {column} for {run.run}: {observed}")
    if frame["subject"].nunique() != EXPECTED_ONHARMONY_SUBJECTS:
        raise ValueError(f"Wrong participant count for {run.run}")
    if frame["scanner_code"].nunique() != EXPECTED_SCANNERS:
        raise ValueError(f"Wrong scanner count for {run.run}")
    if not np.isfinite(frame["age_pred"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Non-finite adapted predictions for {run.run}")
    frame = frame.copy()
    frame["subject_id"] = frame["subject"].astype(str)
    frame["scanner"] = frame["scanner_code"].astype(str)
    return frame, {
        "audit": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": prediction_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "training_lock_sha256": payload.get("training_lock_sha256"),
    }


def frozen_result_dirs(
    brain_age_root: Path,
    *,
    spec: ModelSpec,
    variant: str,
    anatcl_internal_root: Path,
    anatcl_onharmony_root: Path,
) -> tuple[Path, Path]:
    """Resolve frozen baselines without changing any non-AnatCL model."""
    if spec.condition == "anatcl":
        return anatcl_internal_root / variant, anatcl_onharmony_root / variant
    return (
        brain_age_root / "internal_test" / "frozen" / spec.model_dir / variant,
        brain_age_root / "onharmony" / "frozen" / spec.model_dir / variant,
    )


def load_frozen_pair(
    brain_age_root: Path,
    *,
    spec: ModelSpec,
    variant: str,
    anatcl_internal_root: Path,
    anatcl_onharmony_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frozen_internal_dir, frozen_onharmony_dir = frozen_result_dirs(
        brain_age_root,
        spec=spec,
        variant=variant,
        anatcl_internal_root=anatcl_internal_root,
        anatcl_onharmony_root=anatcl_onharmony_root,
    )
    internal_predictions = frozen_internal_dir / "predictions_internal_test.tsv"
    internal_metrics = frozen_internal_dir / "metrics_internal_test.json"
    internal_payload = json.loads(internal_metrics.read_text(encoding="utf-8"))
    expected_internal = {
        "arm": "frozen",
        "model": spec.expected_model,
        "variant": variant,
        "n_predictions": EXPECTED_INTERNAL_TEST_ROWS,
        "n_subjects": EXPECTED_INTERNAL_TEST_ROWS,
        "bias_correction": "none",
        "internal_test_accessed": True,
        "on_harmony_accessed": False,
    }
    mismatch = {
        key: {"expected": value, "observed": internal_payload.get(key)}
        for key, value in expected_internal.items()
        if internal_payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Invalid frozen internal-test metrics for {variant}: {mismatch}")
    internal_hash = sha256_file(internal_predictions)
    if internal_payload.get("predictions_artifact_sha256") != internal_hash:
        raise ValueError(f"Frozen internal-test hash mismatch for {variant}")
    internal = load_internal_test_predictions(internal_predictions)

    onharmony_predictions = frozen_onharmony_dir / "predictions.tsv"
    onharmony_audit = frozen_onharmony_dir / "prediction_audit.json"
    audit_payload = json.loads(onharmony_audit.read_text(encoding="utf-8"))
    expected_audit = {
        "status": "ON_HARMONY_PREDICTED",
        "arm": "frozen",
        "model": spec.expected_model,
        "variant": variant,
        "label_blind": True,
        "outcome_labels_loaded": False,
        "bias_correction": "none",
        "n_predictions": EXPECTED_ONHARMONY_ROWS,
        "n_subjects": EXPECTED_ONHARMONY_SUBJECTS,
        "internal_test_accessed": False,
        "on_harmony_accessed": True,
    }
    mismatch = {
        key: {"expected": value, "observed": audit_payload.get(key)}
        for key, value in expected_audit.items()
        if audit_payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Invalid frozen ON-Harmony audit for {variant}: {mismatch}")
    onharmony_hash = sha256_file(onharmony_predictions)
    if audit_payload.get("predictions_artifact_sha256") != onharmony_hash:
        raise ValueError(f"Frozen ON-Harmony hash mismatch for {variant}")
    onharmony = pd.read_csv(onharmony_predictions, sep="\t")
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
    if not required.issubset(onharmony.columns):
        raise ValueError(f"Frozen ON-Harmony columns missing for {variant}")
    if len(onharmony) != EXPECTED_ONHARMONY_ROWS or onharmony["acquisition_id"].duplicated().any():
        raise ValueError(f"Invalid frozen ON-Harmony rows for {variant}")
    for column, expected_value in {
        "model": spec.expected_model,
        "variant": variant,
        "arm": "frozen",
    }.items():
        if set(onharmony[column].unique()) != {expected_value}:
            raise ValueError(f"Unexpected frozen {column} for {variant}")
    if not np.isfinite(onharmony["age_pred"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Non-finite frozen ON-Harmony predictions for {variant}")
    onharmony = onharmony.copy()
    onharmony["subject_id"] = onharmony["subject"].astype(str)
    onharmony["scanner"] = onharmony["scanner_code"].astype(str)
    return internal, onharmony, {
        "internal_metrics": str(internal_metrics.resolve()),
        "internal_metrics_sha256": sha256_file(internal_metrics),
        "internal_predictions": str(internal_predictions.resolve()),
        "internal_predictions_sha256": internal_hash,
        "onharmony_audit": str(onharmony_audit.resolve()),
        "onharmony_audit_sha256": sha256_file(onharmony_audit),
        "onharmony_predictions": str(onharmony_predictions.resolve()),
        "onharmony_predictions_sha256": onharmony_hash,
    }


def aligned_vector(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    keys: list[str],
    column: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    left = reference.sort_values(keys, kind="stable").reset_index(drop=True)
    right = candidate.sort_values(keys, kind="stable").reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError(f"Paired row identity mismatch for {label}")
    return (
        left[column].to_numpy(dtype=np.float64),
        right[column].to_numpy(dtype=np.float64),
    )


def dbicc_metric(frame: pd.DataFrame, samples: np.ndarray) -> dict:
    cells = participant_scanner_cells(frame)
    return compute_dbicc(
        cells,
        ["age_pred"],
        samples,
    )


def absolute_scanner_metric(frame: pd.DataFrame, samples: np.ndarray) -> tuple[dict, pd.Series]:
    cells = participant_scanner_cells(frame)
    values = pairwise_absolute_difference_by_subject(cells)
    return scalar_metric(values.to_numpy(dtype=np.float64), samples, name="scanner difference"), values


def validate_figure3_dependency(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    analysis_script = Path(payload.get("code", {}).get("analysis_script", ""))
    expected_hash = payload.get("code", {}).get("analysis_script_sha256")
    if not analysis_script.is_file() or sha256_file(analysis_script) != expected_hash:
        raise ValueError("Figure 3 analysis dependency is absent or has drifted")
    if set(payload.get("conditions", {})) != {spec.condition for spec in MODEL_SPECS}:
        raise ValueError("Figure 3 does not contain the same seven model families")
    return payload


def build_analysis(args: argparse.Namespace) -> dict:
    brain_age_root = Path(args.brain_age_root).expanduser().resolve()
    anatcl_internal_root = Path(args.anatcl_internal_test_root).expanduser().resolve()
    anatcl_onharmony_root = Path(args.anatcl_onharmony_root).expanduser().resolve()
    figure3_path = Path(args.figure3_results).expanduser().resolve()
    adapted_audit_path = Path(args.adapted_audit).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    figure3 = validate_figure3_dependency(figure3_path)
    global_audit, source_map = validate_global_adapted_audit(adapted_audit_path)

    internal_contract = figure3["cohorts"]["internal_test"]
    onharmony_contract = figure3["cohorts"]["onharmony"]
    n_bootstrap = int(internal_contract["bootstrap_n"])
    internal_samples = bootstrap_indices(
        EXPECTED_INTERNAL_TEST_ROWS,
        n_bootstrap,
        int(internal_contract["bootstrap_seed"]),
    )
    internal_hash = hashlib.sha256(
        internal_samples.astype(np.int64).tobytes()
    ).hexdigest()
    if internal_hash != internal_contract["bootstrap_samples_sha256"]:
        raise ValueError("Figure 3 internal-test bootstrap cannot be reproduced")
    if int(onharmony_contract["dbicc_bootstrap_n"]) != n_bootstrap:
        raise ValueError("Figure 3 bootstrap counts differ across cohorts")
    onharmony_samples = bootstrap_indices(
        EXPECTED_ONHARMONY_SUBJECTS,
        n_bootstrap,
        int(onharmony_contract["dbicc_bootstrap_seed"]),
    )
    onharmony_hash = hashlib.sha256(
        onharmony_samples.astype(np.int64).tobytes()
    ).hexdigest()
    if onharmony_hash != onharmony_contract["dbicc_bootstrap_samples_sha256"]:
        raise ValueError("Figure 3 ON-Harmony bootstrap cannot be reproduced")
    participants_path = Path(onharmony_contract["participants_file"]).resolve()
    if (
        not participants_path.is_file()
        or sha256_file(participants_path) != onharmony_contract["participants_sha256"]
    ):
        raise ValueError("Figure 3 participant table is absent or has drifted")
    participants = load_participants(participants_path)
    scanner_effect_seed = int(onharmony_contract["scanner_effect_bootstrap_seed"])
    scanner_samples = stratified_phase_bootstrap(
        participants,
        n_bootstrap,
        scanner_effect_seed,
    )

    frozen_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame, dict]] = {}
    conditions: dict[str, dict] = {}
    run_rows = []
    scanner_rows = []
    scanner_cells_by_condition: dict[str, list[pd.DataFrame]] = {}
    reference_internal = None
    reference_acquisitions = None

    for spec in MODEL_SPECS:
        run_results: dict[str, dict] = {}
        primary_records = {endpoint: [] for endpoint in ENDPOINTS}
        primary_frozen = {"mae": [], "dbicc": [], "scanner_difference": []}
        primary_adapted = {"mae": [], "dbicc": [], "scanner_difference": []}
        primary_adapted_cells: list[pd.DataFrame] = []

        for run in spec.runs:
            cache_key = (spec.condition, run.frozen_variant)
            if cache_key not in frozen_cache:
                frozen_cache[cache_key] = load_frozen_pair(
                    brain_age_root,
                    spec=spec,
                    variant=run.frozen_variant,
                    anatcl_internal_root=anatcl_internal_root,
                    anatcl_onharmony_root=anatcl_onharmony_root,
                )
            frozen_internal, frozen_onharmony, frozen_provenance = frozen_cache[cache_key]

            target = f"{spec.model_dir}/{run.run}"
            source = source_map[target]
            adapted_internal_dir = (
                brain_age_root / "internal_test" / "adapted" / spec.model_dir / run.run
            )
            adapted_internal, adapted_internal_provenance = validate_adapted_internal_test(
                adapted_internal_dir / "metrics_internal_test.json",
                adapted_internal_dir / "predictions_internal_test.tsv",
                spec=spec,
                run=run,
                source=source,
            )
            adapted_onharmony_dir = (
                brain_age_root / "onharmony" / "adapted" / spec.model_dir / run.run
            )
            adapted_onharmony, adapted_onharmony_provenance = validate_adapted_onharmony(
                adapted_onharmony_dir / "prediction_audit.json",
                adapted_onharmony_dir / "predictions.tsv",
                spec=spec,
                run=run,
                checkpoint_sha256=source["checkpoint_sha256"],
            )

            internal_identity = internal_test_table(adapted_internal)
            acquisitions = acquisition_table(adapted_onharmony)
            if reference_internal is None:
                reference_internal = internal_identity
            elif not internal_identity.equals(reference_internal):
                raise ValueError(f"Internal-test cohort mismatch for {target}")
            if reference_acquisitions is None:
                reference_acquisitions = acquisitions
            elif not acquisitions.equals(reference_acquisitions):
                raise ValueError(f"ON-Harmony acquisition mismatch for {target}")
            if not internal_test_table(frozen_internal).equals(reference_internal):
                raise ValueError(f"Frozen internal-test cohort mismatch for {target}")
            if not acquisition_table(frozen_onharmony).equals(reference_acquisitions):
                raise ValueError(f"Frozen ON-Harmony acquisition mismatch for {target}")

            frozen_errors, adapted_errors = aligned_vector(
                frozen_internal,
                adapted_internal,
                keys=["dataset", "subject"],
                column="absolute_error_years",
                label=f"{target} internal test",
            )
            frozen_mae = scalar_metric(
                frozen_errors, internal_samples, name=f"{target} frozen MAE"
            )
            adapted_mae = scalar_metric(
                adapted_errors, internal_samples, name=f"{target} adapted MAE"
            )
            accuracy_gain = contrast_metric(
                frozen_mae,
                adapted_mae,
                adapted_minus_frozen=False,
                name=f"{target} accuracy gain",
            )

            frozen_dbicc = dbicc_metric(frozen_onharmony, onharmony_samples)
            adapted_dbicc = dbicc_metric(adapted_onharmony, onharmony_samples)
            reliability_gain = contrast_metric(
                frozen_dbicc,
                adapted_dbicc,
                adapted_minus_frozen=True,
                name=f"{target} reliability gain",
            )

            frozen_difference, frozen_subject_difference = absolute_scanner_metric(
                frozen_onharmony, onharmony_samples
            )
            adapted_difference, adapted_subject_difference = absolute_scanner_metric(
                adapted_onharmony, onharmony_samples
            )
            if not frozen_subject_difference.index.equals(adapted_subject_difference.index):
                raise ValueError(f"ON-Harmony participant mismatch for {target}")
            stability_gain = contrast_metric(
                frozen_difference,
                adapted_difference,
                adapted_minus_frozen=False,
                name=f"{target} stability gain",
            )

            run_result = {
                "run": run.run,
                "seed": run.seed,
                "fold": run.fold,
                "variant": run.adapted_variant,
                "frozen_variant": run.frozen_variant,
                "primary": run.primary,
                "absolute": {
                    "frozen": {
                        "mae_years": frozen_mae["estimate"],
                        "prediction_dbicc": frozen_dbicc["estimate"],
                        "scanner_difference_years": frozen_difference["estimate"],
                    },
                    "adapted": {
                        "mae_years": adapted_mae["estimate"],
                        "prediction_dbicc": adapted_dbicc["estimate"],
                        "scanner_difference_years": adapted_difference["estimate"],
                    },
                },
                "gains": {
                    "accuracy_gain": accuracy_gain["estimate"],
                    "reliability_gain": reliability_gain["estimate"],
                    "stability_gain": stability_gain["estimate"],
                },
                "inputs": {
                    "frozen": frozen_provenance,
                    "adapted_internal_test": adapted_internal_provenance,
                    "adapted_onharmony": adapted_onharmony_provenance,
                },
            }
            run_results[run.run] = run_result
            run_rows.append(
                {
                    "condition": spec.condition,
                    "label": spec.label,
                    "run": run.run,
                    "seed": run.seed,
                    "fold": run.fold,
                    "primary": run.primary,
                    **run_result["absolute"]["frozen"],
                    **{
                        f"adapted_{key}": value
                        for key, value in run_result["absolute"]["adapted"].items()
                    },
                    **run_result["gains"],
                }
            )

            if run.primary:
                primary_adapted_cells.append(
                    participant_scanner_cells(adapted_onharmony)
                )
                for endpoint, value in (
                    ("accuracy_gain", accuracy_gain),
                    ("reliability_gain", reliability_gain),
                    ("stability_gain", stability_gain),
                ):
                    primary_records[endpoint].append(value)
                primary_frozen["mae"].append(frozen_mae)
                primary_frozen["dbicc"].append(frozen_dbicc)
                primary_frozen["scanner_difference"].append(frozen_difference)
                primary_adapted["mae"].append(adapted_mae)
                primary_adapted["dbicc"].append(adapted_dbicc)
                primary_adapted["scanner_difference"].append(adapted_difference)

        primary_gains = {
            endpoint: aggregate_contrasts(
                primary_records[endpoint], name=f"{spec.label} {endpoint}"
            )
            for endpoint in ENDPOINTS
        }
        absolute = {
            "frozen": {
                metric: aggregate_metric_records(
                    primary_frozen[metric], name=f"{spec.label} frozen {metric}"
                )
                for metric in primary_frozen
            },
            "adapted": {
                metric: aggregate_metric_records(
                    primary_adapted[metric], name=f"{spec.label} adapted {metric}"
                )
                for metric in primary_adapted
            },
        }
        scanner_cells_by_condition[spec.condition] = primary_adapted_cells

        reference = figure3["conditions"][spec.condition]
        expected_frozen = {
            "mae": float(reference["internal_test_mae"]["estimate"]),
            "dbicc": float(reference["prediction_dbicc"]["estimate"]),
            "scanner_difference": float(
                reference["cross_scanner_absolute_difference"]["estimate"]
            ),
        }
        for metric, expected_value in expected_frozen.items():
            observed = float(absolute["frozen"][metric]["estimate"])
            if not np.isclose(observed, expected_value, atol=1e-10):
                raise ValueError(
                    f"Figure 3 frozen {metric} mismatch for {spec.label}: "
                    f"{observed} != {expected_value}"
                )

        if spec.condition == "anatcl":
            sensitivity_names = (
                "fold0_seed0_high",
                "fold0_seed1_high",
                "fold0_seed2_high",
            )
            sensitivity = {
                "type": "initialization sensitivity at fixed fold 0",
                "runs": [run_results[name] for name in sensitivity_names],
                "not_pooled_with_primary_fold_sensitivity": True,
            }
        else:
            sensitivity = {
                "type": "initialization sensitivity across selected-LR seeds 0-2",
                "runs": [run_results[run.run] for run in spec.primary_runs],
                "same_runs_as_primary": True,
            }

        conditions[spec.condition] = {
            "label": spec.label,
            "primary_definition": spec.primary_definition,
            "primary_runs": [run.run for run in spec.primary_runs],
            "primary_gains": primary_gains,
            "absolute": absolute,
            "run_results": run_results,
            "training_sensitivity": sensitivity,
        }

    if reference_internal is None or reference_acquisitions is None:
        raise RuntimeError("No Figure 4 inputs were loaded")

    scanner_effects_by_condition = compute_scanner_effect_bootstraps_batched(
        scanner_cells_by_condition,
        scanner_samples,
        SCANNER_ORDER,
    )
    for spec in MODEL_SPECS:
        adapted_scanner_effects = scanner_effects_by_condition[spec.condition]
        adapted_scanner_effects["run_aggregation"] = (
            f"{spec.primary_definition}; effects are averaged inside each shared "
            "participant bootstrap"
        )
        conditions[spec.condition][
            "adapted_scanner_effects"
        ] = adapted_scanner_effects
        for effect in adapted_scanner_effects["effects"]:
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

    within_endpoint_p = {
        endpoint: {
            condition: float(record["primary_gains"][endpoint]["p_value_two_sided"])
            for condition, record in conditions.items()
        }
        for endpoint in ENDPOINTS
    }
    global_p = {
        f"{endpoint}:{condition}": value
        for endpoint, values in within_endpoint_p.items()
        for condition, value in values.items()
    }
    global_q = benjamini_hochberg(global_p)
    for endpoint, values in within_endpoint_p.items():
        endpoint_q = benjamini_hochberg(values)
        for condition, q_value in endpoint_q.items():
            gain = conditions[condition]["primary_gains"][endpoint]
            gain["q_value_bh_within_endpoint"] = q_value
            gain["q_value_bh_across_21_contrasts"] = global_q[
                f"{endpoint}:{condition}"
            ]

    payload = {
        "schema_version": 1,
        "analysis": "Figure 4: paired effect of controlled adaptation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Does controlled adaptation improve held-out brain-age accuracy "
            "without sacrificing cross-acquisition robustness?"
        ),
        "primary_estimand": (
            "mean single-run metric under the prespecified retained-run hierarchy; "
            "not an ensemble"
        ),
        "gain_definitions": {
            "accuracy_gain": "frozen MAE minus adapted MAE, in years",
            "reliability_gain": "adapted predicted-age dbICC minus frozen predicted-age dbICC",
            "stability_gain": (
                "frozen minus adapted participant-macro mean absolute pairwise "
                "scanner difference, in years"
            ),
            "orientation": "positive values always indicate improvement after adaptation",
        },
        "cohorts": {
            "internal_test": {
                "subjects": EXPECTED_INTERNAL_TEST_ROWS,
                "bootstrap_n": n_bootstrap,
                "bootstrap_seed": int(internal_contract["bootstrap_seed"]),
                "bootstrap_samples_sha256": internal_hash,
            },
            "onharmony": {
                "sessions": EXPECTED_ONHARMONY_ROWS,
                "participants": EXPECTED_ONHARMONY_SUBJECTS,
                "scanners": EXPECTED_SCANNERS,
                "bootstrap_n": n_bootstrap,
                "bootstrap_seed": int(onharmony_contract["dbicc_bootstrap_seed"]),
                "bootstrap_samples_sha256": onharmony_hash,
                "scanner_effect_bootstrap_seed": scanner_effect_seed,
                "scanner_effect_bootstrap": "stratified 10/10 by phase",
                "repeat_handling": (
                    "acquisition 001 retained in each participant-scanner cell"
                ),
            },
        },
        "training_hierarchy": {
            "non_anatcl": "selected-LR seeds 0, 1, and 2",
            "anatcl_primary": "matched folds 0-4 at seed 0",
            "anatcl_initialization_sensitivity": "fold 0 seeds 0, 1, and 2",
            "aggregation": (
                "per-run metrics are averaged inside each shared participant bootstrap; "
                "predictions are never averaged"
            ),
        },
        "multiplicity": {
            "primary_reporting": "95% paired-bootstrap confidence intervals",
            "p_values": "two-sided empirical paired-bootstrap tests against zero",
            "q_values": (
                "BH-FDR is reported both within each seven-model endpoint family "
                "and across all 21 model-endpoint contrasts"
            ),
        },
        "conditions": conditions,
        "inputs": {
            "figure3_results": str(figure3_path),
            "figure3_results_sha256": sha256_file(figure3_path),
            "adapted_internal_test_audit": str(adapted_audit_path),
            "adapted_internal_test_audit_sha256": sha256_file(adapted_audit_path),
            "adapted_internal_test_summary": global_audit["summary_artifact"],
            "participants": str(participants_path),
            "participants_sha256": sha256_file(participants_path),
        },
        "scanner_order": list(SCANNER_ORDER),
        "code": {
            "analysis_script": str(SCRIPT_PATH),
            "analysis_script_sha256": sha256_file(SCRIPT_PATH),
        },
    }
    result_path = output_root / "figure4_adaptation_results.json"
    atomic_json(result_path, payload)

    summary_rows = []
    for spec in MODEL_SPECS:
        record = conditions[spec.condition]
        row = {
            "condition": spec.condition,
            "label": spec.label,
            "primary_definition": spec.primary_definition,
            "n_primary_runs": len(spec.primary_runs),
        }
        for endpoint in ENDPOINTS:
            gain = record["primary_gains"][endpoint]
            row.update(
                {
                    endpoint: gain["estimate"],
                    f"{endpoint}_ci_lower": gain["ci_lower"],
                    f"{endpoint}_ci_upper": gain["ci_upper"],
                    f"{endpoint}_p": gain["p_value_two_sided"],
                    f"{endpoint}_q_within": gain["q_value_bh_within_endpoint"],
                    f"{endpoint}_q_global": gain[
                        "q_value_bh_across_21_contrasts"
                    ],
                }
            )
        summary_rows.append(row)
    atomic_csv(output_root / "figure4_adaptation_summary.csv", pd.DataFrame(summary_rows))
    atomic_csv(output_root / "figure4_run_metrics.csv", pd.DataFrame(run_rows))
    atomic_csv(
        output_root / "figure4_adapted_scanner_effects.csv",
        pd.DataFrame(scanner_rows),
    )
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
    argument_parser.add_argument("--figure3-results", default=DEFAULT_FIGURE3_RESULTS)
    argument_parser.add_argument("--adapted-audit", default=DEFAULT_ADAPTED_AUDIT)
    argument_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    result = build_analysis(args)
    print("Figure 4 analysis complete")
    for spec in MODEL_SPECS:
        gains = result["conditions"][spec.condition]["primary_gains"]
        print(
            f"  {spec.label:22s} "
            f"MAE gain={gains['accuracy_gain']['estimate']:+.2f} y; "
            f"dbICC gain={gains['reliability_gain']['estimate']:+.3f}; "
            f"stability gain={gains['stability_gain']['estimate']:+.2f} y"
        )


if __name__ == "__main__":
    main()
