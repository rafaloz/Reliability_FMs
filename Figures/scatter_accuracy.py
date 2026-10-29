"""Participant-bootstrap accuracy summaries shared by supplementary scatterplots."""

import hashlib
import json
from pathlib import Path

import numpy as np

from Analyses.figure2_reliability import bootstrap_indices


def accuracy_summary(frames, *, external=False):
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / 'Analyses').is_dir())
    contract_path = root / "Results/Figure3/figure3_brain_age_results.json"
    contract = json.loads(contract_path.read_text())["cohorts"]
    cohort = contract["onharmony" if external else "internal_test"]
    prefix = "dbicc_" if external else ""
    identity = ["subject_id", "scanner"] if external else ["dataset", "subject"]
    ordered = [frame.sort_values(identity).reset_index(drop=True) for frame in frames]
    reference = ordered[0][identity + ["age_true"]]
    if any(not frame[reference.columns].equals(reference) for frame in ordered):
        raise ValueError("Scatterplot runs have different participants, cells or ages")
    truth = ordered[0]["age_true"].to_numpy(dtype=float)
    predictions = np.stack([frame["age_pred"].to_numpy(dtype=float) for frame in ordered])
    if not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("Non-finite scatterplot inputs")
    if external:
        groups = ordered[0].groupby("subject_id", sort=True).indices
        if len(groups) != 20 or any(len(rows) != 6 for rows in groups.values()):
            raise ValueError("Expected 20 participants with six scanner cells each")
        rows = np.stack(list(groups.values()))
    else:
        if len(reference) != 140 or reference.duplicated(identity).any():
            raise ValueError("Expected 140 unique internal-test participants")
        rows = np.arange(len(truth))[:, None]
    samples = bootstrap_indices(
        len(rows), cohort[prefix + "bootstrap_n"], cohort[prefix + "bootstrap_seed"]
    )
    digest = hashlib.sha256(samples.tobytes()).hexdigest()
    if digest != cohort[prefix + "bootstrap_samples_sha256"]:
        raise ValueError("Scatterplot bootstrap differs from the main analysis")

    def metrics(indices):
        observed = truth[indices]
        error = predictions[:, indices] - observed
        denominator = np.square(observed - observed.mean()).sum()
        if denominator <= 0:
            raise ValueError("Undefined R-squared in participant bootstrap")
        return [np.abs(error).mean(), (1 - np.square(error).sum(axis=1) / denominator).mean()]

    estimate = metrics(np.arange(len(truth)))
    # Repeat whole participants and average run metrics, never run predictions.
    replicates = np.asarray([metrics(rows[sample].ravel()) for sample in samples])
    lower, upper = np.percentile(replicates, [2.5, 97.5], axis=0)
    return {
        "metrics": {
            name: {"estimate": float(estimate[i]), "ci_lower": float(lower[i]), "ci_upper": float(upper[i])}
            for i, name in enumerate(["mae_years", "r2"])
        },
        "bootstrap_n": len(samples),
        "bootstrap_samples_sha256": digest,
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aggregation": "mean of run metrics within each participant resample",
    }


def accuracy_text(summary):
    metrics = summary["metrics"]
    def line(label, key, units=""):
        value = metrics[key]
        return f"{label} {value['estimate']:.2f}{units}\n[{value['ci_lower']:.2f}, {value['ci_upper']:.2f}]"
    return line("MAE", "mae_years", " y") + "\n" + line("$R^2$", "r2")
