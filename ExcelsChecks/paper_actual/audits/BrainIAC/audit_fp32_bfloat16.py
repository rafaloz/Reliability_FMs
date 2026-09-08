#!/usr/bin/env python3
"""Paired FP32/BF16 audit for frozen BrainIAC ON-Harmony inference.

The same loaded checkpoint and the same 165 model-ready inputs are forwarded
once in FP32 and once under CUDA BF16 autocast. The script compares all four
BrainIAC readouts and applies the already locked first-patch Ridge without
fitting or selecting anything. Chronological age is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import Emb_extractors.extract_BrainIAC as extractor  # noqa: E402
from FineTuning.shared.cohort import sha256_file  # noqa: E402
from FineTuning.shared.onharmony import (  # noqa: E402
    acquisition_from_path,
    load_onharmony_manifest,
)
from FineTuning.shared.ridge import (  # noqa: E402
    MODEL_FILENAME,
    load_locked_ridge_artifact,
    predict_ridge,
)
from FineTuning.shared.runtime import (  # noqa: E402
    atomic_json,
    atomic_tsv,
    autocast_context,
    resolve_device,
    seed_torch,
)


EXPECTED_ACQUISITIONS = 165
EXPECTED_SUBJECTS = 20
N_BOOTSTRAP_DEFAULT = 2000
BOOTSTRAP_SEED = 20260830
LOCK_NAME = "LOCKED.json"
HISTORICAL_EXTRACTOR_SHA256 = (
    "f50c5a26571eb3c3a565c7cf558e69371c43252c400e620f060112606e135ca5"
)


def difference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Summarize paired vector or scalar differences."""

    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.ndim not in (1, 2):
        raise ValueError(f"Incompatible paired arrays: {left.shape}, {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Paired comparison contains non-finite values")
    delta = right - left
    absolute = np.abs(delta)
    result = {
        "signed_mean": float(delta.mean()),
        "absolute_mean": float(absolute.mean()),
        "absolute_median": float(np.median(absolute)),
        "absolute_maximum": float(absolute.max()),
    }
    if left.ndim == 1:
        result["pearson"] = float(np.corrcoef(left, right)[0, 1])
        return result
    numerator = np.linalg.norm(delta, axis=1)
    denominator = np.linalg.norm(left, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Reference embeddings contain a zero-norm row")
    left_norm = left / denominator[:, None]
    right_denominator = np.linalg.norm(right, axis=1)
    if np.any(right_denominator <= 0):
        raise ValueError("Candidate embeddings contain a zero-norm row")
    cosine = np.einsum("ij,ij->i", left_norm, right / right_denominator[:, None])
    result.update(
        cosine_mean=float(cosine.mean()),
        cosine_minimum=float(cosine.min()),
        relative_l2_mean=float((numerator / denominator).mean()),
        relative_l2_maximum=float((numerator / denominator).max()),
    )
    return result


def _squared_euclidean(values: np.ndarray) -> np.ndarray:
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


def _distance_blocks(
    subjects: np.ndarray, scanners: np.ndarray, values: np.ndarray
) -> DistanceBlocks:
    feature_columns = [f"feature_{index}" for index in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=feature_columns)
    frame.insert(0, "scanner", scanners.astype(str))
    frame.insert(0, "subject", subjects.astype(str))
    cells = (
        frame.groupby(["subject", "scanner"], as_index=False, sort=True)[
            feature_columns
        ]
        .mean()
        .sort_values(["subject", "scanner"], kind="stable")
        .reset_index(drop=True)
    )
    per_subject = cells.groupby("subject", sort=True).size()
    if len(per_subject) != EXPECTED_SUBJECTS or (per_subject < 2).any():
        raise ValueError("dbICC requires 20 participants with at least two scanner cells")

    labels = cells["subject"].astype(str).to_numpy()
    distances = _squared_euclidean(cells[feature_columns].to_numpy(dtype=np.float64))
    unique_subjects = tuple(sorted(np.unique(labels).tolist()))
    indices = [np.flatnonzero(labels == subject) for subject in unique_subjects]
    count = len(unique_subjects)
    within_sums = np.zeros(count, dtype=np.float64)
    within_counts = np.zeros(count, dtype=np.int64)
    between_sums = np.zeros((count, count), dtype=np.float64)
    between_counts = np.zeros((count, count), dtype=np.int64)
    for left, index in enumerate(indices):
        upper = np.triu_indices(len(index), k=1)
        within = distances[np.ix_(index, index)]
        within_sums[left] = within[upper].sum()
        within_counts[left] = len(upper[0])
        for right in range(left + 1, count):
            cross = distances[np.ix_(index, indices[right])]
            between_sums[left, right] = between_sums[right, left] = cross.sum()
            between_counts[left, right] = between_counts[right, left] = cross.size
    return DistanceBlocks(
        unique_subjects,
        within_sums,
        within_counts,
        between_sums,
        between_counts,
    )


def _dbicc_from_totals(
    within_sum: float,
    within_count: int,
    between_sum: float,
    between_count: int,
) -> float:
    if within_count <= 0 or between_count <= 0:
        raise ValueError("dbICC requires within- and between-participant pairs")
    between = between_sum / between_count
    if not np.isfinite(between) or between <= 0:
        raise ValueError("Between-participant mean squared distance is not positive")
    return float(1.0 - (within_sum / within_count) / between)


def _point_dbicc(blocks: DistanceBlocks) -> float:
    upper = np.triu_indices(len(blocks.subjects), k=1)
    return _dbicc_from_totals(
        float(blocks.within_sums.sum()),
        int(blocks.within_counts.sum()),
        float(blocks.between_sums[upper].sum()),
        int(blocks.between_counts[upper].sum()),
    )


def _bootstrap_dbicc(blocks: DistanceBlocks, samples: np.ndarray) -> np.ndarray:
    n_subjects = len(blocks.subjects)
    pair_left, pair_right = np.triu_indices(n_subjects, k=1)
    estimates = np.full(len(samples), np.nan, dtype=np.float64)
    for index, sampled in enumerate(samples):
        source_left = sampled[pair_left]
        source_right = sampled[pair_right]
        distinct = source_left != source_right
        estimates[index] = _dbicc_from_totals(
            float(blocks.within_sums[sampled].sum()),
            int(blocks.within_counts[sampled].sum()),
            float(blocks.between_sums[source_left[distinct], source_right[distinct]].sum()),
            int(blocks.between_counts[source_left[distinct], source_right[distinct]].sum()),
        )
    if not np.isfinite(estimates).all():
        raise RuntimeError("One or more dbICC bootstrap estimates failed")
    return estimates


def paired_dbicc(
    subjects: np.ndarray,
    scanners: np.ndarray,
    fp32: np.ndarray,
    bfloat16: np.ndarray,
    samples: np.ndarray,
) -> dict[str, Any]:
    """Calculate paired dbICC estimates and a participant-bootstrap delta CI."""

    left = np.asarray(fp32, dtype=np.float64)
    right = np.asarray(bfloat16, dtype=np.float64)
    if left.ndim == 1:
        left = left[:, None]
        right = right[:, None]
    left_blocks = _distance_blocks(subjects, scanners, left)
    right_blocks = _distance_blocks(subjects, scanners, right)
    if left_blocks.subjects != right_blocks.subjects:
        raise RuntimeError("FP32 and BF16 dbICC participant orders differ")
    left_bootstrap = _bootstrap_dbicc(left_blocks, samples)
    right_bootstrap = _bootstrap_dbicc(right_blocks, samples)
    delta = right_bootstrap - left_bootstrap
    lower, upper = np.percentile(delta, [2.5, 97.5])
    left_point = _point_dbicc(left_blocks)
    right_point = _point_dbicc(right_blocks)
    return {
        "fp32": left_point,
        "bfloat16": right_point,
        "delta": right_point - left_point,
        "delta_ci95": [float(lower), float(upper)],
        "n_bootstrap": int(len(samples)),
    }


def _pairwise_scanner_difference(
    subjects: np.ndarray, scanners: np.ndarray, predictions: np.ndarray
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "subject": subjects.astype(str),
            "scanner": scanners.astype(str),
            "prediction": np.asarray(predictions, dtype=np.float64),
        }
    )
    cells = (
        frame.groupby(["subject", "scanner"], as_index=False, sort=True)["prediction"]
        .mean()
        .sort_values(["subject", "scanner"], kind="stable")
    )
    values = {}
    for subject, group in cells.groupby("subject", sort=True):
        prediction = group["prediction"].to_numpy(dtype=np.float64)
        values[str(subject)] = float(
            np.mean([abs(left - right) for left, right in combinations(prediction, 2)])
        )
    result = pd.Series(values, dtype=np.float64).sort_index()
    if len(result) != EXPECTED_SUBJECTS or not np.isfinite(result).all():
        raise ValueError("Invalid participant-level cross-scanner differences")
    return result


def paired_cross_scanner_difference(
    subjects: np.ndarray,
    scanners: np.ndarray,
    fp32: np.ndarray,
    bfloat16: np.ndarray,
    samples: np.ndarray,
) -> dict[str, Any]:
    left = _pairwise_scanner_difference(subjects, scanners, fp32)
    right = _pairwise_scanner_difference(subjects, scanners, bfloat16)
    if not left.index.equals(right.index):
        raise RuntimeError("FP32 and BF16 participant orders differ")
    left_bootstrap = left.to_numpy()[samples].mean(axis=1)
    right_bootstrap = right.to_numpy()[samples].mean(axis=1)
    delta = right_bootstrap - left_bootstrap
    lower, upper = np.percentile(delta, [2.5, 97.5])
    return {
        "fp32_years": float(left.mean()),
        "bfloat16_years": float(right.mean()),
        "delta_years": float(right.mean() - left.mean()),
        "delta_ci95_years": [float(lower), float(upper)],
        "n_bootstrap": int(len(samples)),
    }


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration as error:
            raise ValueError(f"Empty historical embedding CSV: {path}") from error


def load_historical_embeddings(
    path: Path, manifest_path: Path, expected_ids: list[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load only paths and features from the historical FP32 source."""

    columns = [f"emb_{index}" for index in range(extractor.EXPECTED_EMBEDDING_DIM)]
    header = _csv_header(path)
    loaded = pd.read_csv(path, usecols=["file_path", *columns])
    loaded["acquisition_id"] = [
        acquisition_from_path(value)["acquisition_id"]
        for value in loaded["file_path"].astype(str)
    ]
    if loaded["acquisition_id"].duplicated().any():
        raise ValueError("Historical BrainIAC source contains duplicate acquisitions")
    if set(loaded["acquisition_id"]) != set(expected_ids):
        raise ValueError("Historical BrainIAC source differs from the canonical registry")
    values = loaded.set_index("acquisition_id").loc[expected_ids, columns].to_numpy(float)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != sha256_file(path):
        raise ValueError("Historical BrainIAC CSV differs from its manifest")
    if manifest.get("extractor_sha256") != HISTORICAL_EXTRACTOR_SHA256:
        raise ValueError("Unexpected historical BrainIAC extractor SHA")
    return values, {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "extractor_sha256": manifest["extractor_sha256"],
        "outcome_columns_present": sorted(set(header) & {"age", "sex"}),
        "outcome_columns_loaded": [],
    }


def load_ridge(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    lock_path = run_dir / LOCK_NAME
    model_path = run_dir / MODEL_FILENAME
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = {
        "status": "LOCKED_AFTER_TRAIN_VALIDATION",
        "arm": "frozen",
        "model": "BrainIAC",
        "variant": "canonical",
        "readout": extractor.READOUT,
        "embedding_dim": extractor.EXPECTED_EMBEDDING_DIM,
        "extraction_amp": "bfloat16",
    }
    mismatches = {
        key: {"expected": value, "observed": lock.get(key)}
        for key, value in expected.items()
        if lock.get(key) != value
    }
    if mismatches:
        raise ValueError(f"BrainIAC Ridge lock differs: {mismatches}")
    expected_model_sha = str(lock["model_artifact_sha256"])
    if sha256_file(model_path) != expected_model_sha:
        raise ValueError("BrainIAC Ridge artifact differs from LOCKED.json")
    artifact = load_locked_ridge_artifact(
        model_path,
        expected_sha256=expected_model_sha,
        expected_alpha=float(lock["selected_alpha"]),
        expected_dimension=extractor.EXPECTED_EMBEDDING_DIM,
    )
    return artifact, {
        "run_dir": str(run_dir),
        "lock": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "model": str(model_path),
        "model_sha256": expected_model_sha,
        "alpha": float(lock["selected_alpha"]),
    }


def _forward_readouts(model: Any, volume: Any, device: Any, amp: str) -> dict[str, np.ndarray]:
    with torch.no_grad():
        with autocast_context(torch, device, amp):
            output = model(volume)
            tokens = output[0] if isinstance(output, (tuple, list)) else output
            readouts = {
                name: extractor.apply_readout(tokens, name) for name in extractor.READOUTS
            }
    return {
        name: value[0].detach().float().cpu().numpy()
        for name, value in readouts.items()
    }


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Refusing to serialize object array in audit NPZ: {name}")
        normalized[name] = array
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **normalized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filelist", type=Path, default=extractor.DEFAULT_RAW_FILELIST)
    parser.add_argument("--metadata", type=Path, default=extractor.DEFAULT_METADATA)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "Models/BrainIAC/BrainIAC.ckpt",
    )
    parser.add_argument("--preproc-root", type=Path, default=extractor.DEFAULT_PREPROC_ROOT)
    parser.add_argument("--preproc-log-root", type=Path, default=extractor.DEFAULT_LOG_ROOT)
    parser.add_argument("--ridge-run", type=Path, required=True)
    parser.add_argument(
        "--historical-embeddings",
        type=Path,
        default=PROJECT_ROOT / "DATA/embeddings/embeddings_BrainIAC.csv",
    )
    parser.add_argument("--historical-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=extractor.DEFAULT_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.monotonic()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Audit output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")

    frame, onharmony_manifest = load_onharmony_manifest(
        args.filelist,
        args.metadata,
        expected_acquisitions=EXPECTED_ACQUISITIONS,
        expected_subjects=EXPECTED_SUBJECTS,
    )
    registry, input_contract = extractor.build_registry(
        onharmony_manifest,
        preproc_root=args.preproc_root.expanduser().resolve(),
        log_root=args.preproc_log_root.expanduser().resolve(),
    )
    device = resolve_device(torch, args.device)
    extractor.validate_runtime_contract(device, "bfloat16", args.seed)
    seed_torch(torch, args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = args.checkpoint.expanduser().resolve()
    adapter, frozen = extractor.build_model(checkpoint, registry, device)
    model = adapter._require_loaded()

    records = frame.to_dict(orient="records")
    paired = {
        precision: {
            readout: np.empty(
                (len(records), extractor.EXPECTED_EMBEDDING_DIM), dtype=np.float32
            )
            for readout in extractor.READOUTS
        }
        for precision in ("fp32", "bfloat16")
    }
    for index, record in enumerate(records):
        frozen_provenance = registry.freeze(record)
        volume = adapter.load_input(record).to(device, non_blocking=True)
        fp32 = _forward_readouts(model, volume, device, "off")
        bfloat16 = _forward_readouts(model, volume, device, "bfloat16")
        for name in extractor.READOUTS:
            paired["fp32"][name][index] = fp32[name]
            paired["bfloat16"][name][index] = bfloat16[name]
        if adapter.last_input_provenance != frozen_provenance:
            raise RuntimeError(f"Input provenance changed for {record['acquisition_id']}")
        if (index + 1) % 10 == 0 or index + 1 == len(records):
            print(f"  [{index + 1}/{len(records)}] {record['acquisition_id']}", flush=True)
    torch.cuda.synchronize(device)

    expected_ids = frame["acquisition_id"].astype(str).tolist()
    historical_path = args.historical_embeddings.expanduser().resolve()
    historical_manifest = (
        args.historical_manifest.expanduser().resolve()
        if args.historical_manifest is not None
        else historical_path.with_name(f"{historical_path.stem}_manifest.json")
    )
    historical, historical_provenance = load_historical_embeddings(
        historical_path, historical_manifest, expected_ids
    )
    ridge, ridge_provenance = load_ridge(args.ridge_run.expanduser().resolve())
    prediction_fp32 = predict_ridge(ridge, paired["fp32"]["first_patch"])
    prediction_bfloat16 = predict_ridge(ridge, paired["bfloat16"]["first_patch"])
    prediction_historical = predict_ridge(ridge, historical)

    subjects = frame["subject"].to_numpy(dtype=str)
    scanners = frame["scanner_code"].to_numpy(dtype=str)
    samples = np.random.RandomState(BOOTSTRAP_SEED).randint(
        0, EXPECTED_SUBJECTS, size=(args.n_bootstrap, EXPECTED_SUBJECTS)
    )
    readout_audits = {}
    for name in extractor.READOUTS:
        readout_audits[name] = {
            "difference": difference_metrics(
                paired["fp32"][name], paired["bfloat16"][name]
            ),
            "dbicc": paired_dbicc(
                subjects,
                scanners,
                paired["fp32"][name],
                paired["bfloat16"][name],
                samples,
            ),
        }

    scanner_changes = {}
    prediction_delta = prediction_bfloat16 - prediction_fp32
    for scanner in sorted(np.unique(scanners)):
        values = prediction_delta[scanners == scanner]
        scanner_changes[str(scanner)] = {
            "n": int(len(values)),
            "signed_mean_years": float(values.mean()),
            "absolute_mean_years": float(np.abs(values).mean()),
            "absolute_maximum_years": float(np.abs(values).max()),
        }

    paired_embeddings_path = output_dir / "paired_embeddings.npz"
    paired_predictions_path = output_dir / "paired_predictions.tsv"
    arrays: dict[str, Any] = {
        "acquisition_id": np.asarray(expected_ids),
        "subject": subjects,
        "scanner": scanners,
    }
    for precision in ("fp32", "bfloat16"):
        for name in extractor.READOUTS:
            arrays[f"{name}_{precision}"] = paired[precision][name]
    _atomic_npz(paired_embeddings_path, arrays)
    prediction_frame = frame[
        ["subject", "session", "acquisition_id", "cohort", "scanner_code"]
    ].copy()
    prediction_frame["age_pred_fp32"] = prediction_fp32
    prediction_frame["age_pred_bfloat16"] = prediction_bfloat16
    prediction_frame["bfloat16_minus_fp32_years"] = prediction_delta
    atomic_tsv(prediction_frame, paired_predictions_path)

    historical_embedding_difference = difference_metrics(
        historical, paired["fp32"]["first_patch"]
    )
    historical_prediction_difference = difference_metrics(
        prediction_historical, prediction_fp32
    )
    payload = {
        "audit_schema_version": "brainiac_onharmony_fp32_bfloat16_v1",
        "status": "REVIEW",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Quantify the isolated effect of replacing the historical FP32 frozen "
            "ON-Harmony forward with the BF16 contract used for Ridge training."
        ),
        "label_blind": True,
        "outcome_labels_loaded": False,
        "n_acquisitions": len(frame),
        "n_subjects": int(frame["subject"].nunique()),
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "seed": args.seed,
            "deterministic_algorithms": True,
            "input_dtype": "float32",
            "compared_forward_precisions": ["float32", "bfloat16"],
            "peak_gpu_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3),
        },
        "onharmony_manifest": onharmony_manifest,
        "input_contract": input_contract,
        "historical_source": historical_provenance,
        "exact_adapter": {
            "path": str(extractor.MODEL_IMPLEMENTATION_PATH),
            "sha256": sha256_file(extractor.MODEL_IMPLEMENTATION_PATH),
            "checkpoint_sha256": sha256_file(checkpoint),
            "load": adapter.load_description(),
            "frozen": frozen,
        },
        "ridge": ridge_provenance,
        "historical_fp32_reproduction": {
            "bit_exact_embeddings": bool(
                np.array_equal(historical, paired["fp32"]["first_patch"])
            ),
            "embeddings": historical_embedding_difference,
            "predictions": historical_prediction_difference,
        },
        "fp32_to_bfloat16": {
            "readouts": readout_audits,
            "predictions": difference_metrics(prediction_fp32, prediction_bfloat16),
            "prediction_dbicc": paired_dbicc(
                subjects, scanners, prediction_fp32, prediction_bfloat16, samples
            ),
            "cross_scanner_absolute_difference": paired_cross_scanner_difference(
                subjects, scanners, prediction_fp32, prediction_bfloat16, samples
            ),
            "prediction_change_by_scanner": scanner_changes,
        },
        "outputs": {
            "paired_embeddings": str(paired_embeddings_path),
            "paired_embeddings_sha256": sha256_file(paired_embeddings_path),
            "paired_predictions": str(paired_predictions_path),
            "paired_predictions_sha256": sha256_file(paired_predictions_path),
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "interpretation_required": True,
    }
    audit_path = output_dir / "precision_audit.json"
    atomic_json(payload, audit_path)
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    print(f"Audit: {audit_path}", flush=True)


if __name__ == "__main__":
    main()
