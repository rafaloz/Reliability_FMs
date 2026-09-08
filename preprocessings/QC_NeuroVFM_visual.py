#!/usr/bin/env python3
"""Build and finalize a label-blind visual QC for NeuroVFM cache inputs.

The report covers the definitive brain-age cohort and ON-Harmony. It verifies
each StudyPreprocessor cache, reconstructs the exact retained 4x16x16 patches,
and presents them beside the raw T1w. Automatic scores only prioritise human
review; they are not exclusion decisions or anatomical quality diagnoses.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.NeuroVFM.model import (  # noqa: E402
    CACHE_BATCH_REQUIRED_KEYS,
    CACHE_PREPROCESSOR_CODE_FILES,
    DEFAULT_PREPROCESSING_CACHE_ROOT,
    NEUROVFM_CODE_SHA256,
    NEUROVFM_GITHUB_COMMIT,
    NEUROVFM_MODEL_REPO,
    NEUROVFM_MODEL_REVISION,
    PREPROCESSING_CACHE_VERSION,
)
from FineTuning.NeuroVFM.predict_onharmony import (  # noqa: E402
    DEFAULT_CACHE_ROOT as DEFAULT_ONHARMONY_CACHE_ROOT,
)
from FineTuning.shared.cohort import DEFAULT_MANIFEST, sha256_file  # noqa: E402
from FineTuning.shared.onharmony import (  # noqa: E402
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    METADATA_COLUMNS,
    acquisition_from_path,
    canonical_sha256,
)


EXPECTED_FINE_TUNING = 1411
EXPECTED_ONHARMONY = 165
EXPECTED_ONHARMONY_SUBJECTS = 20
PATCH_SHAPE = (4, 16, 16)
TARGET_SPACING = (1.0, 1.0, 4.0)
MONTAGE_VERSION = "neurovfm_visual_qc_v2"
DECISIONS = {"PASS", "REVIEW", "FAIL", "UNREVIEWED"}
MODEL_CONFIG_SHA256 = "c7a4ef6aa982611f067b36aa3a9847fe4f20798dbd6642ff58d210100bbfaef9"
ROBUST_METRICS = (
    "n_tokens",
    "token_occupancy_fraction",
    "largest_component_fraction",
    "boundary_token_fraction",
    "center_offset_fraction",
    "span_fraction_d",
    "span_fraction_h",
    "span_fraction_w",
    "token_intensity_std",
    "token_dynamic_range",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    scope: str
    metric_group: str
    dataset: str
    subject: str
    session: str
    acquisition_id: str
    split: str
    raw_path: Path
    cache_path: Path
    sidecar_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, sep="\t", index=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_file(path: Path | str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise FileNotFoundError(candidate)
    return candidate


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semicolon_values(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(";") if item.strip()}


def load_fine_tuning_cases(
    manifest_path: Path, cache_root: Path, expected: int | None
) -> tuple[list[Case], dict[str, Any]]:
    """Load only identity/path columns; outcomes never enter memory."""

    manifest = require_file(manifest_path)
    columns = (
        "dataset",
        "subject",
        "raw_t1w_path",
        "split",
        "inclusion_status",
        "raw_runtime_conditions",
    )
    frame = pd.read_csv(
        manifest,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=list(columns),
    )
    frame = frame.loc[frame["inclusion_status"].eq("INCLUDED")].copy()
    if expected is not None and len(frame) != expected:
        raise ValueError(f"Expected {expected} fine-tuning cases, found {len(frame)}")
    if frame.duplicated(["dataset", "subject"]).any():
        raise ValueError("Fine-tuning manifest contains duplicate subjects")
    if (frame[list(columns)] == "").any().any():
        raise ValueError("Fine-tuning identity manifest contains empty values")
    if not frame["raw_runtime_conditions"].map(
        lambda value: "NeuroVFM" in semicolon_values(value)
    ).all():
        raise ValueError("Every fine-tuning case must permit NeuroVFM runtime input")

    root = cache_root.expanduser().resolve()
    cases: list[Case] = []
    for row in frame.to_dict(orient="records"):
        dataset = str(row["dataset"])
        subject = str(row["subject"])
        directory = root / dataset / subject
        cases.append(
            Case(
                case_id=f"finetuning::{dataset}::{subject}",
                scope="finetuning",
                metric_group=f"finetuning::{dataset}",
                dataset=dataset,
                subject=subject,
                session="",
                acquisition_id="",
                split=str(row["split"]),
                raw_path=Path(str(row["raw_t1w_path"])).expanduser().resolve(),
                cache_path=directory / "study_preprocessor.pt",
                sidecar_path=directory / "study_preprocessor.json",
            )
        )
    summary = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "n_cases": len(cases),
        "split_counts": {
            str(key): int(value) for key, value in frame["split"].value_counts().items()
        },
        "dataset_counts": {
            str(key): int(value) for key, value in frame["dataset"].value_counts().items()
        },
        "outcome_columns_loaded": [],
    }
    return cases, summary


def load_onharmony_cases(
    filelist_path: Path,
    metadata_path: Path,
    cache_root: Path,
    expected: int | None,
) -> tuple[list[Case], dict[str, Any]]:
    """Join raw paths to metadata using an explicit label-free column list."""

    filelist = require_file(filelist_path)
    metadata_file = require_file(metadata_path)
    paths = [
        Path(line.strip())
        for line in filelist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if expected is not None and len(paths) != expected:
        raise ValueError(f"Expected {expected} ON-Harmony cases, found {len(paths)}")
    if len(paths) != len({str(path) for path in paths}):
        raise ValueError("ON-Harmony raw filelist contains duplicate paths")

    metadata = pd.read_csv(
        metadata_file,
        dtype=str,
        keep_default_na=False,
        usecols=list(METADATA_COLUMNS),
    )
    if expected is not None and len(metadata) != expected:
        raise ValueError(f"Expected {expected} ON-Harmony metadata rows, found {len(metadata)}")
    if metadata["session"].duplicated().any() or (metadata == "").any().any():
        raise ValueError("ON-Harmony metadata has duplicate sessions or empty values")
    indexed = metadata.set_index("session", drop=False)
    root = cache_root.expanduser().resolve()
    cases: list[Case] = []
    public_records: list[dict[str, str]] = []
    for raw_path in paths:
        if not raw_path.is_absolute():
            raise ValueError(f"ON-Harmony raw path is not absolute: {raw_path}")
        identity = acquisition_from_path(raw_path)
        if raw_path.name != f"{identity['acquisition_id']}.nii.gz":
            raise ValueError(f"Unexpected ON-Harmony raw filename: {raw_path.name}")
        metadata_key = f"{identity['participant']}_{identity['scan_session']}"
        if metadata_key not in indexed.index:
            raise ValueError(f"No ON-Harmony metadata for {metadata_key}")
        row = indexed.loc[metadata_key]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"Ambiguous ON-Harmony metadata for {metadata_key}")
        subject = f"sub-{identity['participant']}"
        if row["subject"] != f"_{identity['participant']}":
            raise ValueError(f"Subject metadata differs for {metadata_key}")
        acquisition = str(identity["acquisition_id"])
        directory = root / acquisition
        cases.append(
            Case(
                case_id=f"onharmony::{acquisition}",
                scope="onharmony",
                metric_group=f"onharmony::{row['scanner_code']}",
                dataset="ON-Harmony",
                subject=subject,
                session=str(identity["scan_session"]),
                acquisition_id=acquisition,
                split="external",
                raw_path=raw_path.expanduser().resolve(),
                cache_path=directory / "study_preprocessor.pt",
                sidecar_path=directory / "study_preprocessor.json",
            )
        )
        public_records.append(
            {
                "subject": subject,
                "session": str(identity["scan_session"]),
                "acquisition_id": acquisition,
                "raw_t1w_path": str(raw_path),
                "scanner_code": str(row["scanner_code"]),
                "cohort": str(row["cohort"]),
            }
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("ON-Harmony acquisition identifiers are not unique")
    if (
        expected == EXPECTED_ONHARMONY
        and len({case.subject for case in cases}) != EXPECTED_ONHARMONY_SUBJECTS
    ):
        raise ValueError("Expected 20 ON-Harmony participants")
    summary = {
        "filelist": str(filelist),
        "filelist_sha256": sha256_file(filelist),
        "metadata": str(metadata_file),
        "metadata_sha256": sha256_file(metadata_file),
        "n_acquisitions": len(cases),
        "n_subjects": len({case.subject for case in cases}),
        "ordered_records_sha256": canonical_sha256(public_records),
        "outcome_columns_loaded": [],
    }
    return cases, summary


def validate_preprocessing_contract(contract: Mapping[str, Any]) -> str:
    observed = dict(contract)
    declared_hash = str(observed.pop("contract_sha256", ""))
    if canonical_json_hash(observed) != declared_hash:
        raise ValueError("NeuroVFM preprocessing contract hash is internally inconsistent")
    expected = {
        "cache_version": PREPROCESSING_CACHE_VERSION,
        "model_repository": NEUROVFM_MODEL_REPO,
        "model_revision": NEUROVFM_MODEL_REVISION,
        "config_sha256": MODEL_CONFIG_SHA256,
        "github_commit": NEUROVFM_GITHUB_COMMIT,
        "modality": "mri",
        "batch_keys": sorted(CACHE_BATCH_REQUIRED_KEYS),
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    expected_files = {
        relative: NEUROVFM_CODE_SHA256[relative]
        for relative in CACHE_PREPROCESSOR_CODE_FILES
    }
    if observed.get("preprocessor_files_sha256") != expected_files:
        mismatches["preprocessor_files_sha256"] = {
            "expected": expected_files,
            "observed": observed.get("preprocessor_files_sha256"),
        }
    expected_preprocessor = {
        "patch_size": list(PATCH_SHAPE),
        "target_spacing": list(TARGET_SPACING),
        "remove_background": True,
    }
    if observed.get("study_preprocessor") != expected_preprocessor:
        mismatches["study_preprocessor"] = {
            "expected": expected_preprocessor,
            "observed": observed.get("study_preprocessor"),
        }
    if mismatches:
        raise ValueError(f"NeuroVFM preprocessing contract mismatch: {mismatches}")
    return declared_hash


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
    }


def load_validated_cache(
    case: Case, *, verify_raw_sha256: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    raw_path = require_file(case.raw_path)
    cache_path = require_file(case.cache_path)
    sidecar_path = require_file(case.sidecar_path)
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_identity = {
        "status": "PASS",
        "cache_version": PREPROCESSING_CACHE_VERSION,
        "dataset": case.dataset,
        "subject": case.subject,
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected_identity.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"NeuroVFM cache sidecar identity mismatch: {mismatches}")
    contract_hash = validate_preprocessing_contract(
        metadata.get("preprocessing_contract", {})
    )

    source = metadata.get("raw_source", {})
    stat = raw_path.stat()
    current_source = {
        "path": str(raw_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    for key, value in current_source.items():
        if source.get(key) != value:
            raise ValueError(f"Raw T1w differs from cache metadata ({key}): {raw_path}")
    raw_sha256 = str(source.get("sha256", ""))
    if len(raw_sha256) != 64:
        raise ValueError(f"Cache metadata lacks the raw T1w SHA-256: {sidecar_path}")
    if verify_raw_sha256 and sha256_file(raw_path) != raw_sha256:
        raise ValueError(f"Raw T1w SHA-256 differs from cache metadata: {raw_path}")

    encoded = cache_path.read_bytes()
    cache_sha256 = hashlib.sha256(encoded).hexdigest()
    if cache_sha256 != metadata.get("artifact_sha256"):
        raise ValueError(f"NeuroVFM cache SHA-256 mismatch: {cache_path}")
    payload = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    payload_identity = {
        "cache_version": payload.get("cache_version"),
        "preprocessing_contract_sha256": payload.get(
            "preprocessing_contract_sha256"
        ),
        "dataset": payload.get("dataset"),
        "subject": payload.get("subject"),
    }
    expected_payload_identity = {
        "cache_version": PREPROCESSING_CACHE_VERSION,
        "preprocessing_contract_sha256": contract_hash,
        "dataset": case.dataset,
        "subject": case.subject,
    }
    if payload_identity != expected_payload_identity:
        raise ValueError(
            "NeuroVFM cache payload identity mismatch: "
            f"expected={expected_payload_identity}, observed={payload_identity}"
        )
    if payload.get("raw_source") != source:
        raise ValueError("NeuroVFM cache payload and sidecar disagree on raw source")
    batch = payload.get("batch")
    if not isinstance(batch, dict) or set(batch) != CACHE_BATCH_REQUIRED_KEYS:
        raise ValueError("NeuroVFM cache batch keys differ from the fixed contract")

    for key in ("img", "coords", "series_masks_indices", "series_cu_seqlens"):
        if not isinstance(batch[key], torch.Tensor) or batch[key].device.type != "cpu":
            raise TypeError(f"Cached NeuroVFM field {key!r} is not a CPU tensor")
    img = batch["img"]
    coords = batch["coords"]
    if img.ndim != 2 or int(img.shape[1]) != int(np.prod(PATCH_SHAPE)):
        raise ValueError(f"Unexpected NeuroVFM patch tensor shape: {tuple(img.shape)}")
    if coords.ndim != 2 or tuple(coords.shape) != (int(img.shape[0]), 3):
        raise ValueError(f"Unexpected NeuroVFM coordinate shape: {tuple(coords.shape)}")
    if coords.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"NeuroVFM coordinates are not integer: {coords.dtype}")
    if not torch.isfinite(img).all():
        raise ValueError("NeuroVFM cache contains non-finite patch values")
    if int(img.shape[0]) <= 0 or float(img.std().item()) <= 0.0:
        raise ValueError("NeuroVFM cache contains no variable foreground patches")
    epsilon = 2e-6
    if float(img.min().item()) < -epsilon or float(img.max().item()) > 1.0 + epsilon:
        raise ValueError("NeuroVFM MRI cache values lie outside the expected [0,1] range")
    if int(batch["series_masks_indices"].numel()) != 0:
        raise ValueError("NeuroVFM cache retained background mask indices")
    endpoints = batch["series_cu_seqlens"].reshape(-1)
    if endpoints.numel() != 2 or endpoints.tolist() != [0, int(img.shape[0])]:
        raise ValueError("NeuroVFM cumulative sequence lengths are inconsistent")
    if int(batch["series_max_len"]) != int(img.shape[0]):
        raise ValueError("NeuroVFM series_max_len differs from its token count")
    for key in ("mode", "path", "size"):
        if not isinstance(batch[key], (list, tuple)) or len(batch[key]) != 1:
            raise ValueError(f"NeuroVFM cache requires one {key!r} entry")
    if str(batch["mode"][0]).lower() != "mri":
        raise ValueError("NeuroVFM cache modality is not MRI")
    if Path(str(batch["path"][0])).expanduser().resolve() != raw_path:
        raise ValueError("NeuroVFM cached batch points to another raw T1w")

    shape = tuple(int(value) for value in batch["size"][0])
    if len(shape) != 3 or any(
        size <= 0 or size % patch != 0
        for size, patch in zip(shape, PATCH_SHAPE)
    ):
        raise ValueError(f"Invalid NeuroVFM preprocessed volume size: {shape}")
    grid_shape = tuple(size // patch for size, patch in zip(shape, PATCH_SHAPE))
    coords_np = coords.detach().cpu().numpy().astype(np.int64, copy=False)
    if (coords_np < 0).any() or any(
        (coords_np[:, axis] >= grid_shape[axis]).any() for axis in range(3)
    ):
        raise ValueError("NeuroVFM patch coordinate lies outside the preprocessed grid")
    if len(np.unique(coords_np, axis=0)) != len(coords_np):
        raise ValueError("NeuroVFM cache contains duplicate patch coordinates")

    recorded = metadata.get("batch", {})
    for key in ("img", "coords", "series_masks_indices", "series_cu_seqlens"):
        if recorded.get(key) != _tensor_summary(batch[key]):
            raise ValueError(f"NeuroVFM sidecar summary differs for batch.{key}")
    if int(recorded.get("n_tokens", -1)) != int(img.shape[0]):
        raise ValueError("NeuroVFM sidecar token count differs from its artifact")
    if recorded.get("size") != json.loads(json.dumps(batch["size"])):
        raise ValueError("NeuroVFM sidecar volume size differs from its artifact")

    return batch, {
        "cache_sha256": cache_sha256,
        "sidecar_sha256": sha256_file(sidecar_path),
        "raw_sha256": raw_sha256,
        "raw_sha256_verified": bool(verify_raw_sha256),
        "contract_sha256": contract_hash,
        "shape": shape,
        "grid_shape": grid_shape,
    }


def reconstruct_cache(
    batch: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert the upstream einops layout for retained foreground tokens."""

    tokens = batch["img"].detach().cpu().numpy().astype(np.float32, copy=False)
    coords = batch["coords"].detach().cpu().numpy().astype(np.int64, copy=False)
    shape = tuple(int(value) for value in batch["size"][0])
    grid_shape = tuple(size // patch for size, patch in zip(shape, PATCH_SHAPE))
    volume = np.zeros(shape, dtype=np.float32)
    support = np.zeros(shape, dtype=bool)
    occupancy = np.zeros(grid_shape, dtype=bool)
    slices_by_token: list[tuple[slice, slice, slice]] = []
    for flattened, coordinate in zip(tokens, coords):
        slices = tuple(
            slice(int(index) * width, (int(index) + 1) * width)
            for index, width in zip(coordinate, PATCH_SHAPE)
        )
        volume[slices] = flattened.reshape(PATCH_SHAPE)
        support[slices] = True
        occupancy[tuple(coordinate)] = True
        slices_by_token.append(slices)
    if int(occupancy.sum()) != len(tokens):
        raise RuntimeError("Reconstructed occupancy differs from NeuroVFM token count")
    reconstructed = np.stack(
        [volume[slices].reshape(-1) for slices in slices_by_token], axis=0
    )
    if not np.array_equal(reconstructed, tokens):
        raise RuntimeError("NeuroVFM patch reconstruction is not bit exact")
    return volume, support, occupancy


def cache_metrics(batch: Mapping[str, Any], occupancy: np.ndarray) -> dict[str, Any]:
    tokens = batch["img"].detach().cpu().numpy().astype(np.float64, copy=False)
    coords = batch["coords"].detach().cpu().numpy().astype(np.int64, copy=False)
    labels, components = ndimage.label(
        occupancy, structure=np.ones((3, 3, 3), dtype=int)
    )
    component_sizes = np.bincount(labels.ravel())[1:]
    largest = int(component_sizes.max()) if components else 0
    center = coords.mean(axis=0)
    target = (np.asarray(occupancy.shape, dtype=float) - 1.0) / 2.0
    normalized_offset = np.linalg.norm(
        (center - target) / np.asarray(occupancy.shape)
    )
    boundary = np.zeros_like(occupancy, dtype=bool)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    boundary[:, :, [0, -1]] = True
    spans = coords.max(axis=0) - coords.min(axis=0) + 1
    quantiles = np.percentile(tokens, [1.0, 25.0, 50.0, 75.0, 99.0])
    patch_stds = tokens.std(axis=1)
    n_tokens = int(len(tokens))
    return {
        "n_tokens": n_tokens,
        "grid_tokens": int(occupancy.size),
        "token_occupancy_fraction": float(n_tokens / occupancy.size),
        "coordinate_components": int(components),
        "largest_component_fraction": float(largest / n_tokens),
        "boundary_token_fraction": float((occupancy & boundary).sum() / n_tokens),
        "center_offset_fraction": float(normalized_offset),
        "span_fraction_d": float(spans[0] / occupancy.shape[0]),
        "span_fraction_h": float(spans[1] / occupancy.shape[1]),
        "span_fraction_w": float(spans[2] / occupancy.shape[2]),
        "token_intensity_mean": float(tokens.mean()),
        "token_intensity_std": float(tokens.std()),
        "token_intensity_p01": float(quantiles[0]),
        "token_intensity_p25": float(quantiles[1]),
        "token_intensity_median": float(quantiles[2]),
        "token_intensity_p75": float(quantiles[3]),
        "token_intensity_p99": float(quantiles[4]),
        "token_dynamic_range": float(quantiles[4] - quantiles[0]),
        "constant_patch_fraction": float((patch_stds <= 1e-8).mean()),
    }


def display_limits(
    volume: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float]:
    selected = np.isfinite(volume)
    if mask is not None:
        selected &= mask
    values = volume[selected]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(values.min()), float(values.max() + 1e-6)
    return float(low), float(high)


def expected_model_shape(
    raw_shape_xyz: tuple[int, int, int],
    raw_spacing_xyz: tuple[float, float, float],
    slice_axis_xyz: int,
) -> tuple[int, int, int]:
    """Reproduce NeuroVFM's resample/crop shape calculation."""

    target_spacing = np.ones(3, dtype=float)
    target_spacing[slice_axis_xyz] = 4.0
    resized_xyz = np.asarray(
        [
            int(size * (spacing / target))
            for size, spacing, target in zip(
                raw_shape_xyz, raw_spacing_xyz, target_spacing
            )
        ],
        dtype=int,
    )
    divisors = np.full(3, 16, dtype=int)
    divisors[slice_axis_xyz] = 4
    cropped_xyz = (resized_xyz // divisors) * divisors
    x, y, z = (int(value) for value in cropped_xyz)
    if slice_axis_xyz == 0:
        return x, y, z
    if slice_axis_xyz == 1:
        return y, z, x
    if slice_axis_xyz == 2:
        return z, y, x
    raise ValueError(f"Invalid NeuroVFM slice axis: {slice_axis_xyz}")


def infer_slice_axis_xyz(
    raw_shape_xyz: tuple[int, int, int],
    raw_spacing_xyz: tuple[float, float, float],
    model_shape_dhw: tuple[int, int, int],
) -> int:
    """Recover the acquisition axis used by the official preprocessor."""

    matches = [
        axis
        for axis in range(3)
        if expected_model_shape(
            raw_shape_xyz, raw_spacing_xyz, axis
        )
        == model_shape_dhw
    ]
    if len(matches) == 1:
        return matches[0]

    spacing = np.asarray(raw_spacing_xyz, dtype=float)
    shape = np.asarray(raw_shape_xyz, dtype=int)
    if len(set(spacing.tolist())) == 1:
        preferred = 2
    else:
        values, counts = np.unique(shape, return_counts=True)
        unique_sizes = values[counts == 1]
        preferred = (
            int(np.argmax(spacing))
            if len(unique_sizes) == 0
            else int(np.flatnonzero(shape == unique_sizes[0])[0])
        )
    if not matches or preferred in matches:
        return preferred
    raise ValueError(
        "NeuroVFM model shape is compatible with multiple acquisition axes "
        f"but not the official preference: matches={matches}, preferred={preferred}"
    )


def model_dhw_to_ras(
    array: np.ndarray, slice_axis_xyz: int
) -> np.ndarray:
    """Undo NeuroVFM's D/H/W transpose and convert RPI indices to RAS."""

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D NeuroVFM array, got {array.shape}")
    if slice_axis_xyz == 0:
        rpi_zyx = np.transpose(array, (2, 1, 0))
    elif slice_axis_xyz == 1:
        rpi_zyx = np.transpose(array, (1, 0, 2))
    elif slice_axis_xyz == 2:
        rpi_zyx = array
    else:
        raise ValueError(f"Invalid NeuroVFM slice axis: {slice_axis_xyz}")
    rpi_xyz = np.transpose(rpi_zyx, (2, 1, 0))
    return np.flip(rpi_xyz, axis=(1, 2))


def ras_sections(
    volume: np.ndarray,
    center: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Return sagittal/coronal/axial sections and post-rotation spacing."""

    sections = [
        np.take(volume, center[axis], axis=axis) for axis in range(3)
    ]
    x, y, z = spacing_xyz
    return sections, [(z, y), (z, x), (y, x)]


def scale_slice(values: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip(
        (values - low) / max(high - low, 1e-8), 0.0, 1.0
    )
    return np.rint(normalized * 255.0).astype(np.uint8)


def render_panel(
    section: np.ndarray,
    *,
    low: float,
    high: float,
    support: np.ndarray | None,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    resample: Image.Resampling = Image.Resampling.LANCZOS,
    size: int = 238,
) -> Image.Image:
    section = np.rot90(section)
    gray = scale_slice(section, low, high)
    row_spacing, column_spacing = pixel_spacing
    physical_height = gray.shape[0] * float(row_spacing)
    physical_width = gray.shape[1] * float(column_spacing)
    scale = min(size / physical_width, size / physical_height)
    width = max(1, int(round(physical_width * scale)))
    height = max(1, int(round(physical_height * scale)))
    gray_image = Image.fromarray(gray, mode="L").resize(
        (width, height), resample=resample
    )
    rgb = np.repeat(np.asarray(gray_image)[..., None], 3, axis=2)
    if support is not None:
        support = np.rot90(support.astype(bool))
        support_image = Image.fromarray(
            support.astype(np.uint8) * 255, mode="L"
        ).resize((width, height), resample=Image.Resampling.NEAREST)
        resized_support = np.asarray(support_image) > 0
        contour = resized_support ^ ndimage.binary_erosion(resized_support)
        rgb[contour] = np.asarray([231, 91, 42], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(
        image, ((size - image.width) // 2, (size - image.height) // 2)
    )
    return canvas


def center_indices(mask: np.ndarray) -> tuple[int, int, int]:
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return tuple(int(size // 2) for size in mask.shape)
    center = np.rint(coordinates.mean(axis=0)).astype(int)
    return tuple(
        int(np.clip(value, 0, mask.shape[axis] - 1))
        for axis, value in enumerate(center)
    )


def patch_mean_grid(
    volume: np.ndarray, occupancy: np.ndarray
) -> np.ndarray:
    reshaped = volume.reshape(
        occupancy.shape[0],
        PATCH_SHAPE[0],
        occupancy.shape[1],
        PATCH_SHAPE[1],
        occupancy.shape[2],
        PATCH_SHAPE[2],
    )
    means = reshaped.mean(axis=(1, 3, 5))
    return np.where(occupancy, means, 0.0).astype(np.float32)


def render_montage(
    case: Case,
    raw: np.ndarray,
    volume: np.ndarray,
    support: np.ndarray,
    occupancy: np.ndarray,
    raw_spacing_xyz: tuple[float, float, float],
    slice_axis_xyz: int,
    destination: Path,
) -> None:
    model_ras = model_dhw_to_ras(volume, slice_axis_xyz)
    support_ras = model_dhw_to_ras(support, slice_axis_xyz)
    occupancy_ras = model_dhw_to_ras(occupancy, slice_axis_xyz)
    raw_mask = np.isfinite(raw) & (np.abs(raw) > 1e-8)
    raw_center = center_indices(raw_mask)
    model_center = center_indices(support_ras)
    raw_limits = display_limits(raw, raw_mask)
    model_limits = display_limits(model_ras, support_ras)
    token_means = model_dhw_to_ras(
        patch_mean_grid(volume, occupancy), slice_axis_xyz
    )
    raw_sections, raw_panel_spacing = ras_sections(
        raw, raw_center, raw_spacing_xyz
    )
    model_spacing_xyz = [1.0, 1.0, 1.0]
    model_spacing_xyz[slice_axis_xyz] = 4.0
    model_sections, model_panel_spacing = ras_sections(
        model_ras,
        model_center,
        tuple(model_spacing_xyz),
    )
    support_sections, _ = ras_sections(
        support_ras,
        model_center,
        tuple(model_spacing_xyz),
    )
    token_sections = [
        token_means.max(axis=axis) for axis in range(3)
    ]
    token_support = [
        occupancy_ras.max(axis=axis) for axis in range(3)
    ]
    rows: list[
        tuple[
            str,
            list[np.ndarray],
            list[np.ndarray | None],
            tuple[float, float],
            list[tuple[float, float]],
            Image.Resampling,
        ]
    ] = [
        (
            "1. Raw T1w in closest RAS orientation",
            raw_sections,
            [None, None, None],
            raw_limits,
            raw_panel_spacing,
            Image.Resampling.LANCZOS,
        ),
        (
            "2. Exact model input restored to RAS | orange = retained support",
            model_sections,
            support_sections,
            model_limits,
            model_panel_spacing,
            Image.Resampling.LANCZOS,
        ),
        (
            "3. Maximum projections of retained-token mean intensity",
            token_sections,
            token_support,
            display_limits(token_means, occupancy_ras),
            [(1.0, 1.0)] * 3,
            Image.Resampling.NEAREST,
        ),
    ]
    axis_labels = ("sagittal (S up)", "coronal (S up)", "axial (A up)")
    panel_size = 238
    canvas = Image.new("RGB", (panel_size * 3, 3 * 274 + 38), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (10, 10),
        f"{case.case_id} | NeuroVFM cache QC",
        fill=(20, 28, 36),
        font=font,
    )
    top = 38
    for row_index, (
        label,
        sections,
        masks,
        limits,
        panel_spacings,
        resample,
    ) in enumerate(rows):
        row_top = top + row_index * 274
        draw.text((8, row_top), label, fill=(20, 28, 36), font=font)
        for column, (section, mask, panel_spacing) in enumerate(
            zip(sections, masks, panel_spacings)
        ):
            panel = render_panel(
                section,
                low=limits[0],
                high=limits[1],
                support=mask,
                pixel_spacing=panel_spacing,
                resample=resample,
            )
            x = column * panel_size
            canvas.paste(panel, (x, row_top + 20))
            draw.text(
                (x + 8, row_top + 258),
                axis_labels[column],
                fill=(20, 28, 36),
                font=font,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        canvas.save(handle, format="JPEG", quality=90, optimize=True)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def safe_image_name(case: Case) -> str:
    stem = case.acquisition_id or f"{case.dataset}_{case.subject}"
    normalized = "".join(
        character if character.isalnum() else "_" for character in stem
    )
    digest = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:12]
    return f"{normalized}_{digest}.jpg"


def inspect_case(
    case: Case,
    output_dir: Path,
    *,
    verify_raw_sha256: bool,
) -> dict[str, Any]:
    batch, provenance = load_validated_cache(
        case, verify_raw_sha256=verify_raw_sha256
    )
    volume, support, occupancy = reconstruct_cache(batch)
    raw_image = nib.as_closest_canonical(
        nib.load(str(require_file(case.raw_path)))
    )
    if raw_image.ndim != 3:
        raise ValueError(
            f"Raw T1w is not 3D for {case.case_id}: {raw_image.shape}"
        )
    raw = raw_image.get_fdata(dtype=np.float32)
    if not np.isfinite(raw).all() or float(raw.std()) <= 0.0:
        raise ValueError(
            f"Raw T1w is non-finite or constant for {case.case_id}"
        )
    image_name = safe_image_name(case)
    image_path = output_dir / "images" / image_name
    raw_spacing_xyz = tuple(
        float(value) for value in raw_image.header.get_zooms()[:3]
    )
    slice_axis_xyz = infer_slice_axis_xyz(
        tuple(int(value) for value in raw.shape),
        raw_spacing_xyz,
        tuple(int(value) for value in volume.shape),
    )
    render_montage(
        case,
        raw,
        volume,
        support,
        occupancy,
        raw_spacing_xyz,
        slice_axis_xyz,
        image_path,
    )
    values = asdict(case)
    return {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        },
        **cache_metrics(batch, occupancy),
        "preprocessed_shape": "x".join(
            str(value) for value in provenance["shape"]
        ),
        "patch_grid_shape": "x".join(
            str(value) for value in provenance["grid_shape"]
        ),
        "slice_axis_xyz": int(slice_axis_xyz),
        "display_orientation": "RAS",
        "cache_sha256": provenance["cache_sha256"],
        "sidecar_sha256": provenance["sidecar_sha256"],
        "raw_sha256": provenance["raw_sha256"],
        "raw_sha256_verified": provenance["raw_sha256_verified"],
        "preprocessing_contract_sha256": provenance["contract_sha256"],
        "image_relpath": f"images/{image_name}",
        "image_sha256": sha256_file(image_path),
        "montage_version": MONTAGE_VERSION,
        "technical_status": "PASS",
        "technical_error": "",
    }


def robust_absolute_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    deviation = (numeric - median).abs()
    mad = deviation.median()
    if not np.isfinite(mad) or mad <= 0.0:
        return pd.Series(
            np.zeros(len(values), dtype=float), index=values.index
        )
    return 0.6744897501960817 * deviation / mad


def add_automatic_assessment(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for metric in ROBUST_METRICS:
        result[f"z_{metric}"] = 0.0
    for _, indices in result.groupby("metric_group", sort=False).groups.items():
        for metric in ROBUST_METRICS:
            result.loc[indices, f"z_{metric}"] = robust_absolute_z(
                result.loc[indices, metric]
            )
    z_columns = [f"z_{metric}" for metric in ROBUST_METRICS]
    result["automatic_outlier_count"] = (
        result[z_columns] >= 3.5
    ).sum(axis=1)
    result["automatic_max_robust_z"] = result[z_columns].max(axis=1)
    severity = np.maximum(
        result["automatic_max_robust_z"].to_numpy(float) - 2.0, 0.0
    )
    penalty = (
        1.5 * severity
        + 0.5 * result["automatic_outlier_count"].to_numpy(float)
    )
    result["automatic_typicality_score"] = np.round(
        np.clip(10.0 - penalty, 0.0, 10.0), 1
    )
    priority = np.full(len(result), "ROUTINE", dtype=object)
    review = result["automatic_max_robust_z"].to_numpy(float) >= 3.5
    high = (
        result["automatic_max_robust_z"].to_numpy(float) >= 5.0
    ) | (result["automatic_outlier_count"].to_numpy(int) >= 2)
    priority[review] = "REVIEW"
    priority[high] = "HIGH"
    result["automatic_priority"] = priority
    result["decision"] = "UNREVIEWED"
    result["reviewer_notes"] = ""
    order = pd.Categorical(
        result["automatic_priority"],
        ["HIGH", "REVIEW", "ROUTINE"],
        ordered=True,
    )
    return (
        result.assign(_priority=order)
        .sort_values(
            [
                "_priority",
                "scope",
                "dataset",
                "subject",
                "session",
                "case_id",
            ],
            kind="stable",
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


def report_html(frame: pd.DataFrame, created: str) -> str:
    columns = [
        "case_id",
        "scope",
        "metric_group",
        "dataset",
        "subject",
        "session",
        "acquisition_id",
        "split",
        "image_relpath",
        "automatic_priority",
        "automatic_typicality_score",
        "automatic_max_robust_z",
        "automatic_outlier_count",
        "n_tokens",
        "patch_grid_shape",
        "token_occupancy_fraction",
        "largest_component_fraction",
        "boundary_token_fraction",
        "center_offset_fraction",
        "decision",
        "reviewer_notes",
    ]
    records = frame[columns].to_dict(orient="records")
    payload = json.dumps(
        records, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    report_id = hashlib.sha256(
        (
            MONTAGE_VERSION
            + "|"
            + "|".join(frame["case_id"].astype(str).tolist())
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroVFM visual QC</title><style>
:root{{--bg:#f2f4f5;--paper:#fff;--ink:#182126;--muted:#5e6b71;--line:#cbd3d7;--teal:#087f78;--orange:#b86419;--red:#b63232}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 Arial,sans-serif}}
header{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px}}
.bar{{display:flex;gap:7px;align-items:center;flex-wrap:wrap}}select,input,button{{height:34px;border:1px solid #aeb9be;background:#fff;padding:0 9px;font:inherit}}
button{{cursor:pointer}}button.primary{{background:var(--teal);border-color:var(--teal);color:#fff}}.note{{margin:9px 0 0;color:var(--muted)}}
main{{padding:14px;max-width:1180px;margin:auto}}.status{{font-weight:700;margin:0 0 10px}}.card{{background:var(--paper);border:1px solid var(--line);display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:12px;padding:12px}}
.card img{{display:block;width:100%;height:auto;background:#111}}h1{{font-size:18px;margin:0 10px 0 0}}h2{{font-size:17px;margin:0 0 8px;overflow-wrap:anywhere}}
.meta{{color:var(--muted);margin-bottom:10px;overflow-wrap:anywhere}}.metrics{{font-family:monospace;font-size:12px;white-space:pre-wrap}}
.decision{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:12px 0}}.decision button.selected{{color:#fff}}.pass.selected{{background:var(--teal)}}.review.selected{{background:var(--orange)}}.fail.selected{{background:var(--red)}}
textarea{{width:100%;min-height:84px;padding:7px;border:1px solid #aeb9be}}.error{{padding:28px;color:var(--red)}}@media(max-width:850px){{.card{{grid-template-columns:1fr}}}}
</style></head><body><header><div class="bar"><h1>NeuroVFM visual QC</h1>
<select id="scope"><option value="ALL">All cohorts</option><option value="finetuning">Fine-tuning</option><option value="onharmony">ON-Harmony</option></select>
<select id="priority"><option value="ALL">All priorities</option><option>HIGH</option><option>REVIEW</option><option>ROUTINE</option></select>
<select id="decision"><option value="ALL">All decisions</option><option>UNREVIEWED</option><option>PASS</option><option>REVIEW</option><option>FAIL</option></select>
<input id="search" placeholder="subject, dataset, scanner"><button id="previous" title="Previous case (arrow keys)">Previous</button><button id="next" title="Next case (arrow keys)">Next</button><button id="export" class="primary">Export TSV</button></div>
<p class="note"><b>Human review is authoritative.</b> Row 1 is the raw T1w in closest RAS orientation. Row 2 reconstructs exactly the retained 4x16x16 patches consumed by NeuroVFM, restores them from model D/H/W to RAS and preserves physical aspect ratio. Row 3 enlarges maximum projections of the retained-token mean intensities. Orange marks retained support. The 0-10 score measures within-group typicality only; it is not a quality score and never excludes a scan. No age, sex, diagnosis, prediction, BAG or error was loaded. Created {html.escape(created)}.</p></header>
<main><p class="status" id="status"></p><div id="root"></div></main>
<script>
const rows={payload};
const storageKey='neurovfm-qc-{report_id}';
let saved={{}},storageAvailable=true,visible=[],active=0;
try{{saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}');}}catch(error){{storageAvailable=false;saved={{}};}}
for(const row of rows){{if(saved[row.case_id]){{row.decision=saved[row.case_id].decision||'UNREVIEWED';row.reviewer_notes=saved[row.case_id].reviewer_notes||'';}}}}
function esc(v){{return String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function persist(row){{saved[row.case_id]={{decision:row.decision,reviewer_notes:row.reviewer_notes}};if(storageAvailable){{try{{localStorage.setItem(storageKey,JSON.stringify(saved));}}catch(error){{storageAvailable=false;}}}}}}
function filters(){{const scope=document.getElementById('scope').value,priority=document.getElementById('priority').value,decision=document.getElementById('decision').value,q=document.getElementById('search').value.toLowerCase();return rows.filter(r=>(scope==='ALL'||r.scope===scope)&&(priority==='ALL'||r.automatic_priority===priority)&&(decision==='ALL'||r.decision===decision)&&JSON.stringify(r).toLowerCase().includes(q));}}
function applyFilter(preferredId=null,preferredIndex=null){{visible=filters();const found=preferredId?visible.findIndex(row=>row.case_id===preferredId):-1;if(found>=0)active=found;else if(preferredIndex!==null)active=Math.min(preferredIndex,Math.max(0,visible.length-1));else active=Math.min(active,Math.max(0,visible.length-1));render();}}
function render(){{const root=document.getElementById('root'),reviewed=rows.filter(r=>r.decision!=='UNREVIEWED').length;document.getElementById('status').textContent=`${{visible.length}} visible | ${{reviewed}}/${{rows.length}} reviewed${{visible.length?` | case ${{active+1}}/${{visible.length}}`:''}}`;if(!visible.length){{root.innerHTML='<p>No cases match the filters.</p>';return;}}const row=visible[active];root.innerHTML=`<section class="card"><div><img src="${{esc(row.image_relpath)}}" alt="QC montage for ${{esc(row.case_id)}}"><div class="error" hidden>Image missing. Keep index.html together with its images directory.</div></div><div><h2>${{esc(row.subject)}} ${{esc(row.session)}}</h2><div class="meta">${{esc(row.scope)}} | ${{esc(row.dataset)}} | ${{esc(row.split)}}<br>${{esc(row.metric_group)}}<br><b>Priority: ${{esc(row.automatic_priority)}}</b> | typicality ${{Number(row.automatic_typicality_score).toFixed(1)}}/10</div><div class="metrics">tokens: ${{row.n_tokens}}\ngrid: ${{esc(row.patch_grid_shape)}}\noccupancy: ${{(100*Number(row.token_occupancy_fraction)).toFixed(1)}}%\nlargest component: ${{(100*Number(row.largest_component_fraction)).toFixed(1)}}%\nboundary tokens: ${{(100*Number(row.boundary_token_fraction)).toFixed(1)}}%\ncenter offset: ${{Number(row.center_offset_fraction).toFixed(3)}}\nmax robust z: ${{Number(row.automatic_max_robust_z).toFixed(2)}}\noutlier metrics: ${{row.automatic_outlier_count}}</div><div class="decision"><button class="pass ${{row.decision==='PASS'?'selected':''}}" title="Pass (P or 1)">PASS</button><button class="review ${{row.decision==='REVIEW'?'selected':''}}" title="Review (R or 2)">REVIEW</button><button class="fail ${{row.decision==='FAIL'?'selected':''}}" title="Fail (F or 3)">FAIL</button></div><textarea placeholder="Optional notes">${{esc(row.reviewer_notes)}}</textarea></div></section>`;const img=root.querySelector('img');img.onerror=()=>{{img.hidden=true;root.querySelector('.error').hidden=false;}};const buttons=root.querySelectorAll('.decision button');buttons[0].onclick=()=>decide('PASS');buttons[1].onclick=()=>decide('REVIEW');buttons[2].onclick=()=>decide('FAIL');root.querySelector('textarea').onchange=e=>{{row.reviewer_notes=e.target.value;persist(row);}};}}
function move(delta){{if(!visible.length)return;active=(active+delta+visible.length)%visible.length;render();window.scrollTo({{top:0,behavior:'instant'}});}}
function decide(value){{if(!visible.length)return;const row=visible[active],index=active;row.decision=value;persist(row);applyFilter(null,index);}}
function exportTSV(){{const columns=['case_id','scope','metric_group','dataset','subject','session','acquisition_id','split','automatic_priority','automatic_typicality_score','decision','reviewer_notes'];const clean=v=>String(v??'').replace(/[\\t\\r\\n]+/g,' ');const lines=[columns.join('\\t'),...rows.map(r=>columns.map(c=>clean(r[c])).join('\\t'))];const blob=new Blob([lines.join('\\n')+'\\n'],{{type:'text/tab-separated-values'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='qc_review.tsv';a.click();URL.revokeObjectURL(a.href);}}
function keyboard(event){{if(['TEXTAREA','INPUT','SELECT'].includes(event.target.tagName))return;const key=event.key.toLowerCase();if(event.key==='ArrowRight'||event.key==='ArrowDown'){{event.preventDefault();move(1);}}else if(event.key==='ArrowLeft'||event.key==='ArrowUp'){{event.preventDefault();move(-1);}}else if(key==='p'||key==='1')decide('PASS');else if(key==='r'||key==='2')decide('REVIEW');else if(key==='f'||key==='3')decide('FAIL');}}
['scope','priority','decision'].forEach(id=>document.getElementById(id).addEventListener('change',()=>applyFilter()));document.getElementById('search').addEventListener('input',()=>applyFilter());document.getElementById('previous').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);document.getElementById('export').onclick=exportTSV;document.addEventListener('keydown',keyboard);applyFilter();
</script></body></html>"""


def validate_report_images(frame: pd.DataFrame, report_dir: Path) -> None:
    for row in frame.to_dict(orient="records"):
        path = report_dir / str(row["image_relpath"])
        if sha256_file(require_file(path)) != row["image_sha256"]:
            raise ValueError(f"QC report image SHA-256 mismatch: {path}")


def require_current_report(
    frame: pd.DataFrame, source: Mapping[str, Any]
) -> None:
    if source.get("montage_version") != MONTAGE_VERSION:
        raise ValueError("QC report uses an obsolete NeuroVFM montage")
    if (
        "montage_version" not in frame
        or set(frame["montage_version"]) != {MONTAGE_VERSION}
    ):
        raise ValueError(
            "QC metrics and source manifest use different montage versions"
        )


def command_build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"QC output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fine_tuning, fine_tuning_manifest = load_fine_tuning_cases(
        args.manifest,
        args.fine_tuning_cache_root,
        args.expected_fine_tuning,
    )
    onharmony, onharmony_manifest = load_onharmony_cases(
        args.filelist,
        args.metadata,
        args.onharmony_cache_root,
        args.expected_onharmony,
    )
    cases = [*fine_tuning, *onharmony]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError(
            "NeuroVFM visual-QC case identifiers are not unique"
        )

    def worker(case: Case) -> dict[str, Any]:
        return inspect_case(
            case,
            output_dir,
            verify_raw_sha256=not args.skip_raw_sha256,
        )

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(worker, cases), start=1):
            rows.append(row)
            if index % 25 == 0 or index == len(cases):
                print(
                    f"  [{index}/{len(cases)}] {row['case_id']}",
                    flush=True,
                )
    frame = add_automatic_assessment(pd.DataFrame(rows))
    metrics_path = output_dir / "qc_metrics.tsv"
    source_path = output_dir / "qc_source_manifest.json"
    report_path = output_dir / "index.html"
    atomic_tsv(metrics_path, frame)
    created = utc_now()
    source = {
        "status": "QC_REPORT_BUILT",
        "created_utc": created,
        "label_blind": True,
        "outcome_columns_loaded": [],
        "n_cases": len(frame),
        "n_fine_tuning": len(fine_tuning),
        "n_onharmony": len(onharmony),
        "fine_tuning_manifest": fine_tuning_manifest,
        "onharmony_manifest": onharmony_manifest,
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "ordered_case_ids_sha256": canonical_json_hash(
            frame["case_id"].tolist()
        ),
        "montage_version": MONTAGE_VERSION,
        "patch_shape": list(PATCH_SHAPE),
        "raw_sha256_verified": not args.skip_raw_sha256,
        "automatic_typicality_score_is_not_quality": True,
        "automatic_assessment_never_excludes": True,
    }
    atomic_json(source_path, source)
    atomic_text(report_path, report_html(frame, created))
    validate_report_images(frame, output_dir)
    print(json.dumps(source, indent=2, allow_nan=False), flush=True)
    print(f"Report: {report_path}", flush=True)


def command_rerender(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = require_file(report_dir / "qc_metrics.tsv")
    source_path = require_file(report_dir / "qc_source_manifest.json")
    frame = pd.read_csv(
        metrics_path, sep="\t", keep_default_na=False
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("QC metrics differ from their source manifest")
    require_current_report(frame, source)
    validate_report_images(frame, report_dir)
    atomic_text(
        report_dir / "index.html",
        report_html(frame, str(source["created_utc"])),
    )
    print(f"Report: {report_dir / 'index.html'}")


def final_report_html(
    frame: pd.DataFrame, summary: Mapping[str, Any]
) -> str:
    attention = frame.loc[frame["decision"].isin(["REVIEW", "FAIL"])]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.case_id))}</td>"
        f"<td>{html.escape(str(row.decision))}</td>"
        f"<td>{html.escape(str(row.reviewer_notes))}</td>"
        "</tr>"
        for row in attention.itertuples(index=False)
    )
    if not rows:
        rows = "<tr><td colspan='3'>No REVIEW or FAIL decisions.</td></tr>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>NeuroVFM QC final report</title>
<style>body{{font:14px Arial,sans-serif;max-width:1000px;margin:30px auto;color:#172126}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5d9;padding:7px;text-align:left}}th{{background:#eef2f3}}</style>
</head><body><h1>NeuroVFM visual QC final report</h1>
<p>Status: <b>{html.escape(str(summary["status"]))}</b></p>
<p>Reviewed: {summary["n_reviewed"]} / {summary["n_cases"]}. Decisions: {html.escape(json.dumps(summary["decision_counts"], sort_keys=True))}.</p>
<p>The automatic typicality score only ordered review. Human decisions are authoritative.</p>
<h2>Cases requiring attention</h2>
<table><thead><tr><th>Case</th><th>Decision</th><th>Notes</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


def command_finalize(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = require_file(report_dir / "qc_metrics.tsv")
    source_path = require_file(report_dir / "qc_source_manifest.json")
    review_path = require_file(args.review_tsv)
    metrics = pd.read_csv(
        metrics_path, sep="\t", keep_default_na=False
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("QC metrics differ from their source manifest")
    require_current_report(metrics, source)
    validate_report_images(metrics, report_dir)
    review = pd.read_csv(
        review_path, sep="\t", keep_default_na=False
    )
    required = {"case_id", "decision", "reviewer_notes"}
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError(f"Review TSV lacks columns: {missing}")
    if review["case_id"].duplicated().any():
        raise ValueError(
            "Review TSV contains duplicate case identifiers"
        )
    if set(review["case_id"]) != set(metrics["case_id"]):
        raise ValueError(
            "Review TSV does not contain exactly the report cases"
        )
    invalid = sorted(set(review["decision"]) - DECISIONS)
    if invalid:
        raise ValueError(f"Invalid NeuroVFM QC decisions: {invalid}")
    joined = metrics.drop(
        columns=["decision", "reviewer_notes"]
    ).merge(
        review[["case_id", "decision", "reviewer_notes"]],
        on="case_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if joined["case_id"].tolist() != metrics["case_id"].tolist():
        raise RuntimeError(
            "Final NeuroVFM review order differs from QC metrics"
        )
    decisions = joined["decision"].fillna("UNREVIEWED")
    if (
        not args.allow_incomplete
        and (decisions == "UNREVIEWED").any()
    ):
        raise ValueError(
            "Review is incomplete: "
            f"{(decisions == 'UNREVIEWED').sum()} cases"
        )
    completed_path = report_dir / "qc_review_completed.tsv"
    summary_path = report_dir / "qc_final_summary.json"
    final_report_path = report_dir / "qc_final_report.html"
    atomic_tsv(completed_path, joined)
    summary = {
        "status": (
            "HUMAN_REVIEW_COMPLETED"
            if (decisions != "UNREVIEWED").all()
            else "HUMAN_REVIEW_PARTIAL"
        ),
        "created_utc": utc_now(),
        "label_blind": True,
        "n_cases": len(joined),
        "n_reviewed": int((decisions != "UNREVIEWED").sum()),
        "decision_counts": {
            str(key): int(value)
            for key, value in decisions.value_counts().items()
        },
        "metrics_sha256": sha256_file(metrics_path),
        "source_manifest_sha256": sha256_file(source_path),
        "review_source": str(review_path),
        "review_source_sha256": sha256_file(review_path),
        "completed_review": str(completed_path),
        "completed_review_sha256": sha256_file(completed_path),
        "automatic_typicality_score_is_not_an_exclusion_decision": True,
    }
    atomic_json(summary_path, summary)
    atomic_text(
        final_report_path, final_report_html(joined, summary)
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST
    )
    build.add_argument(
        "--fine-tuning-cache-root",
        type=Path,
        default=DEFAULT_PREPROCESSING_CACHE_ROOT,
    )
    build.add_argument(
        "--filelist", type=Path, default=DEFAULT_RAW_FILELIST
    )
    build.add_argument(
        "--metadata", type=Path, default=DEFAULT_METADATA
    )
    build.add_argument(
        "--onharmony-cache-root",
        type=Path,
        default=DEFAULT_ONHARMONY_CACHE_ROOT,
    )
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument(
        "--expected-fine-tuning",
        type=int,
        default=EXPECTED_FINE_TUNING,
    )
    build.add_argument(
        "--expected-onharmony",
        type=int,
        default=EXPECTED_ONHARMONY,
    )
    build.add_argument("--workers", type=int, default=4)
    build.add_argument(
        "--skip-raw-sha256",
        action="store_true",
        help=(
            "Use path/size/mtime binding without rehashing raw T1w files"
        ),
    )
    build.set_defaults(func=command_build)
    rerender = commands.add_parser("rerender")
    rerender.add_argument("--report-dir", type=Path, required=True)
    rerender.set_defaults(func=command_rerender)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--report-dir", type=Path, required=True)
    finalize.add_argument("--review-tsv", type=Path, required=True)
    finalize.add_argument("--allow-incomplete", action="store_true")
    finalize.set_defaults(func=command_finalize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("--workers must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
