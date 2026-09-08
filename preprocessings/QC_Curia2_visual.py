#!/usr/bin/env python3
"""Build and finalize label-blind visual QC for Curia-2 raw-T1 inputs.

Curia-2 has no persistent model-ready volume.  The report therefore verifies
the raw files used by the definitive brain-age and ON-Harmony cohorts,
reconstructs the exact LAS -> axial PL -> 512x512 processor path for selected
slices, and presents both the 3-D anatomy and processed slices for human
review.  Automatic scores only order cases; they never exclude a scan.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import nibabel as nib
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
    ornt_transform,
)
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.Curia2.model import (  # noqa: E402
    MODEL_NAME,
    TARGET_AXCODES,
    default_model_dir,
    las_slice_to_pl,
    validate_curia2_release,
)
from FineTuning.shared.cohort import (  # noqa: E402
    DEFAULT_MANIFEST,
    MANIFEST_SHA256,
    sha256_file,
)
from FineTuning.shared.onharmony import (  # noqa: E402
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    load_onharmony_manifest,
)


EXPECTED_FINE_TUNING = 1411
EXPECTED_ONHARMONY = 165
EXPECTED_TOTAL = EXPECTED_FINE_TUNING + EXPECTED_ONHARMONY
MONTAGE_VERSION = "curia2_visual_qc_v1"
PROCESSOR_SHAPE = (1, 512, 512)
PROCESSOR_EPS = 1e-6
SLICE_QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.90)
DECISIONS = {"PASS", "REVIEW", "FAIL", "UNREVIEWED"}
ROBUST_METRICS = (
    "n_slices",
    "raw_dynamic_range",
    "active_slice_fraction",
    "core_active_fraction",
    "active_span_fraction",
    "active_center_offset_fraction",
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


def stable_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semicolon_values(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(";") if item.strip()}


def load_fine_tuning_cases(
    manifest_path: Path, expected: int | None
) -> tuple[list[Case], dict[str, Any]]:
    """Load identity and paths only; age, sex and outcomes never enter memory."""

    manifest = require_file(manifest_path)
    digest = sha256_file(manifest)
    if digest != MANIFEST_SHA256:
        raise ValueError(
            f"Definitive manifest drift: expected {MANIFEST_SHA256}, observed {digest}"
        )
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
        raise ValueError("Fine-tuning identity columns contain empty values")
    permitted = frame["raw_runtime_conditions"].map(semicolon_values)
    if not permitted.map(lambda values: MODEL_NAME in values).all():
        raise ValueError(f"Every fine-tuning case must permit {MODEL_NAME}")

    cases = [
        Case(
            case_id=f"finetuning::{row.dataset}::{row.subject}",
            scope="finetuning",
            metric_group=f"finetuning::{row.dataset}",
            dataset=str(row.dataset),
            subject=str(row.subject),
            session="",
            acquisition_id="",
            split=str(row.split),
            raw_path=Path(str(row.raw_t1w_path)).expanduser().resolve(),
        )
        for row in frame.itertuples(index=False)
    ]
    return cases, {
        "manifest": str(manifest),
        "manifest_sha256": digest,
        "n_cases": len(cases),
        "split_counts": {
            str(key): int(value) for key, value in frame["split"].value_counts().items()
        },
        "dataset_counts": {
            str(key): int(value) for key, value in frame["dataset"].value_counts().items()
        },
        "outcome_columns_loaded": [],
    }


def load_onharmony_cases(
    filelist: Path, metadata: Path
) -> tuple[list[Case], dict[str, Any]]:
    frame, summary = load_onharmony_manifest(
        filelist,
        metadata,
        expected_acquisitions=EXPECTED_ONHARMONY,
        expected_subjects=20,
    )
    cases = [
        Case(
            case_id=f"onharmony::{row.acquisition_id}",
            scope="onharmony",
            metric_group=f"onharmony::{row.scanner_code}",
            dataset="ON-Harmony",
            subject=str(row.subject),
            session=str(row.session),
            acquisition_id=str(row.acquisition_id),
            split="external",
            raw_path=Path(str(row.raw_t1w_path)).expanduser().resolve(),
        )
        for row in frame.itertuples(index=False)
    ]
    result = dict(summary)
    result["outcome_columns_loaded"] = []
    return cases, result


def reorient_array(
    volume: np.ndarray, affine: np.ndarray, target_axcodes: Sequence[str]
) -> np.ndarray:
    current = io_orientation(affine)
    target = axcodes2ornt(tuple(target_axcodes))
    return np.ascontiguousarray(
        apply_orientation(volume, ornt_transform(current, target)), dtype=np.float32
    )


def selected_slice_indices(n_slices: int) -> list[int]:
    if n_slices < len(SLICE_QUANTILES):
        raise ValueError(f"Curia-2 raw volume has too few axial slices: {n_slices}")
    indices = [int(round((n_slices - 1) * value)) for value in SLICE_QUANTILES]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Curia-2 QC slice indices are not unique: {indices}")
    return indices


def process_pl_slices(slices: Sequence[np.ndarray]) -> np.ndarray:
    """Reproduce the pinned processor's tensor branch exactly on CPU."""

    import torch
    import torch.nn.functional as functional

    processed: list[np.ndarray] = []
    for array in slices:
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        tensor = functional.interpolate(
            tensor[None, None],
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0, 0]
        torch.clamp_min(tensor, -1000.0, out=tensor)
        mean = float(tensor.mean())
        std = float(tensor.std())
        tensor = tensor - mean if std < PROCESSOR_EPS else (tensor - mean) / std
        processed.append(tensor.numpy())
    output = np.stack(processed).astype(np.float32, copy=False)[:, None]
    if tuple(output.shape[1:]) != PROCESSOR_SHAPE:
        raise RuntimeError(f"Unexpected Curia-2 processor reconstruction: {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("Curia-2 processor reconstruction contains non-finite values")
    return output


def validate_processor_equivalence(model_dir: Path, case: Case) -> dict[str, Any]:
    """Compare our report reconstruction with the exact pinned remote-code processor."""

    import torch
    from transformers import AutoImageProcessor

    image = nib.load(str(require_file(case.raw_path)))
    if len(image.shape) != 3:
        raise ValueError(f"Curia-2 requires a 3-D raw T1w: {case.raw_path}")
    source = image.get_fdata().astype(np.float32)
    las = reorient_array(source, image.affine, TARGET_AXCODES)
    indices = selected_slice_indices(las.shape[-1])
    slices = [las_slice_to_pl(las, index) for index in indices]
    reconstructed = process_pl_slices(slices)
    processor = AutoImageProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    expected = processor(
        [torch.from_numpy(array) for array in slices], return_tensors="pt"
    )["pixel_values"].detach().cpu().numpy()
    difference = np.abs(reconstructed - expected)
    if not np.array_equal(reconstructed, expected):
        raise RuntimeError(
            "Curia-2 QC processor differs from the pinned processor: "
            f"max={float(difference.max())}"
        )
    return {
        "status": "PASS",
        "case_id": case.case_id,
        "selected_indices": indices,
        "shape": list(reconstructed.shape),
        "maximum_absolute_difference": float(difference.max()),
        "bit_exact": True,
    }


def display_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def scale_slice(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    scaled = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def render_panel(
    values: np.ndarray,
    *,
    limits: tuple[float, float],
    pixel_spacing: tuple[float, float],
    orientation: tuple[str, str, str, str],
    size: int = 190,
) -> Image.Image:
    gray = scale_slice(values, limits)
    row_spacing, column_spacing = pixel_spacing
    physical_height = gray.shape[0] * row_spacing
    physical_width = gray.shape[1] * column_spacing
    factor = min((size - 4) / physical_width, (size - 4) / physical_height)
    width = max(1, int(round(physical_width * factor)))
    height = max(1, int(round(physical_height * factor)))
    resized = Image.fromarray(gray, mode="L").resize(
        (width, height), resample=Image.Resampling.LANCZOS
    ).convert("RGB")
    panel = Image.new("RGB", (size, size), "black")
    panel.paste(resized, ((size - width) // 2, (size - height) // 2))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    top, bottom, left, right = orientation
    draw.text((size // 2 - 3, 3), top, fill=(250, 206, 70), font=font)
    draw.text((size // 2 - 3, size - 13), bottom, fill=(250, 206, 70), font=font)
    draw.text((3, size // 2 - 5), left, fill=(250, 206, 70), font=font)
    draw.text((size - 11, size // 2 - 5), right, fill=(250, 206, 70), font=font)
    return panel


def center_index(volume: np.ndarray, axis: int) -> int:
    finite = np.isfinite(volume)
    values = np.abs(volume)
    threshold = max(float(np.percentile(values[finite], 20.0)), 1e-8)
    support = finite & (values > threshold)
    coordinates = np.argwhere(support)
    if not len(coordinates):
        return int(volume.shape[axis] // 2)
    return int(np.clip(round(float(coordinates[:, axis].mean())), 0, volume.shape[axis] - 1))


def render_montage(
    case: Case,
    ras: np.ndarray,
    las: np.ndarray,
    ras_spacing: tuple[float, float, float],
    selected: Sequence[int],
    processed: np.ndarray,
    destination: Path,
) -> None:
    x, y, z = (center_index(ras, axis) for axis in range(3))
    raw_limits = display_limits(ras)
    row1 = [
        (np.rot90(ras[x, :, :]), (ras_spacing[2], ras_spacing[1]), ("S", "I", "P", "A"), "sagittal"),
        (np.rot90(ras[:, y, :]), (ras_spacing[2], ras_spacing[0]), ("S", "I", "L", "R"), "coronal"),
        (np.rot90(ras[:, :, max(0, z - ras.shape[2] // 6)]), (ras_spacing[1], ras_spacing[0]), ("A", "P", "L", "R"), "axial low"),
        (np.rot90(ras[:, :, z]), (ras_spacing[1], ras_spacing[0]), ("A", "P", "L", "R"), "axial center"),
        (np.rot90(ras[:, :, min(ras.shape[2] - 1, z + ras.shape[2] // 6)]), (ras_spacing[1], ras_spacing[0]), ("A", "P", "L", "R"), "axial high"),
    ]
    pl_slices = [las_slice_to_pl(las, index) for index in selected]
    pl_spacing = (ras_spacing[1], ras_spacing[0])
    pl_orientation = ("A", "P", "R", "L")
    panel_size = 190
    title_height = 32
    row_height = panel_size + 28
    canvas = Image.new("RGB", (panel_size * 5, title_height + 3 * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (8, 8),
        f"{case.case_id} | Curia-2 raw and exact processor QC",
        fill=(20, 28, 36),
        font=font,
    )

    labels = (
        "1. Raw T1w in closest RAS orientation",
        "2. Exact axial PL slices before processor (all slices are consumed)",
        "3. Same PL slices after bicubic 512 + clamp + per-slice z-score",
    )
    rows: list[list[tuple[np.ndarray, tuple[float, float], tuple[str, str, str, str], str, tuple[float, float]]]] = []
    rows.append([(array, spacing, orientation, name, raw_limits) for array, spacing, orientation, name in row1])
    rows.append([
        (array, pl_spacing, pl_orientation, f"z={index}/{las.shape[-1] - 1}", raw_limits)
        for array, index in zip(pl_slices, selected)
    ])
    rows.append([
        (array[0], (1.0, 1.0), pl_orientation, f"z={index}/{las.shape[-1] - 1}", (-3.0, 3.0))
        for array, index in zip(processed, selected)
    ])
    for row_index, (label, panels) in enumerate(zip(labels, rows)):
        top = title_height + row_index * row_height
        draw.text((8, top + 2), label, fill=(20, 28, 36), font=font)
        for column, (array, spacing, orientation, name, limits) in enumerate(panels):
            panel = render_panel(
                array,
                limits=limits,
                pixel_spacing=spacing,
                orientation=orientation,
                size=panel_size,
            )
            x_position = column * panel_size
            canvas.paste(panel, (x_position, top + 18))
            draw.text(
                (x_position + 7, top + panel_size + 19),
                name,
                fill=(20, 28, 36),
                font=font,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
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
    normalized = "".join(character if character.isalnum() else "_" for character in stem)
    digest = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:12]
    return f"{normalized}_{digest}.jpg"


def slice_activity_metrics(las: np.ndarray) -> dict[str, Any]:
    flattened = las.reshape(-1, las.shape[-1])
    low = np.percentile(flattened, 0.5, axis=0)
    high = np.percentile(flattened, 99.5, axis=0)
    dynamic = high - low
    global_dynamic = float(np.percentile(las, 99.5) - np.percentile(las, 0.5))
    threshold = max(global_dynamic * 1e-3, 1e-8)
    active = dynamic > threshold
    n_slices = las.shape[-1]
    core_start, core_stop = int(0.25 * n_slices), int(np.ceil(0.75 * n_slices))
    indices = np.flatnonzero(active)
    if len(indices):
        span = int(indices[-1] - indices[0] + 1)
        center = float(indices.mean() / max(n_slices - 1, 1))
    else:
        span, center = 0, 0.5
    return {
        "raw_dynamic_range": global_dynamic,
        "active_slice_fraction": float(active.mean()),
        "core_active_fraction": float(active[core_start:core_stop].mean()),
        "active_span_fraction": float(span / n_slices),
        "active_center_offset_fraction": float(abs(center - 0.5)),
        "inferior_active_fraction": float(active[: max(1, n_slices // 5)].mean()),
        "superior_active_fraction": float(active[-max(1, n_slices // 5) :].mean()),
    }


def inspect_case(case: Case, output_dir: Path, verify_raw_sha256: bool) -> dict[str, Any]:
    raw_path = require_file(case.raw_path)
    before = stable_signature(raw_path)
    raw_sha256 = sha256_file(raw_path) if verify_raw_sha256 else "NOT_COMPUTED"
    image = nib.load(str(raw_path))
    if len(image.shape) != 3:
        raise ValueError(f"Curia-2 requires exactly 3-D input: {case.case_id} {image.shape}")
    source = image.get_fdata().astype(np.float32)
    if not np.isfinite(source).all() or float(source.std()) <= 0.0:
        raise ValueError(f"Raw T1w is non-finite or constant: {case.case_id}")
    ras = reorient_array(source, image.affine, ("R", "A", "S"))
    las = reorient_array(source, image.affine, TARGET_AXCODES)
    indices = selected_slice_indices(las.shape[-1])
    pl_slices = [las_slice_to_pl(las, index) for index in indices]
    processed = process_pl_slices(pl_slices)
    after = stable_signature(raw_path)
    if before != after:
        raise RuntimeError(f"Raw input changed during Curia-2 QC: {raw_path}")

    ras_image = nib.as_closest_canonical(image)
    ras_spacing = tuple(float(value) for value in ras_image.header.get_zooms()[:3])
    image_name = safe_image_name(case)
    image_path = output_dir / "images" / image_name
    render_montage(case, ras, las, ras_spacing, indices, processed, image_path)
    processed_means = processed.astype(np.float64).mean(axis=(1, 2, 3))
    processed_stds = processed.astype(np.float64).std(axis=(1, 2, 3), ddof=1)
    selected_raw_stds = np.asarray([float(array.std()) for array in pl_slices])
    result = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(case).items()
        },
        "raw_sha256": raw_sha256,
        "raw_sha256_verified": bool(verify_raw_sha256),
        "raw_shape": "x".join(str(int(value)) for value in source.shape),
        "ras_shape": "x".join(str(int(value)) for value in ras.shape),
        "raw_axcodes": "".join(str(value) for value in nib.aff2axcodes(image.affine)),
        "ras_voxel_spacing_mm": "x".join(f"{value:.6g}" for value in ras_spacing),
        "voxel_volume_mm3": float(np.prod(ras_spacing)),
        "n_slices": int(las.shape[-1]),
        "selected_slice_indices": ",".join(str(value) for value in indices),
        **slice_activity_metrics(las),
        "selected_constant_fraction": float((selected_raw_stds < PROCESSOR_EPS).mean()),
        "processed_abs_mean_max": float(np.abs(processed_means).max()),
        "processed_std_deviation_max": float(
            np.max(np.abs(processed_stds[selected_raw_stds >= PROCESSOR_EPS] - 1.0))
            if (selected_raw_stds >= PROCESSOR_EPS).any()
            else 0.0
        ),
        "image_relpath": f"images/{image_name}",
        "image_sha256": sha256_file(image_path),
        "montage_version": MONTAGE_VERSION,
        "technical_status": "PASS",
        "technical_error": "",
    }
    return result


def robust_absolute_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    deviation = (numeric - median).abs()
    mad = deviation.median()
    if not np.isfinite(mad) or mad <= 0.0:
        standard_deviation = numeric.std(ddof=1)
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
        return deviation / standard_deviation
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
    result["automatic_outlier_count"] = (result[z_columns] >= 3.5).sum(axis=1)
    result["automatic_max_robust_z"] = result[z_columns].max(axis=1)
    severity = np.maximum(result["automatic_max_robust_z"].to_numpy(float) - 2.0, 0.0)
    penalty = 1.5 * severity + 0.5 * result["automatic_outlier_count"].to_numpy(float)
    result["automatic_typicality_score"] = np.round(
        np.clip(10.0 - penalty, 0.0, 10.0), 1
    )
    priority = np.full(len(result), "ROUTINE", dtype=object)
    review = result["automatic_max_robust_z"].to_numpy(float) >= 3.5
    high = (result["automatic_max_robust_z"].to_numpy(float) >= 5.0) | (
        result["automatic_outlier_count"].to_numpy(int) >= 2
    )
    priority[review] = "REVIEW"
    priority[high] = "HIGH"
    result["automatic_priority"] = priority
    result["decision"] = "UNREVIEWED"
    result["reviewer_notes"] = ""
    order = pd.Categorical(priority, ["HIGH", "REVIEW", "ROUTINE"], ordered=True)
    return (
        result.assign(_priority=order)
        .sort_values(
            ["_priority", "scope", "dataset", "subject", "session", "case_id"],
            kind="stable",
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


def public_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
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
        "raw_shape",
        "raw_axcodes",
        "ras_voxel_spacing_mm",
        "n_slices",
        "active_slice_fraction",
        "core_active_fraction",
        "active_span_fraction",
        "selected_constant_fraction",
        "processed_abs_mean_max",
        "processed_std_deviation_max",
        "decision",
        "reviewer_notes",
    ]
    return frame[columns].replace({np.nan: None}).to_dict(orient="records")


def report_html(frame: pd.DataFrame, created: str) -> str:
    payload = json.dumps(
        public_records(frame), separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    report_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Curia-2 visual QC</title>
<style>
body{{margin:0;background:#eef2f3;color:#172126;font:14px Arial,sans-serif}}header{{position:sticky;top:0;z-index:2;background:#244c55;color:white;padding:10px 16px}}h1{{font-size:20px;margin:0 0 7px;letter-spacing:0}}.tools{{display:flex;gap:7px;flex-wrap:wrap;align-items:center}}button,select,input{{font:inherit;padding:6px 8px;border:1px solid #9aa8ad;background:white;color:#172126}}main{{max-width:1080px;margin:14px auto;padding:0 10px}}.note{{line-height:1.4}}.card{{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:12px;background:white;border:1px solid #c7d0d3;padding:10px}}.card img{{width:100%;height:auto;display:block;background:#111}}.meta{{line-height:1.55}}.metrics{{white-space:pre-line;background:#f2f5f6;padding:8px;margin:8px 0}}.decision{{display:flex;gap:6px}}.decision button{{flex:1}}.pass.selected{{outline:3px solid #238348}}.review.selected{{outline:3px solid #d28a16}}.fail.selected{{outline:3px solid #bc3b31}}textarea{{box-sizing:border-box;width:100%;height:85px;margin-top:8px}}.HIGH{{color:#a51d18;font-weight:bold}}.REVIEW{{color:#a05a00;font-weight:bold}}.ROUTINE{{color:#22663c;font-weight:bold}}.error{{padding:30px;color:#a51d18}}@media(max-width:800px){{.card{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Curia-2 visual QC</h1><div class="tools"><b id="status"></b>
<button id="previous" title="Previous (left/up/K)">Previous</button><button id="next" title="Next (right/down/J)">Next</button>
<select id="scope"><option value="ALL">All cohorts</option><option value="finetuning">Fine-tuning</option><option value="onharmony">ON-Harmony</option></select>
<select id="priority"><option value="ALL">All priorities</option><option>HIGH</option><option>REVIEW</option><option>ROUTINE</option></select>
<select id="decision"><option value="ALL">All decisions</option><option value="UNREVIEWED">Unreviewed</option><option>PASS</option><option>REVIEW</option><option>FAIL</option></select>
<input id="search" placeholder="subject, scanner, dataset"><button id="export">Export qc_review.tsv</button></div></header>
<main><p class="note"><b>Human review is authoritative.</b> Automatic typicality only orders cases within dataset/scanner groups; it is not a diagnostic quality score and never excludes a scan. Row 1 shows raw anatomy in RAS. Rows 2-3 show the exact PL slices before and after the pinned Curia-2 processor. Every axial slice, including peripheral slices, is used by the model. No age, sex, diagnosis, prediction, BAG or error was loaded. Created {html.escape(created)}.</p><div id="root"></div></main>
<script>
const rows={payload};const KEY='curia2-qc-{report_id}';let saved=JSON.parse(localStorage.getItem(KEY)||'{{}}'),visible=[],active=0;
function esc(x){{return String(x??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function state(row){{return saved[row.case_id]||{{decision:'UNREVIEWED',reviewer_notes:'',reviewed_at_utc:''}};}}
function persist(row){{localStorage.setItem(KEY,JSON.stringify(saved));}}
function filter(){{const s=document.getElementById('scope').value,p=document.getElementById('priority').value,d=document.getElementById('decision').value,q=document.getElementById('search').value.toLowerCase();visible=rows.filter(row=>{{const current=state(row).decision;return(s==='ALL'||row.scope===s)&&(p==='ALL'||row.automatic_priority===p)&&(d==='ALL'||current===d)&&(!q||JSON.stringify(row).toLowerCase().includes(q));}});active=Math.min(active,Math.max(visible.length-1,0));render();}}
function render(){{const root=document.getElementById('root'),reviewed=rows.filter(row=>state(row).decision!=='UNREVIEWED').length;document.getElementById('status').textContent=`${{visible.length}} visible | ${{reviewed}}/${{rows.length}} reviewed${{visible.length?` | case ${{active+1}}/${{visible.length}}`:''}}`;if(!visible.length){{root.innerHTML='<p>No cases match the filters.</p>';return;}}const row=visible[active],s=state(row);root.innerHTML=`<section class="card"><div><img src="${{esc(row.image_relpath)}}" alt="QC montage for ${{esc(row.case_id)}}"><div class="error" hidden>Image missing. Keep index.html together with its images directory.</div></div><div><h2>${{esc(row.subject)}} ${{esc(row.session)}}</h2><div class="meta">${{esc(row.scope)}} | ${{esc(row.dataset)}} | ${{esc(row.split)}}<br>${{esc(row.metric_group)}}<br><b class="${{esc(row.automatic_priority)}}">Priority: ${{esc(row.automatic_priority)}}</b> | typicality ${{Number(row.automatic_typicality_score).toFixed(1)}}/10</div><div class="metrics">raw: ${{esc(row.raw_shape)}} | ${{esc(row.raw_axcodes)}} | ${{esc(row.ras_voxel_spacing_mm)}} mm\naxial slices: ${{row.n_slices}}\nactive slices: ${{(100*Number(row.active_slice_fraction)).toFixed(1)}}%\nactive core: ${{(100*Number(row.core_active_fraction)).toFixed(1)}}%\nactive span: ${{(100*Number(row.active_span_fraction)).toFixed(1)}}%\nselected constant: ${{(100*Number(row.selected_constant_fraction)).toFixed(1)}}%\nprocessor max |mean|: ${{Number(row.processed_abs_mean_max).toExponential(2)}}\nprocessor max |SD-1|: ${{Number(row.processed_std_deviation_max).toExponential(2)}}\nmax robust z: ${{Number(row.automatic_max_robust_z).toFixed(2)}} | outliers: ${{row.automatic_outlier_count}}</div><div class="decision"><button class="pass ${{s.decision==='PASS'?'selected':''}}">PASS</button><button class="review ${{s.decision==='REVIEW'?'selected':''}}">REVIEW</button><button class="fail ${{s.decision==='FAIL'?'selected':''}}">FAIL</button></div><textarea placeholder="Optional reviewer notes">${{esc(s.reviewer_notes)}}</textarea></div></section>`;const img=root.querySelector('img');img.onerror=()=>{{img.hidden=true;root.querySelector('.error').hidden=false;}};const buttons=root.querySelectorAll('.decision button');buttons[0].onclick=()=>decide('PASS');buttons[1].onclick=()=>decide('REVIEW');buttons[2].onclick=()=>decide('FAIL');root.querySelector('textarea').onchange=e=>{{s.reviewer_notes=e.target.value;saved[row.case_id]=s;persist();}};}}
function move(step){{if(!visible.length)return;active=Math.max(0,Math.min(visible.length-1,active+step));render();window.scrollTo(0,0);}}
function decide(value){{if(!visible.length)return;const row=visible[active],s=state(row);s.decision=value;s.reviewed_at_utc=new Date().toISOString();saved[row.case_id]=s;persist();if(document.getElementById('decision').value==='UNREVIEWED')filter();else{{move(1);}}}}
function clean(value){{return String(value??'').replace(/[\\t\\r\\n]+/g,' ');}}
function exportTSV(){{const columns=['case_id','decision','reviewer_notes','reviewed_at_utc'],lines=[columns.join('\\t')];rows.forEach(row=>{{const s=state(row);lines.push([row.case_id,s.decision,s.reviewer_notes,s.reviewed_at_utc].map(clean).join('\\t'));}});const blob=new Blob([lines.join('\\n')+'\\n'],{{type:'text/tab-separated-values'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='qc_review.tsv';a.click();URL.revokeObjectURL(a.href);}}
function keyboard(event){{if(event.defaultPrevented||event.altKey||event.ctrlKey||event.metaKey||['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;const key=event.key.toLowerCase();let handled=true;if(key==='arrowright'||key==='arrowdown'||key==='j')move(1);else if(key==='arrowleft'||key==='arrowup'||key==='k')move(-1);else if(key==='p'||key==='1')decide('PASS');else if(key==='r'||key==='2')decide('REVIEW');else if(key==='f'||key==='3')decide('FAIL');else if(key==='e')exportTSV();else handled=false;if(handled)event.preventDefault();}}
['scope','priority','decision'].forEach(id=>document.getElementById(id).addEventListener('change',()=>{{active=0;filter();}}));document.getElementById('search').addEventListener('input',()=>{{active=0;filter();}});document.getElementById('previous').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);document.getElementById('export').onclick=exportTSV;document.addEventListener('keydown',keyboard);filter();
</script></body></html>"""


def validate_report_images(frame: pd.DataFrame, report_dir: Path) -> None:
    for row in frame.itertuples(index=False):
        path = (report_dir / str(row.image_relpath)).resolve()
        if not path.is_relative_to(report_dir) or not path.is_file():
            raise FileNotFoundError(f"Missing Curia-2 QC image: {path}")
        if sha256_file(path) != str(row.image_sha256):
            raise ValueError(f"Curia-2 QC image SHA-256 mismatch: {path}")


def require_current_report(frame: pd.DataFrame, source: Mapping[str, Any]) -> None:
    if source.get("montage_version") != MONTAGE_VERSION:
        raise ValueError("QC report uses an obsolete Curia-2 montage")
    if "montage_version" not in frame or set(frame["montage_version"]) != {
        MONTAGE_VERSION
    }:
        raise ValueError("QC metrics and source manifest use different montage versions")


def command_build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"QC output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = default_model_dir(args.model_dir).resolve()
    release = validate_curia2_release(model_dir)
    fine_tuning, fine_tuning_manifest = load_fine_tuning_cases(
        args.manifest, EXPECTED_FINE_TUNING
    )
    onharmony, onharmony_manifest = load_onharmony_cases(
        args.filelist, args.metadata
    )
    cases = [*fine_tuning, *onharmony]
    if len(cases) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} cases, found {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Curia-2 visual-QC case identifiers are not unique")
    processor_equivalence = validate_processor_equivalence(model_dir, cases[0])

    def worker(case: Case) -> dict[str, Any]:
        return inspect_case(case, output_dir, not args.skip_raw_sha256)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(worker, cases), start=1):
            rows.append(row)
            if index % 25 == 0 or index == len(cases):
                print(f"  [{index}/{len(cases)}] {row['case_id']}", flush=True)
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
        "release": release,
        "processor_equivalence": processor_equivalence,
        "processor_contract": {
            "input": "all axial slices from LAS volume rotated LA to PL",
            "resize": "bicubic 512x512, align_corners=false, antialias=true",
            "clip_minimum": -1000.0,
            "normalization": "independent full-slice z-score",
        },
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "ordered_case_ids_sha256": canonical_json_hash(frame["case_id"].tolist()),
        "montage_version": MONTAGE_VERSION,
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
    frame = pd.read_csv(metrics_path, sep="\t", keep_default_na=False)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("QC metrics differ from their source manifest")
    require_current_report(frame, source)
    validate_report_images(frame, report_dir)
    atomic_text(report_dir / "index.html", report_html(frame, str(source["created_utc"])))
    print(f"Report: {report_dir / 'index.html'}")


def command_reassess(args: argparse.Namespace) -> None:
    """Recompute automatic ordering without reopening raw data or montages."""

    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = require_file(report_dir / "qc_metrics.tsv")
    source_path = require_file(report_dir / "qc_source_manifest.json")
    frame = pd.read_csv(metrics_path, sep="\t", keep_default_na=False)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("QC metrics differ from their source manifest")
    require_current_report(frame, source)
    validate_report_images(frame, report_dir)
    reassessed = add_automatic_assessment(frame)
    atomic_tsv(metrics_path, reassessed)
    source.update(
        {
            "metrics_sha256": sha256_file(metrics_path),
            "ordered_case_ids_sha256": canonical_json_hash(
                reassessed["case_id"].tolist()
            ),
            "automatic_assessment_recomputed_utc": utc_now(),
            "automatic_assessment_metrics": list(ROBUST_METRICS),
            "script_sha256": sha256_file(SCRIPT_PATH),
        }
    )
    atomic_json(source_path, source)
    atomic_text(
        report_dir / "index.html",
        report_html(reassessed, str(source["created_utc"])),
    )
    print(
        json.dumps(
            {
                "status": "AUTOMATIC_ORDER_REASSESSED",
                "n_cases": len(reassessed),
                "priority_counts": {
                    str(key): int(value)
                    for key, value in reassessed[
                        "automatic_priority"
                    ].value_counts().items()
                },
                "metrics_sha256": source["metrics_sha256"],
                "raw_or_montage_reopened": False,
            },
            indent=2,
        ),
        flush=True,
    )


def final_report_html(frame: pd.DataFrame, summary: Mapping[str, Any]) -> str:
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
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Curia-2 QC final report</title>
<style>body{{font:14px Arial,sans-serif;max-width:1000px;margin:30px auto;color:#172126}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5d9;padding:7px;text-align:left}}th{{background:#eef2f3}}</style></head><body>
<h1>Curia-2 visual QC final report</h1><p>Status: <b>{html.escape(str(summary['status']))}</b></p>
<p>Reviewed: {summary['n_reviewed']} / {summary['n_cases']}. Decisions: {html.escape(json.dumps(summary['decision_counts'], sort_keys=True))}.</p>
<p>Automatic typicality only ordered review. Human decisions are authoritative.</p>
<h2>Cases requiring attention</h2><table><thead><tr><th>Case</th><th>Decision</th><th>Notes</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


def command_finalize(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = require_file(report_dir / "qc_metrics.tsv")
    source_path = require_file(report_dir / "qc_source_manifest.json")
    review_path = require_file(args.review_tsv)
    metrics = pd.read_csv(metrics_path, sep="\t", keep_default_na=False)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("QC metrics differ from their source manifest")
    require_current_report(metrics, source)
    validate_report_images(metrics, report_dir)
    review = pd.read_csv(review_path, sep="\t", keep_default_na=False)
    required = {"case_id", "decision", "reviewer_notes", "reviewed_at_utc"}
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError(f"Review TSV lacks columns: {missing}")
    if review["case_id"].duplicated().any():
        raise ValueError("Review TSV contains duplicate case identifiers")
    if set(review["case_id"]) != set(metrics["case_id"]):
        raise ValueError("Review TSV does not contain exactly the report cases")
    invalid = sorted(set(review["decision"]) - DECISIONS)
    if invalid:
        raise ValueError(f"Invalid Curia-2 QC decisions: {invalid}")
    joined = metrics.drop(columns=["decision", "reviewer_notes"]).merge(
        review[["case_id", "decision", "reviewer_notes", "reviewed_at_utc"]],
        on="case_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if joined["case_id"].tolist() != metrics["case_id"].tolist():
        raise RuntimeError("Final Curia-2 review order differs from QC metrics")
    decisions = joined["decision"].fillna("UNREVIEWED")
    if not args.allow_incomplete and (decisions == "UNREVIEWED").any():
        raise ValueError(
            f"Review is incomplete: {(decisions == 'UNREVIEWED').sum()} cases"
        )
    completed_path = report_dir / "qc_review_completed.tsv"
    summary_path = report_dir / "qc_final_summary.json"
    final_report_path = report_dir / "qc_final_report.html"
    for path in (completed_path, summary_path, final_report_path):
        if path.exists():
            raise FileExistsError(f"Final Curia-2 QC artifact already exists: {path}")
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
            str(key): int(value) for key, value in decisions.value_counts().items()
        },
        "metrics_sha256": sha256_file(metrics_path),
        "source_manifest_sha256": sha256_file(source_path),
        "review_export": str(review_path),
        "review_export_sha256": sha256_file(review_path),
        "completed_review": str(completed_path),
        "completed_review_sha256": sha256_file(completed_path),
        "automatic_typicality_score_is_not_quality": True,
    }
    atomic_json(summary_path, summary)
    atomic_text(final_report_path, final_report_html(joined, summary))
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Generate montages and interactive report")
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--filelist", type=Path, default=DEFAULT_RAW_FILELIST)
    build.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    build.add_argument("--model-dir", type=Path)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--skip-raw-sha256", action="store_true")
    build.set_defaults(func=command_build)
    rerender = commands.add_parser("rerender", help="Rebuild only the HTML shell")
    rerender.add_argument("--report-dir", type=Path, required=True)
    rerender.set_defaults(func=command_rerender)
    reassess = commands.add_parser(
        "reassess", help="Recompute automatic ordering without rebuilding images"
    )
    reassess.add_argument("--report-dir", type=Path, required=True)
    reassess.set_defaults(func=command_reassess)
    finalize = commands.add_parser("finalize", help="Validate and freeze human review")
    finalize.add_argument("--report-dir", type=Path, required=True)
    finalize.add_argument("--review-tsv", type=Path, required=True)
    finalize.add_argument("--allow-incomplete", action="store_true")
    finalize.set_defaults(func=command_finalize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "workers", 1) <= 0:
        raise ValueError("--workers must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
