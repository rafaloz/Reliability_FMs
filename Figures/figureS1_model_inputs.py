#!/usr/bin/env python3
"""Render representative triplanar views of the seven exact model inputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-figure-s1-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
    ornt_transform,
)
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figures"
DEFAULT_PROVENANCE = PROJECT_ROOT / "Results" / "FigureS1" / "model_inputs_manifest.json"
DEFAULT_DATA_ROOTS = (
    PROJECT_ROOT,
    Path("/home/rafa/workRafa/FoundationModels_OH_T1w"),
    Path("/datos/work/rnavgon/FoundationModels_OH_T1w"),
)
DEFAULT_DATASET = "CamCAN"
DEFAULT_SUBJECT = "sub-CC110033"
PATCH_SHAPE = (4, 16, 16)
CURIA_PROCESSOR_EPS = 1e-6


@dataclass(frozen=True)
class Panel:
    model: str
    display_name: str
    images: tuple[np.ndarray, np.ndarray, np.ndarray]
    input_shape: str
    transform: str
    sources: tuple[Path, ...]
    display_aspects: tuple[float, float, float] = (1.0, 1.0, 1.0)
    foregrounds: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] = (
        None,
        None,
        None,
    )
    outlines: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] = (
        None,
        None,
        None,
    )


@dataclass(frozen=True)
class AnatomicalReference:
    path: Path
    provenance: dict[str, Any]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_data_root(override: Path | None) -> Path:
    candidates = (override,) if override is not None else DEFAULT_DATA_ROOTS
    markers = (
        Path("preprocessings/FineTunningDataset/preproc_yAware"),
        Path("preprocessings/FineTunningDataset/preproc_AnatCL"),
        Path("preprocessings/FineTunningDataset/preproc_3DNeuroSimCLR"),
        Path("preprocessings/FineTunningDataset/preproc_BrainIAC"),
        Path("preprocessings/FineTunningDataset/preproc_BrainSegFounder"),
        Path("preprocessings/FineTunningDataset/preproc_NeuroVFM"),
    )
    for candidate in candidates:
        if candidate is not None:
            resolved = candidate.expanduser().resolve()
            if all((resolved / marker).is_dir() for marker in markers):
                return resolved
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "Could not locate a project data tree containing all required model-input "
        f"directories. Searched: {searched}"
    )


def load_subject_record(manifest: Path, dataset: str, subject: str) -> dict[str, str]:
    frame = pd.read_csv(require_file(manifest), sep="\t", dtype=str, keep_default_na=False)
    required = {"dataset", "subject", "raw_t1w_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest lacks columns: {missing}")
    selected = frame.loc[(frame["dataset"] == dataset) & (frame["subject"] == subject)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one manifest row for {dataset}/{subject}, observed {len(selected)}"
        )
    return {str(key): str(value) for key, value in selected.iloc[0].items()}


def resolve_anatomical_reference(
    data_root: Path, record: dict[str, str], override: Path | None
) -> AnatomicalReference:
    if override is not None:
        path = require_file(override)
        return AnatomicalReference(
            path=path,
            provenance={
                "kind": "explicit_raw_t1w_override",
                "path": str(path),
                "sha256": sha256_file(path),
            },
        )
    recorded = Path(record["raw_t1w_path"])
    if recorded.is_file():
        path = recorded.resolve()
        return AnatomicalReference(
            path=path,
            provenance={
                "kind": "manifest_raw_t1w",
                "path": str(path),
                "sha256": sha256_file(path),
            },
        )

    subject_dir = (
        data_root
        / "preprocessings"
        / "FineTunningDataset"
        / "preproc_BrainSegFounder"
        / record["dataset"]
        / record["subject"]
    )
    candidate = require_file(subject_dir / f"{record['subject']}_ukb_reoriented.nii.gz")
    qc_path = require_file(subject_dir / "preprocessing_qc.json")
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if qc.get("status") != "PASS":
        raise ValueError(f"BrainSegFounder QC is not PASS: {qc_path}")
    if qc.get("dataset") != record["dataset"] or qc.get("subject") != record["subject"]:
        raise ValueError(f"BrainSegFounder QC identity mismatch: {qc_path}")
    raw_record = qc.get("raw_input", {})
    if raw_record.get("path") != record["raw_t1w_path"]:
        raise ValueError(f"BrainSegFounder QC raw-source mismatch: {qc_path}")
    observed_sha = sha256_file(candidate)
    expected_sha = qc.get("output_sha256", {}).get("reoriented_t1")
    if observed_sha != expected_sha:
        raise ValueError(
            f"Losslessly reoriented T1w SHA mismatch: {observed_sha} != {expected_sha}"
        )

    reorientation = qc.get("input_reorientation", {})
    required_keys = {
        "input_shape",
        "output_shape",
        "input_spacing_mm",
        "output_spacing_mm",
        "input_axcodes",
        "output_axcodes",
    }
    if not required_keys.issubset(reorientation):
        raise ValueError(f"Incomplete BrainSegFounder reorientation evidence: {qc_path}")
    input_shape = tuple(int(value) for value in reorientation["input_shape"])
    output_shape = tuple(int(value) for value in reorientation["output_shape"])
    input_spacing = tuple(float(value) for value in reorientation["input_spacing_mm"])
    output_spacing = tuple(float(value) for value in reorientation["output_spacing_mm"])
    image = nib.load(str(candidate))
    observed_shape = tuple(int(value) for value in image.shape)
    observed_spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    observed_axcodes = tuple(str(value) for value in nib.aff2axcodes(image.affine))
    if int(np.prod(input_shape)) != int(np.prod(output_shape)):
        raise ValueError("The recorded reorientation changed the voxel count")
    if not np.allclose(sorted(input_spacing), sorted(output_spacing), rtol=0.0, atol=1e-5):
        raise ValueError("The recorded reorientation changed the voxel spacing")
    if observed_shape != output_shape:
        raise ValueError(f"Reoriented T1w shape mismatch: {observed_shape} != {output_shape}")
    if not np.allclose(observed_spacing, output_spacing, rtol=0.0, atol=1e-5):
        raise ValueError(
            f"Reoriented T1w spacing mismatch: {observed_spacing} != {output_spacing}"
        )
    if observed_axcodes != tuple(reorientation["output_axcodes"]):
        raise ValueError(
            f"Reoriented T1w orientation mismatch: {observed_axcodes} != "
            f"{tuple(reorientation['output_axcodes'])}"
        )

    return AnatomicalReference(
        path=candidate,
        provenance={
            "kind": "lossless_fslreorient2std_equivalent",
            "path": str(candidate),
            "sha256": observed_sha,
            "qc": str(qc_path),
            "qc_sha256": sha256_file(qc_path),
            "manifest_raw_t1w": record["raw_t1w_path"],
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "input_spacing_mm": list(input_spacing),
            "output_spacing_mm": list(output_spacing),
            "input_axcodes": list(reorientation["input_axcodes"]),
            "output_axcodes": list(reorientation["output_axcodes"]),
            "equivalence": (
                "FSL axis permutation/90-degree reorientation without interpolation; "
                "Curia-2 subsequently targets LAS and NeuroVFM uses only canonical geometry"
            ),
        },
    )


def load_nifti(path: Path) -> tuple[np.ndarray, Any]:
    image = nib.load(str(require_file(path)))
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI at {path}, got {image.shape}")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or float(np.std(data)) <= 0.0:
        raise ValueError(f"NIfTI is non-finite or constant: {path}")
    return np.ascontiguousarray(data), image


def coronal_xyz(volume: np.ndarray, index: int | None = None) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected [X,Y,Z], got {volume.shape}")
    y = volume.shape[1] // 2 if index is None else int(index)
    return np.rot90(np.asarray(volume[:, y, :], dtype=np.float32))


def sagittal_xyz(volume: np.ndarray, index: int | None = None) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected [X,Y,Z], got {volume.shape}")
    x = volume.shape[0] // 2 if index is None else int(index)
    return np.rot90(np.asarray(volume[x, :, :], dtype=np.float32))


def axial_xyz(volume: np.ndarray, index: int | None = None) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected [X,Y,Z], got {volume.shape}")
    z = volume.shape[2] // 2 if index is None else int(index)
    return np.rot90(np.asarray(volume[:, :, z], dtype=np.float32))


def coronal_dhw(volume: np.ndarray, index: int | None = None) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected [D,H,W], got {volume.shape}")
    h = volume.shape[1] // 2 if index is None else int(index)
    return np.flipud(np.asarray(volume[:, h, :], dtype=np.float32))


def sagittal_dhw(volume: np.ndarray, index: int | None = None) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected [D,H,W], got {volume.shape}")
    w = volume.shape[2] // 2 if index is None else int(index)
    return np.flipud(np.asarray(volume[:, :, w], dtype=np.float32))


def axial_dhw(volume: np.ndarray, index: int | None = None) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected [D,H,W], got {volume.shape}")
    d = volume.shape[0] // 2 if index is None else int(index)
    return np.flipud(np.asarray(volume[d, :, :], dtype=np.float32))


def robust_limits(
    values: np.ndarray, foreground: np.ndarray | None = None
) -> tuple[float, float]:
    selected = np.isfinite(values)
    if foreground is not None:
        if foreground.shape != values.shape:
            raise ValueError("Display foreground and image shapes differ")
        selected &= foreground
    finite = np.asarray(values, dtype=np.float32)[selected]
    if finite.size == 0:
        raise ValueError("Cannot display an image without finite values")
    low, high = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def load_yaware(base: Path, dataset: str, subject: str) -> Panel:
    path = require_file(base / "preproc_yAware" / dataset / subject / f"{subject}_yaware_ready.npy")
    array = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if tuple(array.shape) != (1, 121, 145, 121):
        raise ValueError(f"Unexpected y-Aware input shape: {array.shape}")
    if not np.isfinite(array).all() or float(array.std()) <= 0.0:
        raise ValueError("Invalid y-Aware model input")
    qc = require_file(base / "preproc_AnatCL" / dataset / subject / "preprocessing_qc.json")
    images = (
        sagittal_xyz(array[0]),
        coronal_xyz(array[0]),
        axial_xyz(array[0]),
    )
    background = float(array[0, 0, 0, 0])
    foregrounds = tuple(
        ~np.isclose(image, background, rtol=0.0, atol=1e-6) for image in images
    )
    return Panel(
        model="y-Aware",
        display_name="y-Aware",
        images=images,
        input_shape="1 x 121 x 145 x 121",
        transform="CAT12 GM; whole-volume z-score",
        sources=(path, qc),
        foregrounds=foregrounds,
    )


def load_anatcl(base: Path, dataset: str, subject: str) -> Panel:
    path = require_file(base / "preproc_AnatCL" / dataset / subject / f"{subject}_anatcl_ready.nii.gz")
    data, _ = load_nifti(path)
    if tuple(data.shape) != (121, 145, 121):
        raise ValueError(f"Unexpected AnatCL shared input shape: {data.shape}")
    runtime = np.ascontiguousarray(data[:, 8:136, :], dtype=np.float32)
    if tuple(runtime.shape) != (121, 128, 121):
        raise RuntimeError(f"Unexpected AnatCL crop shape: {runtime.shape}")
    qc = require_file(path.parent / "preprocessing_qc.json")
    images = (
        sagittal_xyz(runtime),
        coronal_xyz(runtime),
        axial_xyz(runtime),
    )
    return Panel(
        model="AnatCL",
        display_name="AnatCL",
        images=images,
        input_shape="1 x 121 x 128 x 121",
        transform="CAT12 GM; fixed Y crop",
        sources=(path, qc),
        foregrounds=tuple(
            ~np.isclose(image, 0.0, rtol=0.0, atol=1e-8) for image in images
        ),
    )


def load_simclr(base: Path, dataset: str, subject: str) -> Panel:
    directory = base / "preproc_3DNeuroSimCLR" / dataset / subject
    image_path = require_file(directory / f"{subject}_simclr_ready.nii.gz")
    mask_path = require_file(directory / f"{subject}_mask_mni.nii.gz")
    image, _ = load_nifti(image_path)
    mask, _ = load_nifti(mask_path)
    if tuple(image.shape) != (193, 229, 193) or image.shape != mask.shape:
        raise ValueError(f"Unexpected 3D-Neuro-SimCLR image/mask shapes: {image.shape}, {mask.shape}")
    module = importlib.import_module("FineTuning.3DNeuroSimCLR.model")
    cropped = module.transpose_crop(image[np.newaxis, ...])
    cropped_mask = module.transpose_crop((mask > 0.5).astype(np.float32)[np.newaxis, ...])
    runtime = module.masked_standardize(cropped[None, ...], cropped_mask[None, ...])[0, 0]
    qc = require_file(directory / "preprocessing_qc.json")
    images = (
        sagittal_dhw(runtime),
        coronal_dhw(runtime),
        axial_dhw(runtime),
    )
    foregrounds = (
        sagittal_dhw(cropped_mask[0]) > 0.5,
        coronal_dhw(cropped_mask[0]) > 0.5,
        axial_dhw(cropped_mask[0]) > 0.5,
    )
    return Panel(
        model="3D-Neuro-SimCLR",
        display_name="3D-Neuro-\nSimCLR",
        images=images,
        input_shape="1 x 150 x 192 x 192",
        transform="registered T1w; masked z-score",
        sources=(image_path, mask_path, qc),
        foregrounds=foregrounds,
    )


def load_brainiac(base: Path, dataset: str, subject: str) -> Panel:
    directory = base / "preproc_BrainIAC" / dataset / subject
    path = require_file(directory / f"{subject}_brainiac_ready.nii.gz")
    data, _ = load_nifti(path)
    if tuple(data.shape) != (96, 96, 96):
        raise ValueError(f"Unexpected BrainIAC input shape: {data.shape}")
    images = (sagittal_xyz(data), coronal_xyz(data), axial_xyz(data))
    return Panel(
        model="BrainIAC",
        display_name="BrainIAC",
        images=images,
        input_shape="1 x 96 x 96 x 96",
        transform="registered/skull-stripped T1w",
        sources=(path, require_file(directory / "preprocessing_qc.json")),
        foregrounds=tuple(
            ~np.isclose(image, 0.0, rtol=0.0, atol=1e-8) for image in images
        ),
    )


def load_brainsegfounder(base: Path, dataset: str, subject: str) -> Panel:
    directory = base / "preproc_BrainSegFounder" / dataset / subject
    path = require_file(directory / f"{subject}_brainsegfounder_ready.nii.gz")
    data, _ = load_nifti(path)
    if tuple(data.shape) != (96, 96, 96):
        raise ValueError(f"Unexpected BrainSegFounder input shape: {data.shape}")
    images = (sagittal_xyz(data), coronal_xyz(data), axial_xyz(data))
    return Panel(
        model="BrainSegFounder",
        display_name="BrainSeg-\nFounder",
        images=images,
        input_shape="1 x 96 x 96 x 96",
        transform="UKB-derived MNI view",
        sources=(path, require_file(directory / "preprocessing_qc.json")),
        foregrounds=tuple(
            ~np.isclose(image, 0.0, rtol=0.0, atol=1e-8) for image in images
        ),
    )


def reconstruct_neurovfm(
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    payload = torch.load(require_file(cache_path), map_location="cpu", weights_only=True)
    batch = payload.get("batch")
    if not isinstance(batch, dict):
        raise ValueError("NeuroVFM cache lacks its batch mapping")
    tokens = batch["img"].detach().cpu().numpy().astype(np.float32, copy=False)
    coords = batch["coords"].detach().cpu().numpy().astype(np.int64, copy=False)
    shape = tuple(int(value) for value in batch["size"][0])
    if tokens.ndim != 2 or tokens.shape[1] != int(np.prod(PATCH_SHAPE)):
        raise ValueError(f"Unexpected NeuroVFM patch tensor: {tokens.shape}")
    if coords.shape != (tokens.shape[0], 3):
        raise ValueError(f"Unexpected NeuroVFM coordinates: {coords.shape}")
    volume = np.zeros(shape, dtype=np.float32)
    support = np.zeros(shape, dtype=bool)
    reconstructed: list[np.ndarray] = []
    for flattened, coordinate in zip(tokens, coords):
        slices = tuple(
            slice(int(index) * width, (int(index) + 1) * width)
            for index, width in zip(coordinate, PATCH_SHAPE)
        )
        patch = flattened.reshape(PATCH_SHAPE)
        volume[slices] = patch
        support[slices] = True
        reconstructed.append(volume[slices].reshape(-1).copy())
    if not np.array_equal(np.stack(reconstructed), tokens):
        raise RuntimeError("NeuroVFM patch reconstruction is not bit exact")
    return volume, support, batch


def support_center(mask: np.ndarray) -> tuple[int, int, int]:
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        raise ValueError("NeuroVFM retained-patch support is empty")
    center = np.rint(coordinates.mean(axis=0)).astype(int)
    return tuple(
        int(np.clip(value, 0, mask.shape[axis] - 1))
        for axis, value in enumerate(center)
    )


def expected_neurovfm_shape(
    raw_shape: tuple[int, int, int], spacing: tuple[float, float, float], axis: int
) -> tuple[int, int, int]:
    target = np.ones(3, dtype=float)
    target[axis] = 4.0
    resized = np.asarray(
        [int(size * (voxel / desired)) for size, voxel, desired in zip(raw_shape, spacing, target)],
        dtype=int,
    )
    divisors = np.full(3, 16, dtype=int)
    divisors[axis] = 4
    x, y, z = (int(value) for value in (resized // divisors) * divisors)
    return ((x, y, z), (y, z, x), (z, y, x))[axis]


def infer_neurovfm_axis(
    raw_shape: tuple[int, int, int], spacing: tuple[float, float, float], model_shape: tuple[int, int, int]
) -> int:
    matches = [axis for axis in range(3) if expected_neurovfm_shape(raw_shape, spacing, axis) == model_shape]
    if len(matches) == 1:
        return matches[0]
    preferred = 2 if len(set(spacing)) == 1 else int(np.argmax(spacing))
    if not matches or preferred in matches:
        return preferred
    raise ValueError(f"Cannot infer NeuroVFM acquisition axis: matches={matches}, preferred={preferred}")


def neurovfm_dhw_to_ras(array: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        rpi_zyx = np.transpose(array, (2, 1, 0))
    elif axis == 1:
        rpi_zyx = np.transpose(array, (1, 0, 2))
    elif axis == 2:
        rpi_zyx = array
    else:
        raise ValueError(axis)
    return np.flip(np.transpose(rpi_zyx, (2, 1, 0)), axis=(1, 2))


def load_neurovfm(
    base: Path, dataset: str, subject: str, raw_path: Path
) -> Panel:
    directory = base / "preproc_NeuroVFM" / dataset / subject
    cache = require_file(directory / "study_preprocessor.pt")
    sidecar = require_file(directory / "study_preprocessor.json")
    volume, support, batch = reconstruct_neurovfm(cache)
    raw_image = nib.as_closest_canonical(nib.load(str(raw_path)))
    spacing = tuple(float(value) for value in raw_image.header.get_zooms()[:3])
    raw_shape = tuple(int(value) for value in raw_image.shape)
    axis = infer_neurovfm_axis(raw_shape, spacing, tuple(int(value) for value in volume.shape))
    ras = neurovfm_dhw_to_ras(volume, axis)
    support_ras = neurovfm_dhw_to_ras(support, axis)
    center = support_center(support_ras)
    model_spacing = [1.0, 1.0, 1.0]
    model_spacing[axis] = 4.0
    images = (
        sagittal_xyz(ras, center[0]),
        coronal_xyz(ras, center[1]),
        axial_xyz(ras, center[2]),
    )
    support_views = (
        sagittal_xyz(support_ras, center[0]) > 0,
        coronal_xyz(support_ras, center[1]) > 0,
        axial_xyz(support_ras, center[2]) > 0,
    )
    aspects = (
        float(model_spacing[2] / model_spacing[1]),
        float(model_spacing[2] / model_spacing[0]),
        float(model_spacing[1] / model_spacing[0]),
    )
    return Panel(
        model="NeuroVFM",
        display_name="NeuroVFM",
        images=images,
        input_shape=f"{int(batch['img'].shape[0])} x 1024 patches",
        transform="cached retained patches before MRI input normalization",
        sources=(cache, sidecar, raw_path),
        display_aspects=aspects,
        foregrounds=support_views,
        outlines=support_views,
    )


def load_las_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(require_file(path)))
    if image.ndim != 3:
        raise ValueError(f"Curia-2 raw T1w must be 3D, got {image.shape}")
    volume = image.get_fdata(dtype=np.float32)
    current = io_orientation(image.affine)
    target = axcodes2ornt(("L", "A", "S"))
    transformed = np.ascontiguousarray(
        apply_orientation(volume, ornt_transform(current, target)), dtype=np.float32
    )
    spacing_xyz = tuple(float(value) for value in nib.affines.voxel_sizes(image.affine))
    orientation = ornt_transform(current, target)
    reordered_spacing = tuple(spacing_xyz[int(row[0])] for row in orientation)
    return transformed, reordered_spacing


def curia_views(
    raw_path: Path,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[int, int, int, int],
    tuple[float, float, float],
]:
    import torch
    import torch.nn.functional as functional

    volume, spacing = load_las_volume(raw_path)
    axial_stack = np.rot90(
        np.moveaxis(volume, 2, 0), k=1, axes=(1, 2)
    ).copy()
    coronal_rows: list[np.ndarray] = []
    sagittal_rows: list[np.ndarray] = []
    central_index = volume.shape[-1] // 2
    central_axial: np.ndarray | None = None
    for start in range(0, len(axial_stack), 32):
        stop = min(start + 32, len(axial_stack))
        tensor = torch.from_numpy(axial_stack[start:stop])[:, None]
        tensor = functional.interpolate(
            tensor,
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        torch.clamp_min(tensor, -1000.0, out=tensor)
        means = tensor.mean(dim=(2, 3), keepdim=True)
        standard_deviations = tensor.std(dim=(2, 3), keepdim=True)
        centered = tensor - means
        tensor = torch.where(
            standard_deviations < CURIA_PROCESSOR_EPS,
            centered,
            centered / standard_deviations,
        )
        processed = tensor[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        coronal_rows.extend(row.copy() for row in processed[:, 256, :])
        sagittal_rows.extend(row.copy() for row in processed[:, :, 256])
        if start <= central_index < stop:
            central_axial = processed[central_index - start].copy()
    if central_axial is None:
        raise RuntimeError("Curia-2 input did not contain a central axial slice")
    sagittal = np.flipud(np.stack(sagittal_rows))
    coronal = np.flipud(np.stack(coronal_rows))
    sagittal_aspect = float(512.0 * spacing[2] / (volume.shape[1] * spacing[1]))
    coronal_aspect = float(512.0 * spacing[2] / (volume.shape[0] * spacing[0]))
    return (
        (sagittal, coronal, central_axial),
        (volume.shape[-1], 1, 512, 512),
        (sagittal_aspect, coronal_aspect, 1.0),
    )


def load_curia(raw_path: Path) -> Panel:
    images, shape, aspects = curia_views(raw_path)
    return Panel(
        model="Curia-2",
        display_name="Curia-2",
        images=images,
        input_shape=" x ".join(str(value) for value in shape),
        transform="stacked processed axial slices",
        sources=(raw_path,),
        display_aspects=aspects,
    )


def build_panels(data_root: Path, record: dict[str, str], raw_path: Path) -> list[Panel]:
    base = data_root / "preprocessings" / "FineTunningDataset"
    dataset = record["dataset"]
    subject = record["subject"]
    return [
        load_simclr(base, dataset, subject),
        load_anatcl(base, dataset, subject),
        load_brainiac(base, dataset, subject),
        load_brainsegfounder(base, dataset, subject),
        load_curia(raw_path),
        load_neurovfm(base, dataset, subject, raw_path),
        load_yaware(base, dataset, subject),
    ]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
        }
    )


def render(panels: list[Panel], dataset: str, subject: str) -> plt.Figure:
    configure_style()
    figure = plt.figure(figsize=(7.35, 8.25), constrained_layout=False)
    outer = figure.add_gridspec(
        3,
        6,
        left=0.025,
        right=0.985,
        top=0.985,
        bottom=0.03,
        hspace=0.24,
        wspace=0.24,
    )
    placements = ((0, 0), (0, 2), (0, 4), (1, 0), (1, 2), (1, 4), (2, 2))
    letters = "abcdefg"
    grayscale = mpl.colormaps["gray"].copy()
    grayscale.set_bad("black")
    for letter, panel, (row, column) in zip(letters, panels, placements):
        cell = outer[row, column : column + 2].subgridspec(
            2, 1, height_ratios=(0.17, 1.0), hspace=0.035
        )
        header = figure.add_subplot(cell[0, 0])
        header.axis("off")
        header.text(
            0.0,
            0.64,
            letter,
            ha="left",
            va="center",
            fontsize=10.0,
            fontweight="bold",
        )
        header.text(
            0.5,
            0.64,
            panel.display_name,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            linespacing=0.88,
        )
        header.text(
            0.5,
            0.0,
            panel.input_shape,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#444444",
        )

        frame = figure.add_subplot(cell[1, 0])
        frame.set_facecolor("black")
        frame.set_xticks([])
        frame.set_yticks([])
        for spine in frame.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

        views = cell[1, 0].subgridspec(
            2,
            2,
            height_ratios=(1.0, 1.0),
            hspace=0.025,
            wspace=0.025,
        )
        view_specs = (views[0, 0], views[0, 1], views[1, :])
        for plane, image, foreground, outline, aspect, view_spec in zip(
            ("Sagittal", "Coronal", "Axial"),
            panel.images,
            panel.foregrounds,
            panel.outlines,
            panel.display_aspects,
            view_specs,
        ):
            axis = figure.add_subplot(view_spec)
            low, high = robust_limits(image, foreground)
            displayed = (
                np.ma.masked_where(~foreground, image)
                if foreground is not None
                else image
            )
            axis.imshow(
                displayed,
                cmap=grayscale,
                vmin=low,
                vmax=high,
                interpolation="nearest",
                aspect=aspect,
            )
            if outline is not None:
                axis.contour(
                    outline.astype(np.uint8),
                    levels=(0.5,),
                    colors=("#E75B2A",),
                    linewidths=0.7,
                )
            axis.set_facecolor("black")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.text(
                0.96,
                0.96,
                plane,
                transform=axis.transAxes,
                ha="right",
                va="top",
                color="white",
                fontsize=6.5,
                fontweight="bold",
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 1.2},
            )
    figure.text(
        0.5,
        0.008,
        f"Representative central views: {dataset} / {subject}",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#333333",
    )
    return figure


def save_figure(figure: plt.Figure, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "FigureS1_model_inputs.png",
        output_dir / "FigureS1_model_inputs.svg",
    ]
    with tempfile.TemporaryDirectory(prefix="figureS1-", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        for output in outputs:
            candidate = temporary_dir / output.name
            if output.suffix == ".png":
                figure.savefig(candidate, dpi=600, facecolor="white", metadata={"Software": "matplotlib"})
            else:
                figure.savefig(candidate, facecolor="white", metadata={"Date": None})
            os.replace(candidate, output)
    return outputs


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", delete=False, encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-project-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--raw-t1w", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_project_root)
    manifest = args.manifest or (
        data_root
        / "preprocessings"
        / "FineTunningDataset"
        / "brainage_splits"
        / "brainage_definitive_subject_manifest.tsv"
    )
    record = load_subject_record(manifest, args.dataset, args.subject)
    anatomical_reference = resolve_anatomical_reference(data_root, record, args.raw_t1w)
    panels = build_panels(data_root, record, anatomical_reference.path)
    figure = render(panels, record["dataset"], record["subject"])
    outputs = save_figure(figure, args.output_dir.expanduser().resolve())
    plt.close(figure)

    sources: dict[str, list[dict[str, Any]]] = {}
    for panel in panels:
        sources[panel.model] = [
            {"path": str(path), "sha256": sha256_file(path)} for path in panel.sources
        ]
    payload = {
        "figure": "Supplementary Figure S1: model-specific input views",
        "dataset": record["dataset"],
        "subject": record["subject"],
        "manifest": str(require_file(manifest)),
        "manifest_sha256": sha256_file(require_file(manifest)),
        "anatomical_reference": anatomical_reference.provenance,
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "display": {
            "planes": ["central sagittal", "central coronal", "central axial"],
            "layout": (
                "three models, three models, one centered model; each model uses a "
                "black two-over-one triplanar frame"
            ),
            "intensity_scaling": "independent 0.5th--99.5th percentile clipping per view for display only",
            "Curia-2": (
                "coronal visualization reconstructed by stacking the exact processed axial "
                "slices; axial visualization is the exact processed central slice"
            ),
            "NeuroVFM": (
                "sagittal, coronal, and axial visualizations reconstructed bit exactly from "
                "cached retained "
                "foreground patches before the released MRI input normalization"
            ),
        },
        "panels": [
            {
                "model": panel.model,
                "input_shape": panel.input_shape,
                "transform": panel.transform,
                "display_array_shapes": {
                    plane: [int(value) for value in image.shape]
                    for plane, image in zip(
                        ("sagittal", "coronal", "axial"), panel.images
                    )
                },
            }
            for panel in panels
        ],
        "sources": sources,
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in outputs
        ],
    }
    atomic_json(args.provenance.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
