#!/usr/bin/env python3
"""Create 3D-Neuro-SimCLR inputs for the definitive brain-age cohort.

Scientific pipeline (raw T1w -> offline model files):

1. N4 bias-field correction with ANTs (shrink factor 3).
2. SynthStrip skull stripping.
3. Rigid ANTs registration to the 1 mm ICBM152 nonlinear symmetric 2009c grid.
4. Fast SynthSeg segmentation on CPU and binary mask extraction at label >= 2.
5. T1 WhiteStripe intensity normalization using the registered mask.

The public 3D-Neuro-SimCLR release instructs users to run TurboPrep with
``-m t1 -r r``. The commands above reproduce that release contract, including
registering to the complete ICBM152 template named by the release. As a uniform
study safeguard, negative post-SynthStrip intensities are clamped to zero before
registration for every session; sessions without negative values are unchanged.

The fixed transpose/crop to 150 x 192 x 192 and masked per-volume z-score are
performed later by the released dataset loader. They are deliberately not
materialized here, so the same offline image and mask can serve frozen and
adapted downstream arms without duplicating model-input logic.

The definitive brain-age manifest is the default input. Outputs are scoped by
dataset and subject, built in staging directories, validated before publication,
and accompanied by retained intermediates, per-subject logs, QC montages and
cryptographic provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

import nibabel as nib
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
FINETUNING_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = FINETUNING_ROOT.parents[1]

DEFAULT_MANIFEST = (
    FINETUNING_ROOT
    / "brainage_splits"
    / "brainage_definitive_subject_manifest.tsv"
)
DEFAULT_OUTPUT_ROOT = FINETUNING_ROOT / "preproc_3DNeuroSimCLR"
DEFAULT_LOG_ROOT = FINETUNING_ROOT / "logs_preprocs" / "3DNeuroSimCLR"
DEFAULT_TEMPLATE = (
    PROJECT_ROOT
    / "templates"
    / "mni_icbm152_t1_tal_nlin_sym_09c.nii"
)

PIPELINE_VERSION = "3d-neuro-simclr-turboprep-finetuning-v5"
OFFICIAL_3D_NEURO_SIMCLR_COMMIT = "e061f3a19d20755d45361cb206b0803a6de804ff"
OFFICIAL_TURBOPREP_COMMIT = "54a504a5201f26292adf75967c9545b7594bdfaf"
EXPECTED_TEMPLATE_SHA256 = (
    "93359ab97c1c027376397612a9b6c30e95406c15bf8695bd4a8efcb2064eaa34"
)
EXPECTED_TEMPLATE_SHAPE = (193, 229, 193)
EXPECTED_TEMPLATE_SPACING = (1.0, 1.0, 1.0)
EXPECTED_TEMPLATE_AFFINE = np.array(
    [
        [1.0, 0.0, 0.0, -96.0],
        [0.0, 1.0, 0.0, -132.0],
        [0.0, 0.0, 1.0, -78.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
PLOT_LOCK = threading.Lock()

REQUIRED_MANIFEST_COLUMNS = {
    "dataset",
    "subject",
    "raw_t1w_path",
    "age",
    "sex",
    "split",
    "inclusion_status",
    "required_offline_groups",
}

CONTRACT_PAYLOAD = {
    "pipeline_version": PIPELINE_VERSION,
    "official_3d_neuro_simclr_commit": OFFICIAL_3D_NEURO_SIMCLR_COMMIT,
    "official_turboprep_commit": OFFICIAL_TURBOPREP_COMMIT,
    "template": {
        "name": "mni_icbm152_t1_tal_nlin_sym_09c.nii",
        "sha256": EXPECTED_TEMPLATE_SHA256,
        "shape": EXPECTED_TEMPLATE_SHAPE,
        "spacing_mm": EXPECTED_TEMPLATE_SPACING,
        "selection": "complete ICBM152 2009c template named by the public release",
    },
    "n4": {
        "implementation": "ANTs N4BiasFieldCorrection",
        "dimension": 3,
        "shrink_factor": 3,
    },
    "skull_stripping": {
        "implementation": "FreeSurfer mri_synthstrip",
        "negative_voxels": (
            "universally clamp post-SynthStrip intensities below zero to zero"
        ),
        "retain_mask": True,
    },
    "registration": {
        "implementation": "ANTs antsRegistrationSyNQuick.sh",
        "transform": "rigid (-t r)",
        "interpolation": "release default",
    },
    "segmentation": {
        "implementation": "FreeSurfer mri_synthseg --fast",
        "device": "cpu (--cpu), enabling controlled subject-level parallelism",
        "mask": "NumPy (segmentation >= 2), equivalent to mri_threshold -B 1 ... 2",
    },
    "normalization": {
        "implementation": (
            "intensity_normalization.normalize.whitestripe.WhiteStripeNormalize"
        ),
        "modality": "t1",
        "width": 0.05,
        "mask": "registered SynthSeg-derived mask",
    },
    "deferred_model_input": {
        "transpose_crop": "(1,X,Y,Z)->(1,Z,Y,X)[:,10:160,19:211,1:]",
        "shape": (1, 150, 192, 192),
        "normalization": "masked per-volume z-score in released dataset loader",
    },
}
PIPELINE_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(CONTRACT_PAYLOAD, sort_keys=True).encode("utf-8")
).hexdigest()


# Return a timezone-aware UTC timestamp for subject QC and run manifests.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Compute a file digest in chunks so large NIfTI files stay out of Python memory.
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Write JSON through an atomic replacement so interrupted jobs leave no partial QC.
def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


# Write the cohort status table atomically for trustworthy resume accounting.
def atomic_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


# Print one progress message and mirror it into the active per-subject log.
def log_message(message: str, handle: TextIO | None = None) -> None:
    print(message, flush=True)
    if handle is not None:
        handle.write(message + "\n")
        handle.flush()


# Record inexpensive input provenance used to reject stale outputs on resume.
def raw_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


# Load the definitive manifest and enforce schema, inclusion, uniqueness and safety.
def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Definitive subject manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    keys: set[tuple[str, str]] = set()
    raw_paths: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        dataset = row["dataset"].strip()
        subject = row["subject"].strip()
        raw_path = row["raw_t1w_path"].strip()
        if not SAFE_COMPONENT.fullmatch(dataset) or not SAFE_COMPONENT.fullmatch(subject):
            raise ValueError(
                f"Unsafe dataset/subject at line {line_number}: "
                f"dataset={dataset!r}, subject={subject!r}"
            )
        key = (dataset, subject)
        if key in keys:
            raise ValueError(f"Duplicate dataset/subject at line {line_number}: {key}")
        if raw_path in raw_paths:
            raise ValueError(f"Duplicate raw_t1w_path at line {line_number}: {raw_path}")
        if row["inclusion_status"].strip().upper() != "INCLUDED":
            raise ValueError(
                f"Non-included subject in definitive manifest at line {line_number}: {key}"
            )
        groups = {
            value.strip()
            for value in row["required_offline_groups"].split(";")
            if value.strip()
        }
        if "preproc_3d_neuro_simclr" not in groups:
            raise ValueError(f"{key} does not request preproc_3d_neuro_simclr")
        row["dataset"] = dataset
        row["subject"] = subject
        row["raw_t1w_path"] = raw_path
        keys.add(key)
        raw_paths.add(raw_path)
    return rows


# Apply optional cohort selectors and reject unknown or empty selections early.
def select_rows(
    rows: list[dict[str, str]],
    datasets: list[str] | None,
    splits: list[str] | None,
    subjects: list[str] | None,
    limit: int | None,
) -> list[dict[str, str]]:
    selected = rows
    if datasets:
        wanted = set(datasets)
        unknown = wanted - {row["dataset"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown --dataset values: {sorted(unknown)}")
        selected = [row for row in selected if row["dataset"] in wanted]
    if splits:
        wanted = set(splits)
        unknown = wanted - {row["split"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown --split values: {sorted(unknown)}")
        selected = [row for row in selected if row["split"] in wanted]
    if subjects:
        wanted = set(subjects)
        available = {row["subject"] for row in selected}
        unknown = wanted - available
        if unknown:
            raise ValueError(f"Unknown --subject values after filtering: {sorted(unknown)}")
        selected = [row for row in selected if row["subject"] in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No subjects remain after applying selectors")
    return selected


# Construct dataset-scoped retained intermediates, model files and QC paths.
def output_paths(output_root: Path, dataset: str, subject: str) -> dict[str, Path]:
    subject_dir = output_root / dataset / subject
    return {
        "subject_dir": subject_dir,
        "n4": subject_dir / f"{subject}_n4.nii.gz",
        "brain": subject_dir / f"{subject}_brain.nii.gz",
        "native_mask": subject_dir / f"{subject}_synthstrip_mask.nii.gz",
        "registered": subject_dir / f"{subject}_mni.nii.gz",
        "transform": subject_dir / f"{subject}_rigid_0GenericAffine.mat",
        "synthseg": subject_dir / f"{subject}_synthseg.nii.gz",
        "mask_mni": subject_dir / f"{subject}_mask_mni.nii.gz",
        "ready": subject_dir / f"{subject}_simclr_ready.nii.gz",
        "qc": subject_dir / "preprocessing_qc.json",
        "qc_png": subject_dir / "preprocessing_qc.png",
    }


# Verify the exact ON-Harmony atlas variant and capture its immutable geometry.
def validate_template(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"3D-Neuro-SimCLR template not found: {path}")
    digest = sha256_file(path)
    if digest != EXPECTED_TEMPLATE_SHA256:
        raise ValueError(
            f"Unexpected template SHA-256 for {path}: {digest}; "
            f"expected {EXPECTED_TEMPLATE_SHA256}"
        )
    image = nib.load(str(path))
    shape = tuple(int(value) for value in image.shape)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    affine = np.asarray(image.affine, dtype=np.float64)
    if shape != EXPECTED_TEMPLATE_SHAPE:
        raise ValueError(f"Unexpected template shape: {shape}, expected {EXPECTED_TEMPLATE_SHAPE}")
    if not np.allclose(spacing, EXPECTED_TEMPLATE_SPACING, rtol=0.0, atol=1e-6):
        raise ValueError(f"Unexpected template spacing: {spacing}")
    if not np.allclose(affine, EXPECTED_TEMPLATE_AFFINE, rtol=0.0, atol=1e-6):
        raise ValueError(f"Unexpected template affine for {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest,
        "shape": list(shape),
        "spacing_mm": list(spacing),
        "affine": affine.tolist(),
    }


# Resolve an explicit executable path or a command on PATH before cohort work begins.
def resolve_executable(command: str) -> str:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or "/" in command:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FileNotFoundError(f"Executable is not runnable: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"Executable not found on PATH: {command}")
    return resolved


# Run one external release command while preserving its complete stdout and stderr.
def run_external(
    command: list[str],
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> None:
    log_handle.write("command=" + shlex.join(command) + "\n")
    log_handle.flush()
    completed = subprocess.run(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {shlex.join(command)}"
        )


# Reproduce TurboPrep's N4 command with its default shrink factor of three.
def n4_bias_correction(
    input_path: Path,
    output_path: Path,
    executable: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> list[str]:
    command = [
        executable,
        "-d", "3",
        "-i", str(input_path),
        "-o", str(output_path),
        "-s", "3",
        "-v",
    ]
    run_external(command, log_handle, timeout_seconds)
    if not output_path.is_file():
        raise RuntimeError(f"N4 did not create {output_path}")
    return command


# Run SynthStrip on CPU or GPU and retain both the stripped image and native mask.
def skull_strip_synthstrip(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    executable: str,
    device: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> list[str]:
    command = [
        executable,
        "-i", str(input_path),
        "-o", str(output_path),
        "-m", str(mask_path),
    ]
    if device == "cuda":
        command.append("--gpu")
    run_external(command, log_handle, timeout_seconds)
    if not output_path.is_file() or not mask_path.is_file():
        raise RuntimeError(f"SynthStrip did not create {output_path} and {mask_path}")
    return command


# Apply the same value-based safeguard to every session before registration.
def clamp_negative_voxels(
    input_path: Path,
    log_handle: TextIO,
) -> dict[str, Any]:
    image = nib.load(str(input_path))
    data = image.get_fdata(dtype=np.float32)
    negative_voxels = int(np.count_nonzero(data < 0))
    minimum_before = float(np.min(data))

    if negative_voxels:
        data = np.maximum(data, 0.0)
        temporary = input_path.with_name(
            f".{input_path.name}.{uuid.uuid4().hex}.tmp.nii.gz"
        )
        try:
            nib.save(
                nib.Nifti1Image(data, image.affine, image.header.copy()),
                str(temporary),
            )
            os.replace(temporary, input_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    log_message(
        f"    negative voxels clamped to zero: {negative_voxels}",
        log_handle,
    )
    return {
        "operation": "maximum(image, 0) after SynthStrip",
        "scope": "applied uniformly to every session",
        "negative_voxels": negative_voxels,
        "minimum_before": minimum_before,
        "minimum_after": max(minimum_before, 0.0),
    }


# Run the exact rigid ANTs wrapper requested by the 3D-Neuro-SimCLR release.
def rigid_register_ants(
    input_path: Path,
    template_path: Path,
    output_path: Path,
    transform_path: Path,
    executable: str,
    threads: int,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> list[str]:
    prefix = output_path.parent / f".{output_path.name}.ants_"
    command = [
        executable,
        "-d", "3",
        "-f", str(template_path),
        "-m", str(input_path),
        "-o", str(prefix),
        "-n", str(threads),
        "-t", "r",
    ]
    run_external(command, log_handle, timeout_seconds)
    warped = Path(str(prefix) + "Warped.nii.gz")
    affine = Path(str(prefix) + "0GenericAffine.mat")
    if not warped.is_file() or not affine.is_file():
        raise RuntimeError(
            f"ANTs did not create the expected warped image and affine: {warped}, {affine}"
        )
    os.replace(warped, output_path)
    os.replace(affine, transform_path)
    inverse = Path(str(prefix) + "InverseWarped.nii.gz")
    if inverse.exists():
        inverse.unlink()
    return command


# Run fast SynthSeg on CPU, matching TurboPrep step three while controlling parallelism.
def run_synthseg(
    input_path: Path,
    output_path: Path,
    executable: str,
    threads: int,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> list[str]:
    command = [
        executable,
        "--i", str(input_path),
        "--o", str(output_path),
        "--fast",
        "--cpu",
        "--threads", str(threads),
    ]
    run_external(command, log_handle, timeout_seconds)
    if not output_path.is_file():
        raise RuntimeError(f"SynthSeg did not create {output_path}")
    return command


# Derive the same binary mask as ``mri_threshold -B 1 ... 2`` using NumPy.
def derive_registered_mask(
    segmentation_path: Path,
    output_path: Path,
    log_handle: TextIO,
) -> dict[str, Any]:
    segmentation = nib.load(str(segmentation_path))
    mask_data = (segmentation.get_fdata() >= 2).astype(np.uint8)
    nib.save(
        nib.Nifti1Image(mask_data, segmentation.affine, segmentation.header),
        str(output_path),
    )
    if not output_path.is_file():
        raise RuntimeError(f"NumPy mask derivation did not create {output_path}")
    foreground_voxels = int(mask_data.sum())
    log_message(
        f"    mask_implementation=NumPy segmentation>=2; "
        f"foreground_voxels={foreground_voxels}",
        log_handle,
    )
    return {
        "implementation": "NumPy",
        "expression": "segmentation >= 2",
        "foreground_voxels": foreground_voxels,
    }


# Apply the same WhiteStripe class used by the ON-Harmony preprocessing script.
def whitestripe_normalize(
    input_path: Path,
    mask_path: Path,
    output_path: Path,
    log_handle: TextIO,
) -> dict[str, Any]:
    try:
        from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
    except ImportError as exc:
        raise RuntimeError(
            "The intensity-normalization package with WhiteStripeNormalize is required"
        ) from exc

    image = nib.load(str(input_path))
    image_data = image.get_fdata().astype(np.float32)
    mask_data = nib.load(str(mask_path)).get_fdata().astype(bool)
    normalized_data = WhiteStripeNormalize()(image_data, mask_data)
    nib.save(
        nib.Nifti1Image(normalized_data, image.affine, image.header),
        str(output_path),
    )
    if not output_path.is_file():
        raise RuntimeError(f"WhiteStripe did not create {output_path}")
    log_message(
        "    whitestripe_implementation=WhiteStripeNormalize; modality=t1; width=0.05",
        log_handle,
    )
    return {
        "implementation": (
            "intensity_normalization.normalize.whitestripe.WhiteStripeNormalize"
        ),
        "modality": "t1",
        "width": 0.05,
    }


# Compare NIfTI shape and affine with strict tolerances suitable for a fixed atlas grid.
def same_grid(image: nib.spatialimages.SpatialImage, reference: nib.spatialimages.SpatialImage) -> bool:
    return (
        tuple(image.shape) == tuple(reference.shape)
        and np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-5)
        and np.allclose(
            image.header.get_zooms()[:3],
            reference.header.get_zooms()[:3],
            rtol=0.0,
            atol=1e-6,
        )
    )


# Validate every retained NIfTI and summarize geometry, range and mask integrity.
def validate_outputs(paths: dict[str, Path], template_path: Path) -> dict[str, Any]:
    template = nib.load(str(template_path))
    images: dict[str, nib.spatialimages.SpatialImage] = {}
    statistics: dict[str, Any] = {}
    for key in ("n4", "brain", "native_mask", "registered", "synthseg", "mask_mni", "ready"):
        path = paths[key]
        if not path.is_file():
            raise FileNotFoundError(f"Required retained output not found: {path}")
        image = nib.load(str(path))
        data = image.get_fdata(dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI for {key}, got shape {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError(f"Non-finite voxels in {path}")
        if key in {"registered", "synthseg", "mask_mni", "ready"} and not same_grid(image, template):
            raise ValueError(f"{key} does not match the fixed ICBM152 grid: {path}")
        nonzero = int(np.count_nonzero(data))
        if nonzero == 0:
            raise ValueError(f"All-zero image: {path}")
        images[key] = image
        statistics[key] = {
            "path": str(path),
            "shape": [int(value) for value in data.shape],
            "spacing_mm": [float(value) for value in image.header.get_zooms()[:3]],
            "dtype_on_disk": str(image.get_data_dtype()),
            "nonzero_voxels": nonzero,
            "minimum": float(np.min(data)),
            "maximum": float(np.max(data)),
        }

    if not same_grid(images["brain"], images["n4"]):
        raise ValueError("SynthStrip brain grid differs from the N4 input grid")
    if not same_grid(images["native_mask"], images["n4"]):
        raise ValueError("SynthStrip mask grid differs from the N4 input grid")

    for key in ("native_mask", "mask_mni"):
        mask = images[key].get_fdata(dtype=np.float32)
        unique = np.unique(mask)
        if not set(float(value) for value in unique).issubset({0.0, 1.0}):
            raise ValueError(f"{key} is not binary: values={unique[:20]}")
        fraction = float(np.count_nonzero(mask) / mask.size)
        if not 0.01 < fraction < 0.80:
            raise ValueError(f"Implausible {key} foreground fraction: {fraction:.6f}")
        statistics[key]["unique_values"] = [float(value) for value in unique]
        statistics[key]["foreground_fraction"] = fraction

    if not paths["transform"].is_file() or paths["transform"].stat().st_size == 0:
        raise FileNotFoundError(f"Rigid transform is absent or empty: {paths['transform']}")

    ready = images["ready"].get_fdata(dtype=np.float32)
    mask = images["mask_mni"].get_fdata(dtype=np.float32) > 0
    values = ready[mask]
    if values.size < 1000 or float(np.std(values)) <= 1e-6:
        raise ValueError("WhiteStripe output has empty or constant masked foreground")
    statistics["ready"]["masked_mean"] = float(np.mean(values))
    statistics["ready"]["masked_standard_deviation"] = float(np.std(values))
    statistics["ready"]["masked_median"] = float(np.median(values))
    return statistics


# Create a compact montage of the stripped, registered and normalized images.
def write_qc_montage(paths: dict[str, Path], destination: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volumes = [
        ("SynthStrip brain", np.asarray(nib.load(str(paths["brain"])).dataobj)),
        ("Rigid ICBM152", np.asarray(nib.load(str(paths["registered"])).dataobj)),
        ("WhiteStripe", np.asarray(nib.load(str(paths["ready"])).dataobj)),
    ]

    def orthogonal_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers = tuple(size // 2 for size in volume.shape)
        return (
            volume[centers[0], :, :],
            volume[:, centers[1], :],
            volume[:, :, centers[2]],
        )

    # Matplotlib maintains process-global state; serialize only this short QC step.
    with PLOT_LOCK:
        figure, axes = plt.subplots(3, 3, figsize=(9, 8))
        for row_index, (label, volume) in enumerate(volumes):
            for column_index, section in enumerate(orthogonal_slices(volume)):
                axes[row_index, column_index].imshow(np.rot90(section), cmap="gray")
                axes[row_index, column_index].set_axis_off()
                if column_index == 0:
                    axes[row_index, column_index].set_title(label, loc="left", fontsize=9)
        figure.suptitle(title, fontsize=10)
        figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=100, bbox_inches="tight")
        plt.close(figure)


# Capture package versions and the resolved release executables used in a run.
def software_versions(executables: dict[str, str]) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": os.sys.version.split()[0],
        "executables": executables,
        "FREESURFER_HOME": os.environ.get("FREESURFER_HOME"),
    }
    for package in ("nibabel", "numpy", "intensity-normalization", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    return versions


# Require hashes and provenance for all published artifacts before reusing a subject.
def qc_matches(paths: dict[str, Path], raw: Path, template: dict[str, Any]) -> bool:
    if not paths["qc"].is_file():
        return False
    required = (
        "n4",
        "brain",
        "native_mask",
        "registered",
        "transform",
        "synthseg",
        "mask_mni",
        "ready",
        "qc_png",
    )
    if any(not paths[key].is_file() for key in required):
        return False
    try:
        qc = json.loads(paths["qc"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    hashes = qc.get("output_sha256", {})
    return (
        qc.get("status") == "PASS"
        and qc.get("pipeline_contract_sha256") == PIPELINE_CONTRACT_SHA256
        and qc.get("template") == template
        and qc.get("raw_input") == raw_fingerprint(raw)
        and all(hashes.get(key) == sha256_file(paths[key]) for key in required)
    )


# Publish staged scientific artifacts only after every validation has passed.
def publish_staged_outputs(staged: dict[str, Path], final: dict[str, Path]) -> None:
    keys = (
        "n4",
        "brain",
        "native_mask",
        "registered",
        "transform",
        "synthseg",
        "mask_mni",
        "ready",
    )
    for key in keys:
        final[key].parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged[key], final[key])


# Run or validate the complete offline contract for one manifest subject.
def process_subject(
    row: dict[str, str],
    output_root: Path,
    log_root: Path,
    template_path: Path,
    template: dict[str, Any],
    executables: dict[str, str],
    synthstrip_device: str,
    threads: int,
    run_id: str,
    force: bool,
    validate_existing: bool,
    timeout_seconds: float | None,
    versions: dict[str, Any],
) -> dict[str, Any]:
    dataset = row["dataset"]
    subject = row["subject"]
    raw = Path(row["raw_t1w_path"])
    paths = output_paths(output_root, dataset, subject)
    subject_log_dir = log_root / dataset
    subject_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = subject_log_dir / f"{subject}_{run_id}.log"
    historical_qc = subject_log_dir / f"{subject}_{run_id}_qc.json"
    historical_png = subject_log_dir / f"{subject}_{run_id}_qc.png"
    record: dict[str, Any] = {
        "dataset": dataset,
        "subject": subject,
        "split": row["split"],
        "age": row["age"],
        "sex": row["sex"],
        "raw_t1w_path": str(raw),
        "output_path": str(paths["ready"]),
        "mask_path": str(paths["mask_mni"]),
        "qc_path": str(paths["qc"]),
        "log_path": str(log_path),
        "started_utc": utc_now(),
        "finished_utc": "",
        "status": "",
        "message": "",
    }
    stage_dir: Path | None = None

    try:
        if not raw.is_file():
            raise FileNotFoundError(f"Raw T1w input not found: {raw}")

        if validate_existing:
            statistics = validate_outputs(paths, template_path)
            if not qc_matches(paths, raw, template):
                raise RuntimeError(f"Outputs are valid but provenance QC is stale: {paths['qc']}")
            record.update(
                status="VALIDATED",
                message=(
                    f"{tuple(statistics['ready']['shape'])}; WhiteStripe masked mean="
                    f"{statistics['ready']['masked_mean']:.4f}, std="
                    f"{statistics['ready']['masked_standard_deviation']:.4f}"
                ),
            )
            return record

        scientific_keys = (
            "n4",
            "brain",
            "native_mask",
            "registered",
            "transform",
            "synthseg",
            "mask_mni",
            "ready",
            "qc",
        )
        existing = [key for key in scientific_keys if paths[key].exists()]
        if existing and not force:
            statistics = validate_outputs(paths, template_path)
            if not qc_matches(paths, raw, template):
                raise RuntimeError(
                    "Existing 3D-Neuro-SimCLR files have stale or incomplete provenance. "
                    "Inspect them and rerun with --force if replacement is intended."
                )
            record.update(
                status="SKIPPED",
                message=f"Current validated output already exists; shape={tuple(statistics['ready']['shape'])}",
            )
            return record

        paths["subject_dir"].mkdir(parents=True, exist_ok=True)
        stage_dir = paths["subject_dir"] / f".staging_{run_id}_{uuid.uuid4().hex[:8]}"
        stage_dir.mkdir(parents=True, exist_ok=False)
        staged = {
            key: stage_dir / paths[key].name
            for key in (
                "n4",
                "brain",
                "native_mask",
                "registered",
                "transform",
                "synthseg",
                "mask_mni",
                "ready",
            )
        }

        with log_path.open("w", encoding="utf-8") as log_handle:
            log_message(f"pipeline_version={PIPELINE_VERSION}", log_handle)
            log_message(f"pipeline_contract_sha256={PIPELINE_CONTRACT_SHA256}", log_handle)
            log_message(f"dataset={dataset}", log_handle)
            log_message(f"subject={subject}", log_handle)
            log_message(f"raw={raw}", log_handle)
            log_message(f"template={template_path}", log_handle)

            log_message("  [Preprocessing 1/5] ANTs N4 (shrink factor 3)", log_handle)
            n4_command = n4_bias_correction(
                raw,
                staged["n4"],
                executables["n4"],
                log_handle,
                timeout_seconds,
            )

            log_message("  [Preprocessing 2/5] SynthStrip skull stripping", log_handle)
            synthstrip_command = skull_strip_synthstrip(
                staged["n4"],
                staged["brain"],
                staged["native_mask"],
                executables["synthstrip"],
                synthstrip_device,
                log_handle,
                timeout_seconds,
            )
            clamp_operation = clamp_negative_voxels(staged["brain"], log_handle)

            log_message("  [Preprocessing 3/5] Rigid ANTs registration to ICBM152 2009c", log_handle)
            registration_command = rigid_register_ants(
                staged["brain"],
                template_path,
                staged["registered"],
                staged["transform"],
                executables["registration"],
                threads,
                log_handle,
                timeout_seconds,
            )

            log_message("  [Preprocessing 4/5] Fast SynthSeg + mask labels >= 2", log_handle)
            synthseg_command = run_synthseg(
                staged["registered"],
                staged["synthseg"],
                executables["synthseg"],
                threads,
                log_handle,
                timeout_seconds,
            )
            mask_operation = derive_registered_mask(
                staged["synthseg"],
                staged["mask_mni"],
                log_handle,
            )

            log_message("  [Preprocessing 5/5] T1 WhiteStripe normalization", log_handle)
            whitestripe_operation = whitestripe_normalize(
                staged["registered"],
                staged["mask_mni"],
                staged["ready"],
                log_handle,
            )
            statistics = validate_outputs(staged, template_path)

            publish_staged_outputs(staged, paths)
            write_qc_montage(
                paths,
                paths["qc_png"],
                f"{dataset}/{subject} - 3D-Neuro-SimCLR PASS",
            )
            hashed_keys = (
                "n4",
                "brain",
                "native_mask",
                "registered",
                "transform",
                "synthseg",
                "mask_mni",
                "ready",
                "qc_png",
            )
            output_hashes = {key: sha256_file(paths[key]) for key in hashed_keys}
            qc = {
                "status": "PASS",
                "created_utc": utc_now(),
                "pipeline_version": PIPELINE_VERSION,
                "pipeline_contract_sha256": PIPELINE_CONTRACT_SHA256,
                "contract": CONTRACT_PAYLOAD,
                "script_path": str(SCRIPT_PATH),
                "script_sha256": sha256_file(SCRIPT_PATH),
                "dataset": dataset,
                "subject": subject,
                "split": row["split"],
                "raw_input": raw_fingerprint(raw),
                "template": template,
                "software": versions,
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
                    )
                },
                "synthstrip_device": synthstrip_device,
                "commands": {
                    "n4": n4_command,
                    "synthstrip": synthstrip_command,
                    "registration": registration_command,
                    "synthseg": synthseg_command,
                },
                "python_operations": {
                    "post_synthstrip_clamp": clamp_operation,
                    "mask": mask_operation,
                    "whitestripe": whitestripe_operation,
                },
                "outputs": statistics,
                "output_sha256": output_hashes,
            }
            atomic_json(paths["qc"], qc)
            atomic_json(historical_qc, qc)
            shutil.copy2(paths["qc_png"], historical_png)
            log_message(f"  [QC] PASS: {paths['qc']}", log_handle)

        record.update(
            status="PASS",
            message=(
                f"{tuple(statistics['ready']['shape'])}; WhiteStripe masked mean="
                f"{statistics['ready']['masked_mean']:.4f}, std="
                f"{statistics['ready']['masked_standard_deviation']:.4f}; "
                "all intermediates retained"
            ),
        )
        if stage_dir.exists():
            stage_dir.rmdir()
    except Exception as error:  # Continue the cohort while preserving diagnostics.
        record.update(status="FAILED", message=f"{type(error).__name__}: {error}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n=== PIPELINE FAILURE ===\n")
            handle.write(traceback.format_exc())
        if stage_dir is not None and stage_dir.exists():
            failed_dir = stage_dir.with_name(stage_dir.name.replace(".staging_", ".failed_", 1))
            try:
                os.replace(stage_dir, failed_dir)
                record["message"] += f"; failed work retained at {failed_dir}"
            except OSError:
                pass
    finally:
        record["finished_utc"] = utc_now()
    return record


# Set process-wide thread limits inherited by ANTs, FreeSurfer and numerical libraries.
def configure_threads(threads_per_worker: int) -> None:
    if threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be positive")
    value = str(threads_per_worker)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        os.environ[variable] = value


# Build the CLI used for one-subject pilots, validation and resumable cohort runs.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TurboPrep-aligned 3D-Neuro-SimCLR preprocessing for brain age"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--subject", action="append", dest="subjects")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--synthstrip-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n4-command", default="N4BiasFieldCorrection")
    parser.add_argument("--synthstrip-command", default="mri_synthstrip")
    parser.add_argument("--registration-command", default="antsRegistrationSyNQuick.sh")
    parser.add_argument("--synthseg-command", default="mri_synthseg")
    parser.add_argument("--step-timeout-seconds", type=float)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


# Validate global inputs, execute a bounded subject pool and write run-level records.
def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.force and args.validate_existing:
        raise ValueError("--force and --validate-existing are mutually exclusive")
    if args.synthstrip_device == "cuda" and args.workers != 1:
        raise ValueError("GPU SynthStrip requires --workers 1 to avoid concurrent model copies")

    configure_threads(args.threads_per_worker)
    manifest_path = args.manifest.resolve()
    rows = load_manifest(manifest_path)
    selected = select_rows(rows, args.datasets, args.splits, args.subjects, args.limit)
    template_path = args.template.resolve()
    template = validate_template(template_path)

    requested_commands = {
        "n4": args.n4_command,
        "synthstrip": args.synthstrip_command,
        "registration": args.registration_command,
        "synthseg": args.synthseg_command,
    }
    executables = (
        requested_commands
        if args.dry_run or args.validate_existing
        else {name: resolve_executable(command) for name, command in requested_commands.items()}
    )
    versions = software_versions(executables)
    output_root = args.output_root.resolve()
    log_root = args.log_root.resolve()

    print("3D-Neuro-SimCLR preprocessing for the definitive brain-age cohort")
    print(f"  manifest : {manifest_path}")
    print(f"  selected : {len(selected)}/{len(rows)}")
    print(f"  outputs  : {output_root}")
    print(f"  logs     : {log_root}")
    print(f"  template : {template_path}")
    print(f"  workers  : {args.workers}")
    print(f"  threads  : {args.threads_per_worker} per worker")
    print(f"  SynthStrip: {args.synthstrip_device}")
    print(f"  contract : {PIPELINE_CONTRACT_SHA256}")
    print("  retained : N4, brain/mask, rigid transform, registered, SynthSeg/mask, final, log and QC")

    if args.dry_run:
        for row in selected[:20]:
            paths = output_paths(output_root, row["dataset"], row["subject"])
            print(
                f"  DRY-RUN {row['dataset']}/{row['subject']}: "
                f"{row['raw_t1w_path']} -> {paths['ready']}"
            )
        if len(selected) > 20:
            print(f"  ... {len(selected) - 20} additional subjects")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    log_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_subject,
                row,
                output_root,
                log_root,
                template_path,
                template,
                executables,
                args.synthstrip_device,
                args.threads_per_worker,
                run_id,
                args.force,
                args.validate_existing,
                args.step_timeout_seconds,
                versions,
            ): row
            for row in selected
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(
                f"  [{completed_count}/{len(selected)}] {record['status']:<9} "
                f"{record['dataset']}/{record['subject']}: {record['message']}",
                flush=True,
            )

    order = {(row["dataset"], row["subject"]): index for index, row in enumerate(selected)}
    records.sort(key=lambda record: order[(record["dataset"], record["subject"])])
    fields = [
        "dataset",
        "subject",
        "split",
        "age",
        "sex",
        "raw_t1w_path",
        "output_path",
        "mask_path",
        "qc_path",
        "log_path",
        "started_utc",
        "finished_utc",
        "status",
        "message",
    ]
    run_manifest = log_root / f"_3dneurosimclr_run_{run_id}.tsv"
    run_summary = log_root / f"_3dneurosimclr_run_{run_id}.json"
    atomic_tsv(run_manifest, records, fields)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    atomic_json(
        run_summary,
        {
            "created_utc": utc_now(),
            "run_id": run_id,
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_contract_sha256": PIPELINE_CONTRACT_SHA256,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "template": template,
            "output_root": str(output_root),
            "log_root": str(log_root),
            "workers": args.workers,
            "threads_per_worker": args.threads_per_worker,
            "synthstrip_device": args.synthstrip_device,
            "selected": len(selected),
            "counts": counts,
            "records_tsv": str(run_manifest),
        },
    )
    print(f"Run manifest: {run_manifest}")
    print(f"Run summary : {run_summary}")
    print(f"Summary     : {counts}")
    if counts.get("FAILED", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
