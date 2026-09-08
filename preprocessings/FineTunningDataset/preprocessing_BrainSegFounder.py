#!/usr/bin/env python3
"""Create BrainSegFounder inputs for the definitive brain-age cohort.

The spatial chain adapts each source T1w to the orientation assumed by the
released UK Biobank structural pipeline and then reproduces that pipeline
through its final masked ``T1_brain_to_MNI`` derivative, using the explicit
``GDC_off`` path from the ON-Harmony fork:

1. Lossless axis reorientation with ``fslreorient2std`` so the UKB z-axis
   crop is valid across source cohorts with heterogeneous storage orders.
2. ``robustfov``-derived 170-slice head crop and recursive FSL BET.
3. ``standard_space_roi`` reduction against the MNI152 1-mm brain.
4. UKB matrix composition and nonlinear FNIRT with ``bb_fnirt.cnf`` and the
   released dilated reference mask. If the released translation-only FLIRT
   step yields an empty MNI image, the crop-to-raw transform is recovered
   deterministically from the NIfTI geometry and FNIRT is repeated.
5. Composition of the reduced-T1-to-MNI warp with the reoriented-to-reduced affine,
   spline resampling of the original T1w, and the released UKB MNI brain mask.
6. RAS orientation, foreground crop, nonzero channel-wise z-score, padding,
   and trilinear resize to 96 x 96 x 96.

Gradient-distortion correction is the only unavailable UKB spatial step,
because the source cohorts do not provide the scanner-specific coefficient
files. The final fixed-size whole-volume view remains a study-defined
deployment view: Stage-1 pretraining used random 96-cube patches, whereas the
public downstream validation transform supplies nonzero normalization.

All intermediates are retained. Subjects are built in staging directories,
validated before publication, and accompanied by per-subject logs, QC
montages and cryptographic provenance.
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
DEFAULT_OUTPUT_ROOT = FINETUNING_ROOT / "preproc_BrainSegFounder"
DEFAULT_LOG_ROOT = FINETUNING_ROOT / "logs_preprocs" / "BrainSegFounder"
DEFAULT_UKB_ASSET_DIR = PROJECT_ROOT / "templates" / "UKBiobank"

PIPELINE_VERSION = "brainsegfounder-ukb-gdcoff-finetuning-v4"
OFFICIAL_BRAINSEGFOUNDER_COMMIT = "0bbf43282447b8f998e4ed47afe26b84e8f05419"
UKB_PIPELINE_COMMIT = "9458b42e23c3476cc9bd4b6a0ae60e1df3807e55"
UKB_ONHARMONY_FORK_COMMIT = "8a86d328c671b6496fd54a28b4905a2d0898e9ab"
EXPECTED_MNI_SHAPE = (182, 218, 182)
EXPECTED_MNI_SPACING = (1.0, 1.0, 1.0)
EXPECTED_READY_SHAPE = (96, 96, 96)
UKB_ROI_DEPTH = 170
EXPECTED_UKB_HASHES = {
    "fnirt_config": "41e2f1b068e4841c8f6d7934375651fbe2cf2e66ac40da3d21f3fbb9b9576e37",
    "mask": "4e599d7d0bd843251bb24391b0ddc636b73216f12a9ccc017a34cb0d17e287e4",
    "dilated_mask": "1365d6747d6878501f97551ce44f143f1662b9dce3ab3f362243e5a456c5725d",
}
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")

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
    "official_brainsegfounder_commit": OFFICIAL_BRAINSEGFOUNDER_COMMIT,
    "ukb_pipeline_commit": UKB_PIPELINE_COMMIT,
    "ukb_onharmony_fork_commit": UKB_ONHARMONY_FORK_COMMIT,
    "scope": "released UK Biobank structural T1 chain with explicit GDC_off",
    "input_orientation_adapter": {
        "implementation": "FSL fslreorient2std",
        "purpose": "make the UKB 170-slice voxel-z crop valid across source storage orders",
        "interpolation": "none; axis permutation and 90-degree rotations only",
        "status": "study-required adapter before the released UKB chain",
    },
    "gradient_distortion_correction": {
        "applied": False,
        "reason": "scanner-specific gradient coefficient files unavailable",
    },
    "field_of_view": {
        "robustfov": True,
        "roi_depth_slices": UKB_ROI_DEPTH,
        "brain_extraction": "FSL bet -R",
        "reduction": "standard_space_roi -maskNONE -altinput raw -d",
    },
    "linear_registration": {
        "primary": "released FSL flirt xyztrans.sch plus convert_xfm composition",
        "recovery_trigger": (
            "fewer than 1000 foreground voxels in the masked MNI image before resize"
        ),
        "recovery": (
            "FSL flirt -applyxfm -usesqform crop-to-raw geometry followed by the "
            "same convert_xfm composition and FNIRT chain"
        ),
        "fixed": "MNI152_T1_1mm_brain",
    },
    "nonlinear_registration": {
        "implementation": "FSL fnirt",
        "moving": "UKB reduced-FOV T1w",
        "fixed": "MNI152_T1_1mm head",
        "config": "released bb_fnirt.cnf",
        "reference_mask": "released MNI152_T1_1mm_brain_mask_dil_GD7",
        "interpolation": "spline",
    },
    "warp_composition": (
        "convertwarp --premat raw-to-reduced --warp2 reduced-to-MNI; "
        "applywarp --rel --interp=spline to the original T1w"
    ),
    "brain_mask": "released UKB MNI152_T1_1mm_brain_mask applied after warping",
    "model_view": {
        "orientation": "RAS",
        "order": [
            "CropForeground(nonzero,k_divisible=96^3)",
            "NormalizeIntensity(nonzero=True,channel_wise=True)",
            "SpatialPad(96^3)",
            "Resize(96^3,trilinear)",
        ],
        "shape": EXPECTED_READY_SHAPE,
        "stored_affine": "MONAI transformed affine",
        "status": "study-defined fixed whole-volume deployment view",
    },
    "unavailable_ukb_steps": ["gradient-distortion correction"],
}
PIPELINE_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(CONTRACT_PAYLOAD, sort_keys=True).encode("utf-8")
).hexdigest()

# Version 4 only adds a conditional recovery after the complete v3 primary
# branch has produced an empty MNI image. A validated v3 PASS therefore
# remains reusable without relabelling or recomputing its artifacts.
REUSABLE_PIPELINE_CONTRACTS = {
    "brainsegfounder-ukb-gdcoff-finetuning-v3": (
        "9aa88a3f0ee244725b6d6326cb77b5996062250bfc3513f0ce0140957fb23e3e"
    ),
    PIPELINE_VERSION: PIPELINE_CONTRACT_SHA256,
}


class InsufficientForegroundError(RuntimeError):
    """Signal a catastrophic registration output eligible for one recovery attempt."""


def reusable_pipeline_qc(qc: dict[str, Any]) -> bool:
    """Accept only a pinned v3 primary result or the current v4 result."""

    version = qc.get("pipeline_version")
    contract = qc.get("pipeline_contract_sha256")
    return (
        version in REUSABLE_PIPELINE_CONTRACTS
        and REUSABLE_PIPELINE_CONTRACTS[version] == contract
    )


# Return a timezone-aware UTC timestamp for logs and provenance records.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Hash large artifacts incrementally so they do not need to be loaded into memory.
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Write JSON via atomic replacement so interrupted jobs leave no partial sidecar.
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


# Write the run table atomically for reliable resume and failure accounting.
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


# Print progress and mirror it into the current subject log when available.
def log_message(message: str, handle: TextIO | None = None) -> None:
    print(message, flush=True)
    if handle is not None:
        handle.write(message + "\n")
        handle.flush()


# Record inexpensive raw-file provenance used to determine whether outputs are current.
def raw_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


# Load the definitive cohort and enforce schema, uniqueness and model requirements.
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
        if "preproc_brainsegfounder" not in groups:
            raise ValueError(f"{key} does not request preproc_brainsegfounder")
        row["dataset"] = dataset
        row["subject"] = subject
        row["raw_t1w_path"] = raw_path
        keys.add(key)
        raw_paths.add(raw_path)
    return rows


# Apply optional selectors and reject unknown or empty cohort subsets.
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


# Define dataset-scoped retained intermediates, final input and QC artifacts.
def output_paths(output_root: Path, dataset: str, subject: str) -> dict[str, Path]:
    subject_dir = output_root / dataset / subject
    return {
        "subject_dir": subject_dir,
        "reoriented_t1": subject_dir / f"{subject}_ukb_reoriented.nii.gz",
        "fov_roi": subject_dir / f"{subject}_ukb_fov.nii.gz",
        "fov_brain": subject_dir / f"{subject}_ukb_fov_brain.nii.gz",
        "reduced_t1": subject_dir / f"{subject}_ukb_T1.nii.gz",
        "roi_to_mni_mat": subject_dir / f"{subject}_roi_to_mni.mat",
        "t1_to_raw_mat": subject_dir / f"{subject}_T1_to_raw.mat",
        "raw_to_t1_mat": subject_dir / f"{subject}_raw_to_T1.mat",
        "linear_mat": subject_dir / f"{subject}_T1_to_MNI_linear.mat",
        "warp_coef": subject_dir / f"{subject}_T1_to_MNI_warp_coef.nii.gz",
        "warp_field": subject_dir / f"{subject}_T1_to_MNI_warp.nii.gz",
        "warp_jac": subject_dir / f"{subject}_T1_to_MNI_warp_jac.nii.gz",
        "fnirt_preview": subject_dir / f"{subject}_T1_fnirt_preview.nii.gz",
        "fnirt_log": subject_dir / f"{subject}_T1_to_MNI_fnirt.log",
        "combined_warp": subject_dir / f"{subject}_T1_orig_to_MNI_warp.nii.gz",
        "warped_unmasked": subject_dir / f"{subject}_T1_to_MNI_unmasked.nii.gz",
        "fnirt_brain": subject_dir / f"{subject}_fnirt_brain.nii.gz",
        "ready": subject_dir / f"{subject}_brainsegfounder_ready.nii.gz",
        "qc": subject_dir / "preprocessing_qc.json",
        "qc_png": subject_dir / "preprocessing_qc.png",
    }


# Resolve the FSL standard directory without depending on the FSL-bundled Python.
def resolve_standard_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    fsldir = os.environ.get("FSLDIR")
    if not fsldir:
        raise RuntimeError("FSLDIR is unset; provide --fsl-standard-dir explicitly")
    return (Path(fsldir).expanduser() / "data" / "standard").resolve()


# Resolve the translation-only FLIRT schedule distributed with FSL.
def resolve_fsl_schedule(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    fsldir = os.environ.get("FSLDIR")
    if not fsldir:
        raise RuntimeError("FSLDIR is unset; provide --fsl-schedule explicitly")
    return (Path(fsldir).expanduser() / "etc" / "flirtsch" / "xyztrans.sch").resolve()


# Verify the FSL references and the exact UKB config/masks used by bb_struct_init.
def validate_mni_assets(
    standard_dir: Path,
    ukb_asset_dir: Path,
    fsl_schedule: Path,
) -> dict[str, Any]:
    nifti_assets = {
        "head": standard_dir / "MNI152_T1_1mm.nii.gz",
        "brain": standard_dir / "MNI152_T1_1mm_brain.nii.gz",
        "mask": ukb_asset_dir / "MNI152_T1_1mm_brain_mask.nii.gz",
        "dilated_mask": ukb_asset_dir / "MNI152_T1_1mm_brain_mask_dil_GD7.nii.gz",
    }
    images: dict[str, nib.spatialimages.SpatialImage] = {}
    records: dict[str, Any] = {}
    for name, path in nifti_assets.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing MNI/UKB asset: {path}")
        image = nib.load(str(path))
        shape = tuple(int(value) for value in image.shape)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if shape != EXPECTED_MNI_SHAPE:
            raise ValueError(f"Unexpected {name} template shape {shape}: {path}")
        if not np.allclose(spacing, EXPECTED_MNI_SPACING, rtol=0.0, atol=1e-6):
            raise ValueError(f"Unexpected {name} template spacing {spacing}: {path}")
        images[name] = image
        records[name] = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "shape": list(shape),
            "spacing_mm": list(spacing),
            "affine": np.asarray(image.affine, dtype=np.float64).tolist(),
        }

    reference_affine = np.asarray(images["head"].affine)
    for name in ("brain", "mask", "dilated_mask"):
        if not np.allclose(images[name].affine, reference_affine, rtol=0.0, atol=1e-5):
            raise ValueError(f"MNI {name} affine differs from the head template")
    for name in ("mask", "dilated_mask"):
        mask_data = np.asarray(images[name].dataobj)
        unique = np.unique(mask_data)
        if not set(float(value) for value in unique).issubset({0.0, 1.0}):
            raise ValueError(f"UKB {name} is not binary: values={unique[:10]}")
        if not np.any(mask_data > 0):
            raise ValueError(f"UKB {name} is empty")
        records[name]["foreground_voxels"] = int(np.count_nonzero(mask_data))

    text_assets = {
        "fnirt_config": ukb_asset_dir / "bb_fnirt.cnf",
        "fsl_schedule": fsl_schedule,
    }
    for name, path in text_assets.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing registration asset: {path}")
        records[name] = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    for name, expected_hash in EXPECTED_UKB_HASHES.items():
        if records[name]["sha256"] != expected_hash:
            raise ValueError(
                f"Unexpected UKB {name} SHA-256: {records[name]['sha256']}"
            )
    records["standard_dir"] = str(standard_dir)
    records["ukb_asset_dir"] = str(ukb_asset_dir)
    return records


# Reject ambiguous raw dimensionality before any expensive FSL operation starts.
def validate_raw_input(path: Path) -> dict[str, Any]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"BrainSegFounder requires a 3D raw T1w NIfTI, got {image.shape}: {path}")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError(f"Invalid raw T1w spacing {spacing}: {path}")
    if not np.isfinite(image.affine).all():
        raise ValueError(f"Invalid raw T1w affine: {path}")
    return {
        "shape": [int(value) for value in image.shape],
        "spacing_mm": list(spacing),
        "affine": np.asarray(image.affine, dtype=np.float64).tolist(),
        "dtype_on_disk": str(image.get_data_dtype()),
    }


# Resolve an explicit executable or the first matching command on PATH.
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


# Execute one external FSL step while preserving its complete output in the log.
def run_external(
    command: list[str],
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> list[str]:
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
    return command


# Run a command whose standard output is also required for the next step.
def run_external_capture(
    command: list[str],
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> tuple[list[str], str]:
    log_handle.write("command=" + shlex.join(command) + "\n")
    log_handle.flush()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    output = completed.stdout or ""
    log_handle.write(output)
    if output and not output.endswith("\n"):
        log_handle.write("\n")
    log_handle.flush()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {shlex.join(command)}"
        )
    return command, output


# Reorient voxel axes to FSL standard storage before applying UKB's z-axis crop.
def reorient_to_fsl_standard(
    input_path: Path,
    output_path: Path,
    executable: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> tuple[list[str], dict[str, Any]]:
    input_image = nib.load(str(input_path))
    input_shape = tuple(int(value) for value in input_image.shape)
    input_spacing = tuple(float(value) for value in input_image.header.get_zooms()[:3])
    input_axcodes = tuple(str(value) for value in nib.aff2axcodes(input_image.affine))
    input_determinant = float(np.linalg.det(np.asarray(input_image.affine)[:3, :3]))

    command = run_external(
        [executable, str(input_path), str(output_path)],
        log_handle,
        timeout_seconds,
    )
    output_image = nib.load(str(output_path))
    if len(output_image.shape) != 3:
        raise ValueError(f"fslreorient2std returned a non-3D image: {output_image.shape}")
    output_shape = tuple(int(value) for value in output_image.shape)
    output_spacing = tuple(float(value) for value in output_image.header.get_zooms()[:3])
    output_axcodes = tuple(str(value) for value in nib.aff2axcodes(output_image.affine))
    output_determinant = float(np.linalg.det(np.asarray(output_image.affine)[:3, :3]))

    if output_axcodes[2] != "S":
        raise ValueError(
            "fslreorient2std did not place inferior-to-superior on voxel z: "
            f"input={input_axcodes}, output={output_axcodes}"
        )
    if int(np.prod(input_shape)) != int(np.prod(output_shape)):
        raise ValueError(
            f"fslreorient2std changed voxel count: {input_shape} -> {output_shape}"
        )
    if not np.allclose(
        sorted(input_spacing), sorted(output_spacing), rtol=0.0, atol=1e-5
    ):
        raise ValueError(
            f"fslreorient2std changed voxel sizes: {input_spacing} -> {output_spacing}"
        )
    if (
        not np.isfinite(output_determinant)
        or output_determinant == 0
        or np.sign(input_determinant) != np.sign(output_determinant)
    ):
        raise ValueError(
            "fslreorient2std changed coordinate handedness or returned a singular affine"
        )

    return command, {
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "input_spacing_mm": list(input_spacing),
        "output_spacing_mm": list(output_spacing),
        "input_axcodes": list(input_axcodes),
        "output_axcodes": list(output_axcodes),
        "input_affine_determinant": input_determinant,
        "output_affine_determinant": output_determinant,
    }


# Extract the fifth robustfov value used as z-start by the released UKB script.
def parse_robustfov_head_top(output: str) -> int:
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 6:
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError:
            continue
        head_top = values[4]
        if not np.isfinite(head_top) or head_top < 0:
            break
        rounded = int(round(head_top))
        if abs(head_top - rounded) > 1e-4:
            raise ValueError(f"robustfov returned non-integral z-start: {head_top}")
        return rounded
    raise RuntimeError(f"Could not parse robustfov output: {output!r}")


# Determine the UKB z-start and retain the exact 170-slice initial head crop.
def crop_field_of_view(
    input_path: Path,
    output_path: Path,
    robustfov_executable: str,
    fslmaths_executable: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> tuple[list[str], list[str], int]:
    robust_command, output = run_external_capture(
        [robustfov_executable, "-i", str(input_path)],
        log_handle,
        timeout_seconds,
    )
    head_top = parse_robustfov_head_top(output)
    crop_command = run_external(
        [
            fslmaths_executable,
            str(input_path),
            "-roi",
            "0",
            "-1",
            "0",
            "-1",
            str(head_top),
            str(UKB_ROI_DEPTH),
            "0",
            "1",
            str(output_path),
        ],
        log_handle,
        timeout_seconds,
    )
    return robust_command, crop_command, head_top


# Apply recursive BET to the robustfov head crop exactly as bb_struct_init.
def skull_strip_bet(
    input_path: Path,
    output_path: Path,
    executable: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> list[str]:
    return run_external(
        [
            executable,
            str(input_path),
            str(output_path),
            "-R",
        ],
        log_handle,
        timeout_seconds,
    )


# Reduce the raw T1 field of view via standard_space_roi and retain its matrix.
def reduce_field_of_view(
    brain_path: Path,
    raw_path: Path,
    fixed_brain: Path,
    output_path: Path,
    roi_to_mni_path: Path,
    executable: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> list[str]:
    command = run_external(
        [
            executable,
            str(brain_path),
            str(output_path),
            "-maskNONE",
            "-ssref",
            str(fixed_brain),
            "-altinput",
            str(raw_path),
            "-d",
        ],
        log_handle,
        timeout_seconds,
    )
    output_name = output_path.name
    for suffix in (".nii.gz", ".nii"):
        if output_name.endswith(suffix):
            output_name = output_name[: -len(suffix)]
            break
    generated_prefix = output_path.parent / output_name
    generated_matrix = Path(f"{generated_prefix}_tmp_to_std.mat")
    if not generated_matrix.is_file():
        raise FileNotFoundError(
            f"standard_space_roi did not create its registration matrix: {generated_matrix}"
        )
    os.replace(generated_matrix, roi_to_mni_path)
    for temporary in output_path.parent.glob(f"{output_name}_tmp*"):
        if temporary.is_file():
            temporary.unlink()
        elif temporary.is_dir():
            shutil.rmtree(temporary)
    return command


# Compose the raw-to-reduced and reduced-to-MNI matrices used by UKB.
def build_linear_transforms(
    reduced_t1: Path,
    raw_t1: Path,
    roi_to_mni: Path,
    t1_to_raw: Path,
    raw_to_t1: Path,
    linear_to_mni: Path,
    flirt_executable: str,
    convert_xfm_executable: str,
    schedule: Path,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> dict[str, list[str]]:
    flirt_command = run_external(
        [
            flirt_executable,
            "-in",
            str(reduced_t1),
            "-ref",
            str(raw_t1),
            "-omat",
            str(t1_to_raw),
            "-schedule",
            str(schedule),
        ],
        log_handle,
        timeout_seconds,
    )
    inverse_command = run_external(
        [
            convert_xfm_executable,
            "-omat",
            str(raw_to_t1),
            "-inverse",
            str(t1_to_raw),
        ],
        log_handle,
        timeout_seconds,
    )
    concat_command = run_external(
        [
            convert_xfm_executable,
            "-omat",
            str(linear_to_mni),
            "-concat",
            str(roi_to_mni),
            str(t1_to_raw),
        ],
        log_handle,
        timeout_seconds,
    )
    return {"flirt": flirt_command, "inverse": inverse_command, "concat": concat_command}


# Recover the exact crop-to-raw mapping from NIfTI geometry after a failed
# intensity-based translation. FLIRT requires an image output even though only
# its matrix is needed, so the temporary resample is removed immediately.
def build_header_geometry_transforms(
    reduced_t1: Path,
    raw_t1: Path,
    roi_to_mni: Path,
    t1_to_raw: Path,
    raw_to_t1: Path,
    linear_to_mni: Path,
    geometry_preview: Path,
    flirt_executable: str,
    convert_xfm_executable: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> dict[str, list[str]]:
    try:
        flirt_command = run_external(
            [
                flirt_executable,
                "-in",
                str(reduced_t1),
                "-ref",
                str(raw_t1),
                "-applyxfm",
                "-usesqform",
                "-omat",
                str(t1_to_raw),
                "-out",
                str(geometry_preview),
            ],
            log_handle,
            timeout_seconds,
        )
    finally:
        geometry_preview.unlink(missing_ok=True)
    inverse_command = run_external(
        [
            convert_xfm_executable,
            "-omat",
            str(raw_to_t1),
            "-inverse",
            str(t1_to_raw),
        ],
        log_handle,
        timeout_seconds,
    )
    concat_command = run_external(
        [
            convert_xfm_executable,
            "-omat",
            str(linear_to_mni),
            "-concat",
            str(roi_to_mni),
            str(t1_to_raw),
        ],
        log_handle,
        timeout_seconds,
    )
    return {
        "flirt_usesqform": flirt_command,
        "inverse": inverse_command,
        "concat": concat_command,
    }


# Nonlinearly register the reduced T1 using the exact UKB config and refmask.
def nonlinear_register_fnirt(
    input_path: Path,
    fixed_head: Path,
    affine_path: Path,
    config_path: Path,
    reference_mask: Path,
    warp_coef: Path,
    warp_field: Path,
    warp_jac: Path,
    preview: Path,
    fnirt_log: Path,
    executable: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> list[str]:
    return run_external(
        [
            executable,
            f"--in={input_path}",
            f"--ref={fixed_head}",
            f"--aff={affine_path}",
            f"--config={config_path}",
            f"--refmask={reference_mask}",
            f"--logout={fnirt_log}",
            f"--cout={warp_coef}",
            f"--fout={warp_field}",
            f"--jout={warp_jac}",
            f"--iout={preview}",
            "--interp=spline",
        ],
        log_handle,
        timeout_seconds,
    )


# Compose the GDC-off warp and resample the original T1w once into MNI space.
def warp_original_to_mni(
    raw_path: Path,
    fixed_head: Path,
    raw_to_t1: Path,
    nonlinear_warp: Path,
    combined_warp: Path,
    output_path: Path,
    convertwarp_executable: str,
    applywarp_executable: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> dict[str, list[str]]:
    combine_command = run_external(
        [
            convertwarp_executable,
            f"--ref={fixed_head}",
            f"--premat={raw_to_t1}",
            f"--warp2={nonlinear_warp}",
            f"--out={combined_warp}",
        ],
        log_handle,
        timeout_seconds,
    )
    apply_command = run_external(
        [
            applywarp_executable,
            "--rel",
            "-i",
            str(raw_path),
            "-r",
            str(fixed_head),
            "-w",
            str(combined_warp),
            "-o",
            str(output_path),
            "--interp=spline",
        ],
        log_handle,
        timeout_seconds,
    )
    return {"convertwarp": combine_command, "applywarp": apply_command}


# Apply the released UKB standard-space brain mask to T1_brain_to_MNI.
def apply_mni_brain_mask(
    input_path: Path,
    mask_path: Path,
    output_path: Path,
    executable: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> list[str]:
    return run_external(
        [executable, str(input_path), "-mul", str(mask_path), str(output_path)],
        log_handle,
        timeout_seconds,
    )


# Materialize the deterministic 96-cube deployment view used for frozen embeddings.
def materialize_model_input(input_path: Path, output_path: Path) -> dict[str, Any]:
    import torch
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        Resized,
        SpatialPadd,
    )

    pre_resize_transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            CropForegroundd(
                keys=["image"],
                source_key="image",
                k_divisible=EXPECTED_READY_SHAPE,
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            SpatialPadd(keys=["image"], spatial_size=EXPECTED_READY_SHAPE),
            EnsureTyped(keys=["image"]),
        ]
    )
    item = pre_resize_transforms({"image": str(input_path)})
    pre_resize = item["image"]
    if torch.is_tensor(pre_resize):
        pre_resize_array = pre_resize.detach().cpu().numpy()
    else:
        pre_resize_array = np.asarray(pre_resize)
    pre_resize_foreground = pre_resize_array != 0
    if int(np.count_nonzero(pre_resize_foreground)) < 1000:
        raise InsufficientForegroundError(
            "Too few foreground voxels before BrainSegFounder resize"
        )
    pre_resize_mean = float(np.mean(pre_resize_array[pre_resize_foreground]))
    pre_resize_standard_deviation = float(
        np.std(pre_resize_array[pre_resize_foreground])
    )
    if (
        abs(pre_resize_mean) > 2e-4
        or abs(pre_resize_standard_deviation - 1.0) > 2e-4
    ):
        raise RuntimeError(
            "MONAI nonzero normalization failed before resize: "
            f"mean={pre_resize_mean}, std={pre_resize_standard_deviation}"
        )

    resize_transform = Compose(
        [
            Resized(
                keys=["image"],
                spatial_size=EXPECTED_READY_SHAPE,
                mode="trilinear",
            ),
            EnsureTyped(keys=["image"]),
        ]
    )
    transformed = resize_transform(item)["image"]
    affine = getattr(transformed, "affine", None)
    if affine is None:
        raise RuntimeError("MONAI did not preserve transformed affine metadata")
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    affine_array = np.asarray(affine, dtype=np.float64)
    if affine_array.shape != (4, 4) or not np.isfinite(affine_array).all():
        raise RuntimeError("MONAI returned an invalid transformed affine")

    if torch.is_tensor(transformed):
        array = transformed.detach().cpu().numpy()
    else:
        array = np.asarray(transformed)
    if tuple(array.shape) != (1, *EXPECTED_READY_SHAPE):
        raise RuntimeError(f"Unexpected MONAI output shape: {array.shape}")
    array = np.asarray(array[0], dtype=np.float32)
    if not np.isfinite(array).all():
        raise RuntimeError("MONAI model input contains non-finite values")

    image = nib.Nifti1Image(array, affine_array)
    image.set_qform(affine_array, code=1)
    image.set_sform(affine_array, code=1)
    nib.save(image, str(output_path))
    return {
        "monai_output_shape": [1, *EXPECTED_READY_SHAPE],
        "transformed_affine": affine_array.tolist(),
        "normalization_before_resize": {
            "mean_nonzero": pre_resize_mean,
            "standard_deviation_nonzero": pre_resize_standard_deviation,
        },
    }


# Validate one NIfTI against a required shape/grid and return JSON-safe statistics.
def validate_nifti(
    path: Path,
    *,
    expected_shape: tuple[int, int, int] | None = None,
    reference: nib.spatialimages.SpatialImage | None = None,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty NIfTI output: {path}")
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI, got {image.shape}: {path}")
    shape = tuple(int(value) for value in image.shape)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"Unexpected shape {shape}, expected {expected_shape}: {path}")
    if reference is not None:
        if shape != tuple(reference.shape):
            raise ValueError(f"Grid shape differs from reference: {path}")
        if not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-4):
            raise ValueError(f"Grid affine differs from reference: {path}")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in output: {path}")
    nonzero = int(np.count_nonzero(data))
    if require_nonempty and nonzero == 0:
        raise ValueError(f"Empty output: {path}")
    return {
        "path": str(path),
        "shape": list(shape),
        "spacing_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "dtype_on_disk": str(image.get_data_dtype()),
        "nonzero_voxels": nonzero,
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
        "affine": np.asarray(image.affine, dtype=np.float64).tolist(),
    }


# Validate all retained FSL outputs, transform matrix, masking and shared MNI grid.
def validate_spatial_outputs(
    paths: dict[str, Path],
    mni_assets: dict[str, Any],
) -> dict[str, Any]:
    reference = nib.load(mni_assets["head"]["path"])
    statistics = {
        "reoriented_t1": validate_nifti(paths["reoriented_t1"]),
        "fov_roi": validate_nifti(paths["fov_roi"]),
        "fov_brain": validate_nifti(paths["fov_brain"]),
        "reduced_t1": validate_nifti(paths["reduced_t1"]),
        "fnirt_preview": validate_nifti(paths["fnirt_preview"], reference=reference),
        "warp_jac": validate_nifti(paths["warp_jac"], reference=reference),
        "warped_unmasked": validate_nifti(paths["warped_unmasked"], reference=reference),
        "fnirt_brain": validate_nifti(paths["fnirt_brain"], reference=reference),
    }
    for key in ("roi_to_mni_mat", "t1_to_raw_mat", "raw_to_t1_mat", "linear_mat"):
        if not paths[key].is_file() or paths[key].stat().st_size == 0:
            raise FileNotFoundError(f"Missing FSL transform matrix: {paths[key]}")
        matrix = np.loadtxt(paths[key])
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid FSL transform matrix: {paths[key]}")
        statistics[key] = {
            "path": str(paths[key]),
            "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
        }

    warp_coef = paths["warp_coef"]
    if not warp_coef.is_file() or warp_coef.stat().st_size == 0:
        raise FileNotFoundError(f"Missing FNIRT coefficient image: {warp_coef}")
    coefficient_image = nib.load(str(warp_coef))
    if len(coefficient_image.shape) not in (3, 4):
        raise ValueError(
            f"Unexpected FNIRT coefficient geometry {coefficient_image.shape}: {warp_coef}"
        )
    if any(int(value) <= 0 for value in coefficient_image.shape):
        raise ValueError(f"Empty FNIRT coefficient dimension: {warp_coef}")
    if not np.isfinite(coefficient_image.affine).all():
        raise ValueError(f"Non-finite FNIRT coefficient affine: {warp_coef}")
    coefficient_values = np.asanyarray(coefficient_image.dataobj)
    if not np.isfinite(coefficient_values).all():
        raise ValueError(f"Non-finite FNIRT coefficients: {warp_coef}")
    statistics["warp_coef"] = {
        "path": str(warp_coef),
        "shape": [int(value) for value in coefficient_image.shape],
        "spacing_mm": [
            float(value) for value in coefficient_image.header.get_zooms()[:3]
        ],
        "dtype_on_disk": str(coefficient_image.get_data_dtype()),
        "geometry": "FNIRT B-spline coefficient lattice",
    }

    for key in ("warp_field", "combined_warp"):
        path = paths[key]
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing FSL warp output: {path}")
        image = nib.load(str(path))
        if len(image.shape) not in (3, 4) or tuple(image.shape[:3]) != EXPECTED_MNI_SHAPE:
            raise ValueError(f"Unexpected FSL warp geometry {image.shape}: {path}")
        if not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-4):
            raise ValueError(f"FSL warp grid affine differs from MNI reference: {path}")
        statistics[key] = {
            "path": str(path),
            "shape": [int(value) for value in image.shape],
            "spacing_mm": [float(value) for value in image.header.get_zooms()[:3]],
            "dtype_on_disk": str(image.get_data_dtype()),
        }
    if not paths["fnirt_log"].is_file() or paths["fnirt_log"].stat().st_size == 0:
        raise FileNotFoundError(f"Missing FNIRT log: {paths['fnirt_log']}")
    statistics["fnirt_log"] = {
        "path": str(paths["fnirt_log"]),
        "size_bytes": int(paths["fnirt_log"].stat().st_size),
    }

    masked = nib.load(str(paths["fnirt_brain"])).get_fdata(dtype=np.float32)
    mask = nib.load(mni_assets["mask"]["path"]).get_fdata(dtype=np.float32) > 0
    outside_maximum = float(np.max(np.abs(masked[~mask])))
    if outside_maximum > 1e-5:
        raise ValueError(
            f"Masked FNIRT output is nonzero outside the MNI mask: {outside_maximum}"
        )
    statistics["fnirt_brain"]["maximum_absolute_outside_mask"] = outside_maximum
    return statistics


# Enforce the final shape and finite intensity contract consumed by the encoder.
def validate_ready_output(path: Path) -> dict[str, Any]:
    statistics = validate_nifti(path, expected_shape=EXPECTED_READY_SHAPE)
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    foreground = data != 0
    count = int(np.count_nonzero(foreground))
    if count < 1000:
        raise ValueError(f"Too few nonzero voxels in model input ({count}): {path}")
    mean = float(np.mean(data[foreground]))
    standard_deviation = float(np.std(data[foreground]))
    # Normalization is verified exactly before resize in materialize_model_input.
    # Trilinear interpolation reduces the final standard deviation (about 0.93
    # in the validated ON-Harmony outputs), so requiring final std==1 would be
    # scientifically incorrect for this transform order.
    if abs(mean) > 1e-3 or not 0.5 < standard_deviation <= 1.05:
        raise ValueError(
            f"Unexpected post-resize BrainSegFounder intensities: "
            f"mean={mean}, std={standard_deviation}"
        )
    statistics.update(
        nonzero_fraction=float(count / data.size),
        mean_nonzero=mean,
        standard_deviation_nonzero=standard_deviation,
    )
    return statistics


# Create a compact montage of UKB FOV, registration and model-ready volumes.
def write_qc_montage(paths: dict[str, Path], destination: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volumes = [
        ("FSL-standard orientation", np.asarray(nib.load(str(paths["reoriented_t1"])).dataobj)),
        ("robustfov + BET", np.asarray(nib.load(str(paths["fov_brain"])).dataobj)),
        ("standard_space_roi", np.asarray(nib.load(str(paths["reduced_t1"])).dataobj)),
        ("UKB T1 brain to MNI", np.asarray(nib.load(str(paths["fnirt_brain"])).dataobj)),
        ("BrainSegFounder 96-cube", np.asarray(nib.load(str(paths["ready"])).dataobj)),
    ]

    def orthogonal_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers = tuple(size // 2 for size in volume.shape)
        return (
            volume[centers[0], :, :],
            volume[:, centers[1], :],
            volume[:, :, centers[2]],
        )

    figure, axes = plt.subplots(len(volumes), 3, figsize=(9, 12))
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


# Capture software and resolved FSL commands used by the run.
def software_versions(executables: dict[str, str]) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": os.sys.version.split()[0],
        "executables": executables,
    }
    for package in ("nibabel", "numpy", "monai", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    fsldir = os.environ.get("FSLDIR")
    version_file = Path(fsldir) / "etc" / "fslversion" if fsldir else None
    if version_file is not None and version_file.is_file():
        versions["FSL"] = version_file.read_text(encoding="utf-8").strip()
    else:
        versions["FSL"] = "VERSION FILE NOT FOUND"
    return versions


# Require hashes and provenance for all artifacts before reusing a subject.
def qc_matches(
    paths: dict[str, Path],
    raw: Path,
    mni_assets: dict[str, Any],
) -> bool:
    if not paths["qc"].is_file():
        return False
    required = (
        "reoriented_t1", "fov_roi", "fov_brain", "reduced_t1", "roi_to_mni_mat",
        "t1_to_raw_mat", "raw_to_t1_mat", "linear_mat", "warp_coef",
        "warp_field", "warp_jac", "fnirt_preview", "fnirt_log",
        "combined_warp", "warped_unmasked", "fnirt_brain", "ready", "qc_png",
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
        and reusable_pipeline_qc(qc)
        and qc.get("mni_assets") == mni_assets
        and qc.get("raw_input") == raw_fingerprint(raw)
        and all(hashes.get(key) == sha256_file(paths[key]) for key in required)
    )


# Publish a validated staging directory without exposing partial subject outputs.
def publish_staged_outputs(staged: dict[str, Path], final: dict[str, Path]) -> None:
    for key in (
        "reoriented_t1", "fov_roi", "fov_brain", "reduced_t1", "roi_to_mni_mat",
        "t1_to_raw_mat", "raw_to_t1_mat", "linear_mat", "warp_coef",
        "warp_field", "warp_jac", "fnirt_preview", "fnirt_log",
        "combined_warp", "warped_unmasked", "fnirt_brain", "ready",
    ):
        final[key].parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged[key], final[key])


# Run or validate the full BrainSegFounder input contract for one manifest subject.
def process_subject(
    row: dict[str, str],
    output_root: Path,
    log_root: Path,
    mni_assets: dict[str, Any],
    executables: dict[str, str],
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
        raw_geometry = validate_raw_input(raw)

        if validate_existing:
            validate_spatial_outputs(paths, mni_assets)
            ready_stats = validate_ready_output(paths["ready"])
            if not qc_matches(paths, raw, mni_assets):
                raise RuntimeError(f"Outputs are valid but provenance QC is stale: {paths['qc']}")
            record.update(
                status="VALIDATED",
                message=(
                    f"{tuple(ready_stats['shape'])}; nonzero mean="
                    f"{ready_stats['mean_nonzero']:.6f}, std="
                    f"{ready_stats['standard_deviation_nonzero']:.6f}"
                ),
            )
            return record

        output_keys = (
            "reoriented_t1", "fov_roi", "fov_brain", "reduced_t1", "roi_to_mni_mat",
            "t1_to_raw_mat", "raw_to_t1_mat", "linear_mat", "warp_coef",
            "warp_field", "warp_jac", "fnirt_preview", "fnirt_log",
            "combined_warp", "warped_unmasked", "fnirt_brain", "ready", "qc",
        )
        existing = [key for key in output_keys if paths[key].exists()]
        if existing and not force:
            validate_spatial_outputs(paths, mni_assets)
            ready_stats = validate_ready_output(paths["ready"])
            if not qc_matches(paths, raw, mni_assets):
                raise RuntimeError(
                    "Existing BrainSegFounder files have stale or incomplete provenance. "
                    "Inspect them and rerun with --force if replacement is intended."
                )
            record.update(
                status="SKIPPED",
                message="Compatible validated v3/v4 output already exists",
            )
            return record

        paths["subject_dir"].mkdir(parents=True, exist_ok=True)
        stage_dir = paths["subject_dir"] / f".staging_{run_id}_{uuid.uuid4().hex[:8]}"
        stage_dir.mkdir(parents=True, exist_ok=False)
        staged = {
            key: stage_dir / paths[key].name
            for key in output_keys
            if key != "qc"
        }

        head = Path(mni_assets["head"]["path"])
        fixed_brain = Path(mni_assets["brain"]["path"])
        mask = Path(mni_assets["mask"]["path"])
        dilated_mask = Path(mni_assets["dilated_mask"]["path"])
        fnirt_config = Path(mni_assets["fnirt_config"]["path"])
        fsl_schedule = Path(mni_assets["fsl_schedule"]["path"])
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_message(f"pipeline_version={PIPELINE_VERSION}", log_handle)
            log_message(f"pipeline_contract_sha256={PIPELINE_CONTRACT_SHA256}", log_handle)
            log_message(f"dataset={dataset}", log_handle)
            log_message(f"subject={subject}", log_handle)
            log_message(f"raw={raw}", log_handle)
            log_message(f"mni_head={head}", log_handle)
            log_message("gradient_distortion_correction=OFF (coefficients unavailable)", log_handle)
            log_message(
                "  [Preprocessing 1/7] lossless fslreorient2std input adapter",
                log_handle,
            )
            reorient_command, reorientation = reorient_to_fsl_standard(
                raw,
                staged["reoriented_t1"],
                executables["fslreorient2std"],
                log_handle,
                timeout_seconds,
            )
            log_message(
                f"    axcodes={reorientation['input_axcodes']} -> "
                f"{reorientation['output_axcodes']}",
                log_handle,
            )
            log_message(
                "  [Preprocessing 2/7] robustfov 170-slice crop + recursive FSL BET",
                log_handle,
            )
            robustfov_command, crop_command, head_top = crop_field_of_view(
                staged["reoriented_t1"],
                staged["fov_roi"],
                executables["robustfov"],
                executables["fslmaths"],
                log_handle,
                timeout_seconds,
            )
            log_message(f"    robustfov_head_top={head_top}", log_handle)
            bet_command = skull_strip_bet(
                staged["fov_roi"],
                staged["fov_brain"],
                executables["bet"],
                log_handle,
                timeout_seconds,
            )
            log_message(
                "  [Preprocessing 3/7] standard_space_roi FOV reduction",
                log_handle,
            )
            standard_space_roi_command = reduce_field_of_view(
                staged["fov_brain"],
                staged["reoriented_t1"],
                fixed_brain,
                staged["reduced_t1"],
                staged["roi_to_mni_mat"],
                executables["standard_space_roi"],
                log_handle,
                timeout_seconds,
            )
            log_message(
                "  [Preprocessing 4/7] UKB FLIRT/convert_xfm matrix composition",
                log_handle,
            )
            matrix_commands = build_linear_transforms(
                staged["reduced_t1"],
                staged["reoriented_t1"],
                staged["roi_to_mni_mat"],
                staged["t1_to_raw_mat"],
                staged["raw_to_t1_mat"],
                staged["linear_mat"],
                executables["flirt"],
                executables["convert_xfm"],
                fsl_schedule,
                log_handle,
                timeout_seconds,
            )
            log_message(
                "  [Preprocessing 5/7] UKB-configured FNIRT nonlinear registration",
                log_handle,
            )
            fnirt_command = nonlinear_register_fnirt(
                staged["reduced_t1"],
                head,
                staged["linear_mat"],
                fnirt_config,
                dilated_mask,
                staged["warp_coef"],
                staged["warp_field"],
                staged["warp_jac"],
                staged["fnirt_preview"],
                staged["fnirt_log"],
                executables["fnirt"],
                log_handle,
                timeout_seconds,
            )
            log_message(
                "  [Preprocessing 6/7] Compose GDC-off warp, resample reoriented T1w, "
                "apply UKB mask",
                log_handle,
            )
            warp_commands = warp_original_to_mni(
                staged["reoriented_t1"],
                head,
                staged["raw_to_t1_mat"],
                staged["warp_field"],
                staged["combined_warp"],
                staged["warped_unmasked"],
                executables["convertwarp"],
                executables["applywarp"],
                log_handle,
                timeout_seconds,
            )
            mask_command = apply_mni_brain_mask(
                staged["warped_unmasked"],
                mask,
                staged["fnirt_brain"],
                executables["fslmaths"],
                log_handle,
                timeout_seconds,
            )
            registration_attempts = [
                {
                    "strategy": "released_ukb_intensity_translation",
                    "linear_transforms": matrix_commands,
                    "fnirt": fnirt_command,
                    "warp_original_to_mni": warp_commands,
                    "fslmaths_mask": mask_command,
                }
            ]
            recovery_trigger: str | None = None
            log_message(
                "  [Preprocessing 7/7] RAS + foreground crop + nonzero z-score + "
                "pad/resize 96^3",
                log_handle,
            )
            try:
                model_view = materialize_model_input(
                    staged["fnirt_brain"], staged["ready"]
                )
            except InsufficientForegroundError as error:
                recovery_trigger = str(error)
                log_message(
                    "  [Recovery] Empty MNI output after the released UKB translation; "
                    "rebuilding crop-to-raw geometry with FLIRT -usesqform",
                    log_handle,
                )
                for key in (
                    "t1_to_raw_mat",
                    "raw_to_t1_mat",
                    "linear_mat",
                    "warp_coef",
                    "warp_field",
                    "warp_jac",
                    "fnirt_preview",
                    "fnirt_log",
                    "combined_warp",
                    "warped_unmasked",
                    "fnirt_brain",
                    "ready",
                ):
                    staged[key].unlink(missing_ok=True)
                matrix_commands = build_header_geometry_transforms(
                    staged["reduced_t1"],
                    staged["reoriented_t1"],
                    staged["roi_to_mni_mat"],
                    staged["t1_to_raw_mat"],
                    staged["raw_to_t1_mat"],
                    staged["linear_mat"],
                    stage_dir / f"{subject}_header_geometry_preview.nii.gz",
                    executables["flirt"],
                    executables["convert_xfm"],
                    log_handle,
                    timeout_seconds,
                )
                fnirt_command = nonlinear_register_fnirt(
                    staged["reduced_t1"],
                    head,
                    staged["linear_mat"],
                    fnirt_config,
                    dilated_mask,
                    staged["warp_coef"],
                    staged["warp_field"],
                    staged["warp_jac"],
                    staged["fnirt_preview"],
                    staged["fnirt_log"],
                    executables["fnirt"],
                    log_handle,
                    timeout_seconds,
                )
                warp_commands = warp_original_to_mni(
                    staged["reoriented_t1"],
                    head,
                    staged["raw_to_t1_mat"],
                    staged["warp_field"],
                    staged["combined_warp"],
                    staged["warped_unmasked"],
                    executables["convertwarp"],
                    executables["applywarp"],
                    log_handle,
                    timeout_seconds,
                )
                mask_command = apply_mni_brain_mask(
                    staged["warped_unmasked"],
                    mask,
                    staged["fnirt_brain"],
                    executables["fslmaths"],
                    log_handle,
                    timeout_seconds,
                )
                registration_attempts.append(
                    {
                        "strategy": "nifti_header_geometry_recovery",
                        "trigger": recovery_trigger,
                        "linear_transforms": matrix_commands,
                        "fnirt": fnirt_command,
                        "warp_original_to_mni": warp_commands,
                        "fslmaths_mask": mask_command,
                    }
                )
                model_view = materialize_model_input(
                    staged["fnirt_brain"], staged["ready"]
                )
            spatial_stats = validate_spatial_outputs(staged, mni_assets)
            ready_stats = validate_ready_output(staged["ready"])

            publish_staged_outputs(staged, paths)
            write_qc_montage(
                paths,
                paths["qc_png"],
                f"{dataset}/{subject} - BrainSegFounder PASS",
            )
            required = (
                "reoriented_t1", "fov_roi", "fov_brain", "reduced_t1", "roi_to_mni_mat",
                "t1_to_raw_mat", "raw_to_t1_mat", "linear_mat", "warp_coef",
                "warp_field", "warp_jac", "fnirt_preview", "fnirt_log",
                "combined_warp", "warped_unmasked", "fnirt_brain", "ready", "qc_png",
            )
            output_hashes = {key: sha256_file(paths[key]) for key in required}
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
                "raw_geometry": raw_geometry,
                "mni_assets": mni_assets,
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
                "commands": {
                    "fslreorient2std": reorient_command,
                    "robustfov": robustfov_command,
                    "fslmaths_roi": crop_command,
                    "bet": bet_command,
                    "standard_space_roi": standard_space_roi_command,
                    "linear_transforms": matrix_commands,
                    "fnirt": fnirt_command,
                    "warp_original_to_mni": warp_commands,
                    "fslmaths_mask": mask_command,
                },
                "registration_recovery": {
                    "used": recovery_trigger is not None,
                    "trigger": recovery_trigger,
                    "effective_strategy": registration_attempts[-1]["strategy"],
                    "attempts": registration_attempts,
                },
                "input_reorientation": reorientation,
                "robustfov_head_top": head_top,
                "spatial_outputs": spatial_stats,
                "model_view": model_view,
                "model_ready_output": ready_stats,
                "output_sha256": output_hashes,
            }
            atomic_json(paths["qc"], qc)
            atomic_json(historical_qc, qc)
            shutil.copy2(paths["qc_png"], historical_png)
            log_message(f"  [QC] PASS: {paths['qc']}", log_handle)

        record.update(
            status="PASS",
            message=(
                f"{tuple(ready_stats['shape'])}; nonzero mean="
                f"{ready_stats['mean_nonzero']:.6f}, std="
                f"{ready_stats['standard_deviation_nonzero']:.6f}; intermediates retained"
            ),
        )
        if stage_dir.exists():
            stage_dir.rmdir()
    except Exception as error:
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


# Fix numerical thread limits before launching bounded subject-level parallelism.
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


# Build the CLI shared by one-subject pilots and resumable cohort runs.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UKB GDC-off BrainSegFounder preprocessing for the brain-age cohort"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--fsl-standard-dir", type=Path)
    parser.add_argument("--ukb-asset-dir", type=Path, default=DEFAULT_UKB_ASSET_DIR)
    parser.add_argument("--fsl-schedule", type=Path)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--subject", action="append", dest="subjects")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--fslreorient2std-command", default="fslreorient2std")
    parser.add_argument("--robustfov-command", default="robustfov")
    parser.add_argument("--bet-command", default="bet")
    parser.add_argument("--standard-space-roi-command", default="standard_space_roi")
    parser.add_argument("--flirt-command", default="flirt")
    parser.add_argument("--convert-xfm-command", default="convert_xfm")
    parser.add_argument("--fnirt-command", default="fnirt")
    parser.add_argument("--convertwarp-command", default="convertwarp")
    parser.add_argument("--applywarp-command", default="applywarp")
    parser.add_argument("--fslmaths-command", default="fslmaths")
    parser.add_argument("--dc-command", default="dc")
    parser.add_argument("--step-timeout-seconds", type=float)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


# Validate global inputs, process the cohort and save run-level status artifacts.
def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.force and args.validate_existing:
        raise ValueError("--force and --validate-existing are mutually exclusive")

    configure_threads(args.threads_per_worker)
    manifest_path = args.manifest.resolve()
    rows = load_manifest(manifest_path)
    selected = select_rows(rows, args.datasets, args.splits, args.subjects, args.limit)
    standard_dir = resolve_standard_dir(args.fsl_standard_dir)
    fsl_schedule = resolve_fsl_schedule(args.fsl_schedule)
    ukb_asset_dir = args.ukb_asset_dir.expanduser().resolve()
    mni_assets = validate_mni_assets(standard_dir, ukb_asset_dir, fsl_schedule)
    command_names = {
        "fslreorient2std": args.fslreorient2std_command,
        "robustfov": args.robustfov_command,
        "bet": args.bet_command,
        "standard_space_roi": args.standard_space_roi_command,
        "flirt": args.flirt_command,
        "convert_xfm": args.convert_xfm_command,
        "fnirt": args.fnirt_command,
        "convertwarp": args.convertwarp_command,
        "applywarp": args.applywarp_command,
        "fslmaths": args.fslmaths_command,
        "dc": args.dc_command,
    }
    executables = (
        command_names
        if args.dry_run or args.validate_existing
        else {name: resolve_executable(command) for name, command in command_names.items()}
    )
    versions = software_versions(executables)
    output_root = args.output_root.resolve()
    log_root = args.log_root.resolve()

    print("BrainSegFounder preprocessing for the definitive brain-age cohort")
    print(f"  manifest : {manifest_path}")
    print(f"  selected : {len(selected)}/{len(rows)}")
    print(f"  outputs  : {output_root}")
    print(f"  logs     : {log_root}")
    print(f"  MNI head : {mni_assets['head']['path']}")
    print(f"  UKB assets: {mni_assets['ukb_asset_dir']}")
    print("  GDC      : OFF (scanner-specific coefficients unavailable)")
    print(f"  workers  : {args.workers}")
    print(f"  threads  : {args.threads_per_worker} per worker")
    print(f"  contract : {PIPELINE_CONTRACT_SHA256}")
    print(
        "  retained : reoriented T1, UKB FOV/BET, matrices, FNIRT warps, "
        "masked MNI T1, final, logs and QC"
    )

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
                mni_assets,
                executables,
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
        "qc_path",
        "log_path",
        "started_utc",
        "finished_utc",
        "status",
        "message",
    ]
    run_manifest = log_root / f"_brainsegfounder_run_{run_id}.tsv"
    run_summary = log_root / f"_brainsegfounder_run_{run_id}.json"
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
            "mni_assets": mni_assets,
            "output_root": str(output_root),
            "log_root": str(log_root),
            "workers": args.workers,
            "threads_per_worker": args.threads_per_worker,
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
