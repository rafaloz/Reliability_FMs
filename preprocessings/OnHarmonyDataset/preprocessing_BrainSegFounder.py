#!/usr/bin/env python3
"""Prepare ON-Harmony T1w scans for BrainSegFounder.

The spatial part first adapts source storage orientation and then follows the
released UK Biobank ``bb_struct_init`` T1 chain with gradient-distortion
correction disabled because ON-Harmony does not provide scanner-specific
gradient coefficient files:

1. Lossless ``fslreorient2std`` axis reorientation so the UKB z-axis crop is
   valid for every ON-Harmony storage order.
2. ``robustfov`` z-start, a 170-slice crop, and recursive FSL BET.
3. ``standard_space_roi`` reduction against MNI152_T1_1mm_brain.
4. UKB matrix composition and FNIRT using the released ``bb_fnirt.cnf`` and
   ``MNI152_T1_1mm_brain_mask_dil_GD7`` reference mask. If the released
   translation-only FLIRT initialization yields an empty MNI image, the
   crop-to-raw transform is recovered deterministically from NIfTI geometry
   and the same FNIRT chain is repeated once.
5. GDC-off warp composition, spline resampling of the reoriented T1w, and the
   released UKB MNI brain mask.
6. The predeclared deployment view: RAS, foreground crop, nonzero z-score,
   padding, and trilinear resize to 96 x 96 x 96.

The final fixed-size view is study-defined; it is not claimed to reproduce the
random 96-cube sampling used in Stage-1 pretraining. All spatial intermediates
and a hash-checked provenance record are retained.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import nibabel as nib
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_UKB_ASSET_DIR = PROJECT_ROOT / "templates" / "UKBiobank"
DEFAULT_RAW_FILELIST = PROJECT_ROOT / "DATA" / "filelists" / "filelist_BrainIAC_raw.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "preproc_BrainSegFounder"
DEFAULT_LOG_DIR = SCRIPT_PATH.parent / "logs_preprocs" / "BrainSegFounder"
DEFAULT_PUBLISHED_FILELIST = (
    PROJECT_ROOT / "DATA" / "filelists" / "filelist_BrainSegFounder.txt"
)

PIPELINE_VERSION = "brainsegfounder-ukb-gdcoff-onharmony-v4"
OFFICIAL_BRAINSEGFOUNDER_COMMIT = "0bbf43282447b8f998e4ed47afe26b84e8f05419"
UKB_PIPELINE_COMMIT = "9458b42e23c3476cc9bd4b6a0ae60e1df3807e55"
UKB_ONHARMONY_FORK_COMMIT = "8a86d328c671b6496fd54a28b4905a2d0898e9ab"
UKB_ROI_DEPTH = 170
EXPECTED_MNI_SHAPE = (182, 218, 182)
EXPECTED_MNI_SPACING = (1.0, 1.0, 1.0)
EXPECTED_READY_SHAPE = (96, 96, 96)
CANONICAL_T1_PATTERN = re.compile(
    r"sub-[0-9]+_ses-[A-Za-z0-9]+_T1w\.nii(?:\.gz)?$"
)
COMMAND_NAMES = (
    "fslreorient2std", "robustfov", "fslmaths", "bet", "standard_space_roi",
    "flirt", "convert_xfm", "fnirt", "convertwarp", "applywarp", "dc",
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)
EXPECTED_UKB_HASHES = {
    "fnirt_config": "41e2f1b068e4841c8f6d7934375651fbe2cf2e66ac40da3d21f3fbb9b9576e37",
    "mask": "4e599d7d0bd843251bb24391b0ddc636b73216f12a9ccc017a34cb0d17e287e4",
    "dilated_mask": "1365d6747d6878501f97551ce44f143f1662b9dce3ab3f362243e5a456c5725d",
}

CONTRACT = {
    "pipeline_version": PIPELINE_VERSION,
    "official_brainsegfounder_commit": OFFICIAL_BRAINSEGFOUNDER_COMMIT,
    "ukb_pipeline_commit": UKB_PIPELINE_COMMIT,
    "ukb_onharmony_fork_commit": UKB_ONHARMONY_FORK_COMMIT,
    "input_orientation_adapter": {
        "implementation": "FSL fslreorient2std",
        "purpose": "make the UKB 170-slice voxel-z crop valid across source storage orders",
        "interpolation": "none; axis permutation and 90-degree rotations only",
        "status": "study-required adapter before the released UKB chain",
    },
    "spatial_chain": [
        "GDC_off",
        "robustfov + fslmaths 170-slice ROI",
        "bet -R",
        "standard_space_roi -maskNONE -altinput raw -d",
        "flirt xyztrans.sch + convert_xfm composition",
        "fnirt bb_fnirt.cnf + UKB dilated reference mask",
        "convertwarp --premat + applywarp --rel --interp=spline",
        "UKB MNI152_T1_1mm_brain_mask",
    ],
    "linear_registration": {
        "primary": "released FSL flirt xyztrans.sch plus convert_xfm composition",
        "recovery_trigger": (
            "fewer than 1000 foreground voxels in the masked MNI image before resize"
        ),
        "recovery": (
            "FSL flirt -applyxfm -usesqform crop-to-raw geometry followed by the "
            "same convert_xfm composition and FNIRT chain"
        ),
    },
    "model_view": [
        "Orientation RAS",
        "CropForeground nonzero",
        "NormalizeIntensity nonzero channel-wise",
        "SpatialPad 96^3",
        "Resize 96^3 trilinear",
    ],
}
CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(CONTRACT, sort_keys=True).encode("utf-8")
).hexdigest()

SPATIAL_KEYS = (
    "reoriented_t1", "fov_roi", "fov_brain", "reduced_t1", "roi_to_mni_mat",
    "t1_to_raw_mat", "raw_to_t1_mat", "linear_mat", "warp_coef",
    "warp_field", "warp_jac", "fnirt_preview", "fnirt_log",
    "combined_warp", "warped_unmasked", "fnirt_brain", "ready",
)


class InsufficientForegroundError(RuntimeError):
    """Signal a catastrophic registration output eligible for one recovery attempt."""


def utc_now() -> str:
    """Return a timezone-aware timestamp for provenance records."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an artifact incrementally without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """Write JSON by atomic replacement so interrupted jobs cannot publish it."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def acquisition_stem(path: Path) -> str:
    """Return the canonical ON-Harmony acquisition identifier."""
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Input is not a NIfTI file: {path}")


def load_inputs(
    *,
    filelist: Path | None = None,
    input_file: Path | None = None,
    input_dir: Path | None = None,
    expected_inputs: int | None = None,
) -> list[Path]:
    """Load one ordered, duplicate-free registry of raw ON-Harmony T1w files."""
    sources = sum(value is not None for value in (filelist, input_file, input_dir))
    if sources != 1:
        raise ValueError("Provide exactly one of --filelist, --input-file, or --input-dir")

    if filelist is not None:
        filelist = Path(filelist).expanduser().resolve()
        if not filelist.is_file():
            raise FileNotFoundError(filelist)
        raw_paths = [
            line.strip()
            for line in filelist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any("/preproc_" in path for path in raw_paths):
            raise ValueError(f"Raw filelist contains preprocessed paths: {filelist}")
        files = [
            Path(path).expanduser()
            if Path(path).expanduser().is_absolute()
            else filelist.parent / Path(path).expanduser()
            for path in raw_paths
        ]
    elif input_file is not None:
        files = [Path(input_file).expanduser()]
    else:
        input_dir = Path(input_dir).expanduser().resolve()
        if not input_dir.is_dir():
            raise NotADirectoryError(input_dir)
        files = sorted(
            path
            for path in input_dir.rglob("*T1w*.nii*")
            if CANONICAL_T1_PATTERN.fullmatch(path.name)
        )

    files = [path.resolve() for path in files]
    if expected_inputs is not None:
        if expected_inputs < 1:
            raise ValueError("--expected-inputs must be positive")
        if len(files) != expected_inputs:
            raise ValueError(f"Expected {expected_inputs} inputs, found {len(files)}")
    if not files:
        raise ValueError("No canonical T1w NIfTI inputs were selected")
    if len({str(path) for path in files}) != len(files):
        raise ValueError("Input registry contains duplicate paths")
    stems = [acquisition_stem(path) for path in files]
    if len(set(stems)) != len(stems):
        raise ValueError("Input registry contains duplicate acquisition identifiers")
    for path in files:
        if not CANONICAL_T1_PATTERN.fullmatch(path.name):
            raise ValueError(f"Unexpected ON-Harmony T1w filename: {path.name}")
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty raw T1w input: {path}")
    return files


def configure_fsl_environment(fsldir: Path | None) -> Path | None:
    """Expose one explicit FSL installation without ever selecting its Python."""
    if fsldir is None:
        configured = os.environ.get("FSLDIR")
        fsldir = Path(configured).expanduser() if configured else None
    if fsldir is None:
        return None
    fsldir = fsldir.resolve()
    bin_dir = fsldir / "bin"
    if not bin_dir.is_dir():
        raise NotADirectoryError(f"Invalid FSLDIR: {fsldir}")
    os.environ["FSLDIR"] = str(fsldir)
    path_entries = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and entry != str(bin_dir)
    ]
    os.environ["PATH"] = os.pathsep.join([str(bin_dir), *path_entries])
    return fsldir


def resolve_resources(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve FSL and UKB assets consistently for single, run, and validate."""
    fsldir = configure_fsl_environment(args.fsldir)
    if args.template is not None:
        template = args.template.expanduser().resolve()
    elif fsldir is not None:
        template = (fsldir / "data" / "standard" / "MNI152_T1_1mm.nii.gz").resolve()
    else:
        raise ValueError("Provide --template or --fsldir, or set FSLDIR")
    if args.fsl_schedule is not None:
        schedule = args.fsl_schedule.expanduser().resolve()
    elif fsldir is not None:
        schedule = (fsldir / "etc" / "flirtsch" / "xyztrans.sch").resolve()
    else:
        raise ValueError("Provide --fsl-schedule or --fsldir, or set FSLDIR")
    return template, schedule, args.ukb_asset_dir.expanduser().resolve()


def worker_environment(threads: int) -> dict[str, str]:
    """Build a child environment with bounded numerical-library threading."""
    if threads < 1:
        raise ValueError("--threads-per-worker must be positive")
    environment = os.environ.copy()
    for name in THREAD_ENVIRONMENT:
        environment[name] = str(threads)
    return environment


def atomic_publish_filelist(
    inputs: list[Path], output_dir: Path, destination: Path
) -> Path:
    """Publish validated outputs atomically and in canonical raw-input order."""
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        lines = []
        for raw in inputs:
            stem = acquisition_stem(raw)
            lines.append(
                str(output_paths(Path(output_dir).resolve() / stem, stem)["ready"])
            )
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def output_paths(subject_dir: Path, stem: str) -> dict[str, Path]:
    """Define all retained UKB intermediates and the final model input."""
    return {
        "subject_dir": subject_dir,
        "reoriented_t1": subject_dir / f"{stem}_ukb_reoriented.nii.gz",
        "fov_roi": subject_dir / f"{stem}_ukb_fov.nii.gz",
        "fov_brain": subject_dir / f"{stem}_ukb_fov_brain.nii.gz",
        "reduced_t1": subject_dir / f"{stem}_ukb_T1.nii.gz",
        "roi_to_mni_mat": subject_dir / f"{stem}_roi_to_mni.mat",
        "t1_to_raw_mat": subject_dir / f"{stem}_T1_to_raw.mat",
        "raw_to_t1_mat": subject_dir / f"{stem}_raw_to_T1.mat",
        "linear_mat": subject_dir / f"{stem}_T1_to_MNI_linear.mat",
        "warp_coef": subject_dir / f"{stem}_T1_to_MNI_warp_coef.nii.gz",
        "warp_field": subject_dir / f"{stem}_T1_to_MNI_warp.nii.gz",
        "warp_jac": subject_dir / f"{stem}_T1_to_MNI_warp_jac.nii.gz",
        "fnirt_preview": subject_dir / f"{stem}_T1_fnirt_preview.nii.gz",
        "fnirt_log": subject_dir / f"{stem}_T1_to_MNI_fnirt.log",
        "combined_warp": subject_dir / f"{stem}_T1_orig_to_MNI_warp.nii.gz",
        "warped_unmasked": subject_dir / f"{stem}_T1_to_MNI_unmasked.nii.gz",
        "fnirt_brain": subject_dir / f"{stem}_fnirt_brain.nii.gz",
        "ready": subject_dir / f"{stem}_brainsegfounder_ready.nii.gz",
        "qc": subject_dir / "preprocessing_qc.json",
        "log": subject_dir / "preprocessing.log",
    }


def resolve_executable(command: str) -> str:
    """Resolve one FSL executable without invoking the FSL Python runtime."""
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"FSL executable not found on PATH: {command}")
    return resolved


def run_external(
    command: list[str], log_handle: TextIO, timeout_seconds: float | None
) -> list[str]:
    """Run one FSL command and preserve complete stdout/stderr in the log."""
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


def run_external_capture(
    command: list[str], log_handle: TextIO, timeout_seconds: float | None
) -> tuple[list[str], str]:
    """Run a command while retaining stdout for both parsing and logging."""
    log_handle.write("command=" + shlex.join(command) + "\n")
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


def reorient_to_fsl_standard(
    input_path: Path,
    output_path: Path,
    executable: str,
    log_handle: TextIO,
    timeout_seconds: float | None,
) -> tuple[list[str], dict[str, Any]]:
    """Losslessly orient voxel axes and require superior on the voxel-z axis."""
    input_image = nib.load(str(input_path))
    input_shape = tuple(int(value) for value in input_image.shape)
    input_spacing = tuple(float(value) for value in input_image.header.get_zooms()[:3])
    input_axcodes = tuple(str(value) for value in nib.aff2axcodes(input_image.affine))
    input_determinant = float(np.linalg.det(np.asarray(input_image.affine)[:3, :3]))
    command = run_external(
        [executable, str(input_path), str(output_path)], log_handle, timeout_seconds
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
        raise ValueError(f"fslreorient2std changed voxel count: {input_shape} -> {output_shape}")
    if not np.allclose(sorted(input_spacing), sorted(output_spacing), rtol=0.0, atol=1e-5):
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


def parse_robustfov_head_top(output: str) -> int:
    """Parse the fifth numeric robustfov field used by bb_struct_init."""
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 6:
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError:
            continue
        value = values[4]
        rounded = int(round(value))
        if value >= 0 and abs(value - rounded) <= 1e-4:
            return rounded
    raise RuntimeError(f"Could not parse robustfov output: {output!r}")


def remove_extensions(path: Path) -> Path:
    """Return the FSL image basename after removing .nii or .nii.gz."""
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path


def validate_assets(
    template: Path, ukb_asset_dir: Path, fsl_schedule: Path
) -> dict[str, Any]:
    """Validate MNI grids and pin all release-specific UKB files by hash."""
    brain = template.with_name("MNI152_T1_1mm_brain.nii.gz")
    if not brain.is_file():
        fsldir = os.environ.get("FSLDIR")
        if fsldir:
            brain = Path(fsldir) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"
    nifti_paths = {
        "head": template,
        "brain": brain,
        "mask": ukb_asset_dir / "MNI152_T1_1mm_brain_mask.nii.gz",
        "dilated_mask": ukb_asset_dir / "MNI152_T1_1mm_brain_mask_dil_GD7.nii.gz",
    }
    reference = None
    records: dict[str, Any] = {}
    for name, path in nifti_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing MNI/UKB asset: {path}")
        image = nib.load(str(path))
        if tuple(image.shape) != EXPECTED_MNI_SHAPE:
            raise ValueError(f"Unexpected {name} shape {image.shape}: {path}")
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if not np.allclose(spacing, EXPECTED_MNI_SPACING, rtol=0.0, atol=1e-5):
            raise ValueError(f"Unexpected {name} spacing {spacing}: {path}")
        if not np.isfinite(image.affine).all():
            raise ValueError(f"Non-finite affine for {name}: {path}")
        if reference is None:
            reference = image
        elif not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-5):
            raise ValueError(f"MNI affine mismatch for {name}: {path}")
        records[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "shape": [int(value) for value in image.shape],
            "spacing_mm": list(spacing),
            "affine": np.asarray(image.affine, dtype=np.float64).tolist(),
        }
    for name in ("mask", "dilated_mask"):
        data = np.asarray(nib.load(records[name]["path"]).dataobj)
        if not set(np.unique(data).astype(float)).issubset({0.0, 1.0}):
            raise ValueError(f"UKB {name} is not binary")
        records[name]["foreground_voxels"] = int(np.count_nonzero(data))
    text_paths = {
        "fnirt_config": ukb_asset_dir / "bb_fnirt.cnf",
        "fsl_schedule": fsl_schedule,
    }
    for name, path in text_paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing registration asset: {path}")
        records[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    for name, expected in EXPECTED_UKB_HASHES.items():
        if records[name]["sha256"] != expected:
            raise ValueError(f"Unexpected UKB {name} SHA-256: {records[name]['sha256']}")
    return records


def materialize_model_input(
    input_path: Path, output_path: Path, target_size: tuple[int, int, int] = EXPECTED_READY_SHAPE
) -> dict[str, Any]:
    """Create the deployment view after checking foreground and normalization."""
    import torch
    from monai.transforms import (
        Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, LoadImaged,
        NormalizeIntensityd, Orientationd, Resized, SpatialPadd,
    )

    pre_resize_transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=target_size),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            SpatialPadd(keys=["image"], spatial_size=target_size),
            EnsureTyped(keys=["image"]),
        ]
    )
    item = pre_resize_transforms({"image": str(input_path)})
    pre_resize = item["image"]
    pre_resize_array = (
        pre_resize.detach().cpu().numpy()
        if torch.is_tensor(pre_resize)
        else np.asarray(pre_resize)
    )
    pre_resize_foreground = pre_resize_array != 0
    foreground_voxels = int(np.count_nonzero(pre_resize_foreground))
    if foreground_voxels < 1000:
        raise InsufficientForegroundError(
            "Too few foreground voxels before BrainSegFounder resize: "
            f"{foreground_voxels}"
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
            Resized(keys=["image"], spatial_size=target_size, mode="trilinear"),
            EnsureTyped(keys=["image"]),
        ]
    )
    tensor = resize_transform(item)["image"]
    affine = tensor.affine
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    array = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    if tuple(array.shape) != (1, *target_size):
        raise RuntimeError(f"Unexpected model-view shape: {array.shape}")
    array = np.asarray(array[0], dtype=np.float32)
    affine = np.asarray(affine, dtype=np.float64)
    if not np.isfinite(array).all() or affine.shape != (4, 4):
        raise RuntimeError("Invalid MONAI output or transformed affine")
    image = nib.Nifti1Image(array, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(output_path))
    foreground = array != 0
    return {
        "shape": list(array.shape),
        "foreground_voxels_before_resize": foreground_voxels,
        "normalization_before_resize": {
            "mean_nonzero": pre_resize_mean,
            "standard_deviation_nonzero": pre_resize_standard_deviation,
        },
        "mean_nonzero": float(np.mean(array[foreground])),
        "standard_deviation_nonzero": float(np.std(array[foreground])),
        "transformed_affine": affine.tolist(),
    }


def validate_mni_output(path: Path, reference_path: Path) -> dict[str, Any]:
    """Require a finite nonempty volume on the exact MNI reference grid."""
    image = nib.load(str(path))
    reference = nib.load(str(reference_path))
    if tuple(image.shape) != tuple(reference.shape) or not np.allclose(
        image.affine, reference.affine, rtol=0.0, atol=1e-4
    ):
        raise ValueError(f"Output is not on the MNI reference grid: {path}")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not np.any(data != 0):
        raise ValueError(f"Invalid or empty MNI output: {path}")
    return {
        "shape": list(image.shape),
        "spacing_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "nonzero_voxels": int(np.count_nonzero(data)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def validate_ready_output(path: Path) -> dict[str, Any]:
    """Require the fixed shape and post-resize intensity range used by the encoder."""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing model-ready output: {path}")
    image = nib.load(str(path))
    if tuple(image.shape) != EXPECTED_READY_SHAPE:
        raise ValueError(f"Unexpected model-ready shape {image.shape}: {path}")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite model-ready values: {path}")
    foreground = data != 0
    count = int(np.count_nonzero(foreground))
    if count < 1000:
        raise ValueError(f"Too few nonzero model-ready voxels ({count}): {path}")
    mean = float(np.mean(data[foreground]))
    standard_deviation = float(np.std(data[foreground]))
    if abs(mean) > 1e-3 or not 0.5 < standard_deviation <= 1.05:
        raise ValueError(
            "Unexpected post-resize BrainSegFounder intensities: "
            f"mean={mean}, std={standard_deviation}"
        )
    return {
        "shape": list(image.shape),
        "spacing_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "nonzero_voxels": count,
        "nonzero_fraction": float(count / data.size),
        "mean_nonzero": mean,
        "standard_deviation_nonzero": standard_deviation,
        "affine": np.asarray(image.affine, dtype=np.float64).tolist(),
    }


def qc_is_current(paths: dict[str, Path], raw: Path, assets: dict[str, Any]) -> bool:
    """Accept reuse only when contract, inputs, assets, and every output match."""
    if not paths["qc"].is_file() or any(not paths[key].is_file() for key in SPATIAL_KEYS):
        return False
    try:
        qc = json.loads(paths["qc"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    raw_stat = raw.stat()
    raw_record = {"path": str(raw.resolve()), "size_bytes": raw_stat.st_size, "mtime_ns": raw_stat.st_mtime_ns}
    hashes = qc.get("output_sha256", {})
    return (
        qc.get("status") == "PASS"
        and qc.get("pipeline_contract_sha256") == CONTRACT_SHA256
        and qc.get("raw_input") == raw_record
        and qc.get("assets") == assets
        and all(hashes.get(key) == sha256_file(paths[key]) for key in SPATIAL_KEYS)
    )


def validate_published_subject(
    raw: Path,
    output_dir: Path,
    template: Path,
    assets: dict[str, Any],
) -> dict[str, Any]:
    """Revalidate exact provenance hashes and both published image contracts."""
    stem = acquisition_stem(raw)
    paths = output_paths(Path(output_dir).resolve() / stem, stem)
    if not qc_is_current(paths, raw.resolve(), assets):
        raise RuntimeError("QC, contract, input, asset, or output hash mismatch")
    mni = validate_mni_output(paths["fnirt_brain"], template)
    ready = validate_ready_output(paths["ready"])
    qc = json.loads(paths["qc"].read_text(encoding="utf-8"))
    return {
        "stem": stem,
        "ready": str(paths["ready"]),
        "qc": str(paths["qc"]),
        "registration_recovery_used": bool(
            qc.get("registration_recovery", {}).get("used", False)
        ),
        "generator_sha256": qc.get("script_sha256"),
        "mni": mni,
        "model_ready": ready,
    }


def run_logged_acquisition(
    raw: Path,
    output_dir: Path,
    log_dir: Path,
    template: Path,
    ukb_asset_dir: Path,
    fsl_schedule: Path,
    threads: int,
    force: bool,
    timeout_seconds: float | None,
) -> tuple[int, Path]:
    """Run one isolated Python child and capture its console output per acquisition."""
    stem = acquisition_stem(raw)
    log_path = Path(log_dir).resolve() / f"brainsegfounder_{stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "single",
        "--input-file",
        str(raw),
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--template",
        str(template),
        "--ukb-asset-dir",
        str(ukb_asset_dir),
        "--fsl-schedule",
        str(fsl_schedule),
        "--threads",
        str(threads),
    ]
    if force:
        command.append("--force")
    if timeout_seconds is not None:
        command.extend(("--step-timeout-seconds", str(timeout_seconds)))
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"=== {stem} ===\n")
        handle.write(f"input    : {raw}\n")
        handle.write(f"template : {template}\n")
        handle.write(f"started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=worker_environment(threads),
            check=False,
        )
        handle.write(f"\nexit_code: {completed.returncode}\n")
    return completed.returncode, log_path


def build_linear_transforms(
    staged: dict[str, Path],
    fsl_schedule: Path,
    executables: dict[str, str],
    log: TextIO,
    timeout_seconds: float | None,
) -> dict[str, list[str]]:
    """Build the released UKB crop-to-raw and raw-to-MNI linear transforms."""
    flirt_command = run_external(
        [
            executables["flirt"],
            "-in",
            str(staged["reduced_t1"]),
            "-ref",
            str(staged["reoriented_t1"]),
            "-omat",
            str(staged["t1_to_raw_mat"]),
            "-schedule",
            str(fsl_schedule),
        ],
        log,
        timeout_seconds,
    )
    inverse_command = run_external(
        [
            executables["convert_xfm"],
            "-omat",
            str(staged["raw_to_t1_mat"]),
            "-inverse",
            str(staged["t1_to_raw_mat"]),
        ],
        log,
        timeout_seconds,
    )
    concat_command = run_external(
        [
            executables["convert_xfm"],
            "-omat",
            str(staged["linear_mat"]),
            "-concat",
            str(staged["roi_to_mni_mat"]),
            str(staged["t1_to_raw_mat"]),
        ],
        log,
        timeout_seconds,
    )
    return {"flirt": flirt_command, "inverse": inverse_command, "concat": concat_command}


def build_header_geometry_transforms(
    staged: dict[str, Path],
    geometry_preview: Path,
    executables: dict[str, str],
    log: TextIO,
    timeout_seconds: float | None,
) -> dict[str, list[str]]:
    """Recover crop-to-raw geometry with FLIRT ``-usesqform`` after an empty result."""
    try:
        flirt_command = run_external(
            [
                executables["flirt"],
                "-in",
                str(staged["reduced_t1"]),
                "-ref",
                str(staged["reoriented_t1"]),
                "-applyxfm",
                "-usesqform",
                "-omat",
                str(staged["t1_to_raw_mat"]),
                "-out",
                str(geometry_preview),
            ],
            log,
            timeout_seconds,
        )
    finally:
        geometry_preview.unlink(missing_ok=True)
    inverse_command = run_external(
        [
            executables["convert_xfm"],
            "-omat",
            str(staged["raw_to_t1_mat"]),
            "-inverse",
            str(staged["t1_to_raw_mat"]),
        ],
        log,
        timeout_seconds,
    )
    concat_command = run_external(
        [
            executables["convert_xfm"],
            "-omat",
            str(staged["linear_mat"]),
            "-concat",
            str(staged["roi_to_mni_mat"]),
            str(staged["t1_to_raw_mat"]),
        ],
        log,
        timeout_seconds,
    )
    return {
        "flirt_usesqform": flirt_command,
        "inverse": inverse_command,
        "concat": concat_command,
    }


def run_nonlinear_chain(
    staged: dict[str, Path],
    template: Path,
    assets: dict[str, Any],
    executables: dict[str, str],
    log: TextIO,
    timeout_seconds: float | None,
    progress_message: str | None = None,
) -> dict[str, list[str]]:
    """Run FNIRT, GDC-off warp composition, resampling, and the UKB brain mask."""
    fnirt_command = run_external(
        [
            executables["fnirt"],
            f"--in={staged['reduced_t1']}",
            f"--ref={template}",
            f"--aff={staged['linear_mat']}",
            f"--config={assets['fnirt_config']['path']}",
            f"--refmask={assets['dilated_mask']['path']}",
            f"--logout={staged['fnirt_log']}",
            f"--cout={staged['warp_coef']}",
            f"--fout={staged['warp_field']}",
            f"--jout={staged['warp_jac']}",
            f"--iout={staged['fnirt_preview']}",
            "--interp=spline",
        ],
        log,
        timeout_seconds,
    )
    if progress_message:
        print(progress_message)
    convertwarp_command = run_external(
        [
            executables["convertwarp"],
            f"--ref={template}",
            f"--premat={staged['raw_to_t1_mat']}",
            f"--warp2={staged['warp_field']}",
            f"--out={staged['combined_warp']}",
        ],
        log,
        timeout_seconds,
    )
    applywarp_command = run_external(
        [
            executables["applywarp"],
            "--rel",
            "-i",
            str(staged["reoriented_t1"]),
            "-r",
            str(template),
            "-w",
            str(staged["combined_warp"]),
            "-o",
            str(staged["warped_unmasked"]),
            "--interp=spline",
        ],
        log,
        timeout_seconds,
    )
    mask_command = run_external(
        [
            executables["fslmaths"],
            str(staged["warped_unmasked"]),
            "-mul",
            assets["mask"]["path"],
            str(staged["fnirt_brain"]),
        ],
        log,
        timeout_seconds,
    )
    return {
        "fnirt": fnirt_command,
        "convertwarp": convertwarp_command,
        "applywarp": applywarp_command,
        "fslmaths_mask": mask_command,
    }


def preprocess_single(
    input_file: str,
    output_dir: str,
    template_path: str,
    ukb_asset_dir: Path,
    fsl_schedule: Path,
    executables: dict[str, str],
    *,
    force: bool = False,
    timeout_seconds: float | None = None,
) -> str:
    """Run the complete UKB GDC-off chain for one ON-Harmony session."""
    raw = Path(input_file).resolve()
    stem = raw.name.removesuffix(".nii.gz").removesuffix(".nii")
    subject_dir = Path(output_dir).resolve() / stem
    subject_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(subject_dir, stem)
    template = Path(template_path).resolve()
    assets = validate_assets(template, ukb_asset_dir, fsl_schedule)

    if qc_is_current(paths, raw, assets) and not force:
        print(f"  SKIP (validated current output): {paths['ready']}")
        return str(paths["ready"])
    if any(paths[key].exists() for key in (*SPATIAL_KEYS, "qc")) and not force:
        raise RuntimeError(
            f"Stale or legacy BrainSegFounder outputs exist for {stem}; rerun with --force"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    stage_dir = subject_dir / f".staging_{run_id}_{uuid.uuid4().hex[:8]}"
    stage_dir.mkdir()
    staged = {key: stage_dir / paths[key].name for key in SPATIAL_KEYS}
    try:
        with paths["log"].open("w", encoding="utf-8") as log:
            log.write(f"pipeline_version={PIPELINE_VERSION}\ncontract={CONTRACT_SHA256}\n")
            log.write(f"raw={raw}\ngradient_distortion_correction=OFF\n")
            print(f"  [Preprocessing 1/7] fslreorient2std input adapter: {stem}")
            reorient_command, reorientation = reorient_to_fsl_standard(
                raw,
                staged["reoriented_t1"],
                executables["fslreorient2std"],
                log,
                timeout_seconds,
            )
            log.write(
                f"axcodes={reorientation['input_axcodes']} -> "
                f"{reorientation['output_axcodes']}\n"
            )
            print(f"  [Preprocessing 2/7] robustfov crop + recursive BET: {stem}")
            robust_command, robust_output = run_external_capture(
                [executables["robustfov"], "-i", str(staged["reoriented_t1"])],
                log,
                timeout_seconds,
            )
            head_top = parse_robustfov_head_top(robust_output)
            crop_command = run_external(
                [executables["fslmaths"], str(staged["reoriented_t1"]), "-roi", "0", "-1", "0", "-1",
                 str(head_top), str(UKB_ROI_DEPTH), "0", "1", str(staged["fov_roi"])],
                log, timeout_seconds,
            )
            bet_command = run_external(
                [executables["bet"], str(staged["fov_roi"]), str(staged["fov_brain"]), "-R"],
                log, timeout_seconds,
            )

            print(f"  [Preprocessing 3/7] standard_space_roi: {stem}")
            ssroi_command = run_external(
                [executables["standard_space_roi"], str(staged["fov_brain"]),
                 str(staged["reduced_t1"]), "-maskNONE", "-ssref", assets["brain"]["path"],
                 "-altinput", str(staged["reoriented_t1"]), "-d"],
                log, timeout_seconds,
            )
            base = remove_extensions(staged["reduced_t1"])
            generated_matrix = Path(f"{base}_tmp_to_std.mat")
            if not generated_matrix.is_file():
                raise FileNotFoundError(f"standard_space_roi matrix missing: {generated_matrix}")
            os.replace(generated_matrix, staged["roi_to_mni_mat"])
            for temporary in stage_dir.glob(f"{base.name}_tmp*"):
                if temporary.is_dir():
                    shutil.rmtree(temporary)
                elif temporary.is_file():
                    temporary.unlink()

            print(f"  [Preprocessing 4/7] UKB affine composition: {stem}")
            matrix_commands = build_linear_transforms(
                staged,
                fsl_schedule,
                executables,
                log,
                timeout_seconds,
            )

            print(f"  [Preprocessing 5/7] UKB-configured FNIRT: {stem}")
            registration_commands = run_nonlinear_chain(
                staged,
                template,
                assets,
                executables,
                log,
                timeout_seconds,
                f"  [Preprocessing 6/7] GDC-off warp + UKB MNI mask: {stem}",
            )
            registration_attempts = [
                {
                    "strategy": "released_ukb_intensity_translation",
                    "linear_transforms": matrix_commands,
                    "nonlinear_chain": registration_commands,
                }
            ]
            recovery_trigger: str | None = None

            print(f"  [Preprocessing 7/7] MONAI 96-cube deployment view: {stem}")
            try:
                model_view = materialize_model_input(
                    staged["fnirt_brain"], staged["ready"]
                )
            except InsufficientForegroundError as error:
                recovery_trigger = str(error)
                print(
                    "  [Recovery] Empty MNI result; rebuilding crop-to-raw "
                    f"geometry with FLIRT -usesqform: {stem}"
                )
                log.write(
                    "registration_recovery=nifti_header_geometry\n"
                    f"registration_recovery_trigger={recovery_trigger}\n"
                )
                log.flush()
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
                    staged,
                    stage_dir / f"{stem}_header_geometry_preview.nii.gz",
                    executables,
                    log,
                    timeout_seconds,
                )
                registration_commands = run_nonlinear_chain(
                    staged,
                    template,
                    assets,
                    executables,
                    log,
                    timeout_seconds,
                    f"  [Recovery] Reapplying GDC-off warp + UKB MNI mask: {stem}",
                )
                registration_attempts.append(
                    {
                        "strategy": "nifti_header_geometry_recovery",
                        "trigger": recovery_trigger,
                        "linear_transforms": matrix_commands,
                        "nonlinear_chain": registration_commands,
                    }
                )
                model_view = materialize_model_input(
                    staged["fnirt_brain"], staged["ready"]
                )
            mni_stats = validate_mni_output(staged["fnirt_brain"], template)
            ready_stats = validate_ready_output(staged["ready"])

        missing = [
            str(staged[key])
            for key in SPATIAL_KEYS
            if not staged[key].is_file() or staged[key].stat().st_size == 0
        ]
        if missing:
            raise FileNotFoundError(
                "The UKB chain did not create all required outputs: " + ", ".join(missing)
            )
        for key in SPATIAL_KEYS:
            os.replace(staged[key], paths[key])
        raw_stat = raw.stat()
        qc = {
            "status": "PASS",
            "created_utc": utc_now(),
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_contract_sha256": CONTRACT_SHA256,
            "contract": CONTRACT,
            "script_sha256": sha256_file(SCRIPT_PATH),
            "raw_input": {"path": str(raw), "size_bytes": raw_stat.st_size, "mtime_ns": raw_stat.st_mtime_ns},
            "assets": assets,
            "input_reorientation": reorientation,
            "robustfov_head_top": head_top,
            "commands": {
                "fslreorient2std": reorient_command,
                "robustfov": robust_command, "fslmaths_roi": crop_command, "bet": bet_command,
                "standard_space_roi": ssroi_command,
                "linear_transforms": matrix_commands,
                "nonlinear_chain": registration_commands,
            },
            "registration_recovery": {
                "used": recovery_trigger is not None,
                "trigger": recovery_trigger,
                "effective_strategy": registration_attempts[-1]["strategy"],
                "attempts": registration_attempts,
            },
            "mni_output": mni_stats,
            "model_view": model_view,
            "model_ready_output": ready_stats,
            "output_sha256": {key: sha256_file(paths[key]) for key in SPATIAL_KEYS},
        }
        atomic_json(paths["qc"], qc)
        stage_dir.rmdir()
        print(f"  Done: {paths['ready']}")
        return str(paths["ready"])
    except Exception:
        if stage_dir.exists():
            failed = stage_dir.with_name(stage_dir.name.replace(".staging_", ".failed_", 1))
            os.replace(stage_dir, failed)
        raise


def require_non_fsl_python() -> None:
    """Reject FSL's bundled Python; model-view materialization requires the conda stack."""
    executable = Path(sys.executable).resolve()
    fsldir = os.environ.get("FSLDIR")
    inside_fsldir = False
    if fsldir:
        try:
            executable.relative_to(Path(fsldir).expanduser().resolve())
            inside_fsldir = True
        except ValueError:
            pass
    fsl_path_component = any(
        re.fullmatch(r"fsl(?:[_-]\d.*)?|fslpython.*", part, flags=re.IGNORECASE)
        for part in executable.parts
    )
    if inside_fsldir or fsl_path_component:
        raise RuntimeError(
            f"Refusing FSL Python {executable}; activate the project conda environment"
        )


def _cohort_inputs(args: argparse.Namespace) -> list[Path]:
    """Resolve cohort inputs, defaulting to the canonical 165-row raw registry."""
    filelist = args.filelist
    expected_inputs = args.expected_inputs
    if filelist is None and args.input_file is None and args.input_dir is None:
        filelist = DEFAULT_RAW_FILELIST
        if expected_inputs is None:
            expected_inputs = 165
    return load_inputs(
        filelist=filelist,
        input_file=args.input_file,
        input_dir=args.input_dir,
        expected_inputs=expected_inputs,
    )


def command_single(args: argparse.Namespace) -> None:
    """Run the unchanged seven-stage pipeline serially for selected inputs."""
    require_non_fsl_python()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = str(args.threads)

    if args.monai_only:
        sources = sorted(Path(args.output_dir).glob("**/*_fnirt_brain.nii.gz"))
        print(f"BrainSegFounder MONAI-only re-run: {len(sources)} files")
        for source in sources:
            stem = source.name.removesuffix("_fnirt_brain.nii.gz")
            target = source.with_name(f"{stem}_brainsegfounder_ready.nii.gz")
            materialize_model_input(source, target)
            print(f"  Done: {target}")
        return

    if args.input_file is None and args.input_dir is None:
        raise ValueError("Provide --input-file or --input-dir")
    files = load_inputs(input_file=args.input_file, input_dir=args.input_dir)
    template, fsl_schedule, ukb_asset_dir = resolve_resources(args)
    executables = {name: resolve_executable(name) for name in COMMAND_NAMES}
    assets = validate_assets(template, ukb_asset_dir, fsl_schedule)
    print(f"BrainSegFounder UKB GDC-off preprocessing: {len(files)} files")
    print(f"  template : {template}")
    print(f"  UKB assets: {ukb_asset_dir}")
    print("  GDC      : OFF")
    print(f"  contract : {CONTRACT_SHA256}")
    print(f"  mask SHA : {assets['mask']['sha256']}")
    for input_file in files:
        preprocess_single(
            str(input_file),
            str(Path(args.output_dir).expanduser().resolve()),
            str(template),
            ukb_asset_dir,
            fsl_schedule,
            executables,
            force=args.force,
            timeout_seconds=args.step_timeout_seconds,
        )
    print("All done.")


def command_validate(args: argparse.Namespace) -> None:
    """Revalidate every subject's hashes and image contracts without preprocessing."""
    require_non_fsl_python()
    inputs = _cohort_inputs(args)
    template, fsl_schedule, ukb_asset_dir = resolve_resources(args)
    assets = validate_assets(template, ukb_asset_dir, fsl_schedule)
    output_dir = Path(args.output_dir).expanduser().resolve()
    failures: list[tuple[Path, Exception]] = []
    recoveries = []
    generators: dict[str, int] = {}
    for index, raw in enumerate(inputs, start=1):
        try:
            result = validate_published_subject(raw, output_dir, template, assets)
            if result["registration_recovery_used"]:
                recoveries.append(result["stem"])
            generator = str(result["generator_sha256"])
            generators[generator] = generators.get(generator, 0) + 1
            if index % 25 == 0 or index == len(inputs):
                print(f"  [{index}/{len(inputs)}] VALIDATED {result['stem']}")
        except Exception as error:
            failures.append((raw, error))
            print(
                f"  [{index}/{len(inputs)}] FAILED    {acquisition_stem(raw)}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
    print(f"  exact current QC : {len(inputs) - len(failures)}/{len(inputs)}")
    print(f"  recovery used    : {len(recoveries)}")
    print(f"  generator hashes : {generators}")
    if failures:
        raise SystemExit(1)
    if args.publish_filelist is not None:
        destination = atomic_publish_filelist(inputs, output_dir, args.publish_filelist)
        print(f"Published extractor filelist: {destination} ({len(inputs)} paths)")


def command_run(args: argparse.Namespace) -> None:
    """Run acquisitions in isolated children, validate all outputs, then publish."""
    require_non_fsl_python()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be positive")
    inputs = _cohort_inputs(args)
    template, fsl_schedule, ukb_asset_dir = resolve_resources(args)
    assets = validate_assets(template, ukb_asset_dir, fsl_schedule)
    executables = {name: resolve_executable(name) for name in COMMAND_NAMES}
    del executables  # preflight only; isolated children resolve the same PATH
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    existing = []
    for raw in inputs:
        stem = acquisition_stem(raw)
        if output_paths(output_dir / stem, stem)["ready"].is_file():
            existing.append(stem)
    if args.require_clean and existing:
        raise RuntimeError(
            f"--require-clean was requested, but {len(existing)} final outputs already exist"
        )

    print("BrainSegFounder preprocessing (UKB GDC-off)")
    print(f"  inputs    : {len(inputs)}")
    print(f"  template  : {template}")
    print(f"  UKB assets: {ukb_asset_dir}")
    print(f"  outputs   : {output_dir}")
    print(f"  logs      : {log_dir}/brainsegfounder_<stem>.log")
    print(f"  Python    : {sys.executable}")
    print(f"  workers   : {args.workers}")
    print(f"  threads   : {args.threads_per_worker} per worker")
    print(f"  pipeline  : {PIPELINE_VERSION}")
    print(f"  contract  : {CONTRACT_SHA256}")
    print("  GDC       : OFF (scanner-specific coefficients unavailable)")

    failures: list[tuple[Path, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_logged_acquisition,
                raw,
                output_dir,
                log_dir,
                template,
                ukb_asset_dir,
                fsl_schedule,
                args.threads_per_worker,
                args.force,
                args.step_timeout_seconds,
            ): raw
            for raw in inputs
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            raw = futures[future]
            try:
                return_code, log_path = future.result()
            except Exception as error:
                failures.append((raw, error))
                print(
                    f"  [{completed_count}/{len(inputs)}] FAIL {acquisition_stem(raw)}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            if return_code == 0:
                print(
                    f"  [{completed_count}/{len(inputs)}] OK   {acquisition_stem(raw)}",
                    flush=True,
                )
            else:
                error = RuntimeError(f"exit code {return_code}; see {log_path}")
                failures.append((raw, error))
                print(
                    f"  [{completed_count}/{len(inputs)}] FAIL {acquisition_stem(raw)} "
                    f"(rc={return_code}, see {log_path})",
                    flush=True,
                )
    if failures:
        print(f"Preprocessing failed: {len(failures)}/{len(inputs)} acquisitions")
        raise SystemExit(1)

    validation_args = argparse.Namespace(**vars(args))
    validation_args.func = command_validate
    command_validate(validation_args)


def add_resource_arguments(parser: argparse.ArgumentParser) -> None:
    """Add FSL and UKB resource options shared by all execution modes."""
    parser.add_argument("--fsldir", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--ukb-asset-dir", type=Path, default=DEFAULT_UKB_ASSET_DIR)
    parser.add_argument("--fsl-schedule", type=Path)


def add_cohort_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ordered cohort registry and publication arguments."""
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--filelist", type=Path)
    source.add_argument("--input-file", "--input_file", dest="input_file", type=Path)
    source.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path)
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--expected-inputs", type=int)
    parser.add_argument(
        "--publish-filelist",
        type=Path,
        help=(
            "Atomically publish ordered validated outputs; canonical destination is "
            f"{DEFAULT_PUBLISHED_FILELIST}"
        ),
    )
    add_resource_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    """Build the integrated single-scan, cohort-run, and validation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser("single", help="Run one file or directory serially")
    source = single.add_mutually_exclusive_group()
    source.add_argument("--input-file", "--input_file", dest="input_file", type=Path)
    source.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path)
    single.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path, required=True
    )
    add_resource_arguments(single)
    single.add_argument("--threads", type=int, default=2)
    single.add_argument("--force", action="store_true")
    single.add_argument("--monai-only", "--monai_only", dest="monai_only", action="store_true")
    single.add_argument("--step-timeout-seconds", type=float)
    single.set_defaults(func=command_single)

    run = commands.add_parser("run", help="Run a cohort with bounded parallelism")
    add_cohort_arguments(run)
    run.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    run.add_argument("--workers", type=int, default=20)
    run.add_argument("--threads-per-worker", type=int, default=2)
    run.add_argument("--force", action="store_true")
    run.add_argument("--require-clean", action="store_true")
    run.add_argument("--step-timeout-seconds", type=float)
    run.set_defaults(func=command_run)

    validate = commands.add_parser("validate", help="Validate existing cohort outputs")
    add_cohort_arguments(validate)
    validate.set_defaults(func=command_validate)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch the integrated CLI while preserving the former single-file syntax."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {"single", "run", "validate"}
    if arguments and arguments[0] not in commands and arguments[0] not in {"-h", "--help"}:
        arguments.insert(0, "single")
    args = build_parser().parse_args(arguments)
    args.func(args)


if __name__ == "__main__":
    main()
