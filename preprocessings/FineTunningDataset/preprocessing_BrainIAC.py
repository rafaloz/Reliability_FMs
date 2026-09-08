#!/usr/bin/env python3
"""Create release-aligned BrainIAC inputs for the brain-age cohort.

Scientific pipeline (raw T1w -> model input):

1. N4 bias-field correction with the SimpleITK procedural overload and no
   explicit mask.
2. Rigid registration to BrainIAC's released ``temp_head.nii.gz`` atlas after
   placing the atlas on a 1 mm isotropic grid.
3. Skull stripping with HD-BET, fast inference and test-time augmentation
   disabled.
4. Trilinear resize to 96 x 96 x 96.
5. Per-channel z-score normalization over nonzero voxels.

Steps 1-3 follow BrainIAC's released preprocessing code. Steps 4-5 follow its
released validation transform. The paper describes N4, 1 mm resampling, rigid
atlas registration and HD-BET, but does not identify the atlas or all inference
parameters precisely; the release therefore supplies the reproducible details.
The fixed registration metric seed is a study reproducibility control and does
not change the registration objective.

The reporting summary additionally lists denoising with kernel size 3, but no
such operation appears in the paper's Methods pipeline or the released
preprocessing/validation code. It is therefore not introduced here.

The definitive brain-age manifest is the default input. Outputs are scoped by
dataset and subject to prevent identifier collisions. N4, registered, transform,
brain, mask and model-ready files are retained together with per-subject logs,
QC montages and cryptographic provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
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
import SimpleITK as sitk


SCRIPT_PATH = Path(__file__).resolve()
FINETUNING_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = FINETUNING_ROOT.parents[1]

DEFAULT_MANIFEST = (
    FINETUNING_ROOT
    / "brainage_splits"
    / "brainage_definitive_subject_manifest.tsv"
)
DEFAULT_OUTPUT_ROOT = FINETUNING_ROOT / "preproc_BrainIAC"
DEFAULT_LOG_ROOT = FINETUNING_ROOT / "logs_preprocs" / "BrainIAC"
DEFAULT_TEMPLATE = PROJECT_ROOT / "templates" / "temp_head.nii.gz"

PIPELINE_VERSION = "brainiac-release-aligned-finetuning-v1"
OFFICIAL_BRAINIAC_COMMIT = "ba60f45bed832ee0e5683c678caeaeefe2072f0d"
EXPECTED_TEMPLATE_SHA256 = (
    "d8eba10e9528a5212b78fb1be477d5e8229751bfec5e1e1a648dc81ff83a622e"
)
EXPECTED_TEMPLATE_SIZE = (170, 206, 162)
EXPECTED_TEMPLATE_SPACING = (1.0, 1.0, 1.0)
EXPECTED_READY_SHAPE = (96, 96, 96)
REGISTRATION_METRIC_SEED = 42
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
    "official_brainiac_commit": OFFICIAL_BRAINIAC_COMMIT,
    "template_sha256": EXPECTED_TEMPLATE_SHA256,
    "n4": {
        "implementation": "SimpleITK.N4BiasFieldCorrection(image)",
        "explicit_mask": False,
        "custom_parameters": False,
    },
    "registration": {
        "transform": "Euler3D rigid",
        "atlas_spacing_mm": EXPECTED_TEMPLATE_SPACING,
        "metric": "Mattes mutual information",
        "histogram_bins": 50,
        "sampling": "RANDOM 0.01",
        "sampling_seed": REGISTRATION_METRIC_SEED,
        "optimizer": "gradient descent",
        "learning_rate": 1.0,
        "iterations": 100,
        "convergence_minimum": 1e-6,
        "convergence_window": 10,
        "shrink_factors": [4, 2, 1],
        "smoothing_sigmas_mm": [2, 1, 0],
        "interpolation": "linear",
    },
    "skull_stripping": {
        "implementation": "HD-BET CLI",
        "mode": "fast/default release network",
        "tta": False,
        "retain_mask": True,
    },
    "model_view": {
        "shape": EXPECTED_READY_SHAPE,
        "resize": "MONAI Resize trilinear",
        "normalization": "MONAI NormalizeIntensity nonzero=True channel_wise=True",
        "stored_affine": "identity",
    },
}
PIPELINE_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(CONTRACT_PAYLOAD, sort_keys=True).encode("utf-8")
).hexdigest()


# Return a timezone-aware UTC timestamp for subject QC and run manifests.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Compute a file digest in chunks so large NIfTI files are not read into memory at once.
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Write JSON via an atomic replacement so interrupted jobs cannot leave partial QC files.
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


# Write a cohort status table atomically for reliable resume and failure accounting.
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


# Print one progress message and mirror it into the subject log when one is open.
def log_message(message: str, handle: TextIO | None = None) -> None:
    print(message, flush=True)
    if handle is not None:
        handle.write(message + "\n")
        handle.flush()


# Record inexpensive source-file provenance used to decide whether outputs remain current.
def raw_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


# Return the stem of a .nii or .nii.gz path without relying on Path.stem twice.
def nifti_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    raise ValueError(f"Expected a NIfTI filename, got {path}")


# Return the mask filename produced by the HD-BET 2 single-file CLI.
def hdbet_generated_mask_path(output_path: Path) -> Path:
    return output_path.with_name(f"{nifti_stem(output_path)}_bet.nii.gz")


# Load the definitive TSV and enforce schema, inclusion, uniqueness and output-path safety.
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
        if "preproc_brainiac" not in groups:
            raise ValueError(f"{key} does not request preproc_brainiac")
        row["dataset"] = dataset
        row["subject"] = subject
        row["raw_t1w_path"] = raw_path
        keys.add(key)
        raw_paths.add(raw_path)
    return rows


# Apply optional cohort selectors and reject unknown or empty selections before processing.
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


# Construct the dataset-scoped retained intermediates, final image and QC filenames.
def output_paths(output_root: Path, dataset: str, subject: str) -> dict[str, Path]:
    subject_dir = output_root / dataset / subject
    return {
        "subject_dir": subject_dir,
        "n4": subject_dir / f"{subject}_n4.nii.gz",
        "registered": subject_dir / f"{subject}_registered.nii.gz",
        "transform": subject_dir / f"{subject}_rigid.tfm",
        "brain": subject_dir / f"{subject}_brain.nii.gz",
        "mask": subject_dir / f"{subject}_brain_mask.nii.gz",
        "ready": subject_dir / f"{subject}_brainiac_ready.nii.gz",
        "qc": subject_dir / "preprocessing_qc.json",
        "qc_png": subject_dir / "preprocessing_qc.png",
    }


# Verify that the exact atlas distributed by BrainIAC is being used, then capture geometry.
def validate_template(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"BrainIAC template not found: {path}")
    digest = sha256_file(path)
    if digest != EXPECTED_TEMPLATE_SHA256:
        raise ValueError(
            f"Unexpected BrainIAC template SHA-256 for {path}: {digest}; "
            f"expected {EXPECTED_TEMPLATE_SHA256}"
        )
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    size = tuple(int(value) for value in image.GetSize())
    spacing = tuple(float(value) for value in image.GetSpacing())
    if size != EXPECTED_TEMPLATE_SIZE:
        raise ValueError(f"Unexpected template size: {size}, expected {EXPECTED_TEMPLATE_SIZE}")
    if not np.allclose(spacing, EXPECTED_TEMPLATE_SPACING, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Unexpected template spacing: {spacing}, expected {EXPECTED_TEMPLATE_SPACING}"
        )
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest,
        "size": list(size),
        "spacing_mm": list(spacing),
        "origin": [float(value) for value in image.GetOrigin()],
        "direction": [float(value) for value in image.GetDirection()],
    }


# Reproduce the release's explicit 1 mm atlas-resampling step and preserve its physical frame.
def prepare_fixed_template(path: Path) -> sitk.Image:
    fixed = sitk.ReadImage(str(path), sitk.sitkFloat32)
    old_size = fixed.GetSize()
    old_spacing = fixed.GetSpacing()
    new_spacing = EXPECTED_TEMPLATE_SPACING
    new_size = [
        int(round(old_size[index] * old_spacing[index] / new_spacing[index]))
        for index in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(fixed.GetOrigin())
    resampler.SetOutputDirection(fixed.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(fixed)


# Apply the exact one-image SimpleITK N4 overload used in the released BrainIAC script.
def n4_bias_correction(input_path: Path, output_path: Path) -> None:
    image = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
    corrected = sitk.N4BiasFieldCorrection(image)
    sitk.WriteImage(corrected, str(output_path))


# Rigidly register one N4 image to the released atlas and retain the fitted transform.
def register_to_template(
    input_path: Path,
    template_path: Path,
    output_path: Path,
    transform_path: Path,
) -> dict[str, Any]:
    fixed = prepare_fixed_template(template_path)
    moving = sitk.ReadImage(str(input_path), sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01, REGISTRATION_METRIC_SEED)
    registration.SetInterpolator(sitk.sitkLinear)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration.Execute(fixed, moving)
    registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    sitk.WriteImage(registered, str(output_path))
    sitk.WriteTransform(final_transform, str(transform_path))
    return {
        "metric_value": float(registration.GetMetricValue()),
        "optimizer_iteration": int(registration.GetOptimizerIteration()),
        "optimizer_stop_condition": registration.GetOptimizerStopConditionDescription(),
        "sampling_seed": REGISTRATION_METRIC_SEED,
    }


# Invoke the modern HD-BET CLI in the mode corresponding to the release's fast/tta=0 call.
def skull_strip_hdbet(
    input_path: Path,
    output_path: Path,
    executable: str,
    device: str,
    timeout_seconds: float | None,
    log_handle: TextIO,
) -> list[str]:
    command = [
        executable,
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-device",
        device,
        "--disable_tta",
        "--save_bet_mask",
    ]
    log_handle.write("HD-BET command=" + " ".join(command) + "\n")
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
        raise RuntimeError(f"HD-BET failed with exit code {completed.returncode}")
    mask_path = hdbet_generated_mask_path(output_path)
    if not output_path.is_file() or not mask_path.is_file():
        raise RuntimeError(
            f"HD-BET did not create the expected brain/mask pair: {output_path}, {mask_path}"
        )
    return command


# Apply the released BrainIAC validation transform and store its model-ready 96-cube.
def materialize_model_input(input_path: Path, output_path: Path) -> None:
    from monai.transforms import EnsureChannelFirst, LoadImage, NormalizeIntensity, Resize

    image = LoadImage(image_only=True)(str(input_path))
    image = EnsureChannelFirst()(image)
    image = Resize(spatial_size=EXPECTED_READY_SHAPE, mode="trilinear")(image)
    image = NormalizeIntensity(nonzero=True, channel_wise=True)(image)
    array = np.asarray(image.squeeze(0).detach().cpu(), dtype=np.float32)
    nib.save(nib.Nifti1Image(array, affine=np.eye(4)), str(output_path))


# Compare SimpleITK grid fields exactly enough to detect registration onto the wrong atlas.
def grid_matches(image: sitk.Image, reference: sitk.Image) -> bool:
    return (
        image.GetSize() == reference.GetSize()
        and np.allclose(image.GetSpacing(), reference.GetSpacing(), rtol=0.0, atol=1e-6)
        and np.allclose(image.GetOrigin(), reference.GetOrigin(), rtol=0.0, atol=1e-5)
        and np.allclose(image.GetDirection(), reference.GetDirection(), rtol=0.0, atol=1e-6)
    )


# Validate retained N4, registration, brain and mask images and summarize their geometry.
def validate_spatial_outputs(paths: dict[str, Path], template_path: Path) -> dict[str, Any]:
    fixed = prepare_fixed_template(template_path)
    statistics: dict[str, Any] = {}
    for key in ("n4", "registered", "brain", "mask"):
        path = paths[key]
        if not path.is_file():
            raise FileNotFoundError(f"Required retained output not found: {path}")
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayViewFromImage(image)
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite voxels in {path}")
        if key in {"registered", "brain", "mask"} and not grid_matches(image, fixed):
            raise ValueError(f"{key} grid does not match the released BrainIAC atlas: {path}")
        nonzero = int(np.count_nonzero(array))
        if nonzero == 0:
            raise ValueError(f"All-zero image: {path}")
        statistics[key] = {
            "path": str(path),
            "shape_sitk_xyz": [int(value) for value in image.GetSize()],
            "spacing_mm": [float(value) for value in image.GetSpacing()],
            "nonzero_voxels": nonzero,
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }

    brain_fraction = (
        statistics["brain"]["nonzero_voxels"]
        / float(np.prod(statistics["brain"]["shape_sitk_xyz"]))
    )
    if not 0.01 < brain_fraction < 0.80:
        raise ValueError(f"Implausible nonzero brain fraction: {brain_fraction:.6f}")
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(paths["mask"])))
    unique = np.unique(mask)
    if not set(float(value) for value in unique).issubset({0.0, 1.0}):
        raise ValueError(f"HD-BET mask is not binary: values={unique[:20]}")
    statistics["brain_nonzero_fraction"] = float(brain_fraction)
    statistics["mask_unique_values"] = [float(value) for value in unique]
    return statistics


# Validate the final 96-cube, including exact nonzero z-score expectations from MONAI.
def validate_ready_output(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"BrainIAC model-ready output not found: {path}")
    image = nib.load(str(path))
    shape = tuple(int(value) for value in image.shape)
    if shape != EXPECTED_READY_SHAPE:
        raise ValueError(f"Unexpected BrainIAC shape for {path}: {shape}")
    if not np.array_equal(np.asarray(image.affine), np.eye(4)):
        raise ValueError(f"BrainIAC model-ready affine is not identity: {path}")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite voxels in {path}")
    nonzero = data != 0
    count = int(np.count_nonzero(nonzero))
    if count == 0:
        raise ValueError(f"BrainIAC model-ready output is all zero: {path}")
    mean = float(np.mean(data[nonzero]))
    standard_deviation = float(np.std(data[nonzero]))
    if abs(mean) > 2e-4 or abs(standard_deviation - 1.0) > 2e-4:
        raise ValueError(
            f"Unexpected BrainIAC normalization: mean={mean}, std={standard_deviation}"
        )
    return {
        "path": str(path),
        "shape": list(shape),
        "dtype_on_disk": str(image.get_data_dtype()),
        "nonzero_voxels": count,
        "nonzero_fraction": float(count / data.size),
        "mean_nonzero": mean,
        "standard_deviation_nonzero": standard_deviation,
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
        "affine": np.asarray(image.affine).tolist(),
    }


# Create a compact registered/brain/model-input montage for manual cohort screening.
def write_qc_montage(paths: dict[str, Path], destination: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volumes = [
        ("registered", np.asarray(nib.load(str(paths["registered"])).dataobj)),
        ("HD-BET brain", np.asarray(nib.load(str(paths["brain"])).dataobj)),
        ("BrainIAC 96-cube", np.asarray(nib.load(str(paths["ready"])).dataobj)),
    ]

    def orthogonal_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers = tuple(size // 2 for size in volume.shape)
        return (
            volume[centers[0], :, :],
            volume[:, centers[1], :],
            volume[:, :, centers[2]],
        )

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


# Capture installed versions relevant to preprocessing without importing every package eagerly.
def software_versions(hdbet_executable: str) -> dict[str, str]:
    versions: dict[str, str] = {"python": os.sys.version.split()[0], "hd_bet_cli": hdbet_executable}
    for package in ("SimpleITK", "nibabel", "numpy", "monai", "torch", "HD-BET"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    return versions


# Resolve an explicit HD-BET path or the executable on PATH before launching the cohort.
def resolve_executable(command: str) -> str:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or "/" in command:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FileNotFoundError(f"HD-BET executable is not executable: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"HD-BET executable not found on PATH: {command}")
    return resolved


# Require every published output hash and provenance field to match before an output is reused.
def qc_matches(
    paths: dict[str, Path],
    raw: Path,
    template: dict[str, Any],
) -> bool:
    if not paths["qc"].is_file():
        return False
    required = ("n4", "registered", "transform", "brain", "mask", "ready", "qc_png")
    if any(not paths[key].is_file() for key in required):
        return False
    try:
        qc = json.loads(paths["qc"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    output_hashes = qc.get("output_sha256", {})
    return (
        qc.get("status") == "PASS"
        and qc.get("pipeline_contract_sha256") == PIPELINE_CONTRACT_SHA256
        and qc.get("template") == template
        and qc.get("raw_input") == raw_fingerprint(raw)
        and all(output_hashes.get(key) == sha256_file(paths[key]) for key in required)
    )


# Publish staged files as one validated subject result without exposing partial NIfTI outputs.
def publish_staged_outputs(staged: dict[str, Path], final: dict[str, Path]) -> None:
    for key in ("n4", "registered", "transform", "brain", "mask", "ready"):
        final[key].parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged[key], final[key])


# Run or validate the complete BrainIAC preprocessing contract for one manifest subject.
def process_subject(
    row: dict[str, str],
    output_root: Path,
    log_root: Path,
    template_path: Path,
    template: dict[str, Any],
    hdbet_executable: str,
    hdbet_device: str,
    run_id: str,
    force: bool,
    validate_existing: bool,
    timeout_seconds: float | None,
    versions: dict[str, str],
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
    started = utc_now()
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
        "started_utc": started,
        "finished_utc": "",
        "status": "",
        "message": "",
    }
    stage_dir: Path | None = None

    try:
        if not raw.is_file():
            raise FileNotFoundError(f"Raw T1w input not found: {raw}")

        if validate_existing:
            validate_spatial_outputs(paths, template_path)
            ready_stats = validate_ready_output(paths["ready"])
            if not qc_matches(paths, raw, template):
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

        existing = [key for key in ("n4", "registered", "transform", "brain", "mask", "ready", "qc") if paths[key].exists()]
        if existing and not force:
            validate_spatial_outputs(paths, template_path)
            ready_stats = validate_ready_output(paths["ready"])
            if not qc_matches(paths, raw, template):
                raise RuntimeError(
                    "Existing BrainIAC files have stale or incomplete provenance. "
                    "Inspect them and rerun with --force if replacement is intended."
                )
            record.update(
                status="SKIPPED",
                message=(
                    f"Current validated output already exists; shape={tuple(ready_stats['shape'])}"
                ),
            )
            return record

        paths["subject_dir"].mkdir(parents=True, exist_ok=True)
        stage_dir = paths["subject_dir"] / f".staging_{run_id}_{uuid.uuid4().hex[:8]}"
        stage_dir.mkdir(parents=True, exist_ok=False)
        staged = {
            "n4": stage_dir / paths["n4"].name,
            "registered": stage_dir / paths["registered"].name,
            "transform": stage_dir / paths["transform"].name,
            "brain": stage_dir / paths["brain"].name,
            "mask": stage_dir / paths["mask"].name,
            "ready": stage_dir / paths["ready"].name,
        }

        with log_path.open("w", encoding="utf-8") as log_handle:
            log_message(f"pipeline_version={PIPELINE_VERSION}", log_handle)
            log_message(f"pipeline_contract_sha256={PIPELINE_CONTRACT_SHA256}", log_handle)
            log_message(f"dataset={dataset}", log_handle)
            log_message(f"subject={subject}", log_handle)
            log_message(f"raw={raw}", log_handle)
            log_message(f"template={template_path}", log_handle)
            log_message("  [Preprocessing 1/5] N4 bias correction (no mask)", log_handle)
            n4_bias_correction(raw, staged["n4"])

            log_message("  [Preprocessing 2/5] Rigid registration to temp_head at 1 mm", log_handle)
            registration_stats = register_to_template(
                staged["n4"],
                template_path,
                staged["registered"],
                staged["transform"],
            )
            log_message(
                f"    metric={registration_stats['metric_value']:.6f}; "
                f"iterations={registration_stats['optimizer_iteration']}",
                log_handle,
            )

            log_message("  [Preprocessing 3/5] HD-BET skull stripping (TTA disabled)", log_handle)
            hdbet_command = skull_strip_hdbet(
                staged["registered"],
                staged["brain"],
                hdbet_executable,
                hdbet_device,
                timeout_seconds,
                log_handle,
            )
            generated_mask = hdbet_generated_mask_path(staged["brain"])
            if generated_mask != staged["mask"]:
                os.replace(generated_mask, staged["mask"])

            log_message("  [Preprocessing 4/5] Trilinear resize to 96 x 96 x 96", log_handle)
            log_message("  [Preprocessing 5/5] Nonzero, channel-wise z-score", log_handle)
            materialize_model_input(staged["brain"], staged["ready"])
            spatial_stats = validate_spatial_outputs(staged, template_path)
            ready_stats = validate_ready_output(staged["ready"])

            publish_staged_outputs(staged, paths)
            write_qc_montage(paths, paths["qc_png"], f"{dataset}/{subject} — BrainIAC PASS")
            output_hashes = {
                key: sha256_file(paths[key])
                for key in ("n4", "registered", "transform", "brain", "mask", "ready", "qc_png")
            }
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
                    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
                },
                "hdbet_device": hdbet_device,
                "hdbet_command": hdbet_command,
                "registration": registration_stats,
                "spatial_outputs": spatial_stats,
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
                f"{ready_stats['standard_deviation_nonzero']:.6f}; all intermediates retained"
            ),
        )
        if stage_dir.exists():
            stage_dir.rmdir()
    except Exception as error:  # Keep processing the cohort while retaining full diagnostics.
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


# Set process-wide numerical thread limits before SimpleITK, PyTorch or BLAS work begins.
def configure_threads(threads_per_worker: int) -> None:
    if threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be positive")
    value = str(threads_per_worker)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = value
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads_per_worker)


# Build the command-line interface used for one-subject pilots and resumable cohort runs.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Release-aligned BrainIAC preprocessing for the definitive brain-age cohort"
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
    parser.add_argument("--hdbet-command", default="hd-bet")
    parser.add_argument("--hdbet-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--hdbet-timeout-seconds", type=float)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


# Validate global inputs, execute subjects in a bounded pool and write cohort-level records.
def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.force and args.validate_existing:
        raise ValueError("--force and --validate-existing are mutually exclusive")
    if args.hdbet_device == "cuda" and args.workers != 1:
        raise ValueError("GPU HD-BET requires --workers 1 to avoid concurrent model copies")

    configure_threads(args.threads_per_worker)
    rows = load_manifest(args.manifest.resolve())
    selected = select_rows(rows, args.datasets, args.splits, args.subjects, args.limit)
    template_path = args.template.resolve()
    template = validate_template(template_path)
    hdbet_executable = (
        args.hdbet_command
        if args.dry_run or args.validate_existing
        else resolve_executable(args.hdbet_command)
    )
    versions = software_versions(hdbet_executable)
    output_root = args.output_root.resolve()
    log_root = args.log_root.resolve()

    print("BrainIAC preprocessing for the definitive brain-age cohort")
    print(f"  manifest : {args.manifest.resolve()}")
    print(f"  selected : {len(selected)}/{len(rows)}")
    print(f"  outputs  : {output_root}")
    print(f"  logs     : {log_root}")
    print(f"  template : {template_path}")
    print(f"  workers  : {args.workers}")
    print(f"  threads  : {args.threads_per_worker} per worker")
    print(f"  HD-BET   : {hdbet_executable} ({args.hdbet_device})")
    print(f"  contract : {PIPELINE_CONTRACT_SHA256}")
    print("  retained : N4, registered, rigid transform, brain, mask, final, log and QC")

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
        future_to_row = {
            executor.submit(
                process_subject,
                row,
                output_root,
                log_root,
                template_path,
                template,
                hdbet_executable,
                args.hdbet_device,
                run_id,
                args.force,
                args.validate_existing,
                args.hdbet_timeout_seconds,
                versions,
            ): row
            for row in selected
        }
        for completed_count, future in enumerate(as_completed(future_to_row), start=1):
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
    run_manifest = log_root / f"_brainiac_run_{run_id}.tsv"
    run_summary = log_root / f"_brainiac_run_{run_id}.json"
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
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest.resolve()),
            "template": template,
            "output_root": str(output_root),
            "log_root": str(log_root),
            "workers": args.workers,
            "threads_per_worker": args.threads_per_worker,
            "hdbet_device": args.hdbet_device,
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
