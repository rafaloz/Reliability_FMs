#!/usr/bin/env python3
"""
Preprocessing pipeline for BrainIAC (ViT-B, 768-dim).

Scientific preprocessing:
  1. N4 bias field correction (SimpleITK procedural defaults, without an
     explicit mask, matching the released BrainIAC preprocessing script)
  2. Rigid registration to temp_head.nii.gz template (from BrainIAC repo,
     resampled to 1mm isotropic before registration)
  3. Skull stripping with HD-BET
  4. Resample to 96x96x96
  5. Z-score intensity normalization (nonzero voxels)

Per-subject QC and optional intermediate-file cleanup are internal controls,
not numbered preprocessing steps from the BrainIAC paper/release protocol.

The template used is temp_head.nii.gz from the official BrainIAC repository
(github.com/AIM-KannLab/BrainIAC, src/preprocessing/atlases/temp_head.nii.gz),
which is the template used in the quickstart notebook and preprocessing script.

Requirements:
  pip install SimpleITK nibabel numpy hd-bet monai torch

Input:  Raw T1w NIfTI files (.nii or .nii.gz)
Output: Preprocessed 96x96x96 NIfTI files ready for BrainIAC inference

Usage:
  python preprocessing_BrainIAC.py --input_dir /path/to/t1w --output_dir /path/to/output \
      --template /path/to/temp_head.nii.gz
  python preprocessing_BrainIAC.py --input_file /path/to/sub-01_T1w.nii.gz --output_dir /path/to/output \
      --template /path/to/temp_head.nii.gz
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_RAW_FILELIST = PROJECT_ROOT / "DATA" / "filelists" / "filelist_BrainIAC_raw.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "preproc_BrainIAC"
DEFAULT_LOG_DIR = SCRIPT_PATH.parent / "logs_preprocs"
DEFAULT_TEMPLATE = PROJECT_ROOT / "templates" / "temp_head.nii.gz"
DEFAULT_PUBLISHED_FILELIST = PROJECT_ROOT / "DATA" / "filelists" / "filelist_BrainIAC.txt"

PIPELINE_VERSION = "brainiac-release-aligned-onharmony-v1"
OFFICIAL_BRAINIAC_COMMIT = "ba60f45bed832ee0e5683c678caeaeefe2072f0d"
EXPECTED_TEMPLATE_SHA256 = "d8eba10e9528a5212b78fb1be477d5e8229751bfec5e1e1a648dc81ff83a622e"
EXPECTED_TEMPLATE_SIZE = (170, 206, 162)
EXPECTED_TEMPLATE_SPACING = (1.0, 1.0, 1.0)
EXPECTED_READY_SHAPE = (96, 96, 96)
REGISTRATION_METRIC_SEED = 42
HISTORICAL_GENERATOR_SHA256 = "f298f8f0efab28e1b6a07746775c6b1ef5a37c16a4c9dbc3df707bdd52d8bdbb"
CANONICAL_T1_PATTERN = re.compile(
    r"sub-[0-9]+_ses-[A-Za-z0-9]+_T1w\.nii(?:\.gz)?$"
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)

CONTRACT = {
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
        "device": "cpu",
        "tta": False,
    },
    "model_view": {
        "shape": EXPECTED_READY_SHAPE,
        "resize": "MONAI Resize trilinear",
        "normalization": "MONAI NormalizeIntensity nonzero=True channel_wise=True",
        "stored_affine": "identity",
    },
}
CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(CONTRACT, sort_keys=True).encode("utf-8")
).hexdigest()


def utc_now() -> str:
    """Return a timezone-aware timestamp for provenance records."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one artifact incrementally without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """Publish JSON atomically so interrupted writes cannot look complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def raw_fingerprint(path: Path) -> dict[str, Any]:
    """Record the stable filesystem identity used for resume decisions."""
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def acquisition_stem(path: Path) -> str:
    """Return the canonical acquisition identifier of a NIfTI path."""
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
    """Load one ordered, canonical and duplicate-free ON-Harmony registry."""
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
            Path(value).expanduser()
            if Path(value).expanduser().is_absolute()
            else filelist.parent / Path(value).expanduser()
            for value in raw_paths
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
        raise ValueError("No canonical T1w inputs were selected")
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


def output_paths(output_dir: Path, stem: str) -> dict[str, Path]:
    """Define historical BrainIAC filenames without changing the data layout."""
    subject_dir = Path(output_dir).resolve() / stem
    return {
        "subject_dir": subject_dir,
        "n4": subject_dir / f"{stem}_n4.nii.gz",
        "registered": subject_dir / f"{stem}_reg.nii.gz",
        "brain": subject_dir / f"{stem}_brain.nii.gz",
        "ready": subject_dir / f"{stem}_brainiac_ready.nii.gz",
        "qc": subject_dir / f"{stem}_qc.json",
        "qc_png": subject_dir / f"{stem}_qc.png",
    }


def validate_template(path: Path) -> dict[str, Any]:
    """Require the exact atlas distributed by the BrainIAC release."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BrainIAC template not found: {path}")
    digest = sha256_file(path)
    if digest != EXPECTED_TEMPLATE_SHA256:
        raise ValueError(
            f"Unexpected BrainIAC template SHA-256 {digest}; expected "
            f"{EXPECTED_TEMPLATE_SHA256}"
        )
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    size = tuple(int(value) for value in image.GetSize())
    spacing = tuple(float(value) for value in image.GetSpacing())
    if size != EXPECTED_TEMPLATE_SIZE or not np.allclose(
        spacing, EXPECTED_TEMPLATE_SPACING, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            f"Unexpected BrainIAC template geometry: size={size}, spacing={spacing}"
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


def require_non_fsl_python() -> None:
    """Reject FSL's bundled Python because MONAI and HD-BET require Conda."""
    executable = Path(sys.executable).resolve()
    fsldir = os.environ.get("FSLDIR")
    inside_fsldir = False
    if fsldir:
        try:
            executable.relative_to(Path(fsldir).expanduser().resolve())
            inside_fsldir = True
        except ValueError:
            pass
    fsl_component = any(
        re.fullmatch(r"fsl(?:[_-]\d.*)?|fslpython.*", part, flags=re.IGNORECASE)
        for part in executable.parts
    )
    if inside_fsldir or fsl_component:
        raise RuntimeError(
            f"Refusing FSL Python {executable}; activate the project conda environment"
        )


def worker_environment(threads: int) -> dict[str, str]:
    """Build a child environment with bounded numerical-library threading."""
    if threads < 1:
        raise ValueError("Thread count must be positive")
    environment = os.environ.copy()
    for name in THREAD_ENVIRONMENT:
        environment[name] = str(threads)
    return environment


def atomic_publish_filelist(
    inputs: list[Path], output_dir: Path, destination: Path
) -> Path:
    """Publish validated model inputs atomically in canonical raw-input order."""
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        lines = [
            str(output_paths(output_dir, acquisition_stem(raw))["ready"])
            for raw in inputs
        ]
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def n4_bias_correction(input_path, output_path):
    """Apply the exact N4 call used by BrainIAC's released preprocessing."""
    img = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
    # mri_preprocess_3d_simple.py calls this one-image procedural overload.
    # Do not add an Otsu mask or custom iteration schedule here: either would
    # make the primary pipeline differ from the released BrainIAC code.
    corrected = sitk.N4BiasFieldCorrection(img)
    sitk.WriteImage(corrected, str(output_path))
    return output_path


def register_to_template(input_path, template_path, output_path):
    """Rigid registration to temp_head.nii.gz template using SimpleITK.

    The template is first resampled to 1mm isotropic spacing, matching the
    original BrainIAC preprocessing (mri_preprocess_3d_simple.py).
    """
    fixed = sitk.ReadImage(str(template_path), sitk.sitkFloat32)

    # Resample template to 1mm isotropic (matching original BrainIAC code)
    old_size = fixed.GetSize()
    old_spacing = fixed.GetSpacing()
    new_spacing = (1.0, 1.0, 1.0)
    new_size = [
        int(round(old_size[i] * old_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    resample_filt = sitk.ResampleImageFilter()
    resample_filt.SetOutputSpacing(new_spacing)
    resample_filt.SetSize(new_size)
    resample_filt.SetOutputOrigin(fixed.GetOrigin())
    resample_filt.SetOutputDirection(fixed.GetDirection())
    resample_filt.SetInterpolator(sitk.sitkLinear)
    resample_filt.SetDefaultPixelValue(0.0)
    resample_filt.SetOutputPixelType(sitk.sitkFloat32)
    fixed = resample_filt.Execute(fixed)

    moving = sitk.ReadImage(str(input_path), sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    # audit H1: deterministic rigid registration. The seed is the second argument
    # of SetMetricSamplingPercentage; SimpleITK exposes no SetMetricSamplingSeed
    # (absent as of 2.5.3), so calling one raised AttributeError here.
    registration.SetMetricSamplingPercentage(0.01, REGISTRATION_METRIC_SEED)
    registration.SetInterpolator(sitk.sitkLinear)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration.Execute(fixed, moving)
    resampled = sitk.Resample(
        moving, fixed, final_transform, sitk.sitkLinear, 0.0,
        moving.GetPixelID()
    )

    # The whole point of registration is that every subject lands on the SAME
    # grid as the template.
    if resampled.GetSize() != fixed.GetSize():
        raise RuntimeError(
            f"Registered volume grid {resampled.GetSize()} != template grid "
            f"{fixed.GetSize()} for {input_path}"
        )

    sitk.WriteImage(resampled, str(output_path))
    return output_path


def skull_strip_hdbet(input_path, output_path):
    """Skull stripping using HD-BET CLI."""
    cmd = [
        "hd-bet", "-i", str(input_path), "-o", str(output_path),
        "-device", "cpu", "--disable_tta",
    ]
    print(f"    Running HD-BET...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"HD-BET failed: {result.stderr[:500]}")
    return output_path


def resample_and_normalize(input_path, output_path, target_shape=(96, 96, 96)):
    """Resample to target shape and apply z-score normalization."""
    from monai.transforms import (
        LoadImage, EnsureChannelFirst, Resize, NormalizeIntensity
    )

    loader = LoadImage(image_only=True)
    img = loader(str(input_path))

    ensure_ch = EnsureChannelFirst()
    img = ensure_ch(img)

    resize = Resize(spatial_size=target_shape, mode="trilinear")
    img = resize(img)

    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)
    img = normalize(img)

    # Save as NIfTI
    img_np = img.squeeze(0).numpy()
    nii = nib.Nifti1Image(img_np, affine=np.eye(4))
    nib.save(nii, str(output_path))
    return output_path


def write_qc(
    raw_input: Path | None,
    brain_path: Path,
    brainiac_ready_path: Path,
    qc_json_path: Path,
    qc_png_path: Path,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-subject QC: JSON stats + a 2x3 ortho montage PNG (_brain + _brainiac_ready).

    Stats capture shape and z-score sanity; PNG is for visual screening of
    skull-strip / registration failures at a glance. `raw_input` is optional
    — pass None when running post-hoc on outputs whose raw source isn't handy.
    """
    raw_shape = None
    if raw_input is not None and Path(raw_input).exists():
        try:
            raw_shape = list(nib.load(str(raw_input)).shape)
        except Exception:
            raw_shape = None
    brain = np.asarray(nib.load(str(brain_path)).dataobj)
    bready = np.asarray(nib.load(str(brainiac_ready_path)).dataobj)

    nz_bready = bready != 0
    n_nonzero = int(nz_bready.sum())
    mean_nz = float(bready[nz_bready].mean()) if n_nonzero else 0.0
    std_nz = float(bready[nz_bready].std()) if n_nonzero else 0.0
    nonzero_frac = round(n_nonzero / bready.size, 4)
    brain_nonzero_voxels = int((brain > 0).sum())

    # bool(...) casts collapse numpy.bool_ -> Python bool so json.dump works
    shape_ok = bool(tuple(bready.shape) == EXPECTED_READY_SHAPE)
    norm_ok = bool(abs(mean_nz) < 0.5 and 0.5 < std_nz < 1.5)
    brain_ok = bool(brain_nonzero_voxels > 50_000)  # >~50k voxels post-skull-strip
    status = "PASS" if (shape_ok and norm_ok and brain_ok) else "WARN"

    qc = {
        "raw_input": str(Path(raw_input).resolve()) if raw_input is not None else None,
        "raw_shape": raw_shape,
        "brain_shape": list(brain.shape),
        "brainiac_shape": list(bready.shape),
        "brain_nonzero_voxels": brain_nonzero_voxels,
        "brainiac_mean_nonzero": round(mean_nz, 4),
        "brainiac_std_nonzero": round(std_nz, 4),
        "brainiac_nonzero_fraction": nonzero_frac,
        "shape_ok": shape_ok,
        "norm_ok": norm_ok,
        "brain_ok": brain_ok,
        "status": status,
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for BrainIAC QC") from error

    def mids(vol):
        z, y, x = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
        return vol[z, :, :], vol[:, y, :], vol[:, :, x]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for row_i, (label, vol) in enumerate([("brain (temp_head grid)", brain),
                                          ("brainiac_ready (96³ z-scored)", bready)]):
        for col_i, sl in enumerate(mids(vol)):
            axes[row_i, col_i].imshow(np.rot90(sl), cmap="gray")
            axes[row_i, col_i].set_axis_off()
            if col_i == 0:
                axes[row_i, col_i].set_ylabel(label, fontsize=9)
    title_stem = Path(raw_input).name if raw_input else Path(brainiac_ready_path).name
    fig.suptitle(f"{title_stem}  status={status}", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(qc_png_path), dpi=80, bbox_inches="tight")
    plt.close(fig)
    if provenance:
        qc.update(provenance)
    qc["output_sha256"] = {
        "brain": sha256_file(brain_path),
        "ready": sha256_file(brainiac_ready_path),
        "qc_png": sha256_file(qc_png_path),
    }
    atomic_json(qc_json_path, qc)
    return qc


def validate_brain_output(path: Path, template_path: Path) -> dict[str, Any]:
    """Require the skull-stripped image on the exact BrainIAC atlas grid."""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing registered brain output: {path}")
    image = nib.load(str(path))
    template = nib.load(str(template_path))
    if tuple(image.shape) != tuple(template.shape) or not np.allclose(
        image.affine, template.affine, rtol=0.0, atol=1e-4
    ):
        raise ValueError(f"Brain output is not on the temp_head grid: {path}")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite registered brain values: {path}")
    nonzero = int(np.count_nonzero(data))
    if nonzero <= 50_000:
        raise ValueError(f"Too few registered brain voxels ({nonzero}): {path}")
    return {
        "shape": list(image.shape),
        "spacing_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "nonzero_voxels": nonzero,
        "affine": np.asarray(image.affine, dtype=np.float64).tolist(),
    }


def validate_ready_output(path: Path) -> dict[str, Any]:
    """Require the fixed 96-cube and nonzero z-score expected by BrainIAC."""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing BrainIAC model input: {path}")
    image = nib.load(str(path))
    if tuple(image.shape) != EXPECTED_READY_SHAPE:
        raise ValueError(f"Unexpected BrainIAC model-input shape {image.shape}: {path}")
    if not np.allclose(image.affine, np.eye(4), rtol=0.0, atol=1e-6):
        raise ValueError(f"BrainIAC model input does not use the stored identity affine: {path}")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite BrainIAC model-input values: {path}")
    foreground = data != 0
    count = int(np.count_nonzero(foreground))
    if count <= 1_000:
        raise ValueError(f"Too few nonzero BrainIAC voxels ({count}): {path}")
    mean = float(np.mean(data[foreground]))
    standard_deviation = float(np.std(data[foreground]))
    if abs(mean) > 1e-3 or not 0.99 <= standard_deviation <= 1.01:
        raise ValueError(
            f"Unexpected BrainIAC normalization: mean={mean}, std={standard_deviation}"
        )
    return {
        "shape": list(image.shape),
        "nonzero_voxels": count,
        "nonzero_fraction": float(count / data.size),
        "mean_nonzero": mean,
        "standard_deviation_nonzero": standard_deviation,
        "affine": np.asarray(image.affine, dtype=np.float64).tolist(),
    }


def validate_success_log(path: Path) -> str:
    """Validate a successful historical or integrated per-acquisition log."""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing BrainIAC preprocessing log: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^script\s*:\s*([0-9a-f]{64})\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"BrainIAC log has no generator SHA-256: {path}")
    generator = match.group(1)
    allowed = {HISTORICAL_GENERATOR_SHA256, sha256_file(SCRIPT_PATH)}
    if generator not in allowed:
        raise ValueError(f"Unrecognized BrainIAC generator SHA-256 {generator}: {path}")
    required = (
        "[Preprocessing 4-5/5] Resample 96",
        "status=PASS",
        "All done.",
        "exit_code: 0",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise ValueError(f"Incomplete BrainIAC preprocessing log {path}: missing={missing}")
    return generator


def validate_published_subject(
    raw: Path,
    output_dir: Path,
    log_dir: Path,
    template_path: Path,
) -> dict[str, Any]:
    """Revalidate one subject's images, QC, provenance, and successful log."""
    raw = Path(raw).resolve()
    stem = acquisition_stem(raw)
    paths = output_paths(output_dir, stem)
    log_path = Path(log_dir).resolve() / f"brainiac_{stem}.log"
    generator = validate_success_log(log_path)
    if not paths["qc"].is_file() or not paths["qc_png"].is_file():
        raise FileNotFoundError(f"Missing BrainIAC QC artifacts for {stem}")
    qc = json.loads(paths["qc"].read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "raw_input": str(raw),
        "brainiac_shape": list(EXPECTED_READY_SHAPE),
        "shape_ok": True,
        "norm_ok": True,
        "brain_ok": True,
    }
    mismatches = {
        key: {"expected": value, "observed": qc.get(key)}
        for key, value in expected.items()
        if qc.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Invalid BrainIAC QC for {stem}: {mismatches}")
    brain = validate_brain_output(paths["brain"], template_path)
    ready = validate_ready_output(paths["ready"])

    is_current = "pipeline_contract_sha256" in qc
    if is_current:
        current_hash = sha256_file(SCRIPT_PATH)
        current_expected = {
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_contract_sha256": CONTRACT_SHA256,
            "script_sha256": current_hash,
            "raw_fingerprint": raw_fingerprint(raw),
        }
        current_mismatches = {
            key: {"expected": value, "observed": qc.get(key)}
            for key, value in current_expected.items()
            if qc.get(key) != value
        }
        if generator != current_hash:
            current_mismatches["log_generator"] = {
                "expected": current_hash,
                "observed": generator,
            }
        expected_hashes = {
            "brain": sha256_file(paths["brain"]),
            "ready": sha256_file(paths["ready"]),
            "qc_png": sha256_file(paths["qc_png"]),
        }
        if qc.get("output_sha256") != expected_hashes:
            current_mismatches["output_sha256"] = {
                "expected": expected_hashes,
                "observed": qc.get("output_sha256"),
            }
        if current_mismatches:
            raise ValueError(
                f"Current BrainIAC provenance mismatch for {stem}: {current_mismatches}"
            )
    elif generator != HISTORICAL_GENERATOR_SHA256:
        raise ValueError(
            f"Legacy BrainIAC QC is only accepted from {HISTORICAL_GENERATOR_SHA256}"
        )
    return {
        "stem": stem,
        "ready": str(paths["ready"]),
        "qc": str(paths["qc"]),
        "generator_sha256": generator,
        "provenance_schema": "current" if is_current else "historical",
        "brain": brain,
        "model_ready": ready,
    }


def preprocess_single(
    input_file: str | Path,
    output_dir: str | Path,
    template_path: str | Path,
    *,
    keep_intermediates: bool = False,
    force: bool = False,
) -> str:
    """Run the unchanged five-stage BrainIAC pipeline for one T1w scan."""
    raw = Path(input_file).resolve()
    template = Path(template_path).resolve()
    template_record = validate_template(template)
    stem = acquisition_stem(raw)
    paths = output_paths(Path(output_dir), stem)
    paths["subject_dir"].mkdir(parents=True, exist_ok=True)
    if paths["ready"].is_file() and paths["ready"].stat().st_size > 0 and not force:
        if not paths["qc"].is_file() or not paths["qc_png"].is_file():
            raise RuntimeError(
                f"Existing BrainIAC output lacks QC artifacts for {stem}; use --force"
            )
        qc = json.loads(paths["qc"].read_text(encoding="utf-8"))
        expected_qc = {
            "status": "PASS",
            "raw_input": str(raw),
            "brainiac_shape": list(EXPECTED_READY_SHAPE),
            "shape_ok": True,
            "norm_ok": True,
            "brain_ok": True,
        }
        mismatches = {
            key: {"expected": value, "observed": qc.get(key)}
            for key, value in expected_qc.items()
            if qc.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing BrainIAC QC is not reusable for {stem}: {mismatches}; "
                "use --force"
            )
        validate_brain_output(paths["brain"], template)
        validate_ready_output(paths["ready"])
        print(f"  SKIP (validated existing output): {paths['ready']}")
        return str(paths["ready"])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    stage_dir = paths["subject_dir"] / f".staging_{run_id}_{uuid.uuid4().hex[:8]}"
    stage_dir.mkdir()
    staged = {
        key: stage_dir / path.name
        for key, path in paths.items()
        if key != "subject_dir"
    }
    try:
        print(f"  [Preprocessing 1/5] N4 bias correction: {stem}")
        n4_bias_correction(raw, staged["n4"])

        print(f"  [Preprocessing 2/5] Rigid registration to temp_head (1mm iso): {stem}")
        register_to_template(staged["n4"], template, staged["registered"])

        print(f"  [Preprocessing 3/5] Skull-strip (HD-BET): {stem}")
        skull_strip_hdbet(staged["registered"], staged["brain"])

        print(
            "  [Preprocessing 4-5/5] Resample 96³ + z-score normalize "
            f"(nonzero, channel_wise): {stem}"
        )
        resample_and_normalize(staged["brain"], staged["ready"])
        validate_brain_output(staged["brain"], template)
        validate_ready_output(staged["ready"])

        print(f"  [QC] Writing {stem}_qc.{{json,png}}")
        qc = write_qc(
            raw,
            staged["brain"],
            staged["ready"],
            staged["qc"],
            staged["qc_png"],
            provenance={
                "created_utc": utc_now(),
                "pipeline_version": PIPELINE_VERSION,
                "pipeline_contract_sha256": CONTRACT_SHA256,
                "contract": CONTRACT,
                "script_sha256": sha256_file(SCRIPT_PATH),
                "raw_fingerprint": raw_fingerprint(raw),
                "template": template_record,
            },
        )
        if qc["status"] != "PASS":
            raise RuntimeError(f"BrainIAC QC returned {qc['status']} for {stem}")
        print(
            f"    status={qc['status']}  brainiac_shape={qc['brainiac_shape']}  "
            f"mean_nz={qc['brainiac_mean_nonzero']:.3f}  "
            f"std_nz={qc['brainiac_std_nonzero']:.3f}"
        )

        publish_keys = ["brain", "ready", "qc_png"]
        if keep_intermediates:
            publish_keys = ["n4", "registered", *publish_keys]
        else:
            print(f"  [Cleanup] Removing disposable N4/registration intermediates: {stem}")
            staged["n4"].unlink(missing_ok=True)
            staged["registered"].unlink(missing_ok=True)
        for key in publish_keys:
            os.replace(staged[key], paths[key])
        os.replace(staged["qc"], paths["qc"])
        stage_dir.rmdir()
        print(f"  Done: {paths['ready']}")
        return str(paths["ready"])
    except Exception:
        if stage_dir.exists():
            failed = stage_dir.with_name(stage_dir.name.replace(".staging_", ".failed_", 1))
            os.replace(stage_dir, failed)
        raise


def _cohort_inputs(args: argparse.Namespace) -> list[Path]:
    """Resolve inputs, defaulting to the canonical 165-row ON-Harmony registry."""
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


def run_logged_acquisition(
    raw: Path,
    output_dir: Path,
    log_dir: Path,
    template: Path,
    threads: int,
    force: bool,
    keep_intermediates: bool,
) -> tuple[int, Path]:
    """Run one isolated child and atomically publish its successful log."""
    stem = acquisition_stem(raw)
    log_path = Path(log_dir).resolve() / f"brainiac_{stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = log_path.with_name(f".{log_path.name}.{uuid.uuid4().hex}.tmp")
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
        "--threads",
        str(threads),
    ]
    if force:
        command.append("--force")
    if keep_intermediates:
        command.append("--keep-intermediates")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(f"=== {stem} ===\n")
            handle.write(f"input    : {raw}\n")
            handle.write(f"template : {template}\n")
            handle.write("N4       : SimpleITK procedural defaults; no explicit mask (release-strict)\n")
            handle.write(f"script   : {sha256_file(SCRIPT_PATH)}\n")
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
        if completed.returncode == 0:
            os.replace(temporary, log_path)
            return completed.returncode, log_path
        failed = log_path.with_name(
            f"{log_path.stem}.failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        os.replace(temporary, failed)
        return completed.returncode, failed
    finally:
        temporary.unlink(missing_ok=True)


def command_single(args: argparse.Namespace) -> None:
    """Run the five-stage pipeline serially for one file or directory."""
    require_non_fsl_python()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = str(args.threads)
    if shutil.which("hd-bet") is None:
        raise FileNotFoundError("hd-bet is not available on PATH")
    files = load_inputs(input_file=args.input_file, input_dir=args.input_dir)
    template = Path(args.template).expanduser().resolve()
    validate_template(template)
    print(f"BrainIAC preprocessing: {len(files)} files")
    print(f"  template : {template}")
    print(f"  output   : {Path(args.output_dir).expanduser().resolve()}")
    print(f"  Python   : {sys.executable}")
    print(f"  threads  : {args.threads}")
    print(f"  pipeline : {PIPELINE_VERSION}")
    print(f"  contract : {CONTRACT_SHA256}")
    for raw in files:
        preprocess_single(
            raw,
            args.output_dir,
            template,
            keep_intermediates=args.keep_intermediates,
            force=args.force,
        )
    print("All done.")


def command_validate(args: argparse.Namespace) -> None:
    """Validate every published image, QC file, and per-acquisition log."""
    require_non_fsl_python()
    inputs = _cohort_inputs(args)
    template = Path(args.template).expanduser().resolve()
    validate_template(template)
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    failures: list[tuple[Path, Exception]] = []
    generators: dict[str, int] = {}
    schemas: dict[str, int] = {}
    for index, raw in enumerate(inputs, start=1):
        try:
            result = validate_published_subject(raw, output_dir, log_dir, template)
            generator = str(result["generator_sha256"])
            schema = str(result["provenance_schema"])
            generators[generator] = generators.get(generator, 0) + 1
            schemas[schema] = schemas.get(schema, 0) + 1
            if index % 25 == 0 or index == len(inputs):
                print(f"  [{index}/{len(inputs)}] VALIDATED {result['stem']}")
        except Exception as error:
            failures.append((raw, error))
            print(
                f"  [{index}/{len(inputs)}] FAILED    {acquisition_stem(raw)}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
    print(f"  valid subjects  : {len(inputs) - len(failures)}/{len(inputs)}")
    print(f"  provenance      : {schemas}")
    print(f"  generator hashes: {generators}")
    if failures:
        raise SystemExit(1)
    if args.publish_filelist is not None:
        destination = atomic_publish_filelist(inputs, output_dir, args.publish_filelist)
        print(f"Published extractor filelist: {destination} ({len(inputs)} paths)")


def command_run(args: argparse.Namespace) -> None:
    """Run missing acquisitions with bounded workers, then validate the cohort."""
    require_non_fsl_python()
    if args.workers < 1 or args.threads_per_worker < 1:
        raise ValueError("Workers and threads per worker must be positive")
    if shutil.which("hd-bet") is None:
        raise FileNotFoundError("hd-bet is not available on PATH")
    inputs = _cohort_inputs(args)
    template = Path(args.template).expanduser().resolve()
    validate_template(template)
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    scheduled: list[Path] = []
    skipped = 0
    for raw in inputs:
        stem = acquisition_stem(raw)
        paths = output_paths(output_dir, stem)
        existing = any(path.exists() for key, path in paths.items() if key != "subject_dir")
        if args.require_clean and existing:
            raise RuntimeError(f"--require-clean found existing artifacts for {stem}")
        if args.force:
            scheduled.append(raw)
            continue
        try:
            validate_published_subject(raw, output_dir, log_dir, template)
            skipped += 1
        except Exception:
            if existing:
                raise RuntimeError(
                    f"Incomplete, stale, or invalid BrainIAC artifacts exist for {stem}; "
                    "inspect them or rerun with --force"
                )
            scheduled.append(raw)

    print("BrainIAC ON-Harmony preprocessing")
    print(f"  inputs    : {len(inputs)}")
    print(f"  scheduled : {len(scheduled)}")
    print(f"  validated : {skipped} existing")
    print(f"  template  : {template}")
    print(f"  outputs   : {output_dir}")
    print(f"  logs      : {log_dir}/brainiac_<stem>.log")
    print(f"  Python    : {sys.executable}")
    print(f"  workers   : {args.workers}")
    print(f"  threads   : {args.threads_per_worker} per worker")
    print(f"  pipeline  : {PIPELINE_VERSION}")
    print(f"  contract  : {CONTRACT_SHA256}")

    failures: list[tuple[Path, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_logged_acquisition,
                raw,
                output_dir,
                log_dir,
                template,
                args.threads_per_worker,
                args.force,
                args.keep_intermediates,
            ): raw
            for raw in scheduled
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            raw = futures[future]
            try:
                return_code, log_path = future.result()
            except Exception as error:
                failures.append((raw, error))
                print(
                    f"  [{completed_count}/{len(scheduled)}] FAIL "
                    f"{acquisition_stem(raw)}: {type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            if return_code == 0:
                print(
                    f"  [{completed_count}/{len(scheduled)}] OK   {acquisition_stem(raw)}",
                    flush=True,
                )
            else:
                error = RuntimeError(f"exit code {return_code}; see {log_path}")
                failures.append((raw, error))
                print(
                    f"  [{completed_count}/{len(scheduled)}] FAIL "
                    f"{acquisition_stem(raw)} (rc={return_code}, see {log_path})",
                    flush=True,
                )
    if failures:
        print(f"Preprocessing failed: {len(failures)}/{len(scheduled)} scheduled")
        raise SystemExit(1)
    command_validate(args)


def add_cohort_arguments(parser: argparse.ArgumentParser) -> None:
    """Add input registry, output, atlas, and optional publication arguments."""
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--filelist", type=Path)
    source.add_argument("--input-file", "--input_file", dest="input_file", type=Path)
    source.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path)
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--expected-inputs", type=int)
    parser.add_argument(
        "--publish-filelist",
        type=Path,
        help=(
            "Atomically publish ordered validated outputs; canonical destination is "
            f"{DEFAULT_PUBLISHED_FILELIST}"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the integrated single-file, cohort-run, and validation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser("single", help="Run one file or directory serially")
    source = single.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-file", "--input_file", dest="input_file", type=Path)
    source.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path)
    single.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path, required=True
    )
    single.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    single.add_argument("--threads", type=int, default=4)
    single.add_argument(
        "--keep-intermediates", "--keep_intermediates",
        dest="keep_intermediates", action="store_true",
    )
    single.add_argument("--force", action="store_true")
    single.set_defaults(func=command_single)

    run = commands.add_parser("run", help="Run a cohort with bounded parallelism")
    add_cohort_arguments(run)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--threads-per-worker", type=int, default=4)
    run.add_argument("--keep-intermediates", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--require-clean", action="store_true")
    run.set_defaults(func=command_run)

    validate = commands.add_parser("validate", help="Validate existing cohort outputs")
    add_cohort_arguments(validate)
    validate.set_defaults(func=command_validate)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch the integrated CLI while retaining the former single-file syntax."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {"single", "run", "validate"}
    if arguments and arguments[0] not in commands and arguments[0] not in {"-h", "--help"}:
        arguments.insert(0, "single")
    args = build_parser().parse_args(arguments)
    args.func(args)


if __name__ == "__main__":
    main()
