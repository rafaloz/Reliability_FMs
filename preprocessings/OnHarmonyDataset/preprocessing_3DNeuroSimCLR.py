#!/usr/bin/env python3
"""
Preprocessing pipeline for 3D-Neuro-SimCLR (Kaczmarek et al., ICCV-W 2025).

Replicates TurboPrep (github.com/LemuelPuglisi/turboprep) invoked by Kaczmarek as
    turboprep $T1 $OUT $TEMPLATE -m t1 -r r
which runs, in order:
  1. N4 bias field correction (ANTs)
  2. Skull stripping (SynthStrip, FreeSurfer)
  3. Rigid registration to MNI152 ICBM 2009c template (ANTs, -t r)
  4. SynthSeg tissue segmentation (FreeSurfer mri_synthseg --fast)
     → binary brain mask = (seg >= 2)
  5. WhiteStripe intensity normalization (masked)

Requirements:
  - ANTs CLI:      N4BiasFieldCorrection, antsRegistrationSyNQuick.sh in PATH
  - FreeSurfer:    mri_synthstrip, mri_synthseg in PATH (>= 7.3)
  - pip:           intensity-normalization, nibabel, numpy

Input:  Raw T1w NIfTI files (.nii or .nii.gz)
Output: Preprocessed NIfTI files ready for 3D-Neuro-SimCLR inference

Usage:
  # One acquisition (also retained for backwards compatibility):
  python preprocessing_3DNeuroSimCLR.py single \\
      --input-file /path/to/sub-00001_ses-01_T1w.nii.gz \\
      --output-dir /path/to/output

  # Parallel cohort run with one log per acquisition:
  python preprocessing_3DNeuroSimCLR.py run \\
      --filelist DATA/filelists/filelist_BrainIAC_raw.txt \\
      --output-dir preprocessings/OnHarmonyDataset/preproc_SimCLR \\
      --log-dir preprocessings/OnHarmonyDataset/logs_preprocs \\
      --workers 7 --threads-per-worker 6 --expected-inputs 165 \\
      --publish-filelist DATA/filelists/filelist_3DNeuroSimCLR.txt

  # Validate an existing cohort without rerunning preprocessing:
  python preprocessing_3DNeuroSimCLR.py validate \\
      --filelist DATA/filelists/filelist_BrainIAC_raw.txt \\
      --output-dir preprocessings/OnHarmonyDataset/preproc_SimCLR \\
      --log-dir preprocessings/OnHarmonyDataset/logs_preprocs \\
      --expected-inputs 165
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import nibabel as nib
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parents[1]
DEFAULT_TEMPLATE = (
    PROJECT_ROOT / "templates" / "mni_icbm152_t1_tal_nlin_sym_09c.nii"
)
DEFAULT_RAW_FILELIST = PROJECT_ROOT / "DATA" / "filelists" / "filelist_BrainIAC_raw.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "preproc_SimCLR"
DEFAULT_LOG_DIR = SCRIPT_PATH.parent / "logs_preprocs"
DEFAULT_PUBLISHED_FILELIST = (
    PROJECT_ROOT / "DATA" / "filelists" / "filelist_3DNeuroSimCLR.txt"
)
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
CANONICAL_T1_PATTERN = re.compile(
    r"sub-[0-9]+_ses-[A-Za-z0-9]+_T1w\.nii(?:\.gz)?$"
)
READY_SUFFIX = "_simclr_ready.nii.gz"
MASK_SUFFIX = "_mask_mni.nii.gz"
REQUIRED_EXECUTABLES = (
    "N4BiasFieldCorrection",
    "mri_synthstrip",
    "antsRegistrationSyNQuick.sh",
    "mri_synthseg",
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)


def sha256_file(path, chunk_size=1024 * 1024):
    """Hash an atlas without loading the complete NIfTI file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_template(path):
    """Require the exact complete ICBM152 2009c atlas named by TurboPrep."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"3D-Neuro-SimCLR template not found: {path}")

    digest = sha256_file(path)
    if digest != EXPECTED_TEMPLATE_SHA256:
        raise ValueError(
            f"Unexpected template SHA-256 for {path}: {digest}; "
            f"expected {EXPECTED_TEMPLATE_SHA256}. Use the complete "
            "mni_icbm152_t1_tal_nlin_sym_09c.nii atlas, not the _brain variant."
        )

    image = nib.load(str(path))
    shape = tuple(int(value) for value in image.shape)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    affine = np.asarray(image.affine, dtype=np.float64)
    if shape != EXPECTED_TEMPLATE_SHAPE:
        raise ValueError(
            f"Unexpected template shape for {path}: {shape}; "
            f"expected {EXPECTED_TEMPLATE_SHAPE}"
        )
    if not np.allclose(spacing, EXPECTED_TEMPLATE_SPACING, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Unexpected template spacing for {path}: {spacing}; "
            f"expected {EXPECTED_TEMPLATE_SPACING}"
        )
    if not np.allclose(affine, EXPECTED_TEMPLATE_AFFINE, rtol=0.0, atol=1e-6):
        raise ValueError(f"Unexpected template affine for {path}")

    return path


def acquisition_stem(path):
    """Return the canonical acquisition identifier from a NIfTI filename."""
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Input is not a NIfTI file: {path}")


def output_paths(input_path, output_dir, log_dir=None):
    """Resolve the final image, registered mask, and optional per-scan log."""
    stem = acquisition_stem(input_path)
    subject_dir = Path(output_dir).expanduser().resolve() / stem
    paths = {
        "ready": subject_dir / f"{stem}{READY_SUFFIX}",
        "mask": subject_dir / f"{stem}{MASK_SUFFIX}",
    }
    if log_dir is not None:
        paths["log"] = Path(log_dir).expanduser().resolve() / f"simclr_{stem}.log"
    return paths


def load_inputs(*, filelist=None, input_file=None, input_dir=None, expected_inputs=None):
    """Load one unambiguous, ordered set of canonical raw ON-Harmony T1w files."""
    sources = sum(value is not None for value in (filelist, input_file, input_dir))
    if sources != 1:
        raise ValueError("Provide exactly one of --filelist, --input-file, or --input-dir")

    if filelist is not None:
        filelist = Path(filelist).expanduser().resolve()
        if not filelist.is_file():
            raise FileNotFoundError(filelist)
        lines = [line.strip() for line in filelist.read_text(encoding="utf-8").splitlines()]
        raw_paths = [line for line in lines if line]
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


def validate_runtime_dependencies():
    """Fail before spawning jobs if a required CLI or WhiteStripe is unavailable."""
    missing = [command for command in REQUIRED_EXECUTABLES if shutil.which(command) is None]
    if missing:
        raise FileNotFoundError(f"Required executables are unavailable: {missing}")
    from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize

    if WhiteStripeNormalize is None:  # pragma: no cover - import is the actual check
        raise RuntimeError("WhiteStripeNormalize import unexpectedly resolved to None")


def worker_environment(threads):
    """Build a child environment with a fixed CPU allocation per acquisition."""
    if threads < 1:
        raise ValueError("--threads-per-worker must be positive")
    environment = os.environ.copy()
    for name in THREAD_ENVIRONMENT:
        environment[name] = str(threads)
    return environment


def validate_preprocessed_output(input_path, output_dir, log_dir, template_path):
    """Validate the published image, mask, spatial grid, normalization, and log."""
    paths = output_paths(input_path, output_dir, log_dir)
    for key in ("ready", "mask", "log"):
        path = paths[key]
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {key} artifact: {path}")

    template = nib.load(str(template_path))
    ready_image = nib.load(str(paths["ready"]))
    mask_image = nib.load(str(paths["mask"]))
    for label, image in (("ready", ready_image), ("mask", mask_image)):
        if tuple(image.shape) != tuple(template.shape):
            raise ValueError(
                f"{label} grid differs from template for {input_path}: {image.shape}"
            )
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if not np.allclose(spacing, EXPECTED_TEMPLATE_SPACING, rtol=0.0, atol=1e-6):
            raise ValueError(f"{label} spacing differs from template for {input_path}: {spacing}")
        if not np.allclose(image.affine, template.affine, rtol=0.0, atol=1e-5):
            raise ValueError(f"{label} affine differs from template for {input_path}")

    ready = ready_image.get_fdata(dtype=np.float32)
    mask_values = mask_image.get_fdata(dtype=np.float32)
    if not np.isfinite(ready).all() or not np.isfinite(mask_values).all():
        raise ValueError(f"Non-finite output values for {input_path}")
    if not np.all((mask_values == 0.0) | (mask_values == 1.0)):
        raise ValueError(f"Registered mask is not binary for {input_path}")
    mask = mask_values > 0
    foreground = ready[mask]
    if foreground.size < 1000 or float(np.std(foreground)) <= 1e-6:
        raise ValueError(f"Empty or constant normalized foreground for {input_path}")
    mean = float(np.mean(foreground))
    standard_deviation = float(np.std(foreground))
    if abs(mean) < 1e-4 and abs(standard_deviation - 1.0) < 1e-4:
        raise ValueError(
            f"Unexpected exact z-score signature in WhiteStripe output: {paths['ready']}"
        )

    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    required_log_fragments = (
        "3D-Neuro-SimCLR preprocessing (TurboPrep-equivalent)",
        f"Template SHA-256: {EXPECTED_TEMPLATE_SHA256}",
        "[Preprocessing 5/5] WhiteStripe normalization",
        "exit_code: 0",
    )
    missing = [fragment for fragment in required_log_fragments if fragment not in log_text]
    expected_done_suffix = f"/{acquisition_stem(input_path)}/{paths['ready'].name}"
    done_paths = [
        line.strip().removeprefix("Done:").strip()
        for line in log_text.splitlines()
        if line.startswith("  Done:") or line.startswith("Done:")
    ]
    if not any(path.endswith(expected_done_suffix) for path in done_paths):
        missing.append(f"Done: *{expected_done_suffix}")
    if missing or "WhiteStripe failed" in log_text:
        raise ValueError(f"Unsuccessful preprocessing log {paths['log']}; missing={missing}")
    return {
        "input": str(Path(input_path)),
        "ready": str(paths["ready"]),
        "mask": str(paths["mask"]),
        "log": str(paths["log"]),
        "shape": list(ready_image.shape),
        "masked_mean": mean,
        "masked_standard_deviation": standard_deviation,
    }


def run_logged_acquisition(input_path, output_dir, log_dir, template_path, threads):
    """Run one isolated Python child and capture its complete output in one log."""
    paths = output_paths(input_path, output_dir, log_dir)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "single",
        "--input-file",
        str(input_path),
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--template",
        str(template_path),
        "--threads",
        str(threads),
    ]
    with paths["log"].open("w", encoding="utf-8") as handle:
        handle.write(f"=== {acquisition_stem(input_path)} ===\n")
        handle.write(f"input    : {input_path}\n")
        handle.write(f"template : {template_path}\n")
        handle.write(f"started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=worker_environment(threads),
            check=False,
        )
        handle.write(f"\nexit_code: {result.returncode}\n")
    return result.returncode, paths["log"]


def atomic_publish_filelist(inputs, output_dir, destination):
    """Publish outputs in raw-input order only after cohort validation passes."""
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        lines = [str(output_paths(path, output_dir)["ready"]) for path in inputs]
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def n4_bias_correction(input_path, output_path):
    """N4 bias field correction using ANTs CLI."""
    cmd = [
        "N4BiasFieldCorrection",
        "-d", "3",
        "-i", str(input_path),
        "-o", str(output_path),
        "-s", "3",  # audit M-3DSimCLR-40: shrink factor 3 to match TurboPrep (ANTs default is 4)
    ]
    print(f"    Running ANTs N4BiasFieldCorrection...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"N4BiasFieldCorrection failed: {result.stderr[:500]}")
    return output_path


def skull_strip_synthstrip(input_path, output_path, mask_path):
    """Skull stripping using FreeSurfer SynthStrip."""
    cmd = [
        "mri_synthstrip",
        "-i", str(input_path),
        "-o", str(output_path),
        "-m", str(mask_path),
    ]
    print(f"    Running SynthStrip...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"SynthStrip failed: {result.stderr[:500]}")
    return output_path


def clamp_negative_voxels(input_path):
    """Clamp negative post-SynthStrip intensities to zero for every session."""
    image = nib.load(str(input_path))
    data = image.get_fdata(dtype=np.float32)
    negative_voxels = int(np.count_nonzero(data < 0))
    if negative_voxels:
        print(f"    Clamping {negative_voxels:,} negative voxels to 0")
        data = np.maximum(data, 0.0)
        nib.save(nib.Nifti1Image(data, image.affine, image.header), str(input_path))
    return negative_voxels


def rigid_register_ants(input_path, template_path, output_path, threads=4):
    """Rigid registration to the MNI template using ANTs; fail on any error."""
    output_prefix = str(output_path).replace(".nii.gz", "_")
    cmd = [
        "antsRegistrationSyNQuick.sh",  # audit H6: match TurboPrep upstream (Quick variant)
        "-d", "3",
        "-f", str(template_path),
        "-m", str(input_path),
        "-o", output_prefix,
        "-t", "r",
        "-n", str(threads),
    ]
    print(f"    Running ANTs rigid registration...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    warped = output_prefix + "Warped.nii.gz"
    ants_ok = result.returncode == 0 and os.path.exists(warped)

    if ants_ok:
        os.rename(warped, str(output_path))
        for suffix in ["InverseWarped.nii.gz", "0GenericAffine.mat"]:
            tmp = output_prefix + suffix
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        # This used to fall back to FSL FLIRT -dof 6. That is not a skip, but it
        # registers the affected subjects with a different optimiser and
        # interpolator than the rest of the cohort, and nothing in the outputs
        # records which subjects took which path — hidden heterogeneity in a
        # reliability study. Fail instead, so the cohort stays homogeneous.
        raise RuntimeError(
            f"antsRegistrationSyNQuick.sh failed for {input_path}. Refusing to fall "
            f"back to FSL FLIRT: mixing registration engines across subjects would "
            f"introduce heterogeneity that no output artifact records.\n"
            f"  stderr: {result.stderr[:500]}"
        )

    return output_path


def synthseg_brain_mask(reg_path, seg_path, mask_path, threads=4):
    """Run mri_synthseg --fast on the MNI-registered volume and derive a
    binary brain mask from all non-background SynthSeg labels.

    Matches upstream TurboPrep's brain-mask derivation (turboprep script:
    mri_synthseg --fast --threads N → mri_threshold -B 1 segm 2, i.e. seg >= 2).
    """
    cmd = [
        "mri_synthseg",
        "--i", str(reg_path),
        "--o", str(seg_path),
        "--fast",
        "--cpu",
        "--threads", str(threads),
    ]
    print(f"    Running SynthSeg (mri_synthseg --fast)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"SynthSeg failed: {result.stderr[:500]}")

    seg_nii = nib.load(str(seg_path))
    # audit L-3DSimCLR-116: seg >= 2 matches TurboPrep (mri_threshold -B 1 segm 2), excludes background+CSF-outside
    mask_data = (seg_nii.get_fdata() >= 2).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask_data, seg_nii.affine, seg_nii.header),
             str(mask_path))
    return mask_path


def whitestripe_normalize(input_path, mask_path, output_path):
    """Apply mandatory WhiteStripe intensity normalization inside the brain mask."""
    img_nii = nib.load(str(input_path))
    img_data = img_nii.get_fdata().astype(np.float32)

    mask_nii = nib.load(str(mask_path))
    mask_data = mask_nii.get_fdata().astype(bool)

    # WhiteStripe is mandatory; import and normalization errors must propagate.
    from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
    ws = WhiteStripeNormalize()
    normalized_data = ws(img_data, mask_data)

    out_nii = nib.Nifti1Image(normalized_data, img_nii.affine, img_nii.header)
    nib.save(out_nii, str(output_path))
    return output_path


def preprocess_single(input_file, output_dir, template_path, threads=4):
    """Full TurboPrep-equivalent preprocessing pipeline for a single T1w scan."""
    stem = Path(input_file).name.replace(".nii.gz", "").replace(".nii", "")
    subj_dir = os.path.join(output_dir, stem)
    os.makedirs(subj_dir, exist_ok=True)

    # Resume support: skip if final output already exists (idempotent re-runs).
    _final_out = os.path.join(subj_dir, f"{stem}_simclr_ready.nii.gz")
    if os.path.exists(_final_out) and os.path.getsize(_final_out) > 0:
        print(f"  SKIP (already done): {_final_out}")
        return _final_out

    n4_out = os.path.join(subj_dir, f"{stem}_n4.nii.gz")
    print(f"  [Preprocessing 1/5] N4 bias field correction: {stem}")
    n4_bias_correction(input_file, n4_out)

    ss_out = os.path.join(subj_dir, f"{stem}_brain.nii.gz")
    mask_out = os.path.join(subj_dir, f"{stem}_mask.nii.gz")
    print(f"  [Preprocessing 2/5] Skull stripping (SynthStrip): {stem}")
    skull_strip_synthstrip(n4_out, ss_out, mask_out)
    clamp_negative_voxels(ss_out)

    reg_out = os.path.join(subj_dir, f"{stem}_mni.nii.gz")
    print(f"  [Preprocessing 3/5] Rigid registration to MNI (ANTs): {stem}")
    rigid_register_ants(ss_out, template_path, reg_out, threads=threads)

    seg_reg = os.path.join(subj_dir, f"{stem}_synthseg.nii.gz")
    mask_reg = os.path.join(subj_dir, f"{stem}_mask_mni.nii.gz")
    print(f"  [Preprocessing 4/5] SynthSeg brain mask: {stem}")
    synthseg_brain_mask(reg_out, seg_reg, mask_reg, threads=threads)

    final_out = os.path.join(subj_dir, f"{stem}_simclr_ready.nii.gz")
    print(f"  [Preprocessing 5/5] WhiteStripe normalization: {stem}")
    whitestripe_normalize(reg_out, mask_reg, final_out)

    print(f"  Done: {final_out}")
    return final_out


def _cohort_inputs(args):
    """Resolve cohort CLI inputs, defaulting to the canonical raw ON-Harmony list."""
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


def command_single(args):
    """Run the unchanged five-step scientific pipeline serially."""
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    template_path = validate_template(args.template)
    files = load_inputs(
        input_file=args.input_file,
        input_dir=args.input_dir,
        expected_inputs=None,
    )
    validate_runtime_dependencies()
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = str(args.threads)

    print(
        f"3D-Neuro-SimCLR preprocessing (TurboPrep-equivalent): {len(files)} files"
    )
    print(f"Template: {template_path}")
    print(f"Template SHA-256: {EXPECTED_TEMPLATE_SHA256}")
    for input_path in files:
        preprocess_single(
            str(input_path),
            str(Path(args.output_dir).expanduser().resolve()),
            template_path,
            threads=args.threads,
        )
    print("All done.")


def command_validate(args):
    """Validate every selected output and optionally publish its ordered filelist."""
    template_path = validate_template(args.template)
    inputs = _cohort_inputs(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    failures = []
    for index, input_path in enumerate(inputs, start=1):
        try:
            validate_preprocessed_output(input_path, output_dir, log_dir, template_path)
            if index % 25 == 0 or index == len(inputs):
                print(f"  [{index}/{len(inputs)}] VALIDATED {acquisition_stem(input_path)}")
        except Exception as error:
            failures.append((input_path, error))
            print(
                f"  [{index}/{len(inputs)}] FAILED    {acquisition_stem(input_path)}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    if failures:
        print(f"Validation failed: {len(failures)}/{len(inputs)} acquisitions")
        raise SystemExit(1)
    print(f"Validation PASS: {len(inputs)}/{len(inputs)} acquisitions")
    if args.publish_filelist is not None:
        destination = atomic_publish_filelist(inputs, output_dir, args.publish_filelist)
        print(f"Published extractor filelist: {destination} ({len(inputs)} paths)")


def command_run(args):
    """Run independent acquisitions in parallel, validate, then publish atomically."""
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be positive")
    template_path = validate_template(args.template)
    inputs = _cohort_inputs(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        input_path
        for input_path in inputs
        if output_paths(input_path, output_dir)["ready"].is_file()
    ]
    if args.require_clean and existing:
        raise RuntimeError(
            f"--require-clean was requested, but {len(existing)} final outputs already exist"
        )

    skipped = []
    pending = []
    for input_path in inputs:
        ready = output_paths(input_path, output_dir)["ready"]
        if ready.is_file():
            try:
                validate_preprocessed_output(input_path, output_dir, log_dir, template_path)
            except Exception as error:
                raise RuntimeError(
                    f"Existing output cannot be safely resumed for {input_path}: {error}. "
                    "Inspect or remove that acquisition directory before rerunning."
                ) from error
            skipped.append(input_path)
        else:
            pending.append(input_path)

    print("3D-Neuro-SimCLR preprocessing (TurboPrep-equivalent)")
    print(f"  inputs    : {len(inputs)}")
    print(f"  outputs   : {output_dir}")
    print(f"  logs      : {log_dir}/simclr_<stem>.log")
    print(f"  template  : {template_path}")
    print(f"  workers   : {args.workers}")
    print(f"  threads   : {args.threads_per_worker} per worker")
    print(f"  resumable : {len(skipped)} validated existing, {len(pending)} pending")
    if args.publish_filelist is None:
        print("  filelist  : not requested")
    else:
        print(f"  filelist  : {Path(args.publish_filelist).expanduser().resolve()}")

    for input_path in skipped:
        print(f"  SKIP {acquisition_stem(input_path)} (validated existing output)")

    failures = []
    if pending:
        validate_runtime_dependencies()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_logged_acquisition,
                    input_path,
                    output_dir,
                    log_dir,
                    template_path,
                    args.threads_per_worker,
                ): input_path
                for input_path in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                input_path = futures[future]
                try:
                    return_code, log_path = future.result()
                except Exception as error:
                    failures.append((input_path, error))
                    print(
                        f"  [{completed}/{len(pending)}] FAIL {acquisition_stem(input_path)}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    continue
                if return_code == 0:
                    print(
                        f"  [{completed}/{len(pending)}] OK   {acquisition_stem(input_path)}",
                        flush=True,
                    )
                else:
                    error = RuntimeError(f"exit code {return_code}; see {log_path}")
                    failures.append((input_path, error))
                    print(
                        f"  [{completed}/{len(pending)}] FAIL {acquisition_stem(input_path)} "
                        f"(rc={return_code}, see {log_path})",
                        flush=True,
                    )

    if failures:
        print(f"Preprocessing failed: {len(failures)}/{len(inputs)} acquisitions")
        raise SystemExit(1)

    validation_failures = []
    for input_path in inputs:
        try:
            validate_preprocessed_output(input_path, output_dir, log_dir, template_path)
        except Exception as error:
            validation_failures.append((input_path, error))
            print(
                f"  QC FAIL {acquisition_stem(input_path)}: {type(error).__name__}: {error}",
                flush=True,
            )
    if validation_failures:
        print(f"Final validation failed: {len(validation_failures)}/{len(inputs)} acquisitions")
        raise SystemExit(1)

    print(f"Final validation PASS: {len(inputs)}/{len(inputs)} acquisitions")
    if args.publish_filelist is not None:
        destination = atomic_publish_filelist(inputs, output_dir, args.publish_filelist)
        print(f"Published extractor filelist: {destination} ({len(inputs)} paths)")


def add_cohort_arguments(parser):
    """Add the common ordered-input and output contract used by run and validate."""
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--filelist", type=Path)
    source.add_argument("--input-file", "--input_file", dest="input_file", type=Path)
    source.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path,
                        default=DEFAULT_OUTPUT_DIR)
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


def build_parser():
    """Build the integrated single-scan, cohort-run, and validation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "3D-Neuro-SimCLR preprocessing (N4 + SynthStrip + rigid MNI + "
            "SynthSeg mask + WhiteStripe)"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser("single", help="Preprocess one file or directory serially")
    single_source = single.add_mutually_exclusive_group(required=True)
    single_source.add_argument("--input-file", "--input_file", dest="input_file", type=Path)
    single_source.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path)
    single.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path,
                        required=True)
    single.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    single.add_argument("--threads", type=int, default=4)
    single.set_defaults(func=command_single)

    run = commands.add_parser("run", help="Run a cohort with bounded parallelism")
    add_cohort_arguments(run)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--threads-per-worker", type=int, default=4)
    run.add_argument(
        "--require-clean",
        action="store_true",
        help="Refuse any existing final output, for intentional contract migrations",
    )
    run.set_defaults(func=command_run)

    validate = commands.add_parser("validate", help="Validate existing cohort outputs")
    add_cohort_arguments(validate)
    validate.set_defaults(func=command_validate)
    return parser


def main(argv=None):
    """Dispatch the integrated CLI while preserving the former single-file syntax."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {"single", "run", "validate"}
    if arguments and arguments[0] not in commands and arguments[0] not in {"-h", "--help"}:
        arguments.insert(0, "single")
    args = build_parser().parse_args(arguments)
    args.func(args)


if __name__ == "__main__":
    main()
