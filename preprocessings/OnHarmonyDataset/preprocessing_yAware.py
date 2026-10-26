#!/usr/bin/env python3
"""
Preprocessing pipeline for y-Aware (DenseNet-121, 1024-dim) — CORRECTED.

Matches the brainprep CAT12 VBM config used for BHB-10K pretraining
(brainprep default before 2026-01-06):

  - Registration: Dartel
  - Template:     templates_volumes/Template_1_IXI555_MNI152.nii  (IXI-555)
  - Voxel size:   1.5 mm
  - Native output shape: 121 x 145 x 121

No post-hoc resampling: brainprep produces 121x145x121 natively at vox=1.5
with the IXI template, so the previous scipy.ndimage.zoom step is removed.

Scientific preprocessing:
  1. CAT12 VBM (Dartel + IXI + vox=1.5) -> mwp1*.nii at 121x145x121
  2. Z-score intensity normalization

Output artifact (not a numbered preprocessing step):
  .npy array of shape (1, 121, 145, 121)

Requirements:
  - MATLAB + SPM12 + CAT12 toolbox
  - pip install nibabel numpy

Usage:
  python preprocessing_yAware.py --input_file /path/to/sub-01_T1w.nii.gz \
                                 --output_dir /path/to/output \
                                 --matlab_cmd /path/to/matlab
"""

import argparse
import os
import glob
import shutil
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path


# audit H3: default Dartel TPM path (overridable via --darteltpm CLI arg or
# YAWARE_DARTEL_TPM env-var); preserved as the historical default for
# backward compatibility.
DEFAULT_DARTEL_TPM = "/datos/work/rnavgon/ON-Harmony_Preproc/templates/Template_1_IXI555_MNI152.nii"


CAT12_BATCH_TEMPLATE = """
% CAT12 VBM batch script (auto-generated, brainprep-IXI config, y-Aware)
matlabbatch{{1}}.spm.tools.cat.estwrite.data = {{'{input_file}'}};
matlabbatch{{1}}.spm.tools.cat.estwrite.nproc = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.tpm = {{fullfile(spm('dir'),'tpm','TPM.nii')}};
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.affreg = 'mni';
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.registration.dartel.darteltpm = {{'{dartel_tpm}'}};
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.vox = 1.5;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.native = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.mod = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.native = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.mod = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.dartel = 0;
spm('defaults','fmri');
spm_jobman('run', matlabbatch);
"""


def _assert_dartel_in_catlog(work_dir):
    """audit M-yAware-40: verify CAT12 actually ran Dartel (not Shooting).

    CAT12 silently overrides regstr values (see feedback_cat12_batch_not_ground_truth
    memory); the batch source is not ground truth. Only the catlog_*.txt reflects
    the algorithm actually used. This function greps the catlog for the
    "Dartel Registration" marker and raises with the observed regstr if absent.
    """
    catlog_files = sorted(glob.glob(os.path.join(work_dir, "report", "catlog_*.txt"))) \
        or sorted(glob.glob(os.path.join(work_dir, "catlog_*.txt"))) \
        or sorted(glob.glob(os.path.join(work_dir, "**", "catlog_*.txt"), recursive=True))
    if not catlog_files:
        raise RuntimeError(
            f"CAT12 catlog_*.txt not found under {work_dir}; cannot verify "
            f"registration algorithm (Dartel vs Shooting)."
        )
    catlog = catlog_files[-1]
    with open(catlog, "r", errors="replace") as f:
        text = f.read()
    if "dartel registration" in text.lower():
        print(f"    [OK] CAT12 catlog confirms Dartel Registration: {catlog}")
        return
    # Try to surface the actual regstr / registration line for diagnostics.
    observed = []
    for line in text.splitlines():
        low = line.lower()
        if "regstr" in low or "shooting registration" in low or "registration" in low:
            observed.append(line.strip())
    observed_str = "\n      ".join(observed[:20]) if observed else "<no registration lines found>"
    raise RuntimeError(
        f"CAT12 did NOT run Dartel Registration (catlog={catlog}). "
        f"Observed registration lines:\n      {observed_str}"
    )


def run_cat12_matlab(input_file, output_dir, matlab_cmd="matlab",
                     dartel_tpm=DEFAULT_DARTEL_TPM):
    """Run CAT12 VBM via MATLAB + SPM12 + CAT12."""
    work_dir = os.path.join(output_dir, "cat12_work")
    os.makedirs(work_dir, exist_ok=True)

    input_copy = os.path.join(work_dir, Path(input_file).name)
    if input_file.endswith(".nii.gz"):
        img = nib.load(input_file)
        input_copy = input_copy.replace(".nii.gz", ".nii")
        nib.save(img, input_copy)
    else:
        shutil.copy2(input_file, input_copy)

    # audit H3: inject Dartel TPM path (CLI/env-var overridable) into batch.
    batch_script = CAT12_BATCH_TEMPLATE.format(input_file=input_copy,
                                               dartel_tpm=dartel_tpm)
    batch_file = os.path.join(work_dir, "cat12_batch.m")
    with open(batch_file, "w") as f:
        f.write(batch_script)

    cmd = [matlab_cmd, "-nodisplay", "-nosplash", "-batch", f"run('{batch_file}')"]
    print(f"    Running CAT12 via MATLAB...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    if result.stdout:
        print(f"    MATLAB STDOUT:\n{result.stdout[-2000:]}")
    if result.stderr:
        print(f"    MATLAB STDERR:\n{result.stderr[-2000:]}")
    if result.returncode != 0:
        raise RuntimeError(f"CAT12 failed: {result.stderr[-500:]}")

    # audit M-yAware-40: verify Dartel actually ran (CAT12 can silently
    # fall back to Shooting; batch source is not ground truth).
    _assert_dartel_in_catlog(work_dir)

    gm_files = glob.glob(os.path.join(work_dir, "mri", "mwp1*.nii"))
    if not gm_files:
        raise FileNotFoundError(f"CAT12 output not found in {work_dir}/mri/")
    return gm_files[0]


def zscore_normalize(data, eps=1e-8):
    """Z-score normalization as used in y-Aware augmentations.py."""
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / (std_val + eps)


def preprocess_single(input_file, output_dir, matlab_cmd,
                      dartel_tpm=DEFAULT_DARTEL_TPM):
    """Full preprocessing pipeline for a single T1w scan."""
    stem = Path(input_file).name.replace(".nii.gz", "").replace(".nii", "")
    subj_dir = os.path.join(output_dir, stem)
    os.makedirs(subj_dir, exist_ok=True)

    print(f"  [1/2] CAT12 VBM segmentation (Dartel, IXI, vox=1.5): {stem}")
    # Forward the CLI/environment-resolved Dartel template to the validated batch.
    gm_file = run_cat12_matlab(input_file, subj_dir, matlab_cmd,
                               dartel_tpm=dartel_tpm)

    print(f"  [2/2] Z-score normalize + save as .npy: {stem}")
    img = nib.load(gm_file)
    data = img.get_fdata().astype(np.float32)
    shape = data.shape[:3]
    zooms = tuple(round(z, 3) for z in img.header.get_zooms()[:3])
    expected_shape = (121, 145, 121)
    expected_zooms = (1.5, 1.5, 1.5)
    if shape != expected_shape or zooms != expected_zooms:
        raise RuntimeError(
            f"Unexpected CAT12 output for {stem}: shape={shape}, zooms={zooms}; "
            f"expected {expected_shape} @ {expected_zooms}"
        )
    print(f"    [OK] CAT12 native: shape={shape}  zooms={zooms}  "
          f"(expected {expected_shape} @ {expected_zooms})")

    data = zscore_normalize(data)
    data = data[np.newaxis, ...]  # (1, D, H, W)

    final_out = os.path.join(subj_dir, f"{stem}_yaware_ready.npy")
    np.save(final_out, data)

    print(f"  Done: {final_out}  shape={data.shape}")
    return final_out


def main():
    parser = argparse.ArgumentParser(
        description="y-Aware preprocessing — IXI Dartel @ vox=1.5 (corrected)")
    parser.add_argument("--input_dir", type=str, help="Directory with T1w NIfTI files")
    parser.add_argument("--input_file", type=str, help="Single T1w NIfTI file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--matlab_cmd", type=str, required=True,
                        help="Path to MATLAB executable")
    # audit H3: expose Dartel TPM path as CLI arg with env-var fallback,
    # preserving the historical hardcoded path as the ultimate default.
    parser.add_argument("--darteltpm", type=str,
                        default=os.environ.get("YAWARE_DARTEL_TPM",
                                               DEFAULT_DARTEL_TPM),
                        help="Path to Dartel template NIfTI "
                             "(Template_1_IXI555_MNI152.nii). "
                             "Overrides $YAWARE_DARTEL_TPM; falls back to the "
                             "historical isis path when neither is set.")
    args = parser.parse_args()

    if args.input_file:
        files = [args.input_file]
    elif args.input_dir:
        files = sorted(
            glob.glob(os.path.join(args.input_dir, "**", "*T1w*.nii*"), recursive=True)
        )
    else:
        parser.error("Provide either --input_dir or --input_file")

    print(f"y-Aware preprocessing (IXI Dartel, vox=1.5): {len(files)} files")
    print(f"  Dartel TPM: {args.darteltpm}")  # audit H3
    for f in files:
        preprocess_single(f, args.output_dir, args.matlab_cmd,
                          dartel_tpm=args.darteltpm)
    print("All done.")


if __name__ == "__main__":
    main()
