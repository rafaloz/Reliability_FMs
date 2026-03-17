#!/usr/bin/env python3
"""
Preprocessing pipeline for y-Aware (DenseNet-121, 1024-dim).

Steps:
  1. CAT12 VBM preprocessing (via MATLAB/SPM batch or standalone CAT12)
     - Bias correction
     - Registration to MNI template
     - Skull stripping
     - Tissue segmentation
  2. Resample to 121x145x121 (if needed)
  3. Z-score intensity normalization
  4. Save as .npy (NumPy format, shape: 1x121x145x121)

Requirements:
  - MATLAB + SPM12 + CAT12 toolbox  (or CAT12 standalone)
  - pip install nibabel numpy

Input:  Raw T1w NIfTI files (.nii or .nii.gz)
Output: Preprocessed .npy files (shape: 1x121x145x121) ready for y-Aware

Usage:
  python preprocessing_yAware.py --input_dir /path/to/t1w --output_dir /path/to/output --matlab_cmd /path/to/matlab
  python preprocessing_yAware.py --input_file /path/to/sub-01_T1w.nii.gz --output_dir /path/to/output --cat12_standalone /path/to/cat12
"""

import argparse
import os
import glob
import shutil
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path


CAT12_BATCH_TEMPLATE = """
% CAT12 VBM batch script (auto-generated for y-Aware)
matlabbatch{{1}}.spm.tools.cat.estwrite.data = {{'{input_file}'}};
matlabbatch{{1}}.spm.tools.cat.estwrite.nproc = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.tpm = {{fullfile(spm('dir'),'tpm','TPM.nii')}};
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.affreg = 'mni';
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.registration.shooting.shootingtpm = {{fullfile(spm('dir'),'toolbox','cat12','templates_MNI152NLin2009cAsym','Template_0_GS.nii')}};
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.native = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.mod = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.native = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.mod = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.dartel = 0;
spm('defaults','fmri');
spm_jobman('run', matlabbatch);
"""


def run_cat12_matlab(input_file, output_dir, matlab_cmd="matlab"):
    """Run CAT12 VBM via MATLAB + SPM12 + CAT12."""
    work_dir = os.path.join(output_dir, "cat12_work")
    os.makedirs(work_dir, exist_ok=True)

    # CAT12 prefers uncompressed .nii
    input_copy = os.path.join(work_dir, Path(input_file).name)
    if input_file.endswith(".nii.gz"):
        img = nib.load(input_file)
        input_copy = input_copy.replace(".nii.gz", ".nii")
        nib.save(img, input_copy)
    else:
        shutil.copy2(input_file, input_copy)

    batch_script = CAT12_BATCH_TEMPLATE.format(input_file=input_copy)
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

    gm_files = glob.glob(os.path.join(work_dir, "mri", "mwp1*.nii"))
    if not gm_files:
        raise FileNotFoundError(f"CAT12 output not found in {work_dir}/mri/")
    return gm_files[0]


def run_cat12_standalone(input_file, output_dir, cat12_path):
    """Run CAT12 standalone."""
    cmd = [
        os.path.join(cat12_path, "cat_standalone.sh"),
        "-b", os.path.join(cat12_path, "standalone", "cat_standalone_segment.m"),
        str(input_file)
    ]
    print(f"    Running CAT12 standalone...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CAT12 standalone failed: {result.stderr[:500]}")

    input_dir = os.path.dirname(input_file)
    gm_files = glob.glob(os.path.join(input_dir, "mri", "mwp1*.nii"))
    return gm_files[0] if gm_files else None


def resample_to_target(nii_path, target_shape=(121, 145, 121)):
    """Resample volume to target shape if needed."""
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.float32)

    if data.shape[:3] == target_shape:
        return data

    print(f"    Resampling from {data.shape[:3]} to {target_shape}")
    from scipy.ndimage import zoom
    factors = [t / s for t, s in zip(target_shape, data.shape[:3])]
    data = zoom(data, factors, order=3)
    return data


def zscore_normalize(data, eps=1e-8):
    """Z-score normalization as used in y-Aware augmentations.py."""
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / (std_val + eps)


def preprocess_single(input_file, output_dir, matlab_cmd=None, cat12_path=None):
    """Full preprocessing pipeline for a single T1w scan."""
    stem = Path(input_file).name.replace(".nii.gz", "").replace(".nii", "")
    subj_dir = os.path.join(output_dir, stem)
    os.makedirs(subj_dir, exist_ok=True)

    # Step 1: CAT12 VBM
    print(f"  [1/3] CAT12 VBM segmentation: {stem}")
    if cat12_path:
        gm_file = run_cat12_standalone(input_file, subj_dir, cat12_path)
    elif matlab_cmd:
        gm_file = run_cat12_matlab(input_file, subj_dir, matlab_cmd)
    else:
        raise RuntimeError("Provide --matlab_cmd or --cat12_standalone")

    # Step 2: Resample to 121x145x121
    print(f"  [2/3] Resample to 121x145x121: {stem}")
    data = resample_to_target(gm_file, target_shape=(121, 145, 121))

    # Step 3: Z-score normalize and save as .npy
    print(f"  [3/3] Z-score normalize + save as .npy: {stem}")
    data = zscore_normalize(data)
    data = data[np.newaxis, ...]  # Add channel dim: (1, 121, 145, 121)
    data = data.astype(np.float32)

    final_out = os.path.join(subj_dir, f"{stem}_yaware_ready.npy")
    np.save(final_out, data)

    print(f"  Done: {final_out}  shape={data.shape}")
    return final_out


def main():
    parser = argparse.ArgumentParser(description="y-Aware preprocessing pipeline (CAT12)")
    parser.add_argument("--input_dir", type=str, help="Directory with T1w NIfTI files")
    parser.add_argument("--input_file", type=str, help="Single T1w NIfTI file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--matlab_cmd", type=str, default=None,
                        help="Path to MATLAB executable")
    parser.add_argument("--cat12_standalone", type=str, default=None,
                        help="Path to CAT12 standalone installation")
    args = parser.parse_args()

    if args.input_file:
        files = [args.input_file]
    elif args.input_dir:
        files = sorted(
            glob.glob(os.path.join(args.input_dir, "**", "*T1w*.nii*"), recursive=True)
        )
    else:
        parser.error("Provide either --input_dir or --input_file")

    if not args.matlab_cmd and not args.cat12_standalone:
        parser.error("Provide either --matlab_cmd or --cat12_standalone")

    print(f"y-Aware preprocessing (CAT12): {len(files)} files")
    for f in files:
        preprocess_single(f, args.output_dir, args.matlab_cmd, args.cat12_standalone)
    print("All done.")


if __name__ == "__main__":
    main()
