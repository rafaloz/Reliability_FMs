#!/usr/bin/env python3
"""
Preprocessing pipeline for AnatCL (ResNet-18, 512-dim).

Steps:
  1. CAT12 VBM preprocessing (via MATLAB/SPM batch or standalone CAT12)
     - Bias correction
     - Registration to MNI template
     - Skull stripping
     - Tissue segmentation (GM, WM, CSF)
  2. Output: 121x145x121 modulated normalized grey matter volume (native CAT12 MNI)

Requirements:
  - MATLAB + SPM12 + CAT12 toolbox  (or CAT12 standalone)
  - pip install nibabel numpy

  CAT12 download: https://neuro-jena.github.io/cat/
  SPM12 download: https://www.fil.ion.ucl.ac.uk/spm/software/spm12/

Input:  Raw T1w NIfTI files (.nii or .nii.gz)
Output: CAT12-preprocessed 121x145x121 volumes ready for AnatCL (native CAT12 MNI shape)

Usage:
  python preprocessing_AnatCL.py --input_dir /path/to/t1w --output_dir /path/to/output
  python preprocessing_AnatCL.py --input_file /path/to/sub-01_T1w.nii.gz --output_dir /path/to/output

  # With MATLAB/SPM available:
  python preprocessing_AnatCL.py --input_dir /path/to/t1w --output_dir /path/to/output --matlab_cmd /usr/local/MATLAB/R2023a/bin/matlab

  # With CAT12 standalone:
  python preprocessing_AnatCL.py --input_dir /path/to/t1w --output_dir /path/to/output --cat12_standalone /path/to/cat12_standalone
"""

import argparse
import os
import glob
import shutil
import subprocess
import tempfile
import numpy as np
import nibabel as nib
from pathlib import Path


CAT12_BATCH_TEMPLATE = """
% CAT12 VBM batch script (auto-generated)
% Processes a single T1w image through the CAT12 segmentation pipeline
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
    # CAT12 writes output next to input, so copy to output dir first
    stem = Path(input_file).stem.replace(".nii", "")
    work_dir = os.path.join(output_dir, "cat12_work")
    os.makedirs(work_dir, exist_ok=True)

    # Copy input to working directory (CAT12 writes output alongside input)
    input_copy = os.path.join(work_dir, os.path.basename(input_file))
    if not input_copy.endswith(".nii"):
        # CAT12 prefers uncompressed .nii
        img = nib.load(input_file)
        input_nii = input_copy.replace(".nii.gz", ".nii")
        nib.save(img, input_nii)
        input_copy = input_nii
    else:
        shutil.copy2(input_file, input_copy)

    # Write MATLAB batch script
    batch_script = CAT12_BATCH_TEMPLATE.format(input_file=input_copy)
    batch_file = os.path.join(work_dir, "cat12_batch.m")
    with open(batch_file, "w") as f:
        f.write(batch_script)

    # Run MATLAB
    cmd = [
        matlab_cmd, "-nodisplay", "-nosplash", "-batch",
        f"run('{batch_file}')"
    ]
    print(f"    Running CAT12 via MATLAB...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    if result.stdout:
        print(f"    MATLAB STDOUT:\n{result.stdout[-2000:]}")
    if result.stderr:
        print(f"    MATLAB STDERR:\n{result.stderr[-2000:]}")
    if result.returncode != 0:
        raise RuntimeError(f"CAT12 failed for {input_file}")

    # CAT12 outputs modulated normalized GM as mwp1*.nii
    gm_files = glob.glob(os.path.join(work_dir, "mri", "mwp1*.nii"))
    if not gm_files:
        raise FileNotFoundError(f"CAT12 output not found in {work_dir}/mri/")

    return gm_files[0]


def run_cat12_standalone(input_file, output_dir, cat12_path):
    """Run CAT12 standalone (no MATLAB required)."""
    cmd = [
        os.path.join(cat12_path, "cat_standalone.sh"),
        "-b", os.path.join(cat12_path, "standalone", "cat_standalone_segment.m"),
        str(input_file)
    ]
    print(f"    Running CAT12 standalone...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CAT12 standalone failed: {result.stderr[:500]}")

    # Find output
    input_dir = os.path.dirname(input_file)
    gm_files = glob.glob(os.path.join(input_dir, "mri", "mwp1*.nii"))
    return gm_files[0] if gm_files else None


def verify_shape(file_path):
    """Log the output shape (no resampling — model uses AdaptiveAvgPool3d).

    The AnatCL test pipeline feeds native CAT12 VBM output directly without
    any crop or resize. The model uses AdaptiveAvgPool3d(1), so it accepts
    any spatial size. The 121x128x121 shape in the pip package README is a
    training-time random crop augmentation — NOT applied at inference.
    Typical CAT12 output is 121x145x121 (1.5mm iso) or similar depending
    on the CAT12 version and template.
    """
    img = nib.load(file_path)
    actual = img.shape[:3]
    print(f"    Shape: {actual}")
    return file_path


def preprocess_single(input_file, output_dir, matlab_cmd=None, cat12_path=None):
    """Full CAT12 VBM preprocessing for a single T1w scan."""
    stem = Path(input_file).name.replace(".nii.gz", "").replace(".nii", "")
    subj_dir = os.path.join(output_dir, stem)
    os.makedirs(subj_dir, exist_ok=True)

    print(f"  [1/2] CAT12 VBM segmentation: {stem}")
    if cat12_path:
        gm_file = run_cat12_standalone(input_file, subj_dir, cat12_path)
    elif matlab_cmd:
        gm_file = run_cat12_matlab(input_file, subj_dir, matlab_cmd)
    else:
        raise RuntimeError(
            "Provide either --matlab_cmd or --cat12_standalone. "
            "CAT12 requires MATLAB+SPM12 or the standalone version."
        )

    # Copy and verify output
    final_out = os.path.join(subj_dir, f"{stem}_anatcl_ready.nii.gz")
    print(f"  [2/2] Copying CAT12 output (native shape, no resample): {stem}")
    img = nib.load(gm_file)
    nib.save(img, final_out)
    verify_shape(final_out)

    print(f"  Done: {final_out}")
    return final_out


def main():
    parser = argparse.ArgumentParser(description="AnatCL preprocessing pipeline (CAT12 VBM)")
    parser.add_argument("--input_dir", type=str, help="Directory with T1w NIfTI files")
    parser.add_argument("--input_file", type=str, help="Single T1w NIfTI file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--matlab_cmd", type=str, default=None,
                        help="Path to MATLAB executable (e.g., /usr/local/MATLAB/R2023a/bin/matlab)")
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

    print(f"AnatCL preprocessing (CAT12 VBM): {len(files)} files")
    for f in files:
        preprocess_single(f, args.output_dir, args.matlab_cmd, args.cat12_standalone)
    print("All done.")


if __name__ == "__main__":
    main()
