#!/usr/bin/env python3
"""
Preprocessing pipeline for AnatCL (ResNet-18, 512-dim).

Scientific preprocessing:
  1. CAT12 VBM preprocessing via the explicit MATLAB/SPM batch
     - Bias correction
     - Registration to MNI template
     - Skull stripping
     - Tissue segmentation (GM, WM, CSF)
Output artifact (not a numbered preprocessing step):
  121x145x121 modulated normalized grey matter volume
     (Dartel + templates_volumes_old/Template_1_IXI555_MNI152.nii — the legacy
      IXI555 MNI152 Dartel template that produces 121x145x121 at 1.5mm iso,
      matching the output shape reported in the AnatCL paper (PRL 2026 §4.1).
      Template was re-added from CAT12 12.8.2 on GitHub, since it has been
      removed from CAT12 12.9+ distributions which only ship MNI152NLin2009cAsym).

Requirements:
  - MATLAB + SPM12 + CAT12 toolbox
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
"""

import argparse
import os
import glob
import shutil
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = PROJECT_ROOT / "templates" / "Template_1_IXI555_MNI152.nii"

CAT12_BATCH_TEMPLATE = """
% CAT12 VBM batch script (auto-generated)
% Processes a single T1w image through the CAT12 segmentation pipeline
matlabbatch{{1}}.spm.tools.cat.estwrite.data = {{'{input_file}'}};
matlabbatch{{1}}.spm.tools.cat.estwrite.nproc = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.tpm = {{fullfile(spm('dir'),'tpm','TPM.nii')}};
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.affreg = 'mni';
% Exact Dartel syntax preserved in the 165 canonical CAT12 batches. The
% retained CAT12 runtime logs explicitly confirm Dartel Registration.
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.registration.dartel.darteltpm = {{'{template_path}'}};
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.vox = 1.5;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.native = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.mod = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.native = 0;
% audit L-AnatCL-62: WM.mod=0 (extractor only reads mwp1 GM; WM output wastes disk)
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.mod = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.surface = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.ROImenu.noROI = struct([]);
spm('defaults','fmri');
spm_jobman('run', matlabbatch);
"""


def _assert_dartel_in_catlog(search_root):
    """Require CAT12's runtime log to confirm Dartel registration."""
    catlog_files = sorted(
        glob.glob(os.path.join(search_root, "report", "catlog_*.txt"))
    ) or sorted(
        glob.glob(os.path.join(search_root, "catlog_*.txt"))
    ) or sorted(
        glob.glob(os.path.join(search_root, "**", "catlog_*.txt"), recursive=True)
    )
    if not catlog_files:
        raise RuntimeError(
            f"CAT12 catlog_*.txt not found under {search_root}; cannot verify "
            "Dartel registration."
        )

    catlog = catlog_files[-1]
    with open(catlog, "r", errors="replace") as handle:
        text = handle.read()
    if "dartel registration" not in text.lower():
        observed = [
            line.strip() for line in text.splitlines()
            if "registration" in line.lower() or "regstr" in line.lower()
        ]
        raise RuntimeError(
            f"CAT12 did not confirm Dartel Registration in {catlog}. "
            f"Observed registration lines: {observed[:20]}"
        )
    print(f"    [OK] CAT12 catlog confirms Dartel Registration: {catlog}")

def run_cat12_matlab(input_file, output_dir, matlab_cmd="matlab", template_path=None):
    """Run CAT12 VBM via MATLAB + SPM12 + CAT12."""
    if not template_path or not os.path.isfile(template_path):
        raise RuntimeError(f"Dartel template not found: {template_path}")

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
    batch_script = CAT12_BATCH_TEMPLATE.format(
        input_file=input_copy, template_path=template_path)
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

    _assert_dartel_in_catlog(work_dir)

    # CAT12 outputs modulated normalized GM as mwp1*.nii
    gm_files = glob.glob(os.path.join(work_dir, "mri", "mwp1*.nii"))
    if not gm_files:
        raise FileNotFoundError(f"CAT12 output not found in {work_dir}/mri/")

    return gm_files[0]


def verify_shape(file_path):
    """Log the CAT12 output shape. Native VBM output is 121x145x121.

    IMPORTANT — AnatCL expects 121x128x121 at inference (README: "Volumes should
    be 121x128x121 preprocessed with cat12 (vbm)"). The required transform is a
    DETERMINISTIC CENTER CROP of the middle (Y) axis 145->128 — Barbano's eval
    transform is Crop((1,121,128,121), type='center'); random crop is the
    TRAIN-time augmentation only. We do NOT crop here: the center crop is applied
    downstream at extraction time in extract_AnatCL.py::load_volume (mirrors
    FoundationModels_OH_T1w/extract_AnatCL_corrected.py::center_crop_y).

    (Earlier this docstring wrongly claimed 128 was a train-only augmentation and
    that AdaptiveAvgPool3d(1) made input size irrelevant. It is not: the conv
    filters and BatchNorm stats are calibrated to 128-sized feature maps, so
    feeding native 145 RUNS but yields off-distribution embeddings. Bug fixed
    2026-06-23 — feeding 145 made frozen-feature C1 worse than a random net.)
    """
    img = nib.load(file_path)
    actual = img.shape[:3]
    zooms = tuple(round(value, 3) for value in img.header.get_zooms()[:3])
    # audit M-AnatCL-139: fail loudly on unexpected CAT12 output shape
    assert actual == (121, 145, 121), f"Expected (121,145,121), got {actual}"
    assert zooms == (1.5, 1.5, 1.5), f"Expected 1.5 mm isotropic, got {zooms}"
    print(f"    Shape: {actual} @ {zooms} mm  "
          "(extractor center-crops Y->128 for AnatCL)")
    return file_path


def preprocess_single(input_file, output_dir, matlab_cmd, template_path):
    """Full CAT12 VBM preprocessing for a single T1w scan."""
    stem = Path(input_file).name.replace(".nii.gz", "").replace(".nii", "")
    subj_dir = os.path.join(output_dir, stem)
    os.makedirs(subj_dir, exist_ok=True)

    print(f"  [Preprocessing 1/1] CAT12 VBM segmentation: {stem}")
    gm_file = run_cat12_matlab(input_file, subj_dir, matlab_cmd, template_path)

    # Copy and verify output
    final_out = os.path.join(subj_dir, f"{stem}_anatcl_ready.nii.gz")
    print(f"  [Output] Copying CAT12 output (native shape, no resample): {stem}")
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
    parser.add_argument("--matlab_cmd", type=str, required=True,
                        help="Path to MATLAB executable (e.g., /usr/local/MATLAB/R2023a/bin/matlab)")
    # Dartel template path. Default is the legacy IXI555 MNI152 template we
    # keep under <repo>/templates/ because it was dropped from
    # CAT12 12.9+ distributions. This is the template referenced in the
    # AnatCL paper (PRL 2026 §4.1) that produces 121x145x121 @ 1.5mm iso.
    parser.add_argument("--template_path", type=str, default=str(DEFAULT_TEMPLATE),
                        help="Absolute path to the Dartel template .nii file")
    args = parser.parse_args()

    if args.input_file:
        files = [args.input_file]
    elif args.input_dir:
        files = sorted(
            glob.glob(os.path.join(args.input_dir, "**", "*T1w*.nii*"), recursive=True)
        )
    else:
        parser.error("Provide either --input_dir or --input_file")

    print(f"AnatCL preprocessing (CAT12 VBM): {len(files)} files")
    print(f"  Dartel template: {args.template_path}")
    for f in files:
        preprocess_single(f, args.output_dir, args.matlab_cmd, args.template_path)
    print("All done.")


if __name__ == "__main__":
    main()
