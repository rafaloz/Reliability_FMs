#!/usr/bin/env python3
"""
Preprocessing pipeline for BrainSegFounder (Swin-T, 768-dim, T1w-only mode).

Replicates the UK Biobank T1 preprocessing pipeline used for pretraining:
  1. Skull stripping with FSL BET
  2. Affine registration to MNI152 1mm template (FSL FLIRT, 12 DOF)
  3. Nonlinear registration to MNI152 1mm template (FSL FNIRT)
  4. Apply MNI brain mask to FNIRT output (UK Biobank T1_brain_to_MNI is skull-stripped)
  5. Reorient to RAS, scale intensity [0,1], pad, crop foreground, resize 96x96x96

The UK Biobank pipeline produces T1_brain_to_MNI.nii.gz files, which is what
BrainSegFounder was pretrained on (single-channel T1w, in_channels=1).

Requirements:
  - FSL (flirt, fnirt, bet, fslmaths) in PATH
  - pip install monai nibabel numpy torch

Input:  Raw T1w NIfTI files (.nii or .nii.gz)
Output: Preprocessed 96x96x96 NIfTI files ready for BrainSegFounder (in_channels=1)

Usage:
  python preprocessing_BrainSegFounder.py --input_dir /path/to/t1w --output_dir /path/to/output \
      --template /path/to/MNI152_T1_1mm.nii.gz
"""

import argparse
import os
import glob
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path

import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRangePercentilesd, SpatialPadd, CropForegroundd,
    Resized, EnsureTyped
)


def skull_strip_bet(input_path, output_path, frac=0.5):
    """Skull stripping using FSL BET."""
    # Use bet2 directly (avoids 'dc' dependency in bet wrapper script)
    cmd = [
        "bet2",
        str(input_path),
        str(output_path),
        "-f", str(frac),
    ]
    print(f"    Running FSL BET...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FSL BET failed: {result.stderr[:500]}")
    return output_path


def affine_register_flirt(input_path, template_path, output_path, omat_path):
    """Affine (12 DOF) registration to MNI template using FSL FLIRT."""
    cmd = [
        "flirt",
        "-in", str(input_path),
        "-ref", str(template_path),
        "-out", str(output_path),
        "-omat", str(omat_path),
        "-dof", "12",
        "-interp", "trilinear",
    ]
    print(f"    Running FSL FLIRT (affine)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FSL FLIRT failed: {result.stderr[:500]}")
    return output_path


def nonlinear_register_fnirt(input_path, template_path, affine_mat, output_path):
    """Nonlinear registration to MNI template using FSL FNIRT."""
    cmd = [
        "fnirt",
        "--in=" + str(input_path),
        "--ref=" + str(template_path),
        "--aff=" + str(affine_mat),
        "--iout=" + str(output_path),
    ]
    print(f"    Running FSL FNIRT (nonlinear)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FSL FNIRT failed: {result.stderr[:500]}")
    return output_path


def apply_brain_mask(input_path, mask_path, output_path):
    """Apply MNI brain mask to skull-strip the FNIRT output.

    BrainSegFounder was pretrained on UK Biobank T1_brain_to_MNI.nii.gz files,
    which are skull-stripped. FNIRT output retains the skull, so we must mask it.
    """
    cmd = [
        "fslmaths",
        str(input_path),
        "-mas", str(mask_path),
        str(output_path),
    ]
    print(f"    Applying MNI brain mask...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"fslmaths masking failed: {result.stderr[:500]}")
    return output_path


def apply_monai_transforms(input_path, output_path, target_size=(96, 96, 96)):
    """Apply MONAI transforms matching BrainSegFounder data pipeline.

    Order matches the original pretraining code (data_utils.py):
      Orient RAS -> ScaleIntensity -> SpatialPad -> CropForeground -> Resize

    Note: The original BrainSegFounder code defaults to
    ScaleIntensityRanged(a_min=-1000, a_max=1000), inherited from the SwinUNETR
    codebase (designed for CT Hounsfield units). This destroys MRI data with
    scanner-dependent intensity ranges (e.g. Philips scanners with max ~2.5M).
    We use percentile-based scaling instead, which is robust to arbitrary MRI
    intensity ranges and maps each volume to [0, 1].
    """
    transforms = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=target_size),
        CropForegroundd(keys=["image"], source_key="image",
                        k_divisible=target_size),
        Resized(keys=["image"], spatial_size=target_size, mode="trilinear"),
        EnsureTyped(keys=["image"]),
    ])

    data = transforms({"image": str(input_path)})
    img_tensor = data["image"]

    # Save as NIfTI
    img_np = img_tensor.squeeze(0).numpy() if torch.is_tensor(img_tensor) else img_tensor.squeeze(0)
    nii = nib.Nifti1Image(np.array(img_np, dtype=np.float32), affine=np.eye(4))
    nib.save(nii, str(output_path))
    return output_path


def preprocess_single(input_file, output_dir, template_path, monai_only=False):
    """Full UK Biobank-style preprocessing for a single T1w scan.

    If monai_only=True, skip steps 1-4 (FSL) and only re-apply MONAI transforms
    to the existing *_fnirt_brain.nii.gz file. Useful for fixing the intensity
    scaling without re-running the expensive registration steps.
    """
    stem = Path(input_file).name.replace(".nii.gz", "").replace(".nii", "")
    subj_dir = os.path.join(output_dir, stem)
    os.makedirs(subj_dir, exist_ok=True)

    fnirt_brain = os.path.join(subj_dir, f"{stem}_fnirt_brain.nii.gz")

    if monai_only:
        if not os.path.exists(fnirt_brain):
            print(f"  SKIP (no fnirt_brain found): {stem}")
            return None
    else:
        # Derive brain template and mask paths from full template
        # FLIRT needs brain-to-brain; FNIRT needs full head
        template_dir = os.path.dirname(template_path)
        template_brain = os.path.join(template_dir, "MNI152_T1_1mm_brain.nii.gz")
        brain_mask = os.path.join(template_dir, "MNI152_T1_1mm_brain_mask.nii.gz")

        # Step 1: Skull stripping with FSL BET
        bet_out = os.path.join(subj_dir, f"{stem}_brain.nii.gz")
        print(f"  [1/5] Skull stripping (FSL BET): {stem}")
        skull_strip_bet(input_file, bet_out)

        # Step 2: Affine registration to MNI brain (FSL FLIRT)
        flirt_out = os.path.join(subj_dir, f"{stem}_flirt.nii.gz")
        omat = os.path.join(subj_dir, f"{stem}_affine.mat")
        print(f"  [2/5] Affine registration (FSL FLIRT, brain-to-brain): {stem}")
        affine_register_flirt(bet_out, template_brain, flirt_out, omat)

        # Step 3: Nonlinear registration to MNI (FSL FNIRT)
        # FNIRT uses the original (non-skull-stripped) image with the affine mat
        fnirt_out = os.path.join(subj_dir, f"{stem}_fnirt.nii.gz")
        print(f"  [3/5] Nonlinear registration (FSL FNIRT): {stem}")
        nonlinear_register_fnirt(input_file, template_path, omat, fnirt_out)

        # Step 4: Apply MNI brain mask to FNIRT output
        # UK Biobank T1_brain_to_MNI.nii.gz is skull-stripped; we replicate that
        print(f"  [4/5] Apply MNI brain mask: {stem}")
        apply_brain_mask(fnirt_out, brain_mask, fnirt_brain)

    # Step 5: MONAI transforms (orient RAS, scale intensity, pad, crop, resize 96x96x96)
    final_out = os.path.join(subj_dir, f"{stem}_brainsegfounder_ready.nii.gz")
    print(f"  [5/5] MONAI transforms (RAS + scale + pad + crop + 96x96x96): {stem}")
    apply_monai_transforms(fnirt_brain, final_out)

    print(f"  Done: {final_out}")
    return final_out


def main():
    parser = argparse.ArgumentParser(
        description="BrainSegFounder preprocessing (UK Biobank pipeline)"
    )
    parser.add_argument("--input_dir", type=str, help="Directory with T1w NIfTI files")
    parser.add_argument("--input_file", type=str, help="Single T1w NIfTI file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--template", type=str, default=None,
                        help="Path to MNI152_T1_1mm.nii.gz (FSL standard)")
    parser.add_argument("--monai_only", action="store_true",
                        help="Skip FSL steps 1-4, only re-apply MONAI transforms "
                             "to existing *_fnirt_brain.nii.gz files")
    args = parser.parse_args()

    if not args.monai_only and args.template is None:
        parser.error("--template is required unless --monai_only is set")

    if args.monai_only:
        # Find existing fnirt_brain files in output_dir
        fnirt_files = sorted(
            glob.glob(os.path.join(args.output_dir, "**", "*_fnirt_brain.nii.gz"),
                       recursive=True)
        )
        print(f"BrainSegFounder MONAI-only re-run: {len(fnirt_files)} fnirt_brain files")
        for fnirt_brain in fnirt_files:
            subj_dir = os.path.dirname(fnirt_brain)
            stem = Path(fnirt_brain).name.replace("_fnirt_brain.nii.gz", "")
            final_out = os.path.join(subj_dir, f"{stem}_brainsegfounder_ready.nii.gz")
            print(f"  [5/5] MONAI transforms: {stem}")
            apply_monai_transforms(fnirt_brain, final_out)
            print(f"  Done: {final_out}")
    else:
        if args.input_file:
            files = [args.input_file]
        elif args.input_dir:
            files = sorted(
                glob.glob(os.path.join(args.input_dir, "**", "*T1w*.nii*"),
                           recursive=True)
            )
        else:
            parser.error("Provide either --input_dir or --input_file")

        print(f"BrainSegFounder preprocessing (UK Biobank): {len(files)} files")
        for f in files:
            preprocess_single(f, args.output_dir, args.template)
    print("All done.")


if __name__ == "__main__":
    main()
