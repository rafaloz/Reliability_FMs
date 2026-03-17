#!/usr/bin/env python3
"""
Preprocessing pipeline for 3D-Neuro-SimCLR (ResNet-18, 512-dim).

Replicates the TurboPrep pipeline used in the original paper:
  1. N4 bias field correction (ANTs)
  2. Skull stripping (SynthStrip via FreeSurfer)
  3. Rigid registration to MNI152 ICBM 2009c template (ANTs)
  4. WhiteStripe intensity normalization

Requirements:
  - ANTs CLI tools (N4BiasFieldCorrection, antsRegistrationSyN.sh) in PATH
  - FreeSurfer (mri_synthstrip) in PATH
  - pip install intensity-normalization nibabel numpy

Input:  Raw T1w NIfTI files (.nii or .nii.gz)
Output: Preprocessed NIfTI files ready for 3D-Neuro-SimCLR inference

Usage:
  python preprocessing_3DNeuroSimCLR.py --input_dir /path/to/t1w \
      --output_dir /path/to/output \
      --template /path/to/mni_icbm152_t1_tal_nlin_sym_09c.nii
"""

import argparse
import os
import glob
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path


def n4_bias_correction(input_path, output_path):
    """N4 bias field correction using ANTs CLI."""
    cmd = [
        "N4BiasFieldCorrection",
        "-d", "3",
        "-i", str(input_path),
        "-o", str(output_path),
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


def rigid_register_ants(input_path, template_path, output_path):
    """Rigid registration to MNI template using ANTs, with FLIRT fallback."""
    output_prefix = str(output_path).replace(".nii.gz", "_")
    cmd = [
        "antsRegistrationSyN.sh",
        "-d", "3",
        "-f", str(template_path),
        "-m", str(input_path),
        "-o", output_prefix,
        "-t", "r",  # rigid only
        "-n", "4",  # number of threads
    ]
    print(f"    Running ANTs rigid registration...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # antsRegistrationSyN.sh outputs {prefix}Warped.nii.gz
    warped = output_prefix + "Warped.nii.gz"
    ants_ok = result.returncode == 0 and os.path.exists(warped)

    if ants_ok:
        os.rename(warped, str(output_path))
        # Clean ANTs intermediate files
        for suffix in ["InverseWarped.nii.gz", "0GenericAffine.mat"]:
            tmp = output_prefix + suffix
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        print(f"    ANTs failed, falling back to FSL FLIRT...")
        flirt_cmd = [
            "flirt",
            "-in", str(input_path),
            "-ref", str(template_path),
            "-out", str(output_path),
            "-dof", "6",  # rigid (6 DOF)
        ]
        flirt_result = subprocess.run(flirt_cmd, capture_output=True, text=True)
        if flirt_result.returncode != 0:
            raise RuntimeError(
                f"Both ANTs and FLIRT failed for {input_path}:\n"
                f"  ANTs: {result.stderr[:300]}\n"
                f"  FLIRT: {flirt_result.stderr[:300]}"
            )
        # Clean any partial ANTs files
        for suffix in ["Warped.nii.gz", "InverseWarped.nii.gz", "0GenericAffine.mat"]:
            tmp = output_prefix + suffix
            if os.path.exists(tmp):
                os.remove(tmp)

    return output_path


def whitestripe_normalize(input_path, mask_path, output_path):
    """WhiteStripe intensity normalization, with z-score fallback."""
    img_nii = nib.load(str(input_path))
    img_data = img_nii.get_fdata().astype(np.float32)

    mask_nii = nib.load(str(mask_path))
    mask_data = mask_nii.get_fdata().astype(bool)

    try:
        from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
        ws = WhiteStripeNormalize()
        normalized_data = ws(img_data, mask_data)
    except Exception as e:
        print(f"    WhiteStripe failed ({e}), falling back to z-score normalization")
        brain_voxels = img_data[mask_data]
        mu = brain_voxels.mean()
        sigma = brain_voxels.std()
        normalized_data = np.where(mask_data, (img_data - mu) / (sigma + 1e-8), 0.0)

    out_nii = nib.Nifti1Image(normalized_data, img_nii.affine, img_nii.header)
    nib.save(out_nii, str(output_path))
    return output_path


def preprocess_single(input_file, output_dir, template_path):
    """Full TurboPrep preprocessing pipeline for a single T1w scan."""
    stem = Path(input_file).name.replace(".nii.gz", "").replace(".nii", "")
    subj_dir = os.path.join(output_dir, stem)
    os.makedirs(subj_dir, exist_ok=True)

    # Step 1: N4 bias field correction (ANTs)
    n4_out = os.path.join(subj_dir, f"{stem}_n4.nii.gz")
    print(f"  [1/4] N4 bias field correction: {stem}")
    n4_bias_correction(input_file, n4_out)

    # Step 2: Skull stripping (SynthStrip)
    ss_out = os.path.join(subj_dir, f"{stem}_brain.nii.gz")
    mask_out = os.path.join(subj_dir, f"{stem}_mask.nii.gz")
    print(f"  [2/4] Skull stripping (SynthStrip): {stem}")
    skull_strip_synthstrip(n4_out, ss_out, mask_out)

    # Clamp negative values to zero (some scanners, e.g. Oxford Siemens,
    # produce negative intensities after N4 that cause ANTs to fail silently)
    ss_nii = nib.load(ss_out)
    ss_data = ss_nii.get_fdata()
    if (ss_data < 0).any():
        n_neg = (ss_data < 0).sum()
        print(f"    Clamping {n_neg:,} negative voxels to 0")
        ss_data = np.clip(ss_data, 0, None)
        nib.save(nib.Nifti1Image(ss_data, ss_nii.affine, ss_nii.header), ss_out)

    # Step 3: Rigid registration to MNI (ANTs)
    reg_out = os.path.join(subj_dir, f"{stem}_mni.nii.gz")
    print(f"  [3/4] Rigid registration to MNI (ANTs): {stem}")
    rigid_register_ants(ss_out, template_path, reg_out)

    # Derive brain mask in MNI space from the registered skull-stripped image
    # (nonzero voxels = brain). Previous approach of registering binary mask
    # to T1w template was incorrect and failed for ~10% of subjects.
    mask_reg = os.path.join(subj_dir, f"{stem}_mask_mni.nii.gz")
    reg_nii = nib.load(reg_out)
    mask_data = (reg_nii.get_fdata() > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask_data, reg_nii.affine, reg_nii.header), mask_reg)

    # Step 4: WhiteStripe intensity normalization
    final_out = os.path.join(subj_dir, f"{stem}_simclr_ready.nii.gz")
    print(f"  [4/4] WhiteStripe normalization: {stem}")
    whitestripe_normalize(reg_out, mask_reg, final_out)

    print(f"  Done: {final_out}")
    return final_out


def main():
    parser = argparse.ArgumentParser(
        description="3D-Neuro-SimCLR preprocessing (TurboPrep pipeline)"
    )
    parser.add_argument("--input_dir", type=str, help="Directory with T1w NIfTI files")
    parser.add_argument("--input_file", type=str, help="Single T1w NIfTI file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--template", type=str, required=True,
                        help="Path to MNI152 ICBM 2009c nonlinear symmetric template")
    args = parser.parse_args()

    if args.input_file:
        files = [args.input_file]
    elif args.input_dir:
        files = sorted(
            glob.glob(os.path.join(args.input_dir, "**", "*T1w*.nii*"), recursive=True)
        )
    else:
        parser.error("Provide either --input_dir or --input_file")

    print(f"3D-Neuro-SimCLR preprocessing (TurboPrep): {len(files)} files")
    for f in files:
        preprocess_single(f, args.output_dir, args.template)
    print("All done.")


if __name__ == "__main__":
    main()
