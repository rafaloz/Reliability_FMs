#!/usr/bin/env python3
"""
Preprocessing pipeline for BrainIAC (ViT-B, 768-dim).

Steps:
  1. N4 bias field correction (SimpleITK)
  2. Rigid registration to temp_head.nii.gz template (from BrainIAC repo,
     resampled to 1mm isotropic before registration)
  3. Skull stripping with HD-BET
  4. Resample to 96x96x96
  5. Z-score intensity normalization (nonzero voxels)

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

import argparse
import os
import glob
import subprocess
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path


def n4_bias_correction(input_path, output_path):
    """Apply N4 bias field correction using SimpleITK."""
    img = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrected = corrector.Execute(img, mask)
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
    registration.SetMetricSamplingPercentage(0.01)
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


def preprocess_single(input_file, output_dir, template_path=None):
    """Full preprocessing pipeline for a single T1w scan."""
    stem = Path(input_file).name.replace(".nii.gz", "").replace(".nii", "")
    subj_dir = os.path.join(output_dir, stem)
    os.makedirs(subj_dir, exist_ok=True)

    # Step 1: N4 bias correction
    n4_out = os.path.join(subj_dir, f"{stem}_n4.nii.gz")
    print(f"  [1/4] N4 bias correction: {stem}")
    n4_bias_correction(input_file, n4_out)

    # Step 2: Registration to template (temp_head.nii.gz from BrainIAC repo)
    reg_out = os.path.join(subj_dir, f"{stem}_reg.nii.gz")
    print(f"  [2/4] Registration to template: {stem}")
    if template_path and os.path.exists(template_path):
        register_to_template(n4_out, template_path, reg_out)
    else:
        print("    WARNING: No template provided, skipping registration.")
        reg_out = n4_out

    # Step 3: Skull stripping with HD-BET
    bet_file = os.path.join(subj_dir, f"{stem}_brain.nii.gz")
    print(f"  [3/4] Skull stripping (HD-BET): {stem}")
    skull_strip_hdbet(reg_out, bet_file)

    # Step 4: Resample to 96x96x96 + z-score normalize
    final_out = os.path.join(subj_dir, f"{stem}_brainiac_ready.nii.gz")
    print(f"  [4/4] Resample to 96x96x96 + normalize: {stem}")
    resample_and_normalize(bet_file, final_out)

    print(f"  Done: {final_out}")
    return final_out


def main():
    parser = argparse.ArgumentParser(description="BrainIAC preprocessing pipeline")
    parser.add_argument("--input_dir", type=str, help="Directory with T1w NIfTI files")
    parser.add_argument("--input_file", type=str, help="Single T1w NIfTI file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--template", type=str, default=None,
                        help="Path to temp_head.nii.gz (from BrainIAC repo atlases/)")
    args = parser.parse_args()

    if args.input_file:
        files = [args.input_file]
    elif args.input_dir:
        files = sorted(
            glob.glob(os.path.join(args.input_dir, "**", "*T1w*.nii*"), recursive=True)
        )
    else:
        parser.error("Provide either --input_dir or --input_file")

    print(f"BrainIAC preprocessing: {len(files)} files")
    for f in files:
        preprocess_single(f, args.output_dir, args.template)
    print("All done.")


if __name__ == "__main__":
    main()
