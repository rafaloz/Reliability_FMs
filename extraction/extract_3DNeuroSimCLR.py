#!/usr/bin/env python3
"""
Embedding extractor for 3D-Neuro-SimCLR (ResNet-18, 512-dim).

Input:  A .txt file with one path per line to preprocessed NIfTI files
        (*_simclr_ready.nii.gz). Corresponding brain masks are expected at
        *_mask_mni.nii.gz in the same directory.
Output: A .csv in DATA/ with columns: file_path, subject_id, session_id, scanner,
        age, sex, emb_0 ... emb_511

The original 3D-Neuro-SimCLR pipeline (Kaczmarek et al., 2025) applies:
  1. A transpose + center crop to 150x192x192 after MNI registration
  2. Per-sample masked z-score normalization (not WhiteStripe)
Both steps are applied here at extraction time to match the original training.

Requirements:
  pip install torch monai nibabel numpy pandas

Usage:
  python extract_3DNeuroSimCLR.py --input_list /path/to/file_list.txt \
                                  --checkpoint /path/to/simclr_3d_brain_foundation.tar \
                                  --participants /path/to/participants.tsv \
                                  --output ../DATA/embeddings_3DNeuroSimCLR.csv
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from monai.networks.nets import resnet18


def build_model(checkpoint_path, device="cpu"):
    """Build 3D ResNet-18 encoder and load SimCLR pretrained weights.

    The 3D-Neuro-SimCLR checkpoint (Kaczmarek et al.) stores weights under
    "model_state_dict" with keys prefixed "module.encoder.*" (from DDP + SimCLR
    wrapper). We strip both prefixes and skip projector keys.

    Using feed_forward=False omits the final FC layer, matching the original
    SimCLR training where fc was replaced with Identity().
    """
    model = resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]

    # Checkpoint keys: "module.encoder.<key>" and "module.projector.<key>"
    # Strip both "module." and "encoder." prefixes; skip projector weights
    cleaned = {}
    for k, v in state_dict.items():
        if "projector" in k:
            continue
        new_k = k.replace("module.", "").replace("encoder.", "")
        cleaned[new_k] = v

    info = model.load_state_dict(cleaned, strict=True)
    print(f"3D-Neuro-SimCLR weight loading: OK (strict=True)")

    model.eval()
    model.to(device)
    return model


def load_volume(file_path):
    """Load a preprocessed NIfTI volume, apply the original transpose+crop
    and per-sample masked z-score normalization.

    The original 3D-Neuro-SimCLR repo applies (in t1_datasets.py):
      1. _transpose_crop: np.transpose(arr, (0, 3, 2, 1))[:, 10:160, 19:211, 1:]
         This produces a (1, 150, 192, 192) volume from the MNI-registered input.
      2. Per-sample masked z-score: (x - mean_brain) / std_brain, zeroing non-brain.

    The brain mask is loaded from the corresponding *_mask_mni.nii.gz file.
    """
    # Load volume
    img = nib.load(file_path)
    data = img.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]

    # Load corresponding brain mask
    mask_path = _find_mask_path(file_path)
    if mask_path and os.path.exists(mask_path):
        mask = nib.load(mask_path).get_fdata().astype(np.float32)
        if mask.ndim == 4:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
    else:
        # Fallback: derive mask from nonzero voxels of skull-stripped volume
        print(f"    Warning: mask not found at {mask_path}, using nonzero voxels")
        mask = (np.abs(data) > 0).astype(np.float32)

    # Add channel dim: (X, Y, Z) -> (1, X, Y, Z)
    data = data[np.newaxis, ...]
    mask = mask[np.newaxis, ...]

    # Step 1: Transpose + crop (matches original _transpose_crop)
    # Input after TurboPrep is ~(1, 193, 229, 193) in MNI space
    # Transpose (0, 3, 2, 1) swaps spatial axes X and Z
    data = np.transpose(data, (0, 3, 2, 1))  # (1, Z', Y', X')
    mask = np.transpose(mask, (0, 3, 2, 1))

    # Crop to 150x192x192: [:, 10:160, 19:211, 1:]
    data = data[:, 10:160, 19:211, 1:]
    mask = mask[:, 10:160, 19:211, 1:]

    print(f"    Volume after crop: {data.shape}")

    # Step 2: Per-sample masked z-score normalization
    # Matches original standardize() in t1_datasets.py
    data = data * mask
    n_brain = mask.sum()
    if n_brain > 0:
        mean_val = (data * mask).sum() / n_brain
        std_val = np.sqrt((data ** 2 * mask).sum() / n_brain - mean_val ** 2 + 1e-12)
        data = (data - mean_val) / std_val * mask

    # Add batch dim: (1, 150, 192, 192) -> (1, 1, 150, 192, 192)
    tensor = torch.from_numpy(data).unsqueeze(0)
    return tensor


def _find_mask_path(file_path):
    """Derive the brain mask path from the volume path.

    Expects the mask to be in the same directory with the naming convention:
      *_simclr_ready.nii.gz  ->  *_mask_mni.nii.gz
    """
    dirname = os.path.dirname(file_path)
    basename = os.path.basename(file_path)

    # Try replacing _simclr_ready with _mask_mni
    mask_name = basename.replace("_simclr_ready", "_mask_mni")
    if mask_name != basename:
        return os.path.join(dirname, mask_name)

    # Fallback: look for any *_mask_mni* in same directory
    stem = basename.split("_")[0]  # e.g. "sub-16794"
    for f in os.listdir(dirname):
        if "mask_mni" in f and stem in f:
            return os.path.join(dirname, f)

    return None


def parse_bids_path(file_path):
    """Extract subject_id, session_id, and scanner from BIDS-style filename.

    Works with both original BIDS paths and preprocessed filenames like:
      sub-16794_ses-NOT1ACH001_T1w_simclr_ready.nii.gz
    Returns:
      subject_id: "sub-16794"
      session_id: "ses-NOT1ACH001"
      scanner:    "NOT1ACH"  (session code minus 3-digit visit number)
    """
    basename = os.path.basename(file_path)
    sub_match = re.search(r'(sub-\d+)', basename)
    ses_match = re.search(r'ses-([A-Za-z0-9]+?)_', basename)
    subject_id = sub_match.group(1) if sub_match else ""
    if ses_match:
        full_ses = ses_match.group(1)
        session_id = f"ses-{full_ses}"
        scanner = full_ses[:-3]
    else:
        session_id, scanner = "", ""
    return subject_id, session_id, scanner


def load_demographics(participants_file):
    """Load demographics from BIDS participants.tsv."""
    if participants_file and os.path.exists(participants_file):
        return pd.read_csv(participants_file, sep="\t")
    return None


def main():
    parser = argparse.ArgumentParser(description="3D-Neuro-SimCLR embedding extractor")
    parser.add_argument("--input_list", type=str, required=True,
                        help=".txt file with one preprocessed NIfTI path per line")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to simclr_3d_brain_foundation.tar")
    parser.add_argument("--participants", type=str, default=None,
                        help="Path to participants.tsv")
    parser.add_argument("--output", type=str,
                        default="../DATA/embeddings_3DNeuroSimCLR.csv")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.input_list, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]
    print(f"3D-Neuro-SimCLR extractor: {len(file_paths)} files, device={args.device}")

    model = build_model(args.checkpoint, args.device)
    demo_df = load_demographics(args.participants)

    rows = []
    for i, fpath in enumerate(file_paths):
        subject_id, session_id, scanner = parse_bids_path(fpath)

        volume = load_volume(fpath)
        with torch.no_grad():
            embedding = model(volume.to(args.device))
            embedding = embedding.cpu().numpy().flatten()  # 512-dim

        age, sex = "", ""
        if demo_df is not None and subject_id:
            match = demo_df[demo_df["participant_id"] == subject_id]
            if len(match) > 0:
                age = match.iloc[0].get("age", "")
                sex = match.iloc[0].get("sex", "")

        row = {
            "file_path": fpath,
            "subject_id": subject_id,
            "session_id": session_id,
            "scanner": scanner,
            "age": age,
            "sex": sex,
        }
        for d in range(len(embedding)):
            row[f"emb_{d}"] = embedding[d]
        rows.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == len(file_paths):
            print(f"  [{i+1}/{len(file_paths)}] {subject_id} {session_id}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}  ({len(df)} rows, {len(embedding)} embedding dims)")


if __name__ == "__main__":
    main()
