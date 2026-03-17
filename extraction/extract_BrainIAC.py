#!/usr/bin/env python3
"""
Embedding extractor for BrainIAC (ViT-B, 768-dim).

Input:  A .txt file with one path per line to preprocessed 96x96x96 NIfTI files.
Output: A .csv in DATA/ with columns: file_path, subject_id, session_id, scanner,
        age, sex, emb_0 ... emb_767

Requirements:
  pip install torch monai nibabel numpy pandas

Usage:
  python extract_BrainIAC.py --input_list /path/to/file_list.txt \
                             --checkpoint /path/to/BrainIAC.ckpt \
                             --participants /path/to/participants.tsv \
                             --output ../DATA/embeddings_BrainIAC.csv
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from monai.networks.nets import ViT


def build_model(checkpoint_path, device="cpu"):
    """Build BrainIAC ViT-B model and load pretrained weights.

    The official BrainIAC code (github.com/AIM-KannLab/BrainIAC) uses MONAI ViT
    with classification=False (default). This means NO CLS token is created;
    the output is [B, num_patches, hidden_size] = [B, 216, 768].
    The official code extracts features[0][:, 0] (first patch embedding).
    """
    model = ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        # classification=False is the default — no CLS token
    )

    # Load SimCLR checkpoint (Lightning format with "state_dict" key)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Extract only backbone weights (strip "backbone." prefix)
    # The checkpoint has keys: backbone.patch_embedding.*, backbone.blocks.*, etc.
    # plus projection_head.* keys that we skip.
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            cleaned[k[len("backbone."):]] = v

    info = model.load_state_dict(cleaned, strict=False)
    # Newer MONAI adds cross_attn layers to ViT — not in checkpoint, not used
    unexpected = [k for k in info.unexpected_keys]
    missing_other = [k for k in info.missing_keys if "cross_attn" not in k]
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    if missing_other:
        raise RuntimeError(f"Missing non-cross-attn keys: {missing_other}")
    print(f"BrainIAC weight loading: OK ({len(cleaned)} keys loaded, "
          f"{len(info.missing_keys)} cross-attn keys skipped)")

    model.eval()
    model.to(device)
    return model


def load_volume(file_path):
    """Load a preprocessed NIfTI volume as a tensor [1, 1, 96, 96, 96]."""
    if file_path.endswith(".npy"):
        data = np.load(file_path).astype(np.float32)
    else:
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)

    if data.ndim == 4:
        data = data[0]
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # [1, 1, 96, 96, 96]
    return tensor


def parse_bids_path(file_path):
    """Extract subject_id, session_id, and scanner from BIDS-style filename.

    Works with both original BIDS paths and preprocessed filenames like:
      sub-16794_ses-NOT1ACH001_T1w_brainiac_ready.nii.gz
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
        full_ses = ses_match.group(1)  # e.g. "NOT1ACH001"
        session_id = f"ses-{full_ses}"
        scanner = full_ses[:-3]  # strip 3-digit visit → "NOT1ACH"
    else:
        session_id, scanner = "", ""
    return subject_id, session_id, scanner


def load_demographics(participants_file):
    """Load demographics from BIDS participants.tsv."""
    if participants_file and os.path.exists(participants_file):
        df = pd.read_csv(participants_file, sep="\t")
        return df
    return None


def main():
    parser = argparse.ArgumentParser(description="BrainIAC embedding extractor")
    parser.add_argument("--input_list", type=str, required=True,
                        help=".txt file with one preprocessed NIfTI path per line")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to BrainIAC.ckpt")
    parser.add_argument("--participants", type=str, default=None,
                        help="Path to participants.tsv for demographics")
    parser.add_argument("--output", type=str,
                        default="../DATA/embeddings_BrainIAC.csv",
                        help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.input_list, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]
    print(f"BrainIAC extractor: {len(file_paths)} files, device={args.device}")

    model = build_model(args.checkpoint, args.device)
    demo_df = load_demographics(args.participants)

    rows = []
    for i, fpath in enumerate(file_paths):
        subject_id, session_id, scanner = parse_bids_path(fpath)

        volume = load_volume(fpath).to(args.device)
        with torch.no_grad():
            output, _ = model(volume)
            # output shape: [B, 216, 768] — 216 patch embeddings, no CLS token.
            # Official BrainIAC takes the first patch embedding (index 0).
            embedding = output[:, 0, :].cpu().numpy().flatten()  # 768-dim

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
