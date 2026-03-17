#!/usr/bin/env python3
"""
Embedding extractor for BrainSegFounder (Swin-T, 768-dim).

Input:  A .txt file with one path per line to preprocessed 96x96x96 NIfTI files.
Output: A .csv in DATA/ with columns: file_path, subject_id, session_id, scanner,
        age, sex, emb_0 ... emb_767

Note: The Stage 1 SSL checkpoint (64-gpu-model_bestValRMSE.pt) was pretrained with
      single-channel T1w input (in_channels=1). The 2-channel (T1w+T2-FLAIR)
      configuration is used in Stage 2 downstream finetuning, not this checkpoint.

Requirements:
  pip install torch monai nibabel numpy pandas

Usage:
  python extract_BrainSegFounder.py --input_list /path/to/file_list.txt \
                                    --checkpoint /path/to/64-gpu-model_bestValRMSE.pt \
                                    --participants /path/to/participants.tsv \
                                    --output ../DATA/embeddings_BrainSegFounder.csv
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from monai.networks.nets import SwinUNETR


def build_model(checkpoint_path, device="cpu"):
    """Build SwinUNETR encoder and load BrainSegFounder pretrained weights.

    The Stage 1 SSL checkpoint uses in_channels=1 (verified: patch_embed.proj.weight
    shape is [48, 1, 2, 2, 2]). Only swinViT encoder weights are loaded;
    SSL heads (contrastive_head, rotation_head) and decoder are skipped.
    """
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,  # Stage 1 checkpoint uses single channel
        out_channels=1,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Checkpoint keys: "module.swinViT.*" (DDP prefix)
    # Also has SSL head keys (contrastive_head, rotation_head, conv.*) — skip those.
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        if new_k.startswith("swinViT."):
            cleaned[new_k] = v

    info = model.load_state_dict(cleaned, strict=False)
    # Expected missing: encoder/decoder weights (not in SSL checkpoint)
    n_loaded = len(cleaned) - len(info.unexpected_keys)
    print(f"BrainSegFounder weight loading: {n_loaded} swinViT keys loaded, "
          f"{len(info.missing_keys)} missing (decoder/encoder heads, expected)")

    model.eval()
    model.to(device)
    return model


def load_volume(file_path):
    """Load preprocessed NIfTI as single-channel [1, 1, 96, 96, 96]."""
    if file_path.endswith(".npy"):
        data = np.load(file_path).astype(np.float32)
    else:
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)

    if data.ndim == 4:
        data = data[0]

    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # [1, 1, 96, 96, 96]
    return tensor


def extract_embedding(model, volume, device):
    """Extract 768-dim embedding from SwinUNETR encoder (deepest bottleneck).

    model.swinViT(x, normalize) returns a list of 5 spatial feature maps:
        [0] [B, 48,  48, 48, 48]  - after patch embed
        [1] [B, 96,  24, 24, 24]  - after stage 1
        [2] [B, 192, 12, 12, 12]  - after stage 2
        [3] [B, 384,  6,  6,  6]  - after stage 3
        [4] [B, 768,  3,  3,  3]  - after stage 4 (bottleneck)

    We take hidden_states[4] and global-average-pool over spatial dims → [B, 768].
    No hook needed — we call swinViT directly.
    """
    with torch.no_grad():
        hidden_states = model.swinViT(volume.to(device), model.normalize)

    # Bottleneck: [B, 768, 3, 3, 3]
    feat = hidden_states[4]
    feat = feat.mean(dim=[2, 3, 4])  # Global average pool → [B, 768]
    return feat.cpu().numpy().flatten()


def parse_bids_path(file_path):
    """Extract subject_id, session_id, and scanner from BIDS-style filename.

    Works with both original BIDS paths and preprocessed filenames like:
      sub-16794_ses-NOT1ACH001_T1w_brainsegfounder_ready.nii.gz
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
    parser = argparse.ArgumentParser(description="BrainSegFounder embedding extractor")
    parser.add_argument("--input_list", type=str, required=True,
                        help=".txt file with one preprocessed NIfTI path per line")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to 64-gpu-model_bestValRMSE.pt")
    parser.add_argument("--participants", type=str, default=None,
                        help="Path to participants.tsv")
    parser.add_argument("--output", type=str,
                        default="../DATA/embeddings_BrainSegFounder.csv")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.input_list, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]
    print(f"BrainSegFounder extractor: {len(file_paths)} files, device={args.device}")

    model = build_model(args.checkpoint, device=args.device)
    demo_df = load_demographics(args.participants)

    rows = []
    for i, fpath in enumerate(file_paths):
        subject_id, session_id, scanner = parse_bids_path(fpath)

        volume = load_volume(fpath)
        embedding = extract_embedding(model, volume, args.device)

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
