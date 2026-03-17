#!/usr/bin/env python3
"""
Embedding extractor for y-Aware (DenseNet-121, 1024-dim).

Input:  A .txt file with one path per line to CAT12-preprocessed files
        (.npy with shape 1x121x145x121, or NIfTI).
Output: A .csv in DATA/ with columns: file_path, subject_id, session_id, scanner,
        age, sex, emb_0 ... emb_1023

Requirements:
  pip install torch monai nibabel numpy pandas

Usage:
  python extract_yAware.py --input_list /path/to/file_list.txt \
                           --checkpoint /path/to/DenseNet121_BHB-10K_yAwareContrastive.pth \
                           --participants /path/to/participants.tsv \
                           --output ../DATA/embeddings_yAware.csv
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nibabel as nib


class DenseNet121Encoder(nn.Module):
    """3D DenseNet-121 encoder as used in y-Aware contrastive learning."""

    def __init__(self, embedding_dim=1024):
        super().__init__()
        from monai.networks.nets import DenseNet121
        self.backbone = DenseNet121(spatial_dims=3, in_channels=1, out_channels=embedding_dim)

    def forward(self, x):
        return self.backbone(x)


def build_model(checkpoint_path, device="cpu"):
    """Build DenseNet-121 and load y-Aware pretrained weights.

    The y-Aware checkpoint (Dufumier et al.) stores weights under the "model" key
    with DataParallel prefix "module.". Key differences from MONAI:
      - Checkpoint: denselayer1.norm1.weight (no .layers.)
      - MONAI:      denselayer1.layers.norm1.weight
    We remap keys accordingly and add "backbone." prefix for our wrapper.
    """
    model = DenseNet121Encoder(embedding_dim=1024)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # y-Aware stores weights under "model" key
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Key remapping:
    #   1. Strip "module." (DataParallel)
    #   2. Insert ".layers." inside denselayer paths (MONAI convention)
    #   3. Add "backbone." prefix (our wrapper stores DenseNet as self.backbone)
    #   4. Skip projection head keys (hidden_representation.*, head_projection.*)
    cleaned = {}
    for k, v in state_dict.items():
        new_k = re.sub(r"^module\.", "", k)

        # Skip contrastive projection head
        if new_k.startswith("hidden_representation.") or new_k.startswith("head_projection."):
            continue

        # Insert ".layers." for denselayer sub-keys
        # y-Aware:  denselayer1.norm1.weight
        # MONAI:    denselayer1.layers.norm1.weight
        new_k = re.sub(r"(denselayer\d+)\.(norm|relu|conv)", r"\1.layers.\2", new_k)

        # Add "backbone." prefix for our wrapper
        new_k = "backbone." + new_k

        cleaned[new_k] = v

    info = model.load_state_dict(cleaned, strict=False)
    # Expected missing: backbone.features.norm5.* (not in y-Aware encoder-only ckpt)
    #                   backbone.class_layers.* (randomly initialized, used for dim matching)
    print(f"y-Aware weight loading: {len(cleaned)} keys loaded, "
          f"{len(info.missing_keys)} missing, {len(info.unexpected_keys)} unexpected")
    if info.unexpected_keys:
        print(f"  Unexpected: {info.unexpected_keys}")

    model.eval()
    model.to(device)
    return model


def load_volume(file_path):
    """Load a preprocessed volume as tensor [1, 1, 121, 145, 121]."""
    if file_path.endswith(".npy"):
        data = np.load(file_path).astype(np.float32)
        if data.ndim == 4:
            tensor = torch.from_numpy(data).unsqueeze(0)  # [1, 1, 121, 145, 121]
        else:
            tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    else:
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)
        if data.ndim == 4:
            data = data[0]
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return tensor


def extract_embedding(model, volume, device):
    """Extract 1024-dim embedding from DenseNet-121 encoder.

    MONAI DenseNet class_layers: Sequential(relu, pool, flatten, out)
    We hook into class_layers.flatten to get the 1024-dim vector
    after AdaptiveAvgPool3d + Flatten, before the final Linear layer.
    """
    embeddings = {}

    def hook_fn(module, input, output):
        embeddings["feat"] = output

    # Hook on Flatten inside class_layers (after avgpool, before linear)
    hook = model.backbone.class_layers.flatten.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(volume.to(device))

    hook.remove()
    feat = embeddings["feat"].cpu().numpy().flatten()
    return feat


def parse_bids_path(file_path):
    """Extract subject_id, session_id, and scanner from BIDS-style filename.

    Works with both original BIDS paths and preprocessed filenames like:
      sub-16794_ses-NOT1ACH001_T1w_yaware_ready.npy
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
    parser = argparse.ArgumentParser(description="y-Aware embedding extractor")
    parser.add_argument("--input_list", type=str, required=True,
                        help=".txt file with one preprocessed file path per line")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DenseNet121_BHB-10K_yAwareContrastive.pth")
    parser.add_argument("--participants", type=str, default=None,
                        help="Path to participants.tsv")
    parser.add_argument("--output", type=str,
                        default="../DATA/embeddings_yAware.csv")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.input_list, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]
    print(f"y-Aware extractor: {len(file_paths)} files, device={args.device}")

    model = build_model(args.checkpoint, args.device)
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
