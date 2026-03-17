#!/usr/bin/env python3
"""
Embedding extractor for AnatCL (ResNet-18, 512-dim).

Input:  A .txt file with one path per line to CAT12-preprocessed 121x145x121 NIfTI files.
Output: A .csv in DATA/ with columns: file_path, subject_id, session_id, scanner,
        age, sex, emb_0 ... emb_511

Runs all 5 cross-validation folds independently, producing one CSV per fold
(embeddings_AnatCL_fold0.csv ... fold4.csv), matching the AnatCL paper's
evaluation protocol. Downstream analysis computes metrics per fold and
reports mean ± std.

Requirements:
  pip3 install git+https://github.com/carloalbertobarbano/AnatCL.git
  pip install torch nibabel numpy pandas

Usage:
  python extract_AnatCL.py --input_list /path/to/file_list.txt \
                           --checkpoint_dir /path/to/resnet18-anatcl-g3/ \
                           --participants /path/to/participants.tsv \
                           --output_dir ../DATA/
"""

import argparse
import os
import pickle as _pickle
import re
import sys
import types
import numpy as np
import pandas as pd
import torch
import nibabel as nib


# ── Custom unpickler for AnatCL checkpoints ─────────────────────────────────
# The checkpoint was saved in the training env which had a top-level 'models'
# package (models.resnet, models.estimators, ...).  The anatcl pip package only
# ships a subset under 'anatcl.models'.  We override find_class to:
#   1. Try the redirected path  (models.X → anatcl.models.X)
#   2. If that also fails, create a stub class so unpickling succeeds.
# After unpickling, we only need the state_dict (tensor weights), so stubs
# for unknown wrapper classes are harmless.

class _AnatCLUnpickler(_pickle.Unpickler):
    def find_class(self, module, name):
        if module == "models" or module.startswith("models."):
            redirected = "anatcl." + module
            try:
                return super().find_class(redirected, name)
            except (ModuleNotFoundError, AttributeError):
                pass
            # Create a stub class that can be instantiated and unpickled
            stub = type(name, (), {
                "__init__": lambda self, *a, **kw: None,
                "__setstate__": lambda self, state: (
                    self.__dict__.update(state) if isinstance(state, dict) else None
                ),
            })
            return stub
        return super().find_class(module, name)


# Module-like wrapper so we can pass it as torch.load(pickle_module=...)
_custom_pickle = types.ModuleType("_custom_pickle")
for _attr in dir(_pickle):
    setattr(_custom_pickle, _attr, getattr(_pickle, _attr))
_custom_pickle.Unpickler = _AnatCLUnpickler


def build_model(checkpoint_path, descriptor="global", fold=0, device="cpu"):
    """Build AnatCL model architecture and load weights from a local .pth file.

    Uses AnatCL(pretrained=False) to create the model, then loads the checkpoint
    with a custom unpickler to handle missing training-code modules.
    """
    from anatcl import AnatCL

    # Create architecture only — no auto-download
    model = AnatCL(descriptor=descriptor, fold=fold, pretrained=False)

    # Load checkpoint with our custom pickle module
    ckpt = torch.load(checkpoint_path, map_location=device,
                      pickle_module=_custom_pickle, weights_only=False)

    # Extract state_dict depending on checkpoint format
    if hasattr(ckpt, "state_dict") and callable(ckpt.state_dict):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"Cannot extract state_dict from checkpoint type {type(ckpt)}")

    info = model.load_state_dict(state_dict, strict=False)
    print(f"AnatCL weight loading: {len(state_dict)} keys in checkpoint, "
          f"{len(info.missing_keys)} missing, {len(info.unexpected_keys)} unexpected")
    if info.missing_keys:
        print(f"  Missing: {info.missing_keys[:5]}{'...' if len(info.missing_keys) > 5 else ''}")
    if info.unexpected_keys:
        print(f"  Unexpected: {info.unexpected_keys[:5]}{'...' if len(info.unexpected_keys) > 5 else ''}")

    model.eval()
    model.to(device)
    return model


def load_volume(file_path):
    """Load a CAT12-preprocessed NIfTI volume as tensor [1, 1, 121, 145, 121]."""
    if file_path.endswith(".npy"):
        data = np.load(file_path).astype(np.float32)
    else:
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)

    if data.ndim == 4:
        data = data[0]
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return tensor


def extract_embedding(model, volume, device):
    """Extract 512-dim embedding from AnatCL.

    With use_head=False (default), model(x) returns the 512-dim feature vector
    directly (from backbone.encoder: avgpool → flatten). No hook needed.
    """
    with torch.no_grad():
        feat = model(volume.to(device))  # [B, 512]
    return feat.cpu().numpy().flatten()


def parse_bids_path(file_path):
    """Extract subject_id, session_id, and scanner from BIDS-style filename.

    Works with both original BIDS paths and preprocessed filenames like:
      sub-16794_ses-NOT1ACH001_T1w_anatcl_ready.nii.gz
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


def extract_all_images(model, file_paths, demo_df, device):
    """Run extraction for all images with a single model. Returns list of row dicts."""
    rows = []
    for i, fpath in enumerate(file_paths):
        subject_id, session_id, scanner = parse_bids_path(fpath)

        volume = load_volume(fpath)
        embedding = extract_embedding(model, volume, device)

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
            print(f"    [{i+1}/{len(file_paths)}] {subject_id} {session_id}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="AnatCL embedding extractor (5 folds)")
    parser.add_argument("--input_list", type=str, required=True,
                        help=".txt file with one preprocessed NIfTI path per line")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing fold0.pth ... fold4.pth")
    parser.add_argument("--descriptor", type=str, default="global",
                        choices=["global", "local"],
                        help="AnatCL descriptor type (default: global)")
    parser.add_argument("--participants", type=str, default=None,
                        help="Path to participants.tsv")
    parser.add_argument("--output_dir", type=str, default="../DATA",
                        help="Output directory for per-fold CSVs")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.input_list, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]
    print(f"AnatCL extractor ({args.descriptor}, 5 folds): "
          f"{len(file_paths)} files, device={args.device}")

    demo_df = load_demographics(args.participants)
    os.makedirs(args.output_dir, exist_ok=True)

    for fold in range(5):
        ckpt_path = os.path.join(args.checkpoint_dir, f"fold{fold}.pth")
        if not os.path.isfile(ckpt_path):
            print(f"  WARNING: {ckpt_path} not found, skipping fold {fold}")
            continue

        print(f"\n  --- Fold {fold} ---")
        model = build_model(ckpt_path, args.descriptor, fold, args.device)
        rows = extract_all_images(model, file_paths, demo_df, args.device)

        out_path = os.path.join(args.output_dir, f"embeddings_AnatCL_fold{fold}.csv")
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        n_dims = sum(1 for c in df.columns if c.startswith("emb_"))
        print(f"  Saved: {out_path}  ({len(df)} rows, {n_dims} dims)")

        # Free GPU memory before loading next fold
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"\nAll done! Per-fold CSVs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
