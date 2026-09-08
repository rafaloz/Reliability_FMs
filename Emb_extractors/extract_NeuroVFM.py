#!/usr/bin/env python3
"""
Embedding extractor for NeuroVFM (3D ViT-Base, 768-dim, Vol-JEPA pretrained).

Input:  A .txt file with one path per line to RAW T1w NIfTI files (no preproc — NeuroVFM's
        StudyPreprocessor handles the full 8-step internal pipeline: reorient RPI + infer the
        acquisition axis + resample with 4 mm along that axis and 1 mm in-plane + arrange the
        array in the released canonical view + center-crop to a multiple of patch_size + MRI
        normalization (percentile clip [0.5, 99.5] + min-max) + foreground
        mask (10th-percentile threshold) + tokenize into 4x16x16 patches + drop background
        tokens).

        DELIBERATE (audit H9): the NeuroVFM paper describes the foreground mask as an Otsu
        threshold, but the SHIPPED code in neurovfm/data/preprocess.py uses a fixed
        10th-percentile threshold (np.percentile(img_arr, 10)). This extractor is FAITHFUL
        TO THE SHIPPED CODE, not to the paper text — we run the model exactly as its authors
        released it. Do NOT "fix" this to Otsu without also patching the upstream package.
Output: A .csv with columns: file_path, subject_id, session_id, scanner, age, sex, n_tokens,
        emb_0 ... emb_767  (mean-pooled over tokens)

Load hygiene: the shipped neurovfm.pipelines.load_encoder() calls model.load_state_dict(sd,
strict=False) without inspecting the missing/unexpected keys. This wrapper resolves the
same checkpoint, compares its complete key set, shapes, and tensor values with the returned
model, and aborts if the upstream load was incomplete.

Requirements:
  pip install -e /path/to/neurovfm     # (already installed on cluster)
  huggingface-cli login                # first-run only, needs HF_TOKEN with mlinslab access

Usage:
  python extract_NeuroVFM.py --input_list DATA/filelists/filelist_NeuroVFM.txt \
                             --participants /path/to/participants.tsv \
                             --output DATA/embeddings/embeddings_NeuroVFM.csv \
                             --repo mlinslab/neurovfm-encoder
"""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from neurovfm.pipelines import load_encoder


# --------------------------------------------------------------------------- #
# Load hygiene: wrap load_encoder and re-verify weight coverage
# --------------------------------------------------------------------------- #

# If pretraining ckpt is known to drop task-only heads, add substrings here.
# Empty by default — we will discover the true set on first run and update.
LOAD_WHITELIST_MISSING = []
LOAD_WHITELIST_UNEXPECTED = []
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
EXPECTED_ACQUISITIONS = 165
EXPECTED_EMBEDDING_DIM = 768
MANIFEST_SCHEMA_VERSION = "onharmony_embedding_manifest_v1"
PIPELINE_VERSION = "neurovfm_onharmony_embedding_v1"
READOUT = "mean over variable-length final-normalized foreground tokens"


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of one input, release, or output artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(repo):
    if os.path.isdir(repo):
        path = os.path.join(repo, "pytorch_model.bin")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"NeuroVFM checkpoint not found: {path}")
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename="pytorch_model.bin")


def load_encoder_verified(repo, device):
    """Load with the official helper, then verify against its real checkpoint."""
    encoder, preproc = load_encoder(repo, device=device)
    model = encoder.model

    checkpoint = torch.load(_checkpoint_path(repo), map_location="cpu",
                            weights_only=False)
    checkpoint_sd = checkpoint.get("state_dict", checkpoint)
    if not isinstance(checkpoint_sd, dict):
        raise TypeError("NeuroVFM checkpoint does not contain a state_dict mapping")
    model_sd = model.state_dict()

    missing_all = sorted(set(model_sd) - set(checkpoint_sd))
    unexpected_all = sorted(set(checkpoint_sd) - set(model_sd))
    missing = [k for k in missing_all
               if not any(w in k for w in LOAD_WHITELIST_MISSING)]
    unexpected = [k for k in unexpected_all
                  if not any(w in k for w in LOAD_WHITELIST_UNEXPECTED)]

    common = sorted(set(model_sd) & set(checkpoint_sd))
    shape_mismatches = [
        (k, tuple(model_sd[k].shape), tuple(checkpoint_sd[k].shape))
        for k in common if model_sd[k].shape != checkpoint_sd[k].shape
    ]
    value_mismatches = []
    if not shape_mismatches:
        for key in common:
            loaded = model_sd[key].detach().cpu()
            expected = checkpoint_sd[key].detach().cpu()
            if not torch.equal(loaded, expected):
                value_mismatches.append(key)

    print(f"NeuroVFM load report:")
    print(f"  ckpt keys:   {len(checkpoint_sd)}")
    print(f"  model keys:  {len(model_sd)}")
    print(f"  missing:     {len(missing_all)}  (non-whitelisted: {len(missing)})")
    print(f"  unexpected:  {len(unexpected_all)}  (non-whitelisted: {len(unexpected)})")
    print(f"  bad shapes:  {len(shape_mismatches)}")
    print(f"  bad values:  {len(value_mismatches)}")
    if missing:
        print(f"  missing sample: {missing[:5]}")
    if unexpected:
        print(f"  unexpected sample: {unexpected[:5]}")

    if missing:
        raise RuntimeError(
            f"NeuroVFM checkpoint left {len(missing)} model params uninitialised. "
            f"First 10: {missing[:10]}. If this is expected (e.g. task heads not "
            f"included in pretrained ckpt), add matching substrings to "
            f"LOAD_WHITELIST_MISSING at the top of this script.")
    if unexpected:
        raise RuntimeError(
            f"NeuroVFM checkpoint has {len(unexpected)} keys that did not map to "
            f"any model param. First 10: {unexpected[:10]}. Check for a stale "
            f"neurovfm package version.")
    if shape_mismatches:
        raise RuntimeError(
            f"NeuroVFM checkpoint/model shape mismatches: {shape_mismatches[:10]}")
    if value_mismatches:
        raise RuntimeError(
            f"NeuroVFM returned model differs from checkpoint in "
            f"{len(value_mismatches)} tensors. First 10: {value_mismatches[:10]}")

    print("  load verified: every model tensor matches the checkpoint")
    return encoder, preproc


def _atomic_write_csv(df, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = output_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, output_path)


def _atomic_write_json(payload, output_path):
    """Publish one standards-compliant JSON manifest atomically."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)


def _manifest_path(output_path):
    """Return the conventional manifest path paired with one embedding CSV."""
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_manifest.json")


# --------------------------------------------------------------------------- #
# Metadata helpers (same convention as extract_BrainIAC.py)
# --------------------------------------------------------------------------- #

def parse_bids_path(file_path):
    basename = os.path.basename(file_path)
    sub = re.search(r'(sub-\d+)', basename)
    ses = re.search(r'ses-([A-Za-z0-9]+?)_', basename)
    subject_id = sub.group(1) if sub else ""
    if ses:
        full = ses.group(1)
        session_id = f"ses-{full}"
        scanner = full[:-3]
    else:
        session_id, scanner = "", ""
    return subject_id, session_id, scanner


def load_demographics(participants_file):
    if participants_file and os.path.exists(participants_file):
        return pd.read_csv(participants_file, sep="\t")
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _autodetect_input_list():
    # Raw T1w paths; StudyPreprocessor performs the model's internal preprocessing.
    return PROJECT_ROOT / "DATA" / "filelists" / "filelist_NeuroVFM.txt"


def main():
    _def_input_list = _autodetect_input_list()
    ap = argparse.ArgumentParser(description="NeuroVFM embedding extractor")
    ap.add_argument("--input_list", default=_def_input_list,
                    help=f".txt file, one raw T1w NIfTI path per line "
                         f"(default: {_def_input_list})")
    ap.add_argument("--repo", default="mlinslab/neurovfm-encoder",
                    help="HuggingFace repo id (default: mlinslab/neurovfm-encoder)")
    ap.add_argument("--participants",
                    default=str(PROJECT_ROOT / "DATA" / "participants.tsv"),
                    help="Path to participants.tsv for age/sex")
    ap.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "DATA" / "embeddings" /
                    "embeddings_NeuroVFM.csv"),
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--expected_count", type=int, default=EXPECTED_ACQUISITIONS)
    ap.add_argument("--force", action="store_true",
                    help="Replace an existing CSV/manifest intentionally")
    args = ap.parse_args()

    input_list_path = Path(args.input_list).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = _manifest_path(output_path)
    existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing NeuroVFM artifacts; pass --force: "
            + ", ".join(existing)
        )
    with input_list_path.open("r", encoding="utf-8") as f:
        paths = [l.strip() for l in f if l.strip()]
    if len(paths) != args.expected_count or len(paths) != len(set(paths)):
        raise ValueError(
            f"Expected {args.expected_count} unique NeuroVFM inputs, found {len(paths)}"
        )
    missing_paths = [p for p in paths if not os.path.isfile(p)]
    if missing_paths:
        raise FileNotFoundError(
            f"NeuroVFM input list has {len(missing_paths)} missing files. "
            f"First 10: {missing_paths[:10]}")
    print(f"NeuroVFM extractor: {len(paths)} volumes, device={args.device}")

    # Extraction 1/3: build the NeuroVFM encoder and verify its weights.
    print(f"[Extraction 1/3] Load NeuroVFM encoder from HF ({args.repo}) — verified strict-load wrapper")
    encoder, preproc = load_encoder_verified(args.repo, args.device)
    checkpoint_path = Path(_checkpoint_path(args.repo)).expanduser().resolve()
    demo = load_demographics(args.participants)

    rows = []
    failures = []
    emb_dim = None
    n_tokens_ref = None  # audit L-NVFM-196: track token count across subjects
    n_tokens_seen = set()  # audit L-NVFM-196: collect unique values for reporting
    for i, fp in enumerate(paths):
        subject_id, session_id, scanner = parse_bids_path(fp)

        try:
            # Preprocessing 1/1: StudyPreprocessor internal pipeline (8 operations):
            #         RPI reorient + infer acquisition axis + resample at 4 mm along that axis
            #         and 1 mm in-plane + arrange the released canonical view + center-crop
            #         to multiple of (4,16,16) + MRI percentile clip [0.5,99.5]% + min-max +
            #         foreground mask (10th-pct) + tokenize 4×16×16 + drop background tokens.
            batch = preproc.load_study(fp, modality="mri")
            # Extraction 2/3: encoder.embed(batch) returns [N_tokens, 768].
            # Keep the official default BF16-autocast inference path used by the
            # successful canonical extraction. The released pipeline disables
            # FlashAttention independently (use_flash_attn=False); we do not
            # claim bit-exact reproducibility across GPU architectures.
            tokens = encoder.embed(batch)   # [N_tokens, D]  (use_amp=True default)
            # Extraction 3/3: mean-pool tokens → 768-d (no attentive probe).
            emb = tokens.mean(dim=0).float().cpu().numpy()  # [D]
        except Exception as e:
            print(f"  [{i+1}/{len(paths)}] {subject_id} {session_id} FAILED: {e}")
            failures.append((fp, repr(e)))
            continue

        if emb_dim is None:
            emb_dim = emb.shape[0]
            print(f"  embedding dim: {emb_dim}")
        assert emb.shape == (emb_dim,), f"dim drift at {fp}: {emb.shape}"

        # n_tokens varies naturally per subject because input volume shape varies
        # (176×192×36 vs 160×192×40 etc.). After StudyPreprocessor's resampling
        # at 4 mm along the inferred acquisition axis and 1 mm in-plane,
        # center-crop-to-divisible-by-(4,16,16), foreground masking, and
        # tokenize 4×16×16, the number of foreground tokens differs by subject.
        # The mean-pool over tokens still produces a comparable 768-d vector
        # because it collapses the token axis (each token is 768-d). Prior audit
        # L-NVFM-196 asserted constancy and was reverted 2026-07-18 — kept as
        # informational log only.
        n_tok = int(tokens.shape[0])
        n_tokens_seen.add(n_tok)
        if n_tokens_ref is None:
            n_tokens_ref = n_tok
            print(f"  n_tokens (first subject): {n_tokens_ref}  (per-subject variation expected)")

        age, sex = "", ""
        if demo is not None and subject_id:
            m = demo[demo["participant_id"] == subject_id]
            if len(m) > 0:
                age = m.iloc[0].get("age", "")
                sex = m.iloc[0].get("sex", "")

        row = {
            "file_path": fp,
            "subject_id": subject_id,
            "session_id": session_id,
            "scanner": scanner,
            "age": age,
            "sex": sex,
            "n_tokens": int(tokens.shape[0]),
        }
        for d in range(emb_dim):
            row[f"emb_{d}"] = float(emb[d])
        rows.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == len(paths):
            print(f"  [{i+1}/{len(paths)}] {subject_id} {session_id}  "
                  f"n_tokens={tokens.shape[0]}  norm={np.linalg.norm(emb):.3f}")

    # Output validation/serialization is not an extraction step.
    print(f"[Output] Validate and save CSV → {args.output}")
    df = pd.DataFrame(rows)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    invalid_meta = 0 if df.empty else int(
        (df[["subject_id", "session_id", "scanner"]] == "").any(axis=1).sum())
    nonfinite = 0 if not emb_cols else int(
        (~np.isfinite(df[emb_cols].to_numpy(dtype=np.float64))).sum())
    if failures or len(df) != len(paths) or invalid_meta or nonfinite:
        partial_path = str(output_path) + ".partial.csv"
        if not df.empty:
            _atomic_write_csv(df, partial_path)
        raise RuntimeError(
            f"Refusing to replace {args.output}: extracted {len(df)}/{len(paths)} "
            f"rows, failures={len(failures)}, invalid_metadata_rows={invalid_meta}, "
            f"nonfinite_values={nonfinite}. Partial output: {partial_path}")
    if len(emb_cols) != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"NeuroVFM dimension drift: {len(emb_cols)}/{EXPECTED_EMBEDDING_DIM}"
        )
    if df["file_path"].duplicated().any():
        raise RuntimeError("Refusing to save NeuroVFM CSV with duplicate file_path rows")
    _atomic_write_csv(df, str(output_path))
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "NeuroVFM",
        "variant": "canonical",
        "readout": READOUT,
        "feature_prefix": "emb_",
        "embedding_dimensions": EXPECTED_EMBEDDING_DIM,
        "device": args.device,
        "model_source": args.repo,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "input_list": str(input_list_path),
        "input_list_sha256": sha256_file(input_list_path),
        "extractor": str(SCRIPT_PATH),
        "extractor_sha256": sha256_file(SCRIPT_PATH),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "n_rows": len(df),
        "token_counts": sorted(n_tokens_seen),
    }
    if args.participants:
        participants = Path(args.participants).expanduser().resolve()
        manifest.update(
            participants=str(participants),
            participants_sha256=sha256_file(participants),
        )
    _atomic_write_json(manifest, manifest_path)
    print(f"Saved: {output_path}  ({len(df)} rows, {emb_dim} emb dims)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
