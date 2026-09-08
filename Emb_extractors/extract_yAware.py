#!/usr/bin/env python3
"""Extract canonical y-Aware embeddings for ON-Harmony.

The input arrays are stored as float32, exactly as in the fine-tuning cohort.
The model forward uses the same exact released architecture and bfloat16 CUDA
autocast contract as frozen train/validation/test extraction. Embeddings are
cast back to float32 before CSV serialization.

Input:  A .txt file with one path per line to .npy files of shape
        (1, 121, 145, 121) from preprocessing_yAware.py.
Output: DATA/embeddings/embeddings_yAware.csv

Usage:
  python extract_yAware.py --input_list /path/to/file_list.txt \
                           --checkpoint /path/to/DenseNet121_BHB-10K_yAwareContrastive.pth \
                           --participants /path/to/participants.tsv \
                           --output DATA/embeddings/embeddings_yAware.csv
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib


# CuBLAS reads this before its first CUDA workspace is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.shared.runtime import (  # noqa: E402
    autocast_context,
    resolve_device,
    seed_torch,
)
from FineTuning.yAware.model import (  # noqa: E402
    CHECKPOINT_SHA256,
    EMBEDDING_DIM,
    MODEL_INPUT_SHAPE,
    READOUT,
    YAwareModel,
    YAwareOptions,
    make_yaware_input,
)


EXPECTED_SHAPE = MODEL_INPUT_SHAPE
EXPECTED_ACQUISITIONS = 165
EXPECTED_EMBEDDING_DIM = EMBEDDING_DIM
MANIFEST_SCHEMA_VERSION = "onharmony_embedding_manifest_v1"
# This is the semantic source contract consumed by the already locked Ridge.
# Exact implementation and numerical runtime are pinned separately below.
PIPELINE_VERSION = "yaware_onharmony_embedding_v1"
IMPLEMENTATION_VERSION = "yaware_exact_release_adapter_amp_v2"
DEFAULT_AMP = "bfloat16"
DEFAULT_SEED = 42
MODEL_IMPLEMENTATION_PATH = PROJECT_ROOT / "FineTuning" / "yAware" / "model.py"


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of one input, release, or output artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_csv(frame, output_path):
    """Publish one complete CSV without exposing a partially written artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output_path)


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


def build_model(checkpoint_path, device="cpu"):
    """Strictly load the same exact adapter used by frozen cohort extraction."""

    adapter = YAwareModel(YAwareOptions(checkpoint=Path(checkpoint_path)))
    adapter.load(device)
    frozen_summary = adapter.set_frozen()
    return adapter, frozen_summary


def load_volume(file_path):
    """Load a y-Aware preprocessed volume as tensor [1, 1, 121, 145, 121].

    The corrected y-Aware preproc saves arrays of shape (1, 121, 145, 121)
    already z-score normalized — matching y-Aware's training input_size.
    """
    if str(file_path).endswith(".npy"):
        data = np.load(file_path, allow_pickle=False)
    else:
        img = nib.load(file_path)
        data = img.get_fdata(dtype=np.float32)
        if tuple(data.shape) == EXPECTED_SHAPE[1:]:
            data = data[np.newaxis, ...]
    array = make_yaware_input(data)
    return torch.from_numpy(array).unsqueeze(0)


def extract_embedding(adapter, volume, device, amp):
    """Extract the exact 1024-D preprojection readout under locked AMP."""

    model = adapter.model
    if model is None:
        raise RuntimeError("y-Aware adapter has not loaded its checkpoint")
    with torch.no_grad():
        with autocast_context(torch, device, amp):
            embedding = model.preprojection(volume.to(device, non_blocking=True))
    if embedding.ndim == 2 and int(embedding.shape[0]) == 1:
        embedding = embedding[0]
    if tuple(embedding.shape) != (EXPECTED_EMBEDDING_DIM,):
        raise ValueError(f"y-Aware produced shape {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all():
        raise ValueError("y-Aware produced non-finite embedding values")
    return embedding.detach().float().cpu().numpy()


def effective_precision(device, amp):
    """Describe the dtype that autocast actually uses for this device."""

    return amp if device.type == "cuda" and amp != "off" else "float32"


def validate_runtime_contract(device, amp, seed):
    """Require the exact numerical runtime used by frozen cohort extraction."""

    if device.type != "cuda":
        raise RuntimeError("Canonical y-Aware ON-Harmony extraction requires CUDA")
    if amp != DEFAULT_AMP:
        raise ValueError(
            f"Canonical y-Aware ON-Harmony extraction requires amp={DEFAULT_AMP}"
        )
    if seed != DEFAULT_SEED:
        raise ValueError(
            f"Canonical y-Aware ON-Harmony extraction requires seed={DEFAULT_SEED}"
        )


def parse_bids_path(file_path):
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
    if participants_file and os.path.exists(participants_file):
        return pd.read_csv(participants_file, sep="\t")
    return None


def _autodetect_paths():
    return (
        PROJECT_ROOT / "DATA" / "filelists" / "filelist_yAware.txt",
        PROJECT_ROOT / "Models" / "y-Aware" /
        "DenseNet121_BHB-10K_yAwareContrastive.pth",
    )


def parse_args(argv=None):
    _def_input_list, _def_checkpoint = _autodetect_paths()
    parser = argparse.ArgumentParser(
        description="Canonical y-Aware ON-Harmony embedding extractor")
    parser.add_argument("--input_list", type=str, default=_def_input_list,
                        help=f"(default: {_def_input_list})")
    parser.add_argument("--checkpoint", type=str, default=_def_checkpoint,
                        help=f"(default: {_def_checkpoint})")
    parser.add_argument(
        "--participants",
        type=str,
        default=str(PROJECT_ROOT / "DATA" / "participants.tsv"),
    )
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "DATA" / "embeddings" /
                                    "embeddings_yAware.csv"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", choices=("off", "float16", "bfloat16"), default=DEFAULT_AMP
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected_count", type=int, default=EXPECTED_ACQUISITIONS)
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing CSV/manifest intentionally")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    input_list_path = Path(args.input_list).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = _manifest_path(output_path)
    existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing y-Aware artifacts; pass --force: "
            + ", ".join(existing)
        )
    with input_list_path.open("r", encoding="utf-8") as f:
        file_paths = [line.strip() for line in f if line.strip()]
    if len(file_paths) != args.expected_count or len(file_paths) != len(set(file_paths)):
        raise ValueError(
            f"Expected {args.expected_count} unique y-Aware inputs, found {len(file_paths)}"
        )
    missing_inputs = [path for path in file_paths if not Path(path).is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing y-Aware inputs: {missing_inputs[:10]}")
    device = resolve_device(torch, args.device)
    validate_runtime_contract(device, args.amp, args.seed)
    seed_torch(torch, args.seed)
    precision = effective_precision(device, args.amp)
    if device.type == "cuda" and args.amp == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support the locked bfloat16 inference mode")
    print(
        "y-Aware extractor (exact released adapter, 1x121x145x121): "
        f"{len(file_paths)} files, device={device}, amp={args.amp}, "
        f"effective_precision={precision}"
    )

    print("[Extraction 1/3] Load exact released y-Aware architecture and checkpoint")
    adapter, frozen_summary = build_model(args.checkpoint, device)
    demo_df = load_demographics(args.participants)

    print("[Extraction 2/3] Validate inputs and extract 1024-D embeddings")
    rows = []
    for i, fpath in enumerate(file_paths):
        subject_id, session_id, scanner = parse_bids_path(fpath)

        # Input remains float32; autocast affects only eligible model operations.
        volume = load_volume(fpath)
        embedding = extract_embedding(adapter, volume, device, args.amp)

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

    # Output serialization is not an extraction step.
    print(f"[Extraction 3/3] Validate and save CSV: {args.output}")
    df = pd.DataFrame(rows)
    emb_cols = [column for column in df.columns if column.startswith("emb_")]
    if len(df) != len(file_paths) or len(emb_cols) != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"Incomplete y-Aware output: rows={len(df)}/{len(file_paths)}, "
            f"dimensions={len(emb_cols)}/{EXPECTED_EMBEDDING_DIM}"
        )
    if df["file_path"].duplicated().any():
        raise RuntimeError("Duplicate file_path rows in y-Aware output")
    if not np.isfinite(df[emb_cols].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Non-finite y-Aware embeddings")
    _atomic_write_csv(df, output_path)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "y-Aware",
        "variant": "canonical",
        "readout": READOUT,
        "feature_prefix": "emb_",
        "embedding_dimensions": EXPECTED_EMBEDDING_DIM,
        "implementation_version": IMPLEMENTATION_VERSION,
        "architecture": "exact released encoder-mode architecture",
        "device": str(device),
        "amp": args.amp,
        "effective_precision": precision,
        "input_dtype": "float32",
        "embedding_serialization_dtype": "float32",
        "seed": args.seed,
        "deterministic_algorithms": True,
        "input_list": str(input_list_path),
        "input_list_sha256": sha256_file(input_list_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_load": adapter.load_description(),
        "frozen_encoder": frozen_summary,
        "model_implementation": str(MODEL_IMPLEMENTATION_PATH),
        "model_implementation_sha256": sha256_file(MODEL_IMPLEMENTATION_PATH),
        "extractor": str(SCRIPT_PATH),
        "extractor_sha256": sha256_file(SCRIPT_PATH),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "n_rows": len(df),
    }
    if args.participants:
        participants = Path(args.participants).expanduser().resolve()
        manifest.update(
            participants=str(participants),
            participants_sha256=sha256_file(participants),
        )
    _atomic_write_json(manifest, manifest_path)
    print(
        f"Saved: {output_path}  ({len(df)} rows, "
        f"{EXPECTED_EMBEDDING_DIM} embedding dims)"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
