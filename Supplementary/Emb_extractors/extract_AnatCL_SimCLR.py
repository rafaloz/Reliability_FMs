#!/usr/bin/env python3
"""
Canonical extractor for Barbano et al.'s SimCLR training-recipe control.

The control uses the same OpenBHB pretraining data, CAT12 VBM input,
five-fold design, 3D ResNet-18 backbone, and 512-dimensional encoder readout
as AnatCL. It differs in the pretraining objective and augmentations: the
paper's SimCLR baseline uses random cropping and volume cutout, whereas
AnatCL uses its age/anatomy-guided objective without explicit augmentation.
It is therefore a matched training-recipe control, not an objective-only
causal ablation.

At inference, this script deliberately matches the canonical AnatCL
extractor: it reads the same current 121x145x121 CAT12 volumes, center-crops
the middle axis to 121x128x121, applies no additional intensity
normalization, bypasses the projection head, and strictly loads all weights.

Two checkpoint layouts are supported, in order of preference per fold:
  1. CLEAN state_dict file directly in checkpoint_dir:
        weights_clean_fold{0..4}.pth
     (produced by stripping the bundled training artefacts from the originals)
  2. ORIGINAL Carlo bundle inside a per-fold subfolder:
        <checkpoint_dir>/*fold{i}_splits5_trial0/weights.pth
     This is a pickled dict containing model state_dict + AgeEstimator +
     argparse Namespace etc. We stub the missing `models.*` modules so torch
     can unpickle it and we extract ckpt['model'] (the OrderedDict of tensors).

Canonical output:
  DATA/embeddings/embeddings_AnatCL_SimCLR_fold{0..4}.csv

Usage:
  python Supplementary/Emb_extractors/extract_AnatCL_SimCLR.py \
      --input_list DATA/filelists/filelist_AnatCL.txt \
      --checkpoint_dir /path/to/anatcl-pretrained-main-simclr/ \
      --participants /path/to/participants.tsv \
      --output_dir DATA/embeddings
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nibabel as nib


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_INPUT_LIST = os.path.join(
    PROJECT_ROOT, "DATA", "filelists", "filelist_AnatCL.txt"
)
DEFAULT_CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT, "Models", "anatcl-pretrained-main-simclr"
)
DEFAULT_PARTICIPANTS = os.path.join(PROJECT_ROOT, "DATA", "participants.tsv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "DATA", "embeddings")
CANONICAL_OUTPUT_STEM = "embeddings_AnatCL_SimCLR"
SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_ACQUISITIONS = 165
EXPECTED_EMBEDDING_DIM = 512
MANIFEST_SCHEMA_VERSION = "onharmony_embedding_manifest_v1"
PIPELINE_VERSION = "anatcl_matched_simclr_onharmony_embedding_v1"
READOUT = "global-average-pooled 3D ResNet-18 encoder output before the contrastive head"


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of an input, checkpoint, script, or output."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_csv(frame, path):
    """Publish a complete CSV without exposing a partial artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def atomic_write_json(payload, path):
    """Publish a standards-compliant JSON manifest atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def manifest_path(output_path):
    """Return the per-fold manifest path paired with one embedding CSV."""
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def _install_stub_models_module():
    """Stub Carlo's training-time `models.*` modules so torch.load can
    unpickle the original bundled checkpoints (AgeEstimator etc.)."""
    class _Stub:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s):
            self.__dict__.update(s if isinstance(s, dict) else {"_s": s})
        def __reduce_ex__(self, p): return (_Stub, ())

    class StubModule(types.ModuleType):
        __path__ = []  # treat as package
        def __getattr__(self, name):
            cls = type(name, (_Stub,), {})
            setattr(self, name, cls)
            return cls

    for n in ("models", "models.estimators", "models.resnet", "models.simclr",
              "models.networks", "models.builder", "models.encoder",
              "models.utils"):
        sys.modules.setdefault(n, StubModule(n))


def resolve_checkpoint(checkpoint_dir, fold):
    """Return the path to load for this fold. Probes, in order:
      1. {checkpoint_dir}/weights_clean_fold{i}.pth (clean state_dict, flat)
      2. {checkpoint_dir}/fold{i}/weights.pth      (renamed subfolder)
      3. {checkpoint_dir}/*fold{i}_splits5_trial0/weights.pth (original)
    """
    clean = os.path.join(checkpoint_dir, f"weights_clean_fold{fold}.pth")
    if os.path.isfile(clean):
        return clean, "clean"
    short = os.path.join(checkpoint_dir, f"fold{fold}", "weights.pth")
    if os.path.isfile(short):
        return short, "original"
    pattern = os.path.join(
        checkpoint_dir, f"*fold{fold}_splits5_trial0", "weights.pth")
    matches = sorted(glob.glob(pattern))
    if len(matches) == 1:
        return matches[0], "original"
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple original checkpoints match fold {fold}: {matches}")
    return None, None


def load_state_dict(ckpt_path, kind, device):
    """Load just the model state_dict for either checkpoint layout."""
    if kind == "clean":
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        if not isinstance(sd, dict):
            raise RuntimeError(
                f"Clean checkpoint at {ckpt_path} is not a plain state_dict")
        return sd
    if kind == "original":
        _install_stub_models_module()
        bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
        if not (isinstance(bundle, dict) and "model" in bundle):
            raise RuntimeError(
                f"Unexpected bundle layout at {ckpt_path}: "
                f"keys={list(bundle) if isinstance(bundle, dict) else type(bundle)}")
        return bundle["model"]
    raise ValueError(f"Unknown checkpoint kind: {kind!r}")


def build_model(ckpt_path, kind, device="cpu"):
    """Instantiate SupConResNet('resnet18', use_head=False) and load
    Carlo's SimCLR weights. Returns the model in eval mode on `device`.
    """
    from anatcl.models import SupConResNet

    model = SupConResNet(
        name="resnet18", head="mlp", feat_dim=128, use_head=False)

    state_dict = load_state_dict(ckpt_path, kind, device)
    print(f"  [load] {kind:<8s} src={ckpt_path}")
    info = model.load_state_dict(state_dict, strict=True)
    print(f"SimCLR weight loading (strict): {len(state_dict)} keys in checkpoint, "
          f"{len(info.missing_keys)} missing, "
          f"{len(info.unexpected_keys)} unexpected")

    model.eval()
    model.to(device)
    return model


def center_crop_y(data, target_y=128):
    """Symmetric center-crop along Y. (X, 145, Z) -> (X, 128, Z)."""
    y = data.shape[1]
    if y == target_y:
        return data
    if y < target_y:
        raise ValueError(
            f"Cannot crop y={y} to {target_y}: source smaller than target")
    start = (y - target_y) // 2
    return data[:, start:start + target_y, :]


def load_volume(file_path, target_shape=(121, 128, 121), normalize=False):
    """Load CAT12 volume and center-crop Y from 145 → 128. If `normalize`
    is True, applies per-volume z-score (mean=0, std=1) AFTER the crop,
    matching the AnatCL README. Default False keeps the legacy unnormalised
    behaviour. Returns tensor [1, 1, 121, 128, 121]."""
    if file_path.endswith(".npy"):
        data = np.load(file_path).astype(np.float32)
    else:
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    elif data.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume (or singleton trailing channel), found "
            f"shape={data.shape} for {file_path}"
        )

    if data.shape != target_shape:
        if (data.shape[0] == target_shape[0]
                and data.shape[2] == target_shape[2]
                and data.shape[1] >= target_shape[1]):
            data = center_crop_y(data, target_y=target_shape[1])
        else:
            raise ValueError(
                f"Unexpected input shape {data.shape} for {file_path}; "
                f"cannot crop to {target_shape}")

    if normalize:
        mean = float(data.mean())
        std = float(data.std())
        data = (data - mean) / (std + 1e-8)

    return torch.from_numpy(data).unsqueeze(0).unsqueeze(0)


def extract_embedding(model, volume, device):
    """512-dim ResNet-18 encoder output (head is bypassed via use_head=False)."""
    with torch.no_grad():
        feat = model(volume.to(device))
    return feat.cpu().numpy().flatten()


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


def extract_all_images(model, file_paths, demo_df, device, normalize=False):
    rows = []
    for i, fpath in enumerate(file_paths):
        subject_id, session_id, scanner = parse_bids_path(fpath)

        volume = load_volume(fpath, normalize=normalize)
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
    parser = argparse.ArgumentParser(
        description="SimCLR (Carlo) embedding extractor — "
                    "5 folds, 121x128x121 CAT12 VBM input")
    parser.add_argument("--input_list", type=str, default=DEFAULT_INPUT_LIST,
                        help=".txt file with one preprocessed NIfTI path per line")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory containing weights_clean_fold{0..4}.pth")
    parser.add_argument("--participants", type=str,
                        default=DEFAULT_PARTICIPANTS)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out_suffix", type=str, default="",
                        help="Suffix inserted before _fold{i}.csv "
                             "(e.g. '_norm_sensitivity'). Canonical default is empty.")
    parser.add_argument("--normalize", action="store_true",
                        help="If set, apply per-volume z-score (mean=0, std=1) "
                             "after the crop. This is noncanonical and requires "
                             "a nonempty --out_suffix.")
    parser.add_argument("--expected_count", type=int, default=165,
                        help="Fail unless the input list has this many rows.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing canonical fold CSVs and manifests intentionally",
    )
    args = parser.parse_args()
    if args.normalize and not args.out_suffix:
        parser.error(
            "--normalize is a noncanonical sensitivity and requires --out_suffix "
            "to protect canonical outputs"
        )

    input_list_path = Path(args.input_list).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    participants_path = (
        Path(args.participants).expanduser().resolve() if args.participants else None
    )

    with input_list_path.open("r", encoding="utf-8") as f:
        file_paths = [line.strip() for line in f if line.strip()]
    if len(file_paths) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} AnatCL inputs, found {len(file_paths)}"
        )
    if len(set(file_paths)) != len(file_paths):
        raise ValueError("Input list contains duplicate paths")
    missing_inputs = [path for path in file_paths if not os.path.isfile(path)]
    if missing_inputs:
        raise FileNotFoundError(
            f"{len(missing_inputs)} listed inputs do not exist; first: "
            f"{missing_inputs[0]}"
        )

    print(f"AnatCL-matched SimCLR extractor (121x128x121, 5 folds): "
          f"{len(file_paths)} files, device={args.device}, "
          f"normalize={args.normalize}")

    demo_df = load_demographics(str(participants_path) if participants_path else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir
        / f"{CANONICAL_OUTPUT_STEM}{args.out_suffix}_fold{fold}.csv"
        for fold in range(5)
    ]
    final_artifacts = [
        artifact
        for output_path in output_paths
        for artifact in (output_path, manifest_path(output_path))
    ]
    aggregate_manifest_path = output_dir / (
        f"{CANONICAL_OUTPUT_STEM}{args.out_suffix}_manifest.json"
    )
    final_artifacts.append(aggregate_manifest_path)
    existing = [str(path) for path in final_artifacts if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing SimCLR artifacts; pass --force: "
            + ", ".join(existing)
        )

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "AnatCL-matched SimCLR",
        "variant": "five_fold_control",
        "readout": READOUT,
        "feature_prefix": "emb_",
        "embedding_dimensions": EXPECTED_EMBEDDING_DIM,
        "control": "AnatCL-matched SimCLR training-recipe control",
        "interpretation": (
            "Matched on pretraining dataset, CAT12 input, ResNet-18 backbone, "
            "five-fold design, and encoder readout; objective and augmentations differ"
        ),
        "input_list": str(input_list_path),
        "input_list_sha256": sha256_file(input_list_path),
        "n_inputs": len(file_paths),
        "checkpoint_dir": str(checkpoint_dir),
        "strict_checkpoint_loading": True,
        "center_crop": "121x145x121 -> 121x128x121 along middle axis",
        "per_volume_zscore": bool(args.normalize),
        "outputs": [],
    }

    for fold in range(5):
        ckpt_path, kind = resolve_checkpoint(str(checkpoint_dir), fold)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found for fold {fold} under {args.checkpoint_dir}"
            )

        print(f"\n  --- Fold {fold} ({kind}) ---")
        model = build_model(ckpt_path, kind, args.device)
        rows = extract_all_images(model, file_paths, demo_df, args.device,
                                  normalize=args.normalize)

        out_path = output_paths[fold]
        df = pd.DataFrame(rows)
        if len(df) != args.expected_count:
            raise RuntimeError(
                f"Fold {fold} produced {len(df)} rows, expected {args.expected_count}"
            )
        n_dims = sum(1 for c in df.columns if c.startswith("emb_"))
        if n_dims != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                f"Fold {fold} produced {n_dims} embedding dimensions, "
                f"expected {EXPECTED_EMBEDDING_DIM}"
            )
        if df["file_path"].duplicated().any():
            raise RuntimeError(f"Duplicate file_path rows in SimCLR fold{fold}")
        if df[["subject_id", "session_id", "scanner"]].eq("").any().any():
            raise RuntimeError(f"Incomplete BIDS metadata in SimCLR fold{fold}")
        feature_columns = [column for column in df if column.startswith("emb_")]
        if not np.isfinite(df[feature_columns].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite embeddings in SimCLR fold{fold}")

        atomic_write_csv(df, out_path)
        fold_manifest = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": "AnatCL-matched SimCLR",
            "variant": f"fold{fold}",
            "readout": READOUT,
            "feature_prefix": "emb_",
            "embedding_dimensions": EXPECTED_EMBEDDING_DIM,
            "device": args.device,
            "input_list": str(input_list_path),
            "input_list_sha256": sha256_file(input_list_path),
            "checkpoint": str(Path(ckpt_path).resolve()),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "checkpoint_layout": kind,
            "extractor": str(SCRIPT_PATH),
            "extractor_sha256": sha256_file(SCRIPT_PATH),
            "output": str(out_path),
            "output_sha256": sha256_file(out_path),
            "n_rows": len(df),
            "per_volume_zscore": bool(args.normalize),
        }
        if participants_path:
            fold_manifest.update(
                participants=str(participants_path),
                participants_sha256=sha256_file(participants_path),
            )
        atomic_write_json(fold_manifest, manifest_path(out_path))

        manifest["outputs"].append({
            "fold": fold,
            "checkpoint": str(Path(ckpt_path).resolve()),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "checkpoint_layout": kind,
            "csv": str(out_path),
            "csv_sha256": sha256_file(out_path),
            "manifest": str(manifest_path(out_path)),
            "manifest_sha256": sha256_file(manifest_path(out_path)),
            "rows": len(df),
            "embedding_dimensions": n_dims,
        })
        print(f"  Saved: {out_path}  ({len(df)} rows, {n_dims} dims)")

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    manifest["extractor"] = str(SCRIPT_PATH)
    manifest["extractor_sha256"] = sha256_file(SCRIPT_PATH)
    if participants_path:
        manifest.update(
            participants=str(participants_path),
            participants_sha256=sha256_file(participants_path),
        )
    atomic_write_json(manifest, aggregate_manifest_path)
    print(f"\nManifest: {aggregate_manifest_path}")
    print(f"All done! Per-fold CSVs in: {output_dir}/")


if __name__ == "__main__":
    main()
