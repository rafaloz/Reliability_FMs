#!/usr/bin/env python3
"""
Embedding extractor for BrainSegFounder (Swin-T, 768-dim).

Input:  A .txt file with one path per line to preprocessed 96x96x96 NIfTI files.
Output: A .csv in DATA/embeddings/ with columns: file_path, subject_id,
        session_id, scanner, age, sex, emb_0 ... emb_767

Notes:
  - The paper's principal Stage 1 configuration pairs T1w and T2-FLAIR, while
    the released 64-gpu-model_bestValRMSE.pt checkpoint is single-channel
    (in_channels=1). The public loader supports this single-modality branch.
  - Upstream (lab-smile/BrainSegFounder pretrain/models/ssl_head.py) uses
    swinViT(x)[4] → flatten to 27 tokens → token[0] for rotation head, token[1]
    for contrastive head. Upstream exports NO canonical "extract embedding"
    routine. Our global mean-pool over the 3×3×3 bottleneck is a design choice.
  - SSL pretraining applied ScaleIntensityRanged(a_min=-1000, a_max=1000)
    (main_T1T2.py L275-278) on UKB T1w — a CT-oriented range on MRI. At
    inference we use NormalizeIntensityd(nonzero=True, channel_wise=True) as in
    Cox et al. downstream val/test → documented deliberate mismatch.

Requirements:
  pip install torch monai nibabel numpy pandas

Usage:
  python extract_BrainSegFounder.py --input_list /path/to/file_list.txt \
                                    --checkpoint /path/to/64-gpu-model_bestValRMSE.pt \
                                    --participants /path/to/participants.tsv \
                                    --output ../DATA/embeddings_BrainSegFounder.csv
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
import nibabel as nib
from monai.networks.nets import SwinUNETR


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_ACQUISITIONS = 165
EXPECTED_SUBJECTS = 20
EXPECTED_EMBEDDING_DIM = 768
CHECKPOINT_SHA256 = "4be92492ae4f55e934e278700a13a6c18ef8406daba2177c5f9f157dcff5f341"
PREPROCESSING_PIPELINE_VERSION = "brainsegfounder-ukb-gdcoff-onharmony-v4"
PREPROCESSING_CONTRACT_SHA256 = (
    "08f983cc00900d7c57015eb8482eb763f90bd2eeb9bd960aa06563020dc7e30b"
)
CORRECTED_ROOT_MARKER = "/preprocessings/OnHarmonyDataset/preproc_BrainSegFounder/"
READY_PATTERN = re.compile(
    r"^(?P<acquisition>sub-(?P<participant>[0-9]+)_ses-"
    r"(?P<session>[A-Za-z0-9]+)_T1w)_brainsegfounder_ready\.nii\.gz$"
)
RAW_PATTERN = re.compile(
    r"^(?P<acquisition>sub-(?P<participant>[0-9]+)_ses-"
    r"(?P<session>[A-Za-z0-9]+)_T1w)\.nii\.gz$"
)
ALLOWED_NON_ENCODER_PREFIXES = ("rotation_head.", "contrastive_head.", "conv.")


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of one release or output artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(checkpoint_path, device="cpu"):
    """Build SwinUNETR encoder and load BrainSegFounder pretrained weights.

    The Stage 1 SSL checkpoint uses in_channels=1 (verified: patch_embed.proj.weight
    shape is [48, 1, 2, 2, 2]). Only swinViT encoder weights are used at inference —
    we load them INTO model.swinViT with strict=True after stripping the "module."
    (DDP) and "swinViT." prefixes. SSL heads and decoder are dropped by construction:
    they never reach the sub-module and cannot silently fail.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    observed_checkpoint_sha256 = sha256_file(checkpoint_path)
    if observed_checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError(
            "BrainSegFounder checkpoint drift: "
            f"expected {CHECKPOINT_SHA256}, observed {observed_checkpoint_sha256}"
        )

    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,  # Stage 1 checkpoint uses single channel
        out_channels=1,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )

    # Load ckpt tensors to CPU; the model stays on CPU during load_state_dict
    # and moves to `device` only after the proof-of-load block below.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Filter to swinViT-only keys and strip both prefixes so we can load INTO
    # model.swinViT directly with strict=True.
    swin_only = {}
    dropped = []
    for k, v in state_dict.items():
        nk = k.replace("module.", "", 1)
        if nk.startswith("swinViT."):
            swin_only[nk[len("swinViT."):]] = v
        else:
            dropped.append(nk)
    unexpected_dropped = [
        key for key in dropped
        if not key.startswith(ALLOWED_NON_ENCODER_PREFIXES)
    ]
    if unexpected_dropped:
        raise ValueError(
            "BrainSegFounder checkpoint contains unknown non-encoder branches: "
            f"{unexpected_dropped[:10]}"
        )
    patch_key = "patch_embed.proj.weight"
    if patch_key not in swin_only or tuple(swin_only[patch_key].shape) != (48, 1, 2, 2, 2):
        raise ValueError(
            "BrainSegFounder checkpoint is not the released single-channel encoder"
        )

    # Proof-of-load: capture a probe key BEFORE load_state_dict.
    probe_key = next(iter(swin_only))
    before = model.swinViT.state_dict()[probe_key].detach().clone()

    info = model.swinViT.load_state_dict(swin_only, strict=True)
    assert not info.missing_keys, (
        f"BSF swinViT strict-load: {len(info.missing_keys)} missing keys, "
        f"first 10: {info.missing_keys[:10]}")
    assert not info.unexpected_keys, (
        f"BSF swinViT strict-load: {len(info.unexpected_keys)} unexpected keys, "
        f"first 10: {info.unexpected_keys[:10]}")

    # Proof-of-load: after load, the parameter must be bit-exact to the ckpt tensor.
    after = model.swinViT.state_dict()[probe_key].detach().clone()
    assert torch.equal(after, swin_only[probe_key]), (
        f"BSF proof-of-load: probe key {probe_key} does NOT match ckpt tensor "
        f"after load_state_dict — weights did not transfer")
    delta = torch.linalg.norm((after.float() - before.float())).item()
    if delta <= 0:
        raise RuntimeError("BrainSegFounder strict checkpoint load did not change the probe")
    print(f"BrainSegFounder weight load (strict): {len(swin_only)} swinViT keys, "
          f"{len(dropped)} non-swinViT keys dropped (SSL heads / decoder)")
    print(f"  proof-of-load  probe={probe_key}  shape={tuple(before.shape)}  "
          f"‖after-before‖={delta:.4e}  (after == ckpt: bit-exact)")

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

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    elif data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    elif data.ndim != 3:
        raise ValueError(
            f"BrainSegFounder expects a 3D volume or one singleton channel, "
            f"got shape {data.shape}: {file_path}"
        )

    if tuple(data.shape) != (96, 96, 96):
        raise ValueError(
            f"BrainSegFounder expects a 96x96x96 model input, got "
            f"{data.shape}: {file_path}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in BrainSegFounder input: {file_path}")
    if not np.any(data != 0):
        raise ValueError(f"Empty BrainSegFounder input: {file_path}")

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
    """Require one complete age/sex row for each ON-Harmony participant."""
    path = Path(participants_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", dtype={"participant_id": str})
    required = {"participant_id", "age", "sex"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"participants.tsv lacks columns: {missing}")
    if len(frame) != EXPECTED_SUBJECTS or frame["participant_id"].duplicated().any():
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} unique participant rows, found {len(frame)}"
        )
    if frame[list(required)].isna().any().any() or (frame[list(required)] == "").any().any():
        raise ValueError("participants.tsv contains empty required values")
    return frame


def _atomic_write_csv(df, output_path):
    """Publish a complete CSV without exposing a partial final artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, output_path)


def _atomic_write_json(payload, output_path):
    """Publish JSON only after its complete, standards-compliant serialization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, output_path)


def _manifest_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def _acquisition_set(paths, pattern, label):
    """Extract and validate unique ON-Harmony acquisition identifiers."""
    acquisitions = []
    participants = []
    for path in paths:
        match = pattern.fullmatch(Path(path).name)
        if match is None:
            raise ValueError(f"Unexpected {label} filename: {Path(path).name}")
        acquisitions.append(match.group("acquisition"))
        participants.append(match.group("participant"))
    if len(acquisitions) != len(set(acquisitions)):
        raise ValueError(f"{label} contains duplicate ON-Harmony acquisitions")
    return set(acquisitions), set(participants)


def _validate_preprocessing_qc(path, acquisition):
    """Bind one model-ready volume to the exact completed preprocessing contract."""
    qc_path = Path(path).parent / "preprocessing_qc.json"
    if not qc_path.is_file():
        raise FileNotFoundError(qc_path)
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
        "pipeline_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
    }
    mismatches = {
        key: {"expected": value, "observed": qc.get(key)}
        for key, value in expected.items() if qc.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Stale BrainSegFounder preprocessing QC: {mismatches}")
    raw_name = Path(str(qc.get("raw_input", {}).get("path", ""))).name
    raw_match = RAW_PATTERN.fullmatch(raw_name)
    if raw_match is None or raw_match.group("acquisition") != acquisition:
        raise ValueError(f"QC raw input differs from ready acquisition: {qc_path}")
    expected_ready_hash = qc.get("output_sha256", {}).get("ready")
    observed_ready_hash = sha256_file(path)
    if expected_ready_hash != observed_ready_hash:
        raise ValueError(
            f"BrainSegFounder ready hash differs from QC: {path}"
        )
    return {
        "acquisition": acquisition,
        "qc_path": str(qc_path.resolve()),
        "qc_sha256": sha256_file(qc_path),
        "ready_sha256": observed_ready_hash,
        "registration_recovery_used": bool(
            qc.get("registration_recovery", {}).get("used", False)
        ),
    }


def _load_and_validate_inputs(input_list, canonical_raw_list, expected_count):
    """Require the complete corrected ON-Harmony registry and exact per-case QC."""
    input_list = Path(input_list).expanduser().resolve()
    canonical_raw_list = Path(canonical_raw_list).expanduser().resolve()
    if not input_list.is_file() or not canonical_raw_list.is_file():
        raise FileNotFoundError(
            f"Missing ON-Harmony registry: {input_list}, {canonical_raw_list}"
        )
    file_paths = [
        line.strip() for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(file_paths) != expected_count:
        raise ValueError(
            f"Expected {expected_count} BrainSegFounder inputs, found {len(file_paths)}"
        )
    if len(file_paths) != len(set(file_paths)):
        raise ValueError("BrainSegFounder input list contains duplicate paths")
    for path in file_paths:
        if not Path(path).is_absolute():
            raise ValueError(f"Input path is not absolute: {path}")
        if CORRECTED_ROOT_MARKER not in path:
            raise ValueError(f"Stale BrainSegFounder preprocessing root: {path}")
        if not Path(path).is_file() or Path(path).stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty preprocessed input: {path}")

    input_acquisitions, participants = _acquisition_set(
        file_paths, READY_PATTERN, "BrainSegFounder filelist"
    )
    raw_paths = [
        line.strip()
        for line in canonical_raw_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_acquisitions, raw_participants = _acquisition_set(
        raw_paths, RAW_PATTERN, "canonical raw filelist"
    )
    if input_acquisitions != raw_acquisitions:
        missing = sorted(raw_acquisitions - input_acquisitions)
        extra = sorted(input_acquisitions - raw_acquisitions)
        raise ValueError(
            f"BrainSegFounder cohort differs from canonical raw registry: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    if len(participants) != EXPECTED_SUBJECTS or participants != raw_participants:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} canonical subjects, found {len(participants)}"
        )

    qc_records = []
    for path in file_paths:
        acquisition = READY_PATTERN.fullmatch(Path(path).name).group("acquisition")
        qc_records.append(_validate_preprocessing_qc(path, acquisition))
    return file_paths, {
        "input_list": str(input_list),
        "input_list_sha256": sha256_file(input_list),
        "canonical_raw_list": str(canonical_raw_list),
        "canonical_raw_list_sha256": sha256_file(canonical_raw_list),
        "n_acquisitions": len(file_paths),
        "n_subjects": len(participants),
        "preprocessing_pipeline_version": PREPROCESSING_PIPELINE_VERSION,
        "preprocessing_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
        "registration_recoveries": sum(
            int(record["registration_recovery_used"]) for record in qc_records
        ),
        "inputs": qc_records,
    }


def _autodetect_paths():
    return (
        PROJECT_ROOT / "DATA" / "filelists" / "filelist_BrainSegFounder.txt",
        PROJECT_ROOT / "Models" / "BrainSegFounder" / "64-gpu-model_bestValRMSE.pt",
    )


def main():
    _def_input_list, _def_checkpoint = _autodetect_paths()
    parser = argparse.ArgumentParser(description="BrainSegFounder embedding extractor")
    parser.add_argument("--input_list", type=str, default=_def_input_list,
                        help=f".txt file with one preprocessed NIfTI path per line "
                             f"(default: {_def_input_list})")
    parser.add_argument("--checkpoint", type=str, default=_def_checkpoint,
                        help=f"Path to 64-gpu-model_bestValRMSE.pt "
                             f"(default: {_def_checkpoint})")
    parser.add_argument("--participants", type=str,
                        default=str(PROJECT_ROOT / "DATA" / "participants.tsv"),
                        help="Path to participants.tsv")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "DATA" / "embeddings" /
                                    "embeddings_BrainSegFounder.csv"))
    parser.add_argument(
        "--canonical_raw_list", type=str,
        default=str(PROJECT_ROOT / "DATA" / "filelists" / "filelist_BrainIAC_raw.txt"),
        help="Canonical 165-acquisition ON-Harmony raw registry used for identity checks",
    )
    parser.add_argument("--expected_count", type=int, default=EXPECTED_ACQUISITIONS)
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing CSV/manifest intentionally")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.expected_count <= 0:
        parser.error("--expected_count must be positive")
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = _manifest_path(output_path)
    existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing final artifacts; pass --force intentionally: "
            + ", ".join(existing)
        )
    file_paths, input_registry = _load_and_validate_inputs(
        args.input_list, args.canonical_raw_list, args.expected_count
    )
    print(f"BrainSegFounder extractor: {len(file_paths)} files, device={args.device}")

    # Extraction 1/4: build SwinUNETR + strict-load the SSL swinViT weights.
    print(f"[Extraction 1/4] Build MONAI SwinUNETR + strict-load Cox 2024 SSL swinViT ckpt (bit-exact proof)")
    model = build_model(args.checkpoint, device=args.device)
    demo_df = load_demographics(args.participants)

    rows = []
    for i, fpath in enumerate(file_paths):
        subject_id, session_id, scanner = parse_bids_path(fpath)

        # Extraction 2/4: load the preprocessed volume → [1,1,X,Y,Z].
        volume = load_volume(fpath)
        # Extraction 3/4: forward through swinViT → five spatial feature maps.
        # Extraction 4/4: hidden_states[4] [B,768,3,3,3] → spatial mean
        #         spatial → 768-d. Our design choice — upstream ssl_head.py uses tokens[0]/[1]
        #         for SSL heads and exports no canonical pooled embedding.
        embedding = extract_embedding(model, volume, args.device)
        if embedding.shape != (EXPECTED_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
            raise RuntimeError(
                f"Invalid BrainSegFounder embedding {embedding.shape}: {fpath}"
            )

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

    # Output validation/serialization is not an extraction step.
    print(f"[Output] Validate and save CSV → {output_path}")
    df = pd.DataFrame(rows)
    emb_cols = [column for column in df.columns if column.startswith("emb_")]
    if len(df) != len(file_paths) or len(emb_cols) != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"Incomplete BrainSegFounder output: rows={len(df)}/{len(file_paths)}, "
            f"dimensions={len(emb_cols)}/{EXPECTED_EMBEDDING_DIM}"
        )
    if df["file_path"].duplicated().any():
        raise RuntimeError("Refusing to save duplicate BrainSegFounder rows")
    if not np.isfinite(df[emb_cols].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Refusing to save non-finite BrainSegFounder embeddings")
    metadata_columns = ["subject_id", "session_id", "scanner", "age", "sex"]
    invalid_meta = int(
        (df[metadata_columns].isna() | (df[metadata_columns] == ""))
        .any(axis=1)
        .sum()
    )
    if invalid_meta:
        raise RuntimeError(
            f"Refusing to save {invalid_meta} rows with incomplete metadata"
        )
    ages = pd.to_numeric(df["age"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(ages).all() or not set(df["sex"]).issubset({"F", "M"}):
        raise RuntimeError("Refusing to save invalid age/sex metadata")
    _atomic_write_csv(df, output_path)
    manifest = {
        "manifest_schema_version": "onharmony_embedding_manifest_v1",
        "pipeline_version": "brainsegfounder_onharmony_embedding_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "BrainSegFounder",
        "variant": "bottleneck_mean",
        "readout": "spatial mean of the final 3x3x3 Swin bottleneck",
        "feature_prefix": "emb_",
        "embedding_dimensions": EXPECTED_EMBEDDING_DIM,
        "device": args.device,
        "registry": input_registry,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "extractor": str(SCRIPT_PATH),
        "extractor_sha256": sha256_file(SCRIPT_PATH),
        "participants": str(Path(args.participants).expanduser().resolve()),
        "participants_sha256": sha256_file(args.participants),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "n_rows": len(df),
    }
    _atomic_write_json(manifest, manifest_path)
    print(f"Saved: {output_path}  ({len(df)} rows, {len(embedding)} embedding dims)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
