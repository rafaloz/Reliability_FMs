#!/usr/bin/env python3
"""
Embedding extractor for AnatCL (ResNet-18, 512-dim) — canonical.

One configuration only: the NEW anatcl-g3 release checkpoints
(fold{i}_weights.pth), no per-volume z-score, 121x128x121 input via
center-crop of the middle axis from CAT12's 121x145x121 output.

Historical checkpoint and normalization ablations are not part of this
canonical extractor. The pre-cleanup archive retains them if they are ever
needed for forensic reproduction.

Input:  A .txt file with one path per line to 121x145x121 NIfTI files
        from preprocessing_AnatCL.py (brainprep IXI Dartel).
Output: DATA/embeddings/embeddings_AnatCL_fold{0..4}.csv
        (5 folds x 165 rows x 512 dims)

Usage:
  python extract_AnatCL.py --input_list /path/to/file_list.txt \
                           --checkpoint_dir /path/to/NEW_anatcl-g3/ \
                           --participants /path/to/participants.tsv \
                           --output_dir DATA/embeddings/ \
                           --device cuda --amp bfloat16
"""

import argparse
import hashlib
import importlib
import json
import os
import pickle as _pickle
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import nibabel as nib


CKPT_SUFFIX = "_weights"  # NEW anatcl-g3 uses fold{i}_weights.pth (OLD used fold{i}.pth)
OUT_STEM = "embeddings_AnatCL"
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
EXPECTED_ACQUISITIONS = 165
EXPECTED_EMBEDDING_DIM = 512
MANIFEST_SCHEMA_VERSION = "onharmony_embedding_manifest_v1"
PIPELINE_VERSION = "anatcl_onharmony_embedding_v2"
READOUT = "global-average-pooled 3D ResNet-18 encoder output before the contrastive head"
DEFAULT_PREPROCESSING_PROVENANCE = (
    PROJECT_ROOT
    / "ExcelsChecks"
    / "paper_actual"
    / "audits"
    / "AnatCL"
    / "anatcl_preprocessing_provenance.json"
)
EXPECTED_PREPROCESSING_AUDITOR_SHA256 = (
    "c848bf2439c76c5cf0cf98446d4df1edaa0236715d59ac4586982224c5ad1284"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.AnatCL.model import (  # noqa: E402
    CHECKPOINT_SHA256,
    OFFICIAL_SOURCE_SHA256,
)
from FineTuning.shared.runtime import seed_torch  # noqa: E402


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of one input, release, or output artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_csv(frame, path):
    """Publish one complete CSV without exposing a partially written artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def _atomic_write_json(payload, path):
    """Publish one standards-compliant JSON manifest atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def _manifest_path(output_path):
    """Return the conventional manifest path paired with one embedding CSV."""
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def _acquisition_id(value):
    match = re.search(r"(sub-[A-Za-z0-9]+_ses-[A-Za-z0-9]+_T1w)", str(value))
    if match is None:
        raise ValueError(f"Cannot parse AnatCL acquisition from {value!r}")
    return match.group(1)


def validate_preprocessing_provenance(manifest_path, file_paths):
    """Bind every consumed NIfTI to the retained historical CAT12 evidence."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha_before = sha256_file(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha_after = sha256_file(manifest_path)
    if manifest_sha_after != manifest_sha_before:
        raise RuntimeError("AnatCL preprocessing provenance changed while being read")
    if not isinstance(payload, dict):
        raise TypeError("AnatCL preprocessing provenance is not a JSON object")
    expected_header = {
        "schema_version": "anatcl_onharmony_retrospective_provenance_v1",
        "status": "PASS",
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in expected_header.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Invalid AnatCL preprocessing provenance: {mismatches}")
    auditor = payload.get("auditor", {})
    if auditor.get("sha256") != EXPECTED_PREPROCESSING_AUDITOR_SHA256:
        raise ValueError("AnatCL preprocessing provenance uses an unexpected auditor")
    auditor_path = Path(str(auditor.get("path", "")))
    if not auditor_path.is_absolute():
        auditor_path = PROJECT_ROOT / auditor_path
    auditor_path = auditor_path.expanduser().resolve()
    if (
        not auditor_path.is_file()
        or sha256_file(auditor_path) != EXPECTED_PREPROCESSING_AUDITOR_SHA256
    ):
        raise ValueError("AnatCL preprocessing auditor source differs from its contract")
    if payload.get("template_reference", {}).get("identity_matches_expected") is not True:
        raise ValueError("AnatCL preprocessing provenance lacks the expected IXI555 template")
    evidence = payload.get("evidence", {})
    case_table_value = evidence.get("case_table")
    if not isinstance(case_table_value, str) or not case_table_value:
        raise ValueError("AnatCL preprocessing provenance lacks its case table")
    case_table = Path(case_table_value)
    if not case_table.is_absolute():
        case_table = manifest_path.parent / case_table
    case_table = case_table.expanduser().resolve()
    if not case_table.is_file():
        raise FileNotFoundError(case_table)
    case_table_sha = sha256_file(case_table)
    if evidence.get("case_table_sha256") != case_table_sha:
        raise ValueError("AnatCL preprocessing case table differs from its manifest")
    cases = pd.read_csv(case_table, sep="\t", dtype=str, keep_default_na=False)
    required = {
        "acquisition_id",
        "output_path",
        "output_sha256",
        "status",
    }
    if not required.issubset(cases.columns):
        raise ValueError(
            "AnatCL preprocessing case table lacks required columns: "
            f"{sorted(required - set(cases.columns))}"
        )
    if len(cases) != len(file_paths) or cases["acquisition_id"].duplicated().any():
        raise ValueError("AnatCL preprocessing case table has an invalid cohort size")
    if set(cases["status"]) != {"PASS"}:
        raise ValueError("AnatCL preprocessing case table contains failed acquisitions")
    expected_ids = [_acquisition_id(path) for path in file_paths]
    if cases["acquisition_id"].tolist() != expected_ids:
        raise ValueError("AnatCL preprocessing provenance order differs from the filelist")
    for file_path, case in zip(file_paths, cases.to_dict(orient="records")):
        path = Path(file_path).expanduser().resolve()
        if Path(case["output_path"]).name != path.name:
            raise ValueError(
                f"AnatCL provenance output name differs for {case['acquisition_id']}"
            )
        observed = sha256_file(path)
        if case["output_sha256"] != observed:
            raise ValueError(
                f"AnatCL input differs from historical provenance: {path}"
            )
    return {
        "status": "VALIDATED",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha_after,
        "case_table": str(case_table),
        "case_table_sha256": case_table_sha,
        "n_acquisitions": len(cases),
        "historical_generator_source_available": False,
    }


# --- Custom unpickler for AnatCL checkpoints -------------------------------- #
class _AnatCLUnpickler(_pickle.Unpickler):
    def find_class(self, module, name):
        if module == "models" or module.startswith("models."):
            redirected = "anatcl." + module
            try:
                return super().find_class(redirected, name)
            except (ModuleNotFoundError, AttributeError):
                pass
            stub = type(name, (), {
                "__init__": lambda self, *a, **kw: None,
                "__setstate__": lambda self, state: (
                    self.__dict__.update(state) if isinstance(state, dict) else None
                ),
            })
            return stub
        return super().find_class(module, name)


_custom_pickle = types.ModuleType("_custom_pickle")
for _attr in dir(_pickle):
    setattr(_custom_pickle, _attr, getattr(_pickle, _attr))
_custom_pickle.Unpickler = _AnatCLUnpickler


def _validate_installed_source():
    """Require the two installed AnatCL architecture files used at runtime."""

    modules = {
        "anatcl/anatcl.py": importlib.import_module("anatcl.anatcl"),
        "anatcl/models/resnet3d.py": importlib.import_module("anatcl.models.resnet3d"),
    }
    observed = {}
    for relative, module in modules.items():
        source = Path(str(module.__file__)).resolve()
        digest = sha256_file(source)
        if digest != OFFICIAL_SOURCE_SHA256[relative]:
            raise ValueError(
                f"Installed AnatCL source drift for {relative}: "
                f"expected {OFFICIAL_SOURCE_SHA256[relative]}, observed {digest}"
            )
        observed[relative] = digest
    return observed


def build_model(checkpoint_path, descriptor="global", fold=0, device="cpu"):
    """Build AnatCL and load NEW anatcl-g3 weights into the backbone.

    Load hygiene: weights must go into `model.backbone` (the SupConResNet
    inside the AnatCL wrapper), not into the wrapper itself. Loading into the
    wrapper with strict=False silently ignores every 'encoder.*'/'head.*' key
    and leaves the network random-init. We use strict=True on the backbone as an
    explicit guard.
    """
    from anatcl import AnatCL

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    observed_checkpoint = sha256_file(checkpoint_path)
    expected_checkpoint = CHECKPOINT_SHA256[fold]
    if observed_checkpoint != expected_checkpoint:
        raise ValueError(
            f"AnatCL fold{fold} checkpoint drift: expected {expected_checkpoint}, "
            f"observed {observed_checkpoint}"
        )
    installed_source = _validate_installed_source()
    model = AnatCL(
        model="resnet18",
        descriptor=descriptor,
        fold=fold,
        use_head=False,
        pretrained=False,
    )

    ckpt = torch.load(checkpoint_path, map_location=device,
                      pickle_module=_custom_pickle, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif hasattr(ckpt, "state_dict") and callable(ckpt.state_dict):
        state_dict = ckpt.state_dict()
    else:
        raise RuntimeError(
            f"Unexpected checkpoint layout at {checkpoint_path}: "
            f"type={type(ckpt).__name__} "
            f"keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'n/a'}")

    probe_key = next(iter(state_dict))
    if probe_key not in model.backbone.state_dict():
        raise ValueError(f"Checkpoint probe key is absent from AnatCL backbone: {probe_key}")
    before = model.backbone.state_dict()[probe_key].detach().clone()
    info = model.backbone.load_state_dict(state_dict, strict=True)
    after = model.backbone.state_dict()[probe_key].detach()
    changed = float((after.cpu() - before.cpu()).norm().item())
    bit_exact = bool(torch.equal(after.cpu(), state_dict[probe_key].detach().cpu()))
    if info.missing_keys or info.unexpected_keys or changed <= 0.0 or not bit_exact:
        raise RuntimeError(
            "Strict AnatCL checkpoint proof-of-load failed: "
            f"missing={info.missing_keys}, unexpected={info.unexpected_keys}, "
            f"changed_l2={changed}, bit_exact={bit_exact}"
        )
    print(f"AnatCL weight loading (strict): {len(state_dict)} keys, "
          f"missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")

    model.eval()
    model.to(device)
    return model, {
        "strict": True,
        "backbone_keys": len(state_dict),
        "probe_key": probe_key,
        "probe_changed_l2": changed,
        "probe_bit_exact": bit_exact,
        "projection_head_used": False,
        "checkpoint_sha256": observed_checkpoint,
        "installed_source_sha256": installed_source,
    }


def center_crop_y(data, target_y=128):
    """Symmetric center-crop along y from 145 to 128 (removes 8 anterior + 9 posterior)."""
    y = data.shape[1]
    if y == target_y:
        return data
    if y < target_y:
        raise ValueError(f"Cannot crop y={y} to {target_y}: source smaller than target")
    start = (y - target_y) // 2
    return data[:, start:start + target_y, :]


def load_volume(file_path, target_shape=(121, 128, 121)):
    """Load a CAT12-preprocessed volume and crop to AnatCL's expected shape."""
    if file_path.endswith(".npy"):
        data = np.load(file_path).astype(np.float32)
    else:
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)

    if data.ndim == 4:
        data = data[0]

    if data.shape != target_shape:
        if data.shape[0] == target_shape[0] and data.shape[2] == target_shape[2] \
                and data.shape[1] >= target_shape[1]:
            data = center_crop_y(data, target_y=target_shape[1])
        else:
            raise ValueError(
                f"Unexpected input shape {data.shape} for {file_path}; "
                f"cannot crop to {target_shape}"
            )

    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return tensor


def extract_embedding(model, volume, device, amp):
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and amp == "bfloat16",
        ):
            feat = model(volume.to(device))
    return feat.float().cpu().numpy().flatten()


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


def extract_all_images(model, file_paths, demo_df, device, amp):
    rows = []
    for i, fpath in enumerate(file_paths):
        subject_id, session_id, scanner = parse_bids_path(fpath)

        volume = load_volume(fpath)
        embedding = extract_embedding(model, volume, device, amp)

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


def _autodetect_paths():
    return (
        PROJECT_ROOT / "DATA" / "filelists" / "filelist_AnatCL.txt",
        PROJECT_ROOT / "Models" / "AnatCL" / "NEW_anatcl-g3",
    )


def _validate_descriptor_checkpoint_dir(checkpoint_dir, descriptor):
    """Reject checkpoint paths that explicitly name the other descriptor."""
    checkpoint_dir_text = os.fspath(checkpoint_dir).lower()
    other = "local" if descriptor == "global" else "global"
    if other in checkpoint_dir_text and descriptor not in checkpoint_dir_text:
        raise AssertionError(
            f"--descriptor={descriptor} but checkpoint_dir '{checkpoint_dir}' "
            f"appears to be a '{other}' release (name contains '{other}' and not "
            f"'{descriptor}'). Refusing to load a mismatched descriptor."
        )


def main():
    _def_input_list, _def_ckpt_dir = _autodetect_paths()
    parser = argparse.ArgumentParser(
        description="AnatCL embedding extractor — canonical (NEW anatcl-g3, 121x128x121)")
    parser.add_argument("--input_list", type=str, default=_def_input_list,
                        help=f".txt file with one preprocessed NIfTI path per line "
                             f"(default: {_def_input_list})")
    parser.add_argument("--checkpoint_dir", type=str, default=_def_ckpt_dir,
                        help=f"Directory containing fold{{0..4}}_weights.pth from NEW_anatcl-g3 "
                             f"(default: {_def_ckpt_dir})")
    parser.add_argument("--descriptor", type=str, default="global",
                        choices=["global", "local"])
    parser.add_argument(
        "--participants",
        type=str,
        default=str(PROJECT_ROOT / "DATA" / "participants.tsv"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "DATA" / "embeddings"),
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default=OUT_STEM,
        help=(
            "CSV stem before _fold{i}.csv. Use a distinct stem for a "
            "supplementary re-extraction so canonical embeddings are not overwritten."
        ),
    )
    parser.add_argument(
        "--preprocessing_provenance",
        type=str,
        default=str(DEFAULT_PREPROCESSING_PROVENANCE),
        help="Retrospective manifest binding the 165 NIfTI inputs to CAT12 evidence",
    )
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--amp",
        choices=("bfloat16",),
        required=True,
        help="Fixed numerical precision contract matching adapted AnatCL",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_count", type=int, default=EXPECTED_ACQUISITIONS)
    parser.add_argument("--force", action="store_true",
                        help="Replace existing fold CSVs/manifests intentionally")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.output_stem):
        parser.error("--output_stem may contain only letters, digits, dot, underscore, and hyphen")

    input_list_path = Path(args.input_list).expanduser().resolve()
    with input_list_path.open("r", encoding="utf-8") as f:
        file_paths = [line.strip() for line in f if line.strip()]
    if len(file_paths) != args.expected_count or len(file_paths) != len(set(file_paths)):
        raise ValueError(
            f"Expected {args.expected_count} unique AnatCL inputs, found {len(file_paths)}"
        )
    missing_inputs = [path for path in file_paths if not Path(path).is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing AnatCL inputs: {missing_inputs[:10]}")
    preprocessing_provenance = validate_preprocessing_provenance(
        args.preprocessing_provenance, file_paths
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type != "cuda":
        raise ValueError("AnatCL bfloat16 extraction requires a CUDA device")
    seed_torch(torch, args.seed)
    print(f"AnatCL extractor (canonical, {args.descriptor}, 121x128x121, 5 folds): "
          f"{len(file_paths)} files, device={device}, amp={args.amp}")

    demo_df = load_demographics(args.participants)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_artifacts = []
    for fold in range(5):
        output_path = output_dir / f"{args.output_stem}_fold{fold}.csv"
        final_artifacts.extend((output_path, _manifest_path(output_path)))
    existing = [str(path) for path in final_artifacts if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing AnatCL artifacts; pass --force: "
            + ", ".join(existing)
        )

    # audit M-AnatCL-211: cosmetic guard — if the checkpoint path encodes a
    # descriptor (e.g. ".../global/..." or ".../local_..."), ensure it matches
    # --descriptor so a global ckpt dir isn't run with descriptor=local by mistake.
    _validate_descriptor_checkpoint_dir(args.checkpoint_dir, args.descriptor)

    for fold in range(5):
        ckpt_path = Path(args.checkpoint_dir).expanduser().resolve() / f"fold{fold}{CKPT_SUFFIX}.pth"
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint {ckpt_path}. This is the canonical extractor; "
                f"it expects the NEW anatcl-g3 release (fold{{i}}_weights.pth).")

        print(f"\n  --- Fold {fold} ---")
        print(f"  [Extraction 1/3] Build AnatCL wrapper + load fold{fold}_weights.pth into backbone (strict=True)")
        model, checkpoint_load = build_model(
            ckpt_path, args.descriptor, fold, device
        )
        # [Extraction 2/3] Load each volume + center-crop Y (145→128).
        # [Extraction 3/3] Forward pass and obtain the 512-d representation.
        print(f"  [Extraction 2-/3] Load each volume + center-crop Y (145→128) and Forward pass.")
        rows = extract_all_images(model, file_paths, demo_df, device, args.amp)

        print(f"  [Output] Save fold-{fold} CSV")
        out_path = output_dir / f"{args.output_stem}_fold{fold}.csv"
        df = pd.DataFrame(rows)
        emb_cols = [column for column in df.columns if column.startswith("emb_")]
        n_dims = len(emb_cols)
        if len(df) != len(file_paths) or n_dims != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                f"Incomplete AnatCL fold{fold} output: rows={len(df)}/{len(file_paths)}, "
                f"dimensions={n_dims}/{EXPECTED_EMBEDDING_DIM}"
            )
        if df["file_path"].duplicated().any():
            raise RuntimeError(f"Duplicate file_path rows in AnatCL fold{fold}")
        if not np.isfinite(df[emb_cols].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite embeddings in AnatCL fold{fold}")
        _atomic_write_csv(df, out_path)
        manifest = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": "AnatCL",
            "variant": f"fold{fold}",
            "readout": READOUT,
            "feature_prefix": "emb_",
            "embedding_dimensions": EXPECTED_EMBEDDING_DIM,
            "descriptor": args.descriptor,
            "device": str(device),
            "amp": args.amp,
            "seed": args.seed,
            "input_list": str(input_list_path),
            "input_list_sha256": sha256_file(input_list_path),
            "preprocessing_provenance": preprocessing_provenance,
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "checkpoint_load": checkpoint_load,
            "extractor": str(SCRIPT_PATH),
            "extractor_sha256": sha256_file(SCRIPT_PATH),
            "output": str(out_path),
            "output_sha256": sha256_file(out_path),
            "n_rows": len(df),
        }
        if args.participants:
            participants_path = Path(args.participants).expanduser().resolve()
            manifest.update(
                participants=str(participants_path),
                participants_sha256=sha256_file(participants_path),
            )
        manifest_path = _manifest_path(out_path)
        _atomic_write_json(manifest, manifest_path)
        print(f"  Saved: {out_path}  ({len(df)} rows, {n_dims} dims)")
        print(f"  Manifest: {manifest_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nAll done! Per-fold CSVs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
