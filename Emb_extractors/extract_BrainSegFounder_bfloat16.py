#!/usr/bin/env python3
"""Extract canonical label-blind BrainSegFounder ON-Harmony embeddings.

This corrected extractor reuses the audited model and ON-Harmony input adapter
from the brain-age pipeline. Inputs remain float32, the Swin encoder forward
uses CUDA BF16 autocast, and the 768-dimensional bottleneck-mean readout is
serialized as float32. No age, diagnosis, BAG, error, or other outcome is
loaded or written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
MODEL_DIR = PROJECT_ROOT / "FineTuning" / "BrainSegFounder"
for import_root in (PROJECT_ROOT, MODEL_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from model import (  # noqa: E402
    CHECKPOINT_SHA256,
    EMBEDDING_DIM,
    READOUT,
    BrainSegFounderOptions,
)
from predict_onharmony import (  # noqa: E402
    DEFAULT_PREPROC_ROOT,
    EXPECTED_SOURCE_HASHES,
    INPUT_CONTRACT_SHA256,
    INPUT_CONTRACT_VERSION,
    OnHarmonyBrainSegFounderModel,
    PREPROCESSOR,
    UKB_ROOT,
    artifact_paths,
    validate_artifact_set,
)
from FineTuning.shared.cohort import sha256_file  # noqa: E402
from FineTuning.shared.onharmony import (  # noqa: E402
    ArtifactRegistry,
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    effective_precision,
    load_onharmony_manifest,
    source_contract,
)
from FineTuning.shared.runtime import (  # noqa: E402
    autocast_context,
    package_versions,
    resolve_device,
    seed_torch,
)


EXPECTED_ACQUISITIONS = 165
EXPECTED_SUBJECTS = 20
EXPECTED_EMBEDDING_DIM = EMBEDDING_DIM
MANIFEST_SCHEMA_VERSION = "onharmony_embedding_manifest_v1"
# Preserve the semantic version accepted by the already locked Ridge pipeline.
PIPELINE_VERSION = "brainsegfounder_onharmony_embedding_v2"
IMPLEMENTATION_VERSION = "brainsegfounder_exact_adapter_bfloat16_v3"
DEFAULT_AMP = "bfloat16"
DEFAULT_SEED = 42
MODEL_IMPLEMENTATION_PATH = MODEL_DIR / "model.py"
INPUT_ADAPTER_PATH = MODEL_DIR / "predict_onharmony.py"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_manifest.json")


def input_audit_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_inputs.tsv")


def validate_runtime(device: Any, amp: str, seed: int) -> None:
    if device.type != "cuda":
        raise RuntimeError("Canonical BrainSegFounder extraction requires CUDA")
    if amp != DEFAULT_AMP:
        raise ValueError(f"Canonical extraction requires amp={DEFAULT_AMP}")
    if seed != DEFAULT_SEED:
        raise ValueError(f"Canonical extraction requires seed={DEFAULT_SEED}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16 autocast")


def build_registry(
    manifest: Mapping[str, Any], *, preproc_root: Path
) -> tuple[ArtifactRegistry, dict[str, Any]]:
    contract = source_contract(
        contract_version=INPUT_CONTRACT_VERSION,
        paths={
            "preprocessor": PREPROCESSOR,
            "brain_mask": UKB_ROOT / "MNI152_T1_1mm_brain_mask.nii.gz",
            "dilated_mask": UKB_ROOT / "MNI152_T1_1mm_brain_mask_dil_GD7.nii.gz",
            "fnirt_config": UKB_ROOT / "bb_fnirt.cnf",
        },
        expected_sha256=EXPECTED_SOURCE_HASHES,
        details={
            "pipeline_contract_sha256": INPUT_CONTRACT_SHA256,
            "gradient_distortion_correction": "OFF; scanner coefficients unavailable",
            "model_input_shape": [96, 96, 96],
            "raw_filelist_sha256": manifest["filelist_sha256"],
            "orchestration": "integrated run/validate commands in preprocessor",
        },
    )
    registry = ArtifactRegistry(
        contract_version=INPUT_CONTRACT_VERSION,
        resolver=lambda record: artifact_paths(record, preproc_root),
        validator=validate_artifact_set,
        contract=contract,
    )
    return registry, contract


def build_model(
    checkpoint: Path, registry: ArtifactRegistry, device: Any
) -> tuple[OnHarmonyBrainSegFounderModel, dict[str, Any]]:
    adapter = OnHarmonyBrainSegFounderModel(
        BrainSegFounderOptions(checkpoint=checkpoint), registry
    )
    adapter.load(device)
    return adapter, adapter.set_frozen()


def extract_embedding(
    adapter: OnHarmonyBrainSegFounderModel,
    record: Mapping[str, Any],
    *,
    device: Any,
    amp: str,
    seed: int,
) -> np.ndarray:
    adapter.controlled_train_mode()
    with torch.no_grad():
        with autocast_context(torch, device, amp):
            embedding = adapter.encode(record, training=False, epoch=0, seed=seed)
    if tuple(embedding.shape) != (EXPECTED_EMBEDDING_DIM,):
        raise RuntimeError(f"Unexpected embedding shape: {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all():
        raise RuntimeError("BrainSegFounder produced non-finite embedding values")
    return embedding.detach().float().cpu().numpy()


def output_metadata(record: Mapping[str, Any], path: Path) -> dict[str, str]:
    session = str(record["session"])
    if len(session) < 4 or not session[-3:].isdigit():
        raise ValueError(f"Unexpected ON-Harmony session: {session!r}")
    return {
        "file_path": str(path),
        "subject_id": str(record["subject"]),
        "session_id": f"ses-{session}",
        "scanner": session[:-3],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filelist", type=Path, default=DEFAULT_RAW_FILELIST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "Models/BrainSegFounder/64-gpu-model_bestValRMSE.pt",
    )
    parser.add_argument("--preproc-root", type=Path, default=DEFAULT_PREPROC_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=(DEFAULT_AMP,), default=DEFAULT_AMP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_ACQUISITIONS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.expected_count != EXPECTED_ACQUISITIONS:
        raise ValueError(
            f"Canonical extraction requires exactly {EXPECTED_ACQUISITIONS} acquisitions"
        )
    output = args.output.expanduser().resolve()
    audit = input_audit_path(output)
    manifest_output = manifest_path(output)
    existing = [str(path) for path in (output, audit, manifest_output) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite canonical BrainSegFounder artifacts: "
            + ", ".join(existing)
        )

    frame, onharmony_manifest = load_onharmony_manifest(
        args.filelist,
        args.metadata,
        expected_acquisitions=EXPECTED_ACQUISITIONS,
        expected_subjects=EXPECTED_SUBJECTS,
    )
    registry, input_contract = build_registry(
        onharmony_manifest, preproc_root=args.preproc_root.expanduser().resolve()
    )
    device = resolve_device(torch, args.device)
    validate_runtime(device, args.amp, args.seed)
    seed_torch(torch, args.seed)
    precision = effective_precision(device, args.amp)
    checkpoint = args.checkpoint.expanduser().resolve()
    adapter, frozen = build_model(checkpoint, registry, device)

    rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    records = frame.to_dict(orient="records")
    print(
        "BrainSegFounder canonical ON-Harmony extractor: "
        f"n={len(records)}, device={device}, amp={args.amp}",
        flush=True,
    )
    for index, record in enumerate(records, start=1):
        provenance = registry.freeze(record)
        embedding = extract_embedding(
            adapter, record, device=device, amp=args.amp, seed=args.seed
        )
        if adapter.last_input_provenance != provenance:
            raise RuntimeError(
                f"Input provenance changed for {record['acquisition_id']}"
            )
        row: dict[str, Any] = output_metadata(record, adapter.input_path(record))
        row.update(
            {f"emb_{dimension}": value for dimension, value in enumerate(embedding)}
        )
        rows.append(row)
        provenance_rows.append(
            {
                "acquisition_id": record["acquisition_id"],
                "subject_id": row["subject_id"],
                "session_id": row["session_id"],
                "scanner": row["scanner"],
                **provenance,
            }
        )
        if index % 10 == 0 or index == len(records):
            print(f"  [{index}/{len(records)}] {record['acquisition_id']}", flush=True)

    embeddings = pd.DataFrame(rows)
    embedding_columns = [
        f"emb_{dimension}" for dimension in range(EXPECTED_EMBEDDING_DIM)
    ]
    expected_columns = [
        "file_path",
        "subject_id",
        "session_id",
        "scanner",
        *embedding_columns,
    ]
    if embeddings.columns.tolist() != expected_columns:
        raise RuntimeError("Unexpected output-column order")
    if len(embeddings) != EXPECTED_ACQUISITIONS:
        raise RuntimeError(f"Incomplete output: {len(embeddings)} rows")
    if embeddings["file_path"].duplicated().any():
        raise RuntimeError("Duplicate preprocessed input paths")
    if embeddings[["subject_id", "session_id"]].duplicated().any():
        raise RuntimeError("Duplicate acquisition identifiers")
    if not np.isfinite(embeddings[embedding_columns].to_numpy(float)).all():
        raise RuntimeError("Non-finite embeddings")

    provenance_frame = pd.DataFrame(provenance_rows)
    if (
        len(provenance_frame) != EXPECTED_ACQUISITIONS
        or provenance_frame["acquisition_id"].duplicated().any()
    ):
        raise RuntimeError("Incomplete or duplicate input provenance")

    atomic_csv(embeddings, output)
    atomic_tsv(provenance_frame, audit)
    payload = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "BrainSegFounder",
        "variant": "bottleneck_mean",
        "readout": READOUT,
        "feature_prefix": "emb_",
        "embedding_dimensions": EXPECTED_EMBEDDING_DIM,
        "label_blind": True,
        "outcome_labels_loaded": False,
        "output_columns": expected_columns,
        "device": str(device),
        "amp": args.amp,
        "effective_precision": precision,
        "input_dtype": "float32",
        "embedding_serialization_dtype": "float32",
        "seed": args.seed,
        "deterministic_algorithms": True,
        "onharmony_manifest": onharmony_manifest,
        "input_contract": input_contract,
        "input_artifacts": str(audit),
        "input_artifacts_sha256": sha256_file(audit),
        "input_artifacts_n_rows": len(provenance_frame),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_load": adapter.load_description(),
        "frozen_encoder": frozen,
        "model_implementation": str(MODEL_IMPLEMENTATION_PATH),
        "model_implementation_sha256": sha256_file(MODEL_IMPLEMENTATION_PATH),
        "input_adapter": str(INPUT_ADAPTER_PATH),
        "input_adapter_sha256": sha256_file(INPUT_ADAPTER_PATH),
        "extractor": str(SCRIPT_PATH),
        "extractor_sha256": sha256_file(SCRIPT_PATH),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "n_rows": len(embeddings),
        "software": package_versions(),
    }
    atomic_json(payload, manifest_output)
    print(
        f"Saved: {output} ({len(embeddings)} rows, "
        f"{EXPECTED_EMBEDDING_DIM} dimensions)",
        flush=True,
    )
    print(f"Input audit: {audit}", flush=True)
    print(f"Manifest: {manifest_output}", flush=True)


if __name__ == "__main__":
    main()
