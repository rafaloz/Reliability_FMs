#!/usr/bin/env python3
"""Extract canonical 3D-Neuro-SimCLR embeddings for ON-Harmony.

The extractor reuses the same audited model and input adapter as the brain-age
pipeline. Inputs remain float32, the encoder forward uses CUDA BF16 autocast,
and the resulting 512-dimensional embeddings are serialized as float32.

No outcome is loaded or written. The output contains acquisition identifiers,
the consumed preprocessed-image path, and emb_0 ... emb_511.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


# CuBLAS reads this setting before its first CUDA workspace is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
MODEL_DIR = PROJECT_ROOT / "FineTuning" / "3DNeuroSimCLR"
for import_root in (PROJECT_ROOT, MODEL_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from model import (  # noqa: E402
    CHECKPOINT_SHA256,
    EMBEDDING_DIM,
    READOUT,
    NeuroSimCLROptions,
)
from predict_onharmony import (  # noqa: E402
    DEFAULT_LOG_ROOT,
    DEFAULT_PREPROC_ROOT,
    INPUT_CONTRACT_VERSION,
    OnHarmonyNeuroSimCLRModel,
    artifact_paths,
    build_input_contract,
    validate_artifact_set,
)
from FineTuning.shared.cohort import sha256_file  # noqa: E402
from FineTuning.shared.onharmony import (  # noqa: E402
    ArtifactRegistry,
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    effective_precision,
    load_onharmony_manifest,
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
# Keep the semantic source contract consumed by the already locked Ridge.
PIPELINE_VERSION = "3d_neuro_simclr_onharmony_embedding_v2"
IMPLEMENTATION_VERSION = "3d_neuro_simclr_exact_adapter_bfloat16_v3"
DEFAULT_AMP = "bfloat16"
DEFAULT_SEED = 42
MODEL_IMPLEMENTATION_PATH = MODEL_DIR / "model.py"
INPUT_ADAPTER_PATH = MODEL_DIR / "predict_onharmony.py"


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Publish a complete CSV without exposing a partially written file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    """Publish a complete tab-separated provenance table atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Publish standards-compliant JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def _input_audit_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_inputs.tsv")


def validate_runtime_contract(device: Any, amp: str, seed: int) -> None:
    """Require the numerical runtime used for frozen cohort extraction."""

    if device.type != "cuda":
        raise RuntimeError("Canonical 3D-Neuro-SimCLR extraction requires CUDA")
    if amp != DEFAULT_AMP:
        raise ValueError(
            f"Canonical 3D-Neuro-SimCLR extraction requires amp={DEFAULT_AMP}"
        )
    if seed != DEFAULT_SEED:
        raise ValueError(
            f"Canonical 3D-Neuro-SimCLR extraction requires seed={DEFAULT_SEED}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16 autocast")


def build_registry(
    frame_manifest: Mapping[str, Any],
    *,
    preproc_root: Path,
    log_root: Path,
) -> tuple[ArtifactRegistry, dict[str, Any]]:
    """Create the same ON-Harmony input contract used by adapted inference."""

    input_contract = build_input_contract(frame_manifest)
    registry = ArtifactRegistry(
        contract_version=INPUT_CONTRACT_VERSION,
        resolver=lambda record: artifact_paths(record, preproc_root, log_root),
        validator=validate_artifact_set,
        contract=input_contract,
    )
    return registry, input_contract


def build_model(
    checkpoint_path: Path,
    registry: ArtifactRegistry,
    input_contract: Mapping[str, Any],
    device: Any,
) -> tuple[OnHarmonyNeuroSimCLRModel, dict[str, Any]]:
    """Strictly load and freeze the exact common 3D-Neuro-SimCLR adapter."""

    adapter = OnHarmonyNeuroSimCLRModel(
        NeuroSimCLROptions(checkpoint=checkpoint_path),
        registry,
        input_contract,
    )
    adapter.load(device)
    frozen_summary = adapter.set_frozen()
    return adapter, frozen_summary


def extract_embedding(
    adapter: OnHarmonyNeuroSimCLRModel,
    record: Mapping[str, Any],
    *,
    device: Any,
    amp: str,
    seed: int,
) -> np.ndarray:
    """Extract one finite 512-D readout under the locked BF16 contract."""

    adapter.controlled_train_mode()
    with torch.no_grad():
        with autocast_context(torch, device, amp):
            embedding = adapter.encode(
                record,
                training=False,
                epoch=0,
                seed=seed,
            )
    if tuple(embedding.shape) != (EXPECTED_EMBEDDING_DIM,):
        raise RuntimeError(
            f"Unexpected 3D-Neuro-SimCLR embedding shape: {tuple(embedding.shape)}"
        )
    if not torch.isfinite(embedding).all():
        raise RuntimeError("3D-Neuro-SimCLR produced non-finite embedding values")
    return embedding.detach().float().cpu().numpy()


def output_metadata(record: Mapping[str, Any], input_path: Path) -> dict[str, str]:
    """Return label-free metadata in the format used by reliability analyses."""

    session = str(record["session"])
    if len(session) < 4 or not session[-3:].isdigit():
        raise ValueError(f"Unexpected ON-Harmony session identifier: {session!r}")
    return {
        "file_path": str(input_path),
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
        default=(
            PROJECT_ROOT
            / "Models"
            / "3D-Neuro-SimCLR"
            / "simclr_3d_brain_foundation.tar"
        ),
    )
    parser.add_argument("--preproc-root", type=Path, default=DEFAULT_PREPROC_ROOT)
    parser.add_argument("--preproc-log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "DATA"
            / "embeddings"
            / "embeddings_3DNeuroSimCLR.csv"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=(DEFAULT_AMP,), default=DEFAULT_AMP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_ACQUISITIONS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing CSV, input-audit table, and manifest intentionally",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.expected_count != EXPECTED_ACQUISITIONS:
        raise ValueError(
            f"Canonical extraction requires exactly {EXPECTED_ACQUISITIONS} acquisitions"
        )

    output_path = args.output.expanduser().resolve()
    manifest_path = _manifest_path(output_path)
    input_audit_path = _input_audit_path(output_path)
    existing = [
        str(path)
        for path in (output_path, input_audit_path, manifest_path)
        if path.exists()
    ]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing 3D-Neuro-SimCLR artifacts; "
            "pass --force intentionally: " + ", ".join(existing)
        )

    frame, onharmony_manifest = load_onharmony_manifest(
        args.filelist,
        args.metadata,
        expected_acquisitions=EXPECTED_ACQUISITIONS,
        expected_subjects=EXPECTED_SUBJECTS,
    )
    preproc_root = args.preproc_root.expanduser().resolve()
    log_root = args.preproc_log_root.expanduser().resolve()
    registry, input_contract = build_registry(
        onharmony_manifest,
        preproc_root=preproc_root,
        log_root=log_root,
    )

    device = resolve_device(torch, args.device)
    validate_runtime_contract(device, args.amp, args.seed)
    seed_torch(torch, args.seed)
    precision = effective_precision(device, args.amp)
    print(
        "3D-Neuro-SimCLR canonical ON-Harmony extractor: "
        f"n={len(frame)}, device={device}, amp={args.amp}, "
        f"effective_precision={precision}",
        flush=True,
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    adapter, frozen_summary = build_model(
        checkpoint,
        registry,
        input_contract,
        device,
    )

    rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for index, record in enumerate(frame.to_dict(orient="records"), start=1):
        provenance = registry.freeze(record)
        embedding = extract_embedding(
            adapter,
            record,
            device=device,
            amp=args.amp,
            seed=args.seed,
        )
        if adapter.last_input_provenance != provenance:
            raise RuntimeError(
                "Input provenance changed while extracting "
                f"{record['acquisition_id']}"
            )

        row: dict[str, Any] = output_metadata(
            record,
            adapter.input_path(record),
        )
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
        if index % 10 == 0 or index == len(frame):
            print(f"  [{index}/{len(frame)}] {record['acquisition_id']}", flush=True)

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
        raise RuntimeError("Duplicate preprocessed input paths in output")
    if embeddings[["subject_id", "session_id"]].duplicated().any():
        raise RuntimeError("Duplicate acquisition identifiers in output")
    if not np.isfinite(
        embeddings[embedding_columns].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("Non-finite embedding values in output")

    provenance_frame = pd.DataFrame(provenance_rows)
    if (
        len(provenance_frame) != EXPECTED_ACQUISITIONS
        or provenance_frame["acquisition_id"].duplicated().any()
    ):
        raise RuntimeError("Incomplete or duplicate input provenance")

    _atomic_write_csv(embeddings, output_path)
    _atomic_write_tsv(provenance_frame, input_audit_path)
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "3D-Neuro-SimCLR",
        "variant": "canonical",
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
        "input_artifacts": str(input_audit_path),
        "input_artifacts_sha256": sha256_file(input_audit_path),
        "input_artifacts_n_rows": len(provenance_frame),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_load": adapter.load_description(),
        "frozen_encoder": frozen_summary,
        "model_implementation": str(MODEL_IMPLEMENTATION_PATH),
        "model_implementation_sha256": sha256_file(MODEL_IMPLEMENTATION_PATH),
        "input_adapter": str(INPUT_ADAPTER_PATH),
        "input_adapter_sha256": sha256_file(INPUT_ADAPTER_PATH),
        "extractor": str(SCRIPT_PATH),
        "extractor_sha256": sha256_file(SCRIPT_PATH),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "n_rows": len(embeddings),
        "software": package_versions(),
    }
    _atomic_write_json(manifest, manifest_path)
    print(
        f"Saved: {output_path} ({len(embeddings)} rows, "
        f"{EXPECTED_EMBEDDING_DIM} dimensions)",
        flush=True,
    )
    print(f"Input audit: {input_audit_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
