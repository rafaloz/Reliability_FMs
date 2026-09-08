#!/usr/bin/env python3
"""Extract canonical and sensitivity BrainIAC embeddings for ON-Harmony.

This reuses the audited BrainIAC model and ON-Harmony input adapter. Inputs are
float32, the ViT forward uses CUDA BF16 autocast, and the four 768-dimensional
readouts are serialized as float32. No outcome is loaded or written.
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


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
MODEL_DIR = PROJECT_ROOT / "FineTuning" / "BrainIAC"
for import_root in (PROJECT_ROOT, MODEL_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from model import (  # noqa: E402
    CHECKPOINT_SHA256,
    EMBEDDING_DIM,
    N_PATCH_TOKENS,
    READOUT,
    BrainIACOptions,
)
from predict_onharmony import (  # noqa: E402
    DEFAULT_LOG_ROOT,
    DEFAULT_PREPROC_ROOT,
    EXPECTED_SOURCE_HASHES,
    HISTORICAL_PREPROCESSOR_SHA256,
    INPUT_CONTRACT_VERSION,
    OnHarmonyBrainIACModel,
    PREPROCESSOR,
    TEMPLATE,
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


READOUTS = ("first_patch", "mean_pool", "last_patch", "max_pool")
READOUT_DESCRIPTIONS = {
    "first_patch": READOUT,
    "mean_pool": "mean over all released 216 ViT patch tokens",
    "last_patch": "last patch token from the released 216-token ViT output",
    "max_pool": "element-wise maximum over all released 216 ViT patch tokens",
}
EXPECTED_ACQUISITIONS = 165
EXPECTED_SUBJECTS = 20
EXPECTED_EMBEDDING_DIM = EMBEDDING_DIM
MANIFEST_SCHEMA_VERSION = "onharmony_embedding_manifest_v1"
# Keep the semantic source contract consumed by the already locked Ridge.
PIPELINE_VERSION = "brainiac_onharmony_embedding_v1"
IMPLEMENTATION_VERSION = "brainiac_exact_adapter_bfloat16_v2"
DEFAULT_AMP = "bfloat16"
DEFAULT_SEED = 42
MODEL_IMPLEMENTATION_PATH = MODEL_DIR / "model.py"
INPUT_ADAPTER_PATH = MODEL_DIR / "predict_onharmony.py"


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def _input_audit_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.name}_inputs.tsv")


def validate_runtime_contract(device: Any, amp: str, seed: int) -> None:
    """Require the numerical runtime used for frozen brain-age embeddings."""

    if device.type != "cuda":
        raise RuntimeError("Canonical BrainIAC extraction requires CUDA")
    if amp != DEFAULT_AMP:
        raise ValueError(f"Canonical BrainIAC extraction requires amp={DEFAULT_AMP}")
    if seed != DEFAULT_SEED:
        raise ValueError(f"Canonical BrainIAC extraction requires seed={DEFAULT_SEED}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16 autocast")


def build_registry(
    onharmony_manifest: Mapping[str, Any],
    *,
    preproc_root: Path,
    log_root: Path,
) -> tuple[ArtifactRegistry, dict[str, Any]]:
    """Build the same external-input contract used by adapted inference."""

    input_contract = source_contract(
        contract_version=INPUT_CONTRACT_VERSION,
        paths={"preprocessor": PREPROCESSOR, "template": TEMPLATE},
        expected_sha256=EXPECTED_SOURCE_HASHES,
        details={
            "model_input_shape": [96, 96, 96],
            "normalization": "nonzero z-score",
            "raw_filelist_sha256": onharmony_manifest["filelist_sha256"],
            "accepted_historical_generator_sha256": HISTORICAL_PREPROCESSOR_SHA256,
        },
    )
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
    device: Any,
) -> tuple[OnHarmonyBrainIACModel, dict[str, Any]]:
    """Strictly load and freeze the exact common BrainIAC adapter."""

    adapter = OnHarmonyBrainIACModel(
        BrainIACOptions(checkpoint=checkpoint_path),
        registry,
    )
    adapter.load(device)
    return adapter, adapter.set_frozen()


def apply_readout(tokens: Any, readout: str) -> Any:
    """Reduce one complete ViT token sequence to a 768-D representation."""

    if tuple(tokens.shape) != (1, N_PATCH_TOKENS, EXPECTED_EMBEDDING_DIM):
        raise RuntimeError(f"Unexpected BrainIAC token output: {tuple(tokens.shape)}")
    if not torch.isfinite(tokens).all():
        raise RuntimeError("BrainIAC produced non-finite token values")
    if readout == "first_patch":
        return tokens[:, 0, :]
    if readout == "mean_pool":
        return tokens.mean(dim=1)
    if readout == "last_patch":
        return tokens[:, -1, :]
    if readout == "max_pool":
        return tokens.max(dim=1).values
    raise ValueError(f"Unknown BrainIAC readout: {readout}")


def extract_readouts(
    adapter: OnHarmonyBrainIACModel,
    record: Mapping[str, Any],
    *,
    readouts: tuple[str, ...],
    device: Any,
    amp: str,
) -> dict[str, np.ndarray]:
    """Compute all requested readouts from one BF16 encoder forward."""

    adapter.controlled_train_mode()
    volume = adapter.load_input(record).to(device, non_blocking=True)
    model = adapter._require_loaded()
    with torch.no_grad():
        with autocast_context(torch, device, amp):
            output = model(volume)
            tokens = output[0] if isinstance(output, (tuple, list)) else output
            reduced = {name: apply_readout(tokens, name) for name in readouts}

    result: dict[str, np.ndarray] = {}
    for name, embedding in reduced.items():
        if tuple(embedding.shape) != (1, EXPECTED_EMBEDDING_DIM):
            raise RuntimeError(
                f"Unexpected BrainIAC {name} embedding shape: {tuple(embedding.shape)}"
            )
        array = embedding[0].detach().float().cpu().numpy()
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise RuntimeError(f"Invalid serialized BrainIAC {name} embedding")
        result[name] = array
    return result


def output_metadata(record: Mapping[str, Any], input_path: Path) -> dict[str, str]:
    """Return label-free metadata in the reliability-analysis format."""

    session = str(record["session"])
    if len(session) < 4 or not session[-3:].isdigit():
        raise ValueError(f"Unexpected ON-Harmony session identifier: {session!r}")
    return {
        "file_path": str(input_path),
        "subject_id": str(record["subject"]),
        "session_id": f"ses-{session}",
        "scanner": session[:-3],
    }


def _output_paths(output_prefix: Path, readouts: tuple[str, ...]) -> dict[str, Path]:
    paths = {
        name: output_prefix.with_name(f"{output_prefix.name}_{name}.csv")
        for name in readouts
    }
    paths["canonical"] = output_prefix.with_suffix(".csv")
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filelist", type=Path, default=DEFAULT_RAW_FILELIST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "Models" / "BrainIAC" / "BrainIAC.ckpt",
    )
    parser.add_argument("--preproc-root", type=Path, default=DEFAULT_PREPROC_ROOT)
    parser.add_argument("--preproc-log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "DATA" / "embeddings" / "embeddings_BrainIAC",
    )
    parser.add_argument("--readouts", default=",".join(READOUTS))
    parser.add_argument(
        "--canonical-readout", choices=READOUTS, default="first_patch"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=(DEFAULT_AMP,), default=DEFAULT_AMP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_ACQUISITIONS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace all existing BrainIAC embedding artifacts intentionally",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.expected_count != EXPECTED_ACQUISITIONS:
        raise ValueError(
            f"Canonical extraction requires exactly {EXPECTED_ACQUISITIONS} acquisitions"
        )
    readouts = tuple(name.strip() for name in args.readouts.split(",") if name.strip())
    if not readouts or len(readouts) != len(set(readouts)):
        raise ValueError("BrainIAC readouts must be a non-empty unique list")
    unknown = sorted(set(readouts) - set(READOUTS))
    if unknown:
        raise ValueError(f"Unknown BrainIAC readouts: {unknown}")
    if args.canonical_readout not in readouts:
        raise ValueError("--canonical-readout must also be present in --readouts")

    output_prefix = args.output_prefix.expanduser().resolve()
    output_paths = _output_paths(output_prefix, readouts)
    input_audit_path = _input_audit_path(output_prefix)
    cls_probe_path = output_prefix.with_name(f"{output_prefix.name}_cls_probe.json")
    final_artifacts = [input_audit_path, cls_probe_path]
    for path in output_paths.values():
        final_artifacts.extend((path, _manifest_path(path)))
    existing = [str(path) for path in final_artifacts if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing BrainIAC artifacts; pass --force "
            "intentionally: " + ", ".join(existing)
        )

    frame, onharmony_manifest = load_onharmony_manifest(
        args.filelist,
        args.metadata,
        expected_acquisitions=EXPECTED_ACQUISITIONS,
        expected_subjects=EXPECTED_SUBJECTS,
    )
    registry, input_contract = build_registry(
        onharmony_manifest,
        preproc_root=args.preproc_root.expanduser().resolve(),
        log_root=args.preproc_log_root.expanduser().resolve(),
    )
    device = resolve_device(torch, args.device)
    validate_runtime_contract(device, args.amp, args.seed)
    seed_torch(torch, args.seed)
    precision = effective_precision(device, args.amp)
    print(
        "BrainIAC ON-Harmony extractor: "
        f"n={len(frame)}, readouts={','.join(readouts)}, device={device}, "
        f"amp={args.amp}, effective_precision={precision}",
        flush=True,
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    adapter, frozen_summary = build_model(checkpoint, registry, device)
    if adapter.release_provenance()["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("BrainIAC release proof differs from the fixed checkpoint")

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in readouts}
    provenance_rows: list[dict[str, Any]] = []
    for index, record in enumerate(frame.to_dict(orient="records"), start=1):
        provenance = registry.freeze(record)
        embeddings = extract_readouts(
            adapter,
            record,
            readouts=readouts,
            device=device,
            amp=args.amp,
        )
        if adapter.last_input_provenance != provenance:
            raise RuntimeError(
                "Input provenance changed while extracting "
                f"{record['acquisition_id']}"
            )
        metadata = output_metadata(record, adapter.input_path(record))
        for name, embedding in embeddings.items():
            row: dict[str, Any] = dict(metadata)
            row.update(
                {f"emb_{dimension}": value for dimension, value in enumerate(embedding)}
            )
            rows[name].append(row)
        provenance_rows.append(
            {
                "acquisition_id": record["acquisition_id"],
                "subject_id": metadata["subject_id"],
                "session_id": metadata["session_id"],
                "scanner": metadata["scanner"],
                **provenance,
            }
        )
        if index % 10 == 0 or index == len(frame):
            print(f"  [{index}/{len(frame)}] {record['acquisition_id']}", flush=True)

    embedding_columns = [f"emb_{i}" for i in range(EXPECTED_EMBEDDING_DIM)]
    expected_columns = [
        "file_path",
        "subject_id",
        "session_id",
        "scanner",
        *embedding_columns,
    ]
    frames: dict[str, pd.DataFrame] = {}
    for name in readouts:
        result = pd.DataFrame(rows[name])
        if result.columns.tolist() != expected_columns:
            raise RuntimeError(f"Unexpected BrainIAC {name} output-column order")
        if len(result) != EXPECTED_ACQUISITIONS:
            raise RuntimeError(f"Incomplete BrainIAC {name} output: {len(result)} rows")
        if result[["subject_id", "session_id"]].duplicated().any():
            raise RuntimeError(f"Duplicate BrainIAC {name} acquisition identifiers")
        if not np.isfinite(result[embedding_columns].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite BrainIAC {name} embedding values")
        frames[name] = result

    provenance_frame = pd.DataFrame(provenance_rows)
    if (
        len(provenance_frame) != EXPECTED_ACQUISITIONS
        or provenance_frame["acquisition_id"].duplicated().any()
    ):
        raise RuntimeError("Incomplete or duplicate BrainIAC input provenance")
    _atomic_write_tsv(provenance_frame, input_audit_path)

    cls_probe = {
        "has_cls_token_attr": False,
        "pos_embed_len": N_PATCH_TOKENS,
        "n_output_tokens": N_PATCH_TOKENS,
        "n_patches_expected": N_PATCH_TOKENS,
        "cls_token_present": False,
        "evidence": "validated by the exact adapter and every extracted forward",
    }
    _atomic_write_json(cls_probe, cls_probe_path)

    for key, output_path in output_paths.items():
        readout_name = args.canonical_readout if key == "canonical" else key
        result = frames[readout_name]
        _atomic_write_csv(result, output_path)
        manifest = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": "BrainIAC",
            "variant": "canonical" if key == "canonical" else f"readout_{key}",
            "readout": READOUT_DESCRIPTIONS[readout_name],
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
            "cls_probe": cls_probe,
            "model_implementation": str(MODEL_IMPLEMENTATION_PATH),
            "model_implementation_sha256": sha256_file(MODEL_IMPLEMENTATION_PATH),
            "input_adapter": str(INPUT_ADAPTER_PATH),
            "input_adapter_sha256": sha256_file(INPUT_ADAPTER_PATH),
            "extractor": str(SCRIPT_PATH),
            "extractor_sha256": sha256_file(SCRIPT_PATH),
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "n_rows": len(result),
            "software": package_versions(),
        }
        _atomic_write_json(manifest, _manifest_path(output_path))
        print(
            f"Saved: {output_path} ({len(result)} rows, "
            f"{EXPECTED_EMBEDDING_DIM} dimensions; {readout_name})",
            flush=True,
        )
    print(f"Input audit: {input_audit_path}", flush=True)
    print(f"CLS probe: {cls_probe_path}", flush=True)


if __name__ == "__main__":
    main()
