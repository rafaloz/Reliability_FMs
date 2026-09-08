#!/usr/bin/env python3
"""Audit NeuroVFM or extract frozen volume embeddings for one cohort split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.NeuroVFM.model import (  # noqa: E402
    DEFAULT_PREPROCESSING_CACHE_ROOT,
    NEUROVFM_MODEL_REPO,
    NEUROVFM_MODEL_REVISION,
    NeuroVFMModel,
    NeuroVFMOptions,
    load_and_validate_manifest,
    validate_neurovfm_device,
)
from FineTuning.shared.cohort import (  # noqa: E402
    ALLOWED_SPLITS,
    DEFAULT_MANIFEST,
    frame_for_splits,
    sha256_file,
)
from FineTuning.shared.embedding_bundle import (  # noqa: E402
    EMBEDDING_BUNDLE_SCHEMA_VERSION,
    ensure_new_embedding_bundle,
    publish_embedding_bundle,
)
from FineTuning.shared.metrics import (  # noqa: E402
    AgeStandardizer,
    validate_embedding_frame,
)
from FineTuning.shared.runtime import (  # noqa: E402
    atomic_json,
    atomic_tsv,
    autocast_context,
    code_hashes,
    import_torch,
    package_versions,
    resolve_device,
    seed_torch,
)
from FineTuning.shared.test_receipt import (  # noqa: E402
    complete_internal_test_embedding_receipt,
    internal_test_embedding_receipt_path,
    reserve_internal_test_embedding_receipt,
)


PIPELINE_VERSION = "neurovfm_brain_age_v2"


def model_from_args(
    args: argparse.Namespace,
    *,
    require_cache: bool = False,
    memory_cache: bool = True,
) -> NeuroVFMModel:
    """Construct NeuroVFM without importing its GPU dependencies."""

    return NeuroVFMModel(
        NeuroVFMOptions(
            model_source=args.model_source,
            model_revision=args.model_revision,
            allow_download=args.allow_download,
            preprocessing_cache_root=getattr(args, "cache_root", None),
            require_preprocessing_cache=require_cache,
            memory_cache=memory_cache,
        )
    )


def runtime_code_hashes(model: NeuroVFMModel) -> dict[str, str]:
    """Hash only files that determine NeuroVFM frozen extraction."""

    return code_hashes(
        [
            SCRIPT_PATH,
            PROJECT_ROOT / "FineTuning" / "shared" / "cohort.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "metrics.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "runtime.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "embedding_bundle.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "test_receipt.py",
            *model.code_paths(),
        ]
    )


def missing_inputs(model: NeuroVFMModel, frame: pd.DataFrame) -> list[str]:
    """Return missing required model inputs, raw or cached, without loading weights."""

    missing = []
    for record in frame.to_dict(orient="records"):
        missing.extend(str(path) for path in model.required_input_paths(record) if not path.is_file())
    return missing


def embedding_sidecar_path(path: Path | str) -> Path:
    return Path(str(Path(path).resolve()) + ".audit.json")


def command_audit(args: argparse.Namespace) -> None:
    """Validate cohort, files, release, trainable boundary, and optional gradients."""

    if args.smoke_forward and args.split == "internal_test":
        raise ValueError("Smoke-forward is forbidden on the locked internal-test split")
    model = model_from_args(args, require_cache=args.cache_root is not None)
    frame, manifest = load_and_validate_manifest(
        args.manifest,
        check_input_files=False,
    )
    selected = frame if args.split is None else frame_for_splits(frame, [args.split])
    missing = missing_inputs(model, selected) if args.check_input_files else []
    payload: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "model": model.audit_description(),
        "manifest": manifest,
        "selected_split": args.split or "all",
        "selected_subjects": int(len(selected)),
        "input_files_checked": bool(args.check_input_files),
        "missing_model_inputs": len(missing),
        "missing_model_input_examples": missing[:10],
        "release": model.release_provenance(),
        "software": package_versions(),
    }
    if args.inspect_model or args.smoke_forward:
        torch = import_torch()
        device = resolve_device(torch, args.device)
        payload["runtime"] = validate_neurovfm_device(torch, device, args.amp)
        seed_torch(torch, args.seed)
        model.load(device)
        payload["controlled_adaptation"] = model.set_controlled_adaptation()
        payload["code_sha256"] = runtime_code_hashes(model)
        if args.smoke_forward:
            smoke_selected = frame_for_splits(frame, [args.split or "train"])
            record = next(
                (
                    item
                    for item in smoke_selected.to_dict(orient="records")
                    if all(path.is_file() for path in model.required_input_paths(item))
                ),
                None,
            )
            if record is None:
                raise FileNotFoundError("No complete NeuroVFM input is available for smoke-forward")
            age_stats = AgeStandardizer.fit(frame_for_splits(frame, ["train"])["age"])
            head = torch.nn.Linear(model.embedding_dim, 1).to(device)
            model.controlled_train_mode()
            head.train()
            official_embedding = model.encode_official(record).detach().float()
            with autocast_context(torch, device, args.amp):
                embedding = model.encode(record, training=True, epoch=1, seed=args.seed)
                prediction = head(embedding).reshape(())
                target = torch.tensor(
                    age_stats.transform(float(record["age"])),
                    dtype=torch.float32,
                    device=device,
                )
                loss = (prediction.float() - target).square()
            absolute_difference = (embedding.detach().float() - official_embedding).abs()
            equivalent = bool(
                torch.allclose(
                    embedding.detach().float(),
                    official_embedding,
                    rtol=1e-5,
                    atol=1e-6,
                )
            )
            if not equivalent:
                raise RuntimeError(
                    "Differentiable NeuroVFM forward differs from the official frozen path: "
                    f"max_abs_diff={float(absolute_difference.max().item()):.8g}"
                )
            loss.backward()
            named = model.trainable_named_parameters()
            missing_gradient = [name for name, value in named if value.grad is None]
            nonfinite_gradient = [
                name
                for name, value in named
                if value.grad is not None and not torch.isfinite(value.grad).all()
            ]
            nonzero_gradient = [
                name
                for name, value in named
                if value.grad is not None and bool(torch.count_nonzero(value.grad).item())
            ]
            head_missing = [name for name, value in head.named_parameters() if value.grad is None]
            if missing_gradient or nonfinite_gradient or not nonzero_gradient or head_missing:
                raise RuntimeError(
                    "Differentiability smoke test failed: "
                    f"missing={missing_gradient[:10]}, nonfinite={nonfinite_gradient[:10]}, "
                    f"nonzero={len(nonzero_gradient)}, head_missing={head_missing}"
                )
            payload["differentiability_smoke_test"] = {
                "status": "PASS",
                "dataset": str(record["dataset"]),
                "subject": str(record["subject"]),
                "split": str(record["split"]),
                "input_path": str(model.input_path(record)),
                "n_tokens": model.last_token_count,
                "embedding_shape": list(embedding.shape),
                "loss": float(loss.detach().cpu().item()),
                "official_forward_equivalence": {
                    "status": "PASS",
                    "rtol": 1e-5,
                    "atol": 1e-6,
                    "maximum_absolute_difference": float(absolute_difference.max().item()),
                    "mean_absolute_difference": float(absolute_difference.mean().item()),
                },
                "trainable_encoder_parameter_tensors": len(named),
                "encoder_parameter_tensors_with_gradient": len(named)
                - len(missing_gradient),
                "encoder_parameter_tensors_with_nonzero_gradient": len(nonzero_gradient),
                "head_parameter_tensors_with_gradient": len(list(head.parameters()))
                - len(head_missing),
                "optimizer_step_performed": False,
            }
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


def command_cache(args: argparse.Namespace) -> None:
    """Materialize released StudyPreprocessor batches once for reusable training."""

    if "internal_test" in args.splits and not args.acknowledge_locked_test:
        raise ValueError("Pass --acknowledge-locked-test to cache internal-test inputs")
    if (args.dataset is None) != (args.subject is None):
        raise ValueError("--dataset and --subject must be supplied together")
    model = model_from_args(args, memory_cache=False)
    frame, manifest = load_and_validate_manifest(args.manifest)
    selected = frame_for_splits(frame, args.splits)
    if args.dataset is not None:
        selected = selected.loc[
            (selected["dataset"].astype(str) == args.dataset)
            & (selected["subject"].astype(str) == args.subject)
        ].copy()
        if len(selected) != 1:
            raise ValueError(
                f"Expected exactly one manifest row for {args.dataset}/{args.subject}; "
                f"observed {len(selected)}"
            )
    missing = missing_inputs(model, selected)
    if missing:
        raise FileNotFoundError(
            f"NeuroVFM has {len(missing)} missing raw inputs; first 10: {missing[:10]}"
        )

    model.load_preprocessor()
    runtime = {
        "device": "cpu",
        "implementation": "released neurovfm.pipelines.preprocessor.StudyPreprocessor",
        "encoder_loaded": False,
        "random_operations": False,
    }
    contract = model.preprocessing_cache_contract()
    release = model.release_provenance()
    code = runtime_code_hashes(model)

    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + f"_{os.getpid()}"
    run_root = Path(args.cache_root).expanduser().resolve() / "_runs"
    run_manifest = run_root / f"cache_run_{run_id}.tsv"
    run_summary = run_root / f"cache_run_{run_id}.json"
    print(
        "NeuroVFM StudyPreprocessor cache: "
        f"splits={','.join(args.splits)}; n={len(selected)}; device=cpu",
        flush=True,
    )
    print(f"  cache root: {Path(args.cache_root).expanduser().resolve()}", flush=True)
    print(f"  contract  : {contract['contract_sha256']}", flush=True)

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for position, record in enumerate(selected.to_dict(orient="records"), start=1):
        item_started = time.monotonic()
        try:
            result = model.build_preprocessing_cache(record, overwrite=args.overwrite)
            status = str(result["status"])
            message = (
                f"n_tokens={result['n_tokens']}; "
                f"size={int(result.get('artifact_size_bytes', 0))} bytes"
            )
        except Exception as error:  # retain the full cohort ledger before failing
            result = {}
            status = "FAILED"
            message = f"{type(error).__name__}: {error}"
        rows.append(
            {
                "dataset": str(record["dataset"]),
                "subject": str(record["subject"]),
                "split": str(record["split"]),
                "raw_t1w_path": str(model.input_path(record)),
                "status": status,
                "message": message,
                "artifact": result.get("artifact", ""),
                "artifact_sha256": result.get("artifact_sha256", ""),
                "n_tokens": result.get("n_tokens", ""),
                "elapsed_seconds": float(time.monotonic() - item_started),
            }
        )
        print(
            f"  [{position}/{len(selected)}] {status:<7} "
            f"{record['dataset']}/{record['subject']}: {message}",
            flush=True,
        )
        if position % 25 == 0:
            atomic_tsv(pd.DataFrame(rows), run_manifest)

    ledger = pd.DataFrame(rows)
    atomic_tsv(ledger, run_manifest)
    status_counts = {
        str(name): int(value) for name, value in ledger["status"].value_counts().items()
    }
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "operation": "cache_released_study_preprocessor_batches",
        "cache_root": str(Path(args.cache_root).expanduser().resolve()),
        "splits": list(args.splits),
        "selected_subjects": int(len(selected)),
        "status_counts": status_counts,
        "elapsed_seconds": float(time.monotonic() - started),
        "cache_contract": contract,
        "manifest": manifest,
        "release": release,
        "code_sha256": code,
        "software": package_versions(),
        "runtime": runtime,
        "cache_statistics": model.preprocessing_cache_statistics(),
        "internal_test_accessed": "internal_test" in args.splits,
        "on_harmony_accessed": False,
        "run_manifest": str(run_manifest),
    }
    atomic_json(summary, run_summary)
    print(f"Run manifest: {run_manifest}", flush=True)
    print(f"Run summary : {run_summary}", flush=True)
    print(f"Summary     : {status_counts}", flush=True)
    failures = ledger.loc[ledger["status"] == "FAILED"]
    if not failures.empty:
        raise RuntimeError(f"NeuroVFM cache failed for {len(failures)} subject(s)")


def command_extract(args: argparse.Namespace) -> None:
    """Extract the official tokens and study-defined mean vector for one split."""

    if args.split == "internal_test" and not args.acknowledge_locked_test:
        raise ValueError("Pass --acknowledge-locked-test for internal-test extraction")
    model = model_from_args(
        args,
        require_cache=args.cache_root is not None,
        memory_cache=False,
    )
    output = ensure_new_embedding_bundle(
        args.output, embedding_sidecar_path(args.output)
    )
    receipt = internal_test_embedding_receipt_path(model.model_name, model.variant)
    frame, manifest = load_and_validate_manifest(args.manifest)
    selected = frame_for_splits(frame, [args.split])
    missing = missing_inputs(model, selected)
    if missing:
        raise FileNotFoundError(f"NeuroVFM has {len(missing)} missing inputs; first 10: {missing[:10]}")

    torch = import_torch()
    device = resolve_device(torch, args.device)
    runtime = validate_neurovfm_device(torch, device, args.amp)
    seed_torch(torch, args.seed)
    model.load(device)
    frozen_boundary = model.set_frozen()
    release = model.release_provenance()
    code = runtime_code_hashes(model)
    started_receipt = None
    if args.split == "internal_test":
        started_receipt = reserve_internal_test_embedding_receipt(
            receipt,
            {
                "pipeline_version": PIPELINE_VERSION,
                "model": model.model_name,
                "variant": model.variant,
                "readout": model.readout,
                "embedding_dim": model.embedding_dim,
                "split": "internal_test",
                "manifest_artifact": str(manifest["manifest"]),
                "manifest_sha256": manifest["manifest_sha256"],
                "requested_artifact": str(output),
                "requested_sidecar": str(embedding_sidecar_path(output)),
                "requested_n_subjects": int(len(selected)),
                "on_harmony_accessed": False,
            },
        )

    print(
        f"Frozen NeuroVFM extraction: split={args.split}; n={len(selected)}; device={device}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    model.controlled_train_mode()
    with torch.no_grad():
        for position, record in enumerate(selected.to_dict(orient="records"), start=1):
            # The official EncoderPipeline.embed call owns its BF16 autocast and
            # deliberately normalizes tokens in FP32 before entering the encoder.
            embedding = model.encode(record, training=False, epoch=0, seed=args.seed)
            values = embedding.detach().float().cpu().numpy()
            input_path = model.input_path(record)
            cache_path, cache_sidecar_path = model.preprocessing_cache_paths(record)
            cache_metadata = json.loads(cache_sidecar_path.read_text(encoding="utf-8"))
            raw_source = cache_metadata["raw_source"]
            input_sha256 = sha256_file(input_path)
            if raw_source.get("sha256") != input_sha256:
                raise RuntimeError(
                    f"NeuroVFM raw input hash differs from its cache: {input_path}"
                )
            row = {
                "dataset": str(record["dataset"]),
                "subject": str(record["subject"]),
                "split": str(record["split"]),
                "age": float(record["age"]),
                "sex": str(record["sex"]),
                "input_path": str(input_path),
                "input_sha256": input_sha256,
                "preprocessing_cache_path": str(cache_path),
                "preprocessing_cache_sha256": cache_metadata["artifact_sha256"],
                "preprocessing_cache_sidecar_path": str(cache_sidecar_path),
                "preprocessing_cache_sidecar_sha256": sha256_file(cache_sidecar_path),
                "preprocessing_cache_version": cache_metadata["cache_version"],
                "preprocessing_contract_sha256": cache_metadata[
                    "preprocessing_contract"
                ]["contract_sha256"],
                "model": model.model_name,
                "variant": model.variant,
                "n_tokens": int(model.last_token_count or 0),
            }
            row.update({f"emb_{index}": float(value) for index, value in enumerate(values)})
            rows.append(row)
            if position % 25 == 0 or position == len(selected):
                print(
                    f"  [{position}/{len(selected)}] {record['dataset']}/{record['subject']} "
                    f"n_tokens={model.last_token_count}",
                    flush=True,
                )

    embeddings = pd.DataFrame(rows)
    validate_embedding_frame(
        embeddings,
        expected_manifest=selected,
        expected_model=model.model_name,
        expected_variant=model.variant,
    )
    sidecar = {
        "manifest_schema_version": EMBEDDING_BUNDLE_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "arm": "frozen_embedding_extraction",
        "model": model.model_name,
        "variant": model.variant,
        "readout": model.readout,
        "split": args.split,
        "n_subjects": int(len(embeddings)),
        "embedding_dim": model.embedding_dim,
        "artifact": str(output),
        "elapsed_seconds": float(time.monotonic() - started),
        "adapter": model.audit_description(),
        "frozen_boundary": frozen_boundary,
        "manifest": manifest,
        "release": release,
        "preprocessing_cache_contract": model.preprocessing_cache_contract(),
        "preprocessing_cache_statistics": model.preprocessing_cache_statistics(),
        "code_sha256": code,
        "software": package_versions(),
        "runtime": {**runtime, "seed": args.seed, "input_hashes_verified": True},
        "internal_test_accessed": args.split == "internal_test",
        "internal_test_receipt": None
        if started_receipt is None
        else {"path": str(receipt), "attempt_id": started_receipt["attempt_id"]},
        "on_harmony_accessed": False,
    }
    sidecar = publish_embedding_bundle(
        embeddings, output, embedding_sidecar_path(output), sidecar
    )
    if args.split == "internal_test":
        if started_receipt is None:
            raise RuntimeError("Internal-test extraction lacks its exclusive reservation")
        complete_internal_test_embedding_receipt(
            receipt,
            started_receipt,
            {
                "artifact": str(output),
                "artifact_sha256": sidecar["artifact_sha256"],
                "sidecar": str(embedding_sidecar_path(output)),
                "sidecar_sha256": sha256_file(embedding_sidecar_path(output)),
                "n_subjects": int(len(embeddings)),
            },
        )
    print(f"Saved: {output}", flush=True)
    print(f"Audit: {embedding_sidecar_path(output)}", flush=True)


def add_release_options(parser: argparse.ArgumentParser) -> None:
    """Add checkpoint/code options shared by CPU and GPU commands."""

    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-source", default=NEUROVFM_MODEL_REPO)
    parser.add_argument("--model-revision", default=NEUROVFM_MODEL_REVISION)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit a revision-pinned HF download; default is cache-only",
    )


def add_model_options(parser: argparse.ArgumentParser) -> None:
    """Add options for commands that execute the NeuroVFM encoder."""

    add_release_options(parser)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Use verified StudyPreprocessor batches from this cache root",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    add_model_options(audit)
    audit.add_argument("--split", choices=ALLOWED_SPLITS)
    audit.add_argument("--check-input-files", action="store_true")
    audit.add_argument("--inspect-model", action="store_true")
    audit.add_argument("--smoke-forward", action="store_true")
    audit.set_defaults(func=command_audit)

    cache = commands.add_parser("cache")
    add_release_options(cache)
    cache.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_PREPROCESSING_CACHE_ROOT,
        help="Destination for released StudyPreprocessor batches",
    )
    cache.add_argument(
        "--splits",
        choices=ALLOWED_SPLITS,
        nargs="+",
        default=("train", "validation"),
    )
    cache.add_argument("--dataset")
    cache.add_argument("--subject")
    cache.add_argument("--overwrite", action="store_true")
    cache.add_argument("--acknowledge-locked-test", action="store_true")
    cache.set_defaults(func=command_cache)

    extract = commands.add_parser("extract")
    add_model_options(extract)
    extract.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--acknowledge-locked-test", action="store_true")
    extract.set_defaults(
        cache_root=DEFAULT_PREPROCESSING_CACHE_ROOT,
        func=command_extract,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_from_args(args).options.validate()
    args.func(args)


if __name__ == "__main__":
    main()
