#!/usr/bin/env python3
"""Audit AnatCL or extract frozen volume embeddings for one cohort split."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.AnatCL.model import (  # noqa: E402
    AnatCLModel,
    AnatCLOptions,
    load_and_validate_manifest,
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


PIPELINE_VERSION = "anatcl_brain_age_v2"
INTERNAL_TEST_REVISION_ID = "anatcl-frozen-bfloat16-alignment-v1"
INTERNAL_TEST_REVISION_SCHEMA_VERSION = "anatcl_internal_test_revision_v1"
INTERNAL_TEST_REVISION_REASON = (
    "Methodological precision alignment identified during code audit: replace the "
    "original FP32 frozen AnatCL extraction with the study-wide BF16 inference "
    "contract while preserving and linking the original test receipt."
)
INTERNAL_TEST_REVISION_OUTPUT_VERSION = "embeddings_FT_bfloat16_v3"


def _read_stable_json(path: Path | str, label: str) -> tuple[dict[str, Any], str]:
    """Read one JSON artifact while detecting mutation during the read."""

    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {artifact}")
    digest_before = sha256_file(artifact)
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON: {artifact}") from error
    digest_after = sha256_file(artifact)
    if digest_after != digest_before:
        raise RuntimeError(f"{label} changed while it was being read: {artifact}")
    if not isinstance(payload, dict):
        raise TypeError(f"{label} is not a JSON object: {artifact}")
    return payload, digest_after


def _verified_file(
    path: Path | str, expected_sha256: object, label: str
) -> tuple[Path, str]:
    """Resolve one file and require its current digest to match its declaration."""

    artifact = Path(path).expanduser()
    if not artifact.is_absolute():
        raise ValueError(f"{label} path must be absolute: {artifact}")
    artifact = artifact.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {artifact}")
    observed = sha256_file(artifact)
    if not isinstance(expected_sha256, str) or observed != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 differs from its declaration: {artifact}")
    return artifact, observed


def _revision_output_path(variant: str) -> Path:
    """Return the only permitted output for the approved AnatCL test revision."""

    return (
        PROJECT_ROOT
        / "Results"
        / "brain_age"
        / INTERNAL_TEST_REVISION_OUTPUT_VERSION
        / "AnatCL"
        / variant
        / "internal_test.tsv"
    ).resolve()


def _validate_revision_selection_lock(
    lock_path: Path | str,
    *,
    model: AnatCLModel,
    manifest_sha256: str,
    code_sha256: Mapping[str, str],
    fold_dir: Path,
) -> dict[str, Any]:
    """Require validation-only model selection before revised test access."""

    lock, lock_sha256 = _read_stable_json(
        lock_path, "AnatCL BF16 revision selection lock"
    )
    resolved_lock = Path(lock_path).expanduser().resolve()
    expected_lock = (
        PROJECT_ROOT
        / "Results"
        / "brain_age"
        / "frozen_bfloat16_v3"
        / "AnatCL"
        / model.variant
        / "LOCKED.json"
    ).resolve()
    if resolved_lock != expected_lock:
        raise ValueError(
            "AnatCL BF16 revision must use the canonical v3 selection lock: "
            f"expected={expected_lock}, observed={resolved_lock}"
        )
    expected_identity = {
        "status": "LOCKED_AFTER_TRAIN_VALIDATION",
        "pipeline_version": PIPELINE_VERSION,
        "arm": "frozen",
        "model": model.model_name,
        "variant": model.variant,
        "fold": model.fold,
        "readout": model.readout,
        "embedding_dim": model.embedding_dim,
        "internal_test_accessed": False,
        "on_harmony_accessed": False,
    }
    drift = {
        key: {"expected": value, "observed": lock.get(key)}
        for key, value in expected_identity.items()
        if lock.get(key) != value
    }
    if drift:
        raise ValueError(f"Invalid AnatCL BF16 revision selection lock: {drift}")
    if lock.get("manifest", {}).get("manifest_sha256") != manifest_sha256:
        raise ValueError("AnatCL BF16 revision lock uses a different cohort manifest")
    contract = lock.get("extraction_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("AnatCL BF16 revision lock lacks an extraction contract")
    contract_expected = {
        "code_sha256": dict(code_sha256),
        "amp": "bfloat16",
        "fold": model.fold,
    }
    contract_drift = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in contract_expected.items()
        if contract.get(key) != value
    }
    if contract_drift:
        raise ValueError(
            "AnatCL BF16 revision lock extraction contract differs: "
            f"{contract_drift}"
        )

    split_hashes: dict[str, str] = {}
    for split, lock_field in (
        ("train", "train_embeddings_sha256"),
        ("validation", "validation_embeddings_sha256"),
    ):
        artifact = fold_dir / f"{split}.tsv"
        sidecar_path = embedding_sidecar_path(artifact)
        artifact, artifact_sha256 = _verified_file(
            artifact, lock.get(lock_field), f"AnatCL BF16 {split} embeddings"
        )
        sidecar, _ = _read_stable_json(
            sidecar_path, f"AnatCL BF16 {split} embedding sidecar"
        )
        sidecar_expected = {
            "pipeline_version": PIPELINE_VERSION,
            "model": model.model_name,
            "variant": model.variant,
            "split": split,
            "artifact_sha256": artifact_sha256,
            "code_sha256": dict(code_sha256),
        }
        sidecar_drift = {
            key: {"expected": value, "observed": sidecar.get(key)}
            for key, value in sidecar_expected.items()
            if sidecar.get(key) != value
        }
        if sidecar.get("runtime", {}).get("amp") != "bfloat16":
            sidecar_drift["runtime.amp"] = {
                "expected": "bfloat16",
                "observed": sidecar.get("runtime", {}).get("amp"),
            }
        if sidecar.get("manifest", {}).get("manifest_sha256") != manifest_sha256:
            sidecar_drift["manifest.manifest_sha256"] = {
                "expected": manifest_sha256,
                "observed": sidecar.get("manifest", {}).get("manifest_sha256"),
            }
        if sidecar.get("publication", {}).get("status") != "COMMITTED":
            sidecar_drift["publication.status"] = {
                "expected": "COMMITTED",
                "observed": sidecar.get("publication", {}).get("status"),
            }
        if sidecar_drift:
            raise ValueError(
                f"AnatCL BF16 {split} bundle differs from the revision lock: "
                f"{sidecar_drift}"
            )
        split_hashes[split] = artifact_sha256

    try:
        selected_alpha = float(lock["selected_alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("AnatCL BF16 revision lock has an invalid selected alpha") from error
    if not math.isfinite(selected_alpha) or selected_alpha <= 0.0:
        raise ValueError("AnatCL BF16 revision lock has an invalid selected alpha")

    model_artifact, model_sha256 = _verified_file(
        resolved_lock.parent / "frozen_linear_model.npz",
        lock.get("model_artifact_sha256"),
        "AnatCL BF16 revision Ridge model",
    )
    return {
        "selection_lock": str(resolved_lock),
        "selection_lock_sha256": lock_sha256,
        "ridge_model": str(model_artifact),
        "ridge_model_sha256": model_sha256,
        "selected_alpha": selected_alpha,
        "train_embeddings_sha256": split_hashes["train"],
        "validation_embeddings_sha256": split_hashes["validation"],
    }


def resolve_internal_test_receipt(
    args: argparse.Namespace,
    model: AnatCLModel,
    output: Path,
    manifest: Mapping[str, Any],
    code_sha256: Mapping[str, str],
    original_receipt: Path | str,
) -> tuple[Path, dict[str, Any] | None]:
    """Resolve the original receipt or the single approved BF16 revision."""

    original_receipt = Path(original_receipt).expanduser().resolve()
    revision_id = getattr(args, "internal_test_revision", None)
    revision_lock = getattr(args, "internal_test_revision_lock", None)
    if args.split != "internal_test":
        if revision_id is not None or revision_lock is not None:
            raise ValueError(
                "Internal-test revision arguments are forbidden outside internal_test"
            )
        return original_receipt, None
    if revision_id is None:
        if revision_lock is not None:
            raise ValueError(
                "--internal-test-revision-lock requires --internal-test-revision"
            )
        return original_receipt, None
    if revision_id != INTERNAL_TEST_REVISION_ID:
        raise ValueError(f"Unsupported AnatCL internal-test revision: {revision_id!r}")
    if revision_lock is None:
        raise ValueError(
            "The approved AnatCL revision requires --internal-test-revision-lock"
        )
    if getattr(args, "amp", None) != "bfloat16":
        raise ValueError("The approved AnatCL internal-test revision requires BF16")
    expected_output = _revision_output_path(model.variant)
    if output.resolve() != expected_output:
        raise ValueError(
            "The approved AnatCL internal-test revision requires its canonical v3 "
            f"output: expected={expected_output}, observed={output.resolve()}"
        )

    original, original_receipt_sha256 = _read_stable_json(
        original_receipt, "Original AnatCL internal-test receipt"
    )
    original_expected = {
        "status": "INTERNAL_TEST_EMBEDDINGS_EXTRACTED",
        "operation": "embedding extraction",
        "pipeline_version": PIPELINE_VERSION,
        "model": model.model_name,
        "variant": model.variant,
        "split": "internal_test",
        "manifest_sha256": manifest.get("manifest_sha256"),
    }
    original_drift = {
        key: {"expected": value, "observed": original.get(key)}
        for key, value in original_expected.items()
        if original.get(key) != value
    }
    if original_drift:
        raise ValueError(
            f"Original AnatCL internal-test receipt differs: {original_drift}"
        )
    original_artifact, original_artifact_sha256 = _verified_file(
        original["artifact"],
        original.get("artifact_sha256"),
        "Original AnatCL internal-test embeddings",
    )
    original_sidecar, original_sidecar_sha256 = _verified_file(
        original["sidecar"],
        original.get("sidecar_sha256"),
        "Original AnatCL internal-test embedding sidecar",
    )
    original_sidecar_data, stable_sidecar_sha256 = _read_stable_json(
        original_sidecar, "Original AnatCL internal-test embedding sidecar"
    )
    if stable_sidecar_sha256 != original_sidecar_sha256:
        raise RuntimeError("Original AnatCL sidecar changed during revision validation")
    original_sidecar_expected = {
        "pipeline_version": PIPELINE_VERSION,
        "model": model.model_name,
        "variant": model.variant,
        "split": "internal_test",
        "artifact_sha256": original_artifact_sha256,
    }
    sidecar_drift = {
        key: {"expected": value, "observed": original_sidecar_data.get(key)}
        for key, value in original_sidecar_expected.items()
        if original_sidecar_data.get(key) != value
    }
    if original_sidecar_data.get("runtime", {}).get("amp") != "off":
        sidecar_drift["runtime.amp"] = {
            "expected": "off",
            "observed": original_sidecar_data.get("runtime", {}).get("amp"),
        }
    if (
        original_sidecar_data.get("manifest", {}).get("manifest_sha256")
        != manifest.get("manifest_sha256")
    ):
        sidecar_drift["manifest.manifest_sha256"] = {
            "expected": manifest.get("manifest_sha256"),
            "observed": original_sidecar_data.get("manifest", {}).get(
                "manifest_sha256"
            ),
        }
    expected_original_reservation = {
        "path": str(original_receipt),
        "attempt_id": original.get("attempt_id"),
    }
    if original_sidecar_data.get("internal_test_receipt") != expected_original_reservation:
        sidecar_drift["internal_test_receipt"] = {
            "expected": expected_original_reservation,
            "observed": original_sidecar_data.get("internal_test_receipt"),
        }
    if original_sidecar_data.get("publication", {}).get("status") != "COMMITTED":
        sidecar_drift["publication.status"] = {
            "expected": "COMMITTED",
            "observed": original_sidecar_data.get("publication", {}).get("status"),
        }
    if sidecar_drift:
        raise ValueError(
            f"Original AnatCL internal-test sidecar differs: {sidecar_drift}"
        )

    selection = _validate_revision_selection_lock(
        revision_lock,
        model=model,
        manifest_sha256=str(manifest["manifest_sha256"]),
        code_sha256=code_sha256,
        fold_dir=output.parent,
    )
    revision_receipt = (
        original_receipt.parent
        / "revisions"
        / INTERNAL_TEST_REVISION_ID
        / original_receipt.name
    ).resolve()
    context = {
        "schema_version": INTERNAL_TEST_REVISION_SCHEMA_VERSION,
        "revision_id": INTERNAL_TEST_REVISION_ID,
        "reason": INTERNAL_TEST_REVISION_REASON,
        "previous_amp": "off",
        "revised_amp": "bfloat16",
        "scope": "precision-only re-extraction; manifest, fold, checkpoint and readout fixed",
        "original_receipt_preserved": True,
        "supersedes_receipt": str(original_receipt),
        "supersedes_receipt_sha256": original_receipt_sha256,
        "supersedes_artifact": str(original_artifact),
        "supersedes_artifact_sha256": original_artifact_sha256,
        "supersedes_sidecar": str(original_sidecar),
        "supersedes_sidecar_sha256": original_sidecar_sha256,
        "revision_receipt": str(revision_receipt),
        "revised_artifact": str(output.resolve()),
        **selection,
    }
    return revision_receipt, context


def verify_internal_test_revision(context: Mapping[str, Any]) -> None:
    """Recheck immutable revision inputs immediately before publication."""

    checks = (
        ("superseded receipt", "supersedes_receipt", "supersedes_receipt_sha256"),
        ("superseded embeddings", "supersedes_artifact", "supersedes_artifact_sha256"),
        ("superseded sidecar", "supersedes_sidecar", "supersedes_sidecar_sha256"),
        ("selection lock", "selection_lock", "selection_lock_sha256"),
        ("Ridge model", "ridge_model", "ridge_model_sha256"),
    )
    for label, path_key, hash_key in checks:
        _verified_file(context[path_key], context[hash_key], f"Revision {label}")


def model_from_args(args: argparse.Namespace) -> AnatCLModel:
    """Construct AnatCL without importing its GPU dependencies."""

    return AnatCLModel(
        AnatCLOptions(
            fold=args.fold,
            checkpoint_dir=args.checkpoint_dir,
            preproc_root=args.preproc_root,
        )
    )


def runtime_code_hashes(model: AnatCLModel) -> dict[str, str]:
    """Hash only files that determine AnatCL frozen extraction."""

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


def missing_inputs(model: AnatCLModel, frame: pd.DataFrame) -> list[str]:
    """Return missing preprocessed model inputs and QC files without loading the checkpoint."""

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
    model = model_from_args(args)
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
    if args.check_input_files and missing:
        payload["status"] = "FAIL"
        print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
        raise FileNotFoundError(
            f"AnatCL has {len(missing)} missing inputs; first 10: {missing[:10]}"
        )
    payload["status"] = "PASS"
    if args.inspect_model or args.smoke_forward:
        torch = import_torch()
        device = resolve_device(torch, args.device)
        seed_torch(torch, args.seed)
        model.load(device)
        payload["release"] = model.release_provenance()
        payload["checkpoint_load"] = model.load_description()
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
                raise FileNotFoundError("No complete AnatCL input is available for smoke-forward")
            age_stats = AgeStandardizer.fit(frame_for_splits(frame, ["train"])["age"])
            head = torch.nn.Linear(model.embedding_dim, 1).to(device)
            model.controlled_train_mode()
            head.train()
            with autocast_context(torch, device, args.amp):
                embedding = model.encode(record, training=True, epoch=1, seed=args.seed)
                prediction = head(embedding).reshape(())
                target = torch.tensor(
                    age_stats.transform(float(record["age"])),
                    dtype=torch.float32,
                    device=device,
                )
                loss = (prediction.float() - target).square()
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
                "embedding_shape": list(embedding.shape),
                "loss": float(loss.detach().cpu().item()),
                "trainable_encoder_parameter_tensors": len(named),
                "encoder_parameter_tensors_with_gradient": len(named)
                - len(missing_gradient),
                "encoder_parameter_tensors_with_nonzero_gradient": len(nonzero_gradient),
                "head_parameter_tensors_with_gradient": len(list(head.parameters()))
                - len(head_missing),
                "optimizer_step_performed": False,
            }
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


def command_extract(args: argparse.Namespace) -> None:
    """Extract deterministic encoder embeddings for one immutable split."""

    if args.split == "internal_test" and not args.acknowledge_locked_test:
        raise ValueError("Pass --acknowledge-locked-test for internal-test extraction")
    model = model_from_args(args)
    output = ensure_new_embedding_bundle(
        args.output, embedding_sidecar_path(args.output)
    )
    frame, manifest = load_and_validate_manifest(args.manifest)
    selected = frame_for_splits(frame, [args.split])
    missing = missing_inputs(model, selected)
    if missing:
        raise FileNotFoundError(f"AnatCL has {len(missing)} missing inputs; first 10: {missing[:10]}")

    torch = import_torch()
    device = resolve_device(torch, args.device)
    seed_torch(torch, args.seed)
    model.load(device)
    frozen_boundary = model.set_frozen()
    release = model.release_provenance()
    code = runtime_code_hashes(model)
    original_receipt = internal_test_embedding_receipt_path(
        model.model_name, model.variant
    ).resolve()
    receipt, internal_test_revision = resolve_internal_test_receipt(
        args,
        model,
        output,
        manifest,
        code,
        original_receipt,
    )
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
        f"Frozen AnatCL extraction: split={args.split}; n={len(selected)}; device={device}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    model.controlled_train_mode()
    with torch.no_grad():
        for position, record in enumerate(selected.to_dict(orient="records"), start=1):
            with autocast_context(torch, device, args.amp):
                embedding = model.encode(record, training=False, epoch=0, seed=args.seed)
            values = embedding.detach().float().cpu().numpy()
            provenance = model.last_input_provenance
            if provenance is None:
                raise RuntimeError("Input provenance was not captured during extraction")
            row = {
                "dataset": str(record["dataset"]),
                "subject": str(record["subject"]),
                "split": str(record["split"]),
                "age": float(record["age"]),
                "sex": str(record["sex"]),
                "input_path": provenance["input_path"],
                "input_sha256": provenance["input_sha256"],
                "preprocessing_qc_path": provenance["qc_path"],
                "preprocessing_qc_sha256": provenance["qc_sha256"],
                "preprocessing_pipeline_version": provenance["pipeline_version"],
                "preprocessing_contract_sha256": provenance[
                    "pipeline_contract_sha256"
                ],
                "model": model.model_name,
                "variant": model.variant,
            }
            row.update({f"emb_{index}": float(value) for index, value in enumerate(values)})
            rows.append(row)
            if position % 25 == 0 or position == len(selected):
                print(f"  [{position}/{len(selected)}] {record['dataset']}/{record['subject']}", flush=True)

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
        "checkpoint_load": model.load_description(),
        "frozen_boundary": frozen_boundary,
        "manifest": manifest,
        "release": release,
        "code_sha256": code,
        "software": package_versions(),
        "runtime": {
            "device": str(device),
            "amp": args.amp,
            "seed": args.seed,
            "fold": args.fold,
            "input_hashes_verified": True,
        },
        "selected_subjects": int(len(selected)),
        "internal_test_accessed": args.split == "internal_test",
        "internal_test_receipt": None
        if started_receipt is None
        else {"path": str(receipt), "attempt_id": started_receipt["attempt_id"]},
        "internal_test_revision": internal_test_revision,
        "on_harmony_accessed": False,
    }
    if internal_test_revision is not None:
        verify_internal_test_revision(internal_test_revision)
    sidecar = publish_embedding_bundle(
        embeddings, output, embedding_sidecar_path(output), sidecar
    )
    if args.split == "internal_test":
        if started_receipt is None:
            raise RuntimeError("Internal-test extraction lacks its exclusive reservation")
        if internal_test_revision is not None:
            verify_internal_test_revision(internal_test_revision)
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


def add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--preproc-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)


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

    extract = commands.add_parser("extract")
    add_model_options(extract)
    extract.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--acknowledge-locked-test", action="store_true")
    extract.add_argument(
        "--internal-test-revision",
        choices=(INTERNAL_TEST_REVISION_ID,),
        help="Authorize the single audited AnatCL FP32-to-BF16 test revision.",
    )
    extract.add_argument(
        "--internal-test-revision-lock",
        type=Path,
        help="Validation-only v3 Ridge LOCKED.json required by the test revision.",
    )
    extract.set_defaults(func=command_extract)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
