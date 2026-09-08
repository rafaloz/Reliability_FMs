#!/usr/bin/env python3
"""Fit or evaluate the frozen StandardScaler + Ridge head for AnatCL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.AnatCL.model import (  # noqa: E402
    EMBEDDING_DIM,
    MODEL_NAME,
    READOUT,
    load_and_validate_manifest,
    semantic_adapter_contract,
    semantic_release_contract,
    variant_for_fold,
)
from FineTuning.shared.cohort import (  # noqa: E402
    DEFAULT_MANIFEST,
    MANIFEST_SHA256,
    frame_for_splits,
    sha256_file,
)
from FineTuning.shared.embedding_bundle import load_embedding_bundle_sidecar  # noqa: E402
from FineTuning.shared.metrics import (  # noqa: E402
    regression_metrics,
    validate_embedding_frame,
)
from FineTuning.shared.protocol import (  # noqa: E402
    ridge_alpha_argument,
    write_ridge_selection_plot,
)
from FineTuning.shared.ridge import (  # noqa: E402
    MODEL_FILENAME,
    add_frozen_onharmony_arguments,
    add_frozen_onharmony_source_arguments,
    audit_frozen_onharmony_source,
    atomic_npz,
    fit_ridge_grid,
    load_locked_ridge_artifact,
    parse_ridge_alphas,
    predict_ridge,
    run_frozen_onharmony_prediction,
    validate_frozen_embedding_contract,
)
from FineTuning.shared.test_receipt import (  # noqa: E402
    complete_internal_test_receipt,
    prepare_internal_test_output_dir,
    publish_internal_test_result_bundle,
    reserve_internal_test_receipt,
    verify_internal_test_input_integrity,
)
from FineTuning.shared.runtime import (  # noqa: E402
    atomic_json,
    atomic_tsv,
    code_hashes,
    ensure_new_output_dir,
    package_versions,
)


PIPELINE_VERSION = "anatcl_brain_age_v2"
ONHARMONY_EXTRACTOR_PIPELINE_VERSION = "anatcl_onharmony_embedding_v2"
FROZEN_AMP = "bfloat16"
LOCK_NAME = "LOCKED.json"
INTERNAL_TEST_RECEIPT = "INTERNAL_TEST_EVALUATED.json"


def embedding_sidecar_path(path: Path | str) -> Path:
    return Path(str(Path(path).resolve()) + ".audit.json")


def load_embedding_sidecar(
    path: Path | str, *, split: str, variant: str
) -> dict[str, Any]:
    """Verify that an embedding TSV still matches its adjacent audit artifact."""

    artifact = Path(path).resolve()
    return load_embedding_bundle_sidecar(
        artifact,
        embedding_sidecar_path(artifact),
        expected_pipeline_version=PIPELINE_VERSION,
        expected_model=MODEL_NAME,
        expected_variant=variant,
        expected_readout=READOUT,
        expected_split=split,
        expected_dimension=EMBEDDING_DIM,
    )


def read_embedding_table(
    path: Path,
    *,
    split: str,
    variant: str,
    expected_manifest: pd.DataFrame,
    manifest_sha256: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load one AnatCL split after validating rows and extraction provenance."""

    sidecar = load_embedding_sidecar(path, split=split, variant=variant)
    observed = {
        key: sidecar.get(key)
        for key in ("pipeline_version", "model", "variant", "readout", "split")
    }
    expected = {
        "pipeline_version": PIPELINE_VERSION,
        "model": MODEL_NAME,
        "variant": variant,
        "readout": READOUT,
        "split": split,
    }
    if observed != expected:
        raise ValueError(f"Embedding sidecar drift: expected={expected}, observed={observed}")
    if sidecar.get("manifest", {}).get("manifest_sha256") != manifest_sha256:
        raise ValueError("Embedding artifact uses a different subject manifest")
    frame = pd.read_csv(path, sep="\t")
    columns = validate_embedding_frame(
        frame,
        expected_manifest=expected_manifest,
        expected_model=MODEL_NAME,
        expected_variant=variant,
    )
    validate_frozen_embedding_contract(
        frame,
        expected_manifest,
        columns,
        sidecar_dimension=sidecar.get("embedding_dim"),
        expected_dimension=EMBEDDING_DIM,
        model_name=MODEL_NAME,
    )
    if load_embedding_sidecar(path, split=split, variant=variant) != sidecar:
        raise RuntimeError("Embedding bundle changed while its table was being read")
    return frame, columns, sidecar


def extraction_contract(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Return the location-independent extraction identity required across splits."""

    runtime = sidecar.get("runtime", {})
    checkpoint_load = sidecar.get("checkpoint_load", {})
    stable_load_fields = {
        key: checkpoint_load.get(key)
        for key in (
            "strict",
            "backbone_keys",
            "probe_key",
            "probe_bit_exact",
            "projection_head_used",
            "checkpoint_sha256",
            "installed_source_sha256",
        )
    }
    contract = {
        "release": semantic_release_contract(sidecar.get("release", {})),
        "adapter": semantic_adapter_contract(sidecar.get("adapter", {})),
        "code_sha256": sidecar.get("code_sha256"),
        "amp": runtime.get("amp"),
        "fold": runtime.get("fold"),
        "checkpoint_load": stable_load_fields,
    }
    if not contract["release"] or not contract["adapter"] or not contract["code_sha256"]:
        raise ValueError("Embedding sidecar lacks release, adapter, or code provenance")
    if contract["amp"] != FROZEN_AMP:
        raise ValueError(
            "AnatCL frozen embeddings must use the study-wide adapted precision "
            f"{FROZEN_AMP!r}, observed {contract['amp']!r}"
        )
    if contract["fold"] not in range(5):
        raise ValueError(f"Unsupported AnatCL embedding fold contract: {contract['fold']!r}")
    if any(value is None for value in stable_load_fields.values()):
        raise ValueError("Embedding sidecar lacks the strict checkpoint-load proof")
    if (
        stable_load_fields["strict"] is not True
        or stable_load_fields["probe_bit_exact"] is not True
        or stable_load_fields["projection_head_used"] is not False
    ):
        raise ValueError("Embedding sidecar has an invalid checkpoint-load proof")
    return contract


def fit_code_hashes() -> dict[str, str]:
    return code_hashes(
        [
            SCRIPT_PATH,
            PROJECT_ROOT / "FineTuning" / "shared" / "cohort.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "metrics.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "ridge.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "embedding_bundle.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "test_receipt.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "protocol.py",
            PROJECT_ROOT / "FineTuning" / "shared" / "runtime.py",
            PROJECT_ROOT / "FineTuning" / "AnatCL" / "model.py",
        ]
    )


def validate_evaluation_lock(
    lock: dict[str, Any], model_path: Path, *, variant: str
) -> None:
    """Require an untouched validation-only frozen run before test access."""

    if (lock.get("arm"), lock.get("model"), lock.get("variant")) != (
        "frozen",
        MODEL_NAME,
        variant,
    ):
        raise ValueError("LOCKED.json is not the expected frozen AnatCL run")
    if (
        lock.get("status") != "LOCKED_AFTER_TRAIN_VALIDATION"
        or lock.get("internal_test_accessed") is not False
        or lock.get("on_harmony_accessed") is not False
    ):
        raise ValueError("Evaluation requires a validation-only locked frozen run")
    if lock.get("model_artifact_sha256") != sha256_file(model_path):
        raise ValueError("Frozen model hash differs from LOCKED.json")


def command_fit(args: argparse.Namespace) -> None:
    """Select alpha using validation only and lock the resulting linear model."""

    variant = variant_for_fold(args.fold)
    manifest, summary = load_and_validate_manifest(args.manifest)
    train_manifest = frame_for_splits(manifest, ["train"])
    validation_manifest = frame_for_splits(manifest, ["validation"])
    train, train_columns, train_sidecar = read_embedding_table(
        args.train_embeddings,
        split="train",
        variant=variant,
        expected_manifest=train_manifest,
        manifest_sha256=str(summary["manifest_sha256"]),
    )
    validation, validation_columns, validation_sidecar = read_embedding_table(
        args.validation_embeddings,
        split="validation",
        variant=variant,
        expected_manifest=validation_manifest,
        manifest_sha256=str(summary["manifest_sha256"]),
    )
    if train_columns != validation_columns:
        raise ValueError("Train and validation embedding dimensions differ")
    train_contract = extraction_contract(train_sidecar)
    validation_contract = extraction_contract(validation_sidecar)
    if train_contract != validation_contract:
        raise ValueError("Train/validation extraction provenance differs")

    alphas = parse_ridge_alphas(args.ridge_alphas)
    output_dir = ensure_new_output_dir(args.output_dir)
    artifact, candidates, prediction = fit_ridge_grid(
        train[train_columns].to_numpy(dtype=np.float64),
        train["age"],
        validation[validation_columns].to_numpy(dtype=np.float64),
        validation["age"],
        alphas,
    )
    model_path = output_dir / MODEL_FILENAME
    atomic_npz(artifact, model_path)
    atomic_tsv(candidates, output_dir / "ridge_validation_grid.tsv")
    write_ridge_selection_plot(candidates, output_dir)

    predictions = validation[["dataset", "subject", "split", "age", "sex", "input_path"]].copy()
    predictions = predictions.rename(columns={"age": "age_true"})
    predictions["age_pred"] = prediction
    predictions["error_years"] = predictions["age_pred"] - predictions["age_true"]
    predictions["absolute_error_years"] = predictions["error_years"].abs()
    atomic_tsv(predictions, output_dir / "locked_validation_predictions.tsv")
    metrics = regression_metrics(predictions["age_true"], predictions["age_pred"])
    metrics.update({"split": "validation", "bias_correction": "none"})
    atomic_json(metrics, output_dir / "locked_validation_metrics.json")

    lock = {
        "status": "LOCKED_AFTER_TRAIN_VALIDATION",
        "pipeline_version": PIPELINE_VERSION,
        "arm": "frozen",
        "model": MODEL_NAME,
        "variant": variant,
        "readout": READOUT,
        "fold": args.fold,
        "model_artifact": MODEL_FILENAME,
        "model_artifact_sha256": sha256_file(model_path),
        "embedding_dim": len(train_columns),
        "ridge_alphas": alphas,
        "selected_alpha": float(artifact["alpha"][0]),
        "selection_metric": "uncorrected validation MAE in years",
        "locked_validation_metrics": metrics,
        "manifest": summary,
        "extraction_release": train_sidecar["release"],
        "extraction_contract": train_contract,
        "extraction_code_sha256": train_sidecar["code_sha256"],
        "train_embeddings_sha256": train_sidecar["artifact_sha256"],
        "validation_embeddings_sha256": validation_sidecar["artifact_sha256"],
        "fit_code_sha256": fit_code_hashes(),
        "software": package_versions(),
        "internal_test_accessed": False,
        "on_harmony_accessed": False,
    }
    atomic_json(lock, output_dir / LOCK_NAME)
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
    print(f"Locked frozen AnatCL {variant} model: {model_path}", flush=True)


def command_evaluate(args: argparse.Namespace) -> None:
    """Perform the explicitly unlocked one-time internal-test evaluation."""

    variant = variant_for_fold(args.fold)
    if not args.acknowledge_locked_test:
        raise ValueError("Pass --acknowledge-locked-test for internal-test evaluation")
    run_dir = args.run_dir.resolve()
    lock_path = run_dir / LOCK_NAME
    model_path = run_dir / MODEL_FILENAME
    receipt = run_dir / INTERNAL_TEST_RECEIPT
    if receipt.exists():
        raise FileExistsError(f"This model already has an internal-test receipt: {receipt}")
    if not lock_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Expected {lock_path} and {model_path}")
    lock_sha256 = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(lock_path) != lock_sha256:
        raise RuntimeError("LOCKED.json changed while it was being read")
    validate_evaluation_lock(lock, model_path, variant=variant)

    manifest, summary = load_and_validate_manifest(args.manifest)
    if lock.get("manifest", {}).get("manifest_sha256") != summary["manifest_sha256"]:
        raise ValueError("Frozen model uses a different subject manifest")
    if lock.get("fit_code_sha256") != fit_code_hashes():
        raise ValueError("Runtime Ridge code differs from the locked frozen model")
    output_dir = prepare_internal_test_output_dir(args.output_dir)
    test_manifest = frame_for_splits(manifest, ["internal_test"])
    test_embeddings_path = args.test_embeddings.resolve()
    test_sidecar_path = embedding_sidecar_path(test_embeddings_path)
    started_receipt = reserve_internal_test_receipt(
        receipt,
        {
            "pipeline_version": PIPELINE_VERSION,
            "arm": "frozen",
            "model": MODEL_NAME,
            "variant": variant,
            "run_dir": str(run_dir),
            "training_lock_artifact": str(lock_path),
            "training_lock_sha256": lock_sha256,
            "model_artifact": str(model_path),
            "model_artifact_sha256": lock["model_artifact_sha256"],
            "manifest_artifact": str(summary["manifest"]),
            "manifest_sha256": summary["manifest_sha256"],
            "test_embeddings_artifact": str(test_embeddings_path),
            "test_embeddings_audit_artifact": str(test_sidecar_path),
            "output_dir": str(output_dir),
            "on_harmony_accessed": False,
        },
    )
    sidecar_sha256_before = sha256_file(test_sidecar_path)
    test, columns, sidecar = read_embedding_table(
        test_embeddings_path,
        split="internal_test",
        variant=variant,
        expected_manifest=test_manifest,
        manifest_sha256=str(summary["manifest_sha256"]),
    )
    if extraction_contract(sidecar) != lock.get("extraction_contract"):
        raise ValueError("Internal-test embeddings use a different extraction contract")
    if len(columns) != int(lock["embedding_dim"]):
        raise ValueError("Internal-test embedding dimensionality differs")

    test_embeddings_sha256 = sha256_file(test_embeddings_path)
    test_sidecar_sha256 = sha256_file(test_sidecar_path)
    if test_embeddings_sha256 != sidecar.get("artifact_sha256"):
        raise RuntimeError("Internal-test embeddings changed during evaluation")
    if test_sidecar_sha256 != sidecar_sha256_before:
        raise RuntimeError("Internal-test embedding sidecar changed during evaluation")

    ridge_artifact = load_locked_ridge_artifact(
        model_path,
        expected_sha256=str(lock["model_artifact_sha256"]),
        expected_alpha=float(lock["selected_alpha"]),
        expected_dimension=EMBEDDING_DIM,
    )
    prediction = predict_ridge(
        ridge_artifact, test[columns].to_numpy(dtype=np.float64)
    )
    predictions = test[["dataset", "subject", "split", "age", "sex", "input_path"]].copy()
    predictions = predictions.rename(columns={"age": "age_true"})
    predictions["age_pred"] = prediction
    predictions["error_years"] = predictions["age_pred"] - predictions["age_true"]
    predictions["absolute_error_years"] = predictions["error_years"].abs()
    metrics = regression_metrics(predictions["age_true"], predictions["age_pred"])
    metrics.update({"split": "internal_test", "bias_correction": "none"})

    result = {
        "pipeline_version": PIPELINE_VERSION,
        "arm": "frozen",
        "model": MODEL_NAME,
        "variant": variant,
        "fold": args.fold,
        "metrics": metrics,
        "training_lock_artifact": str(lock_path),
        "training_lock_sha256": lock_sha256,
        "model_artifact": str(model_path),
        "model_artifact_sha256": lock["model_artifact_sha256"],
        "test_embeddings_artifact": str(test_embeddings_path),
        "test_embeddings_sha256": test_embeddings_sha256,
        "test_embeddings_audit_artifact": str(test_sidecar_path),
        "test_embeddings_audit_sha256": test_sidecar_sha256,
        "manifest": summary,
        "n_predictions": int(len(predictions)),
        "n_subjects": int(
            len(predictions[["dataset", "subject"]].drop_duplicates())
        ),
        "bias_correction": "none",
        "internal_test_accessed": True,
        "on_harmony_accessed": False,
    }
    verify_internal_test_input_integrity(
        lock_path=lock_path,
        expected_lock_sha256=lock_sha256,
        model_path=model_path,
        expected_model_sha256=str(lock["model_artifact_sha256"]),
        embeddings_path=test_embeddings_path,
        expected_embeddings_sha256=test_embeddings_sha256,
        sidecar_path=test_sidecar_path,
        expected_sidecar_sha256=test_sidecar_sha256,
    )
    publish_internal_test_result_bundle(predictions, output_dir, result)
    result_path = output_dir / "metrics_internal_test.json"
    complete_internal_test_receipt(
        receipt,
        started_receipt,
        {
            "arm": "frozen",
            "model": MODEL_NAME,
            "variant": variant,
            "training_lock_artifact": str(lock_path),
            "training_lock_sha256": lock_sha256,
            "test_embeddings_sha256": test_embeddings_sha256,
            "test_embeddings_audit_sha256": test_sidecar_sha256,
            "result": str(result_path),
            "result_sha256": sha256_file(result_path),
            "on_harmony_accessed": False,
        },
    )
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)


def frozen_onharmony_source_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Return the fold-specific contract shared by source audit and prediction."""

    return {
        "embeddings_path": args.embeddings,
        "source_manifest": args.embedding_manifest,
        "filelist": args.filelist,
        "metadata": args.metadata,
        "model_name": MODEL_NAME,
        "variant": variant_for_fold(args.fold),
        "expected_dimension": EMBEDDING_DIM,
        "feature_prefix": "emb_",
        "extractor_pipeline_version": ONHARMONY_EXTRACTOR_PIPELINE_VERSION,
        "readout": READOUT,
    }


def validate_onharmony_precision_contract(
    args: argparse.Namespace, *, require_locked_ridge: bool
) -> None:
    """Bind the AnatCL ON-Harmony source and Ridge to the fixed BF16 protocol."""

    embeddings = Path(args.embeddings).expanduser().resolve()
    manifest_path = (
        Path(args.embedding_manifest).expanduser().resolve()
        if args.embedding_manifest is not None
        else embeddings.with_name(f"{embeddings.stem}_manifest.json")
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("amp") != FROZEN_AMP:
        raise ValueError(
            "AnatCL ON-Harmony embeddings must declare the fixed bfloat16 "
            f"precision contract, observed {payload.get('amp') if isinstance(payload, dict) else None!r}"
        )
    if require_locked_ridge:
        lock_path = Path(args.run_dir).expanduser().resolve() / LOCK_NAME
        if not lock_path.is_file():
            raise FileNotFoundError(lock_path)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        locked_amp = (
            lock.get("extraction_contract", {}).get("amp")
            if isinstance(lock, dict)
            else None
        )
        if locked_amp != FROZEN_AMP:
            raise ValueError(
                "AnatCL ON-Harmony embedding precision differs from the locked "
                f"frozen Ridge: expected {FROZEN_AMP!r}, observed {locked_amp!r}"
            )


def command_audit_onharmony(args: argparse.Namespace) -> None:
    """Validate one fold-specific label-blind ON-Harmony embedding source."""

    validate_onharmony_precision_contract(args, require_locked_ridge=False)
    audit_frozen_onharmony_source(**frozen_onharmony_source_contract(args))


def command_predict_onharmony(args: argparse.Namespace) -> None:
    """Apply one locked AnatCL fold-specific Ridge head to ON-Harmony."""

    validate_onharmony_precision_contract(args, require_locked_ridge=True)
    run_frozen_onharmony_prediction(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        pipeline_version=PIPELINE_VERSION,
        expected_training_manifest_sha256=MANIFEST_SHA256,
        current_fit_code_sha256=fit_code_hashes(),
        **frozen_onharmony_source_contract(args),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit")
    fit.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fit.add_argument("--fold", type=int, choices=range(5), required=True)
    fit.add_argument("--train-embeddings", type=Path, required=True)
    fit.add_argument("--validation-embeddings", type=Path, required=True)
    fit.add_argument("--ridge-alphas", default=ridge_alpha_argument())
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.set_defaults(func=command_fit)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    evaluate.add_argument("--fold", type=int, choices=range(5), required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--test-embeddings", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--acknowledge-locked-test", action="store_true")
    evaluate.set_defaults(func=command_evaluate)

    audit_onharmony = commands.add_parser("audit-onharmony")
    audit_onharmony.add_argument("--fold", type=int, choices=range(5), required=True)
    add_frozen_onharmony_source_arguments(audit_onharmony)
    audit_onharmony.set_defaults(func=command_audit_onharmony)

    predict_onharmony = commands.add_parser("predict-onharmony")
    predict_onharmony.add_argument("--fold", type=int, choices=range(5), required=True)
    add_frozen_onharmony_arguments(predict_onharmony)
    predict_onharmony.set_defaults(func=command_predict_onharmony)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
