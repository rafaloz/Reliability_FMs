#!/usr/bin/env python3
"""Cache inputs and run label-blind ON-Harmony inference with adapted NeuroVFM."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.NeuroVFM.finetune import (  # noqa: E402
    restore_checkpoint,
    runtime_code_hashes,
)
from FineTuning.NeuroVFM.model import (  # noqa: E402
    NEUROVFM_MODEL_REPO,
    NEUROVFM_MODEL_REVISION,
    PREPROCESSING_CACHE_VERSION,
    NeuroVFMModel,
    NeuroVFMOptions,
    load_and_validate_manifest,
    validate_neurovfm_device,
)
from FineTuning.shared.cohort import sha256_file  # noqa: E402
from FineTuning.shared.onharmony import (  # noqa: E402
    ArtifactRegistry,
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    adapted_runtime_migration,
    canonical_sha256,
    load_adapted_model,
    load_onharmony_manifest,
    run_standard_audit,
    run_standard_prediction,
    select_audit_records,
    validate_lock_header,
    validate_run_binding,
)
from FineTuning.shared.runtime import (  # noqa: E402
    atomic_json,
    atomic_tsv,
    package_versions,
)


PIPELINE_VERSION = "neurovfm_onharmony_prediction_v1"
INPUT_CONTRACT_VERSION = "onharmony_neurovfm_study_preprocessor_cache_v1"
DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT / "preprocessings" / "OnHarmonyDataset" / "preproc_NeuroVFM"
)


def external_cache_paths(
    root: Path | str, record: Mapping[str, Any]
) -> tuple[Path, Path]:
    """Keep one official StudyPreprocessor artifact per ON-Harmony acquisition."""

    acquisition = str(record["acquisition_id"]).strip()
    if not acquisition or Path(acquisition).name != acquisition:
        raise ValueError(f"Unsafe ON-Harmony acquisition identifier: {acquisition!r}")
    directory = Path(root).expanduser().resolve() / acquisition
    return directory / "study_preprocessor.pt", directory / "study_preprocessor.json"


class OnHarmonyNeuroVFMModel(NeuroVFMModel):
    """Preserve the locked adapter while redirecting its cache by acquisition."""

    def __init__(self, options: NeuroVFMOptions, external_cache_root: Path | str) -> None:
        super().__init__(options)
        self.external_cache_root = Path(external_cache_root).expanduser().resolve()

    def preprocessing_cache_paths(self, record: Any) -> tuple[Path, Path]:
        return external_cache_paths(self.external_cache_root, record)

    def required_input_paths(self, record: Any) -> list[Path]:
        artifact, sidecar = self.preprocessing_cache_paths(record)
        return [artifact, sidecar]

    def preprocessing_cache_statistics(self) -> dict[str, Any]:
        result = super().preprocessing_cache_statistics()
        result["external_onharmony_root"] = str(self.external_cache_root)
        return result


def artifact_paths(
    model: OnHarmonyNeuroVFMModel, record: Mapping[str, Any]
) -> dict[str, Path]:
    artifact, sidecar = model.preprocessing_cache_paths(record)
    return {
        "input": artifact,
        "cache_metadata": sidecar,
        "raw_t1w": model.input_path(record),
    }


def validate_artifact_set(
    model: OnHarmonyNeuroVFMModel,
    record: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Validate the official cache payload and bind it to the unchanged raw T1w."""

    batch = model.load_cached_preprocessing(record, retain_in_memory=True)
    metadata = json.loads(paths["cache_metadata"].read_text(encoding="utf-8"))
    raw_hash = sha256_file(paths["raw_t1w"])
    official_contract = model.preprocessing_cache_contract()
    expected = {
        "status": "PASS",
        "cache_version": PREPROCESSING_CACHE_VERSION,
        "dataset": "ON-Harmony",
        "subject": str(record["subject"]),
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if metadata.get("preprocessing_contract") != official_contract:
        mismatches["preprocessing_contract"] = {
            "expected": official_contract,
            "observed": metadata.get("preprocessing_contract"),
        }
    raw_source = metadata.get("raw_source", {})
    if raw_source.get("path") != str(record["raw_t1w_path"]):
        mismatches["raw_source.path"] = {
            "expected": str(record["raw_t1w_path"]),
            "observed": raw_source.get("path"),
        }
    if raw_source.get("sha256") != raw_hash:
        mismatches["raw_source.sha256"] = {
            "expected": raw_hash,
            "observed": raw_source.get("sha256"),
        }
    if metadata.get("artifact_sha256") != sha256_file(paths["input"]):
        mismatches["artifact_sha256"] = {
            "expected": sha256_file(paths["input"]),
            "observed": metadata.get("artifact_sha256"),
        }
    if mismatches:
        raise ValueError(f"Invalid ON-Harmony NeuroVFM cache: {mismatches}")
    return {
        "preprocessing_status": "PASS",
        "preprocessing_cache_version": PREPROCESSING_CACHE_VERSION,
        "preprocessing_contract_sha256": official_contract["contract_sha256"],
        "n_tokens": int(batch["img"].shape[0]),
    }


def model_from_locked_run(
    lock: Mapping[str, Any], external_cache_root: Path
) -> OnHarmonyNeuroVFMModel:
    """Reconstruct the exact training adapter while changing only external inputs."""

    locked_args = lock.get("run_config", {}).get("arguments", {})
    adapter = lock.get("run_config", {}).get("model", {})
    training_cache = adapter.get("preprocessing_cache", {}).get("root")
    if training_cache is None:
        raise ValueError("Locked NeuroVFM run lacks its training-cache root")
    return OnHarmonyNeuroVFMModel(
        NeuroVFMOptions(
            model_source=str(locked_args["model_source"]),
            model_revision=str(locked_args["model_revision"]),
            allow_download=bool(locked_args["allow_download"]),
            preprocessing_cache_root=Path(str(training_cache)),
            require_preprocessing_cache=True,
            memory_cache=True,
        ),
        external_cache_root,
    )


def build_input_contract(
    model: OnHarmonyNeuroVFMModel,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    official = model.preprocessing_cache_contract()
    result: dict[str, Any] = {
        "contract_version": INPUT_CONTRACT_VERSION,
        "external_cache_root": str(model.external_cache_root),
        "cache_path_scope": "one artifact per ON-Harmony acquisition",
        "official_study_preprocessor_contract": official,
        "raw_filelist": manifest["filelist"],
        "raw_filelist_sha256": manifest["filelist_sha256"],
    }
    result["contract_sha256"] = canonical_sha256(result)
    return result


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    frame, manifest = load_onharmony_manifest(args.filelist, args.metadata)
    lock, _, _ = validate_lock_header(
        args.run_dir, expected_model="NeuroVFM", expected_variant="canonical"
    )
    model = model_from_locked_run(lock, args.cache_root)
    input_contract = build_input_contract(model, manifest)
    registry = ArtifactRegistry(
        contract_version=INPUT_CONTRACT_VERSION,
        resolver=lambda record: artifact_paths(model, record),
        validator=lambda record, paths: validate_artifact_set(model, record, paths),
        contract=input_contract,
    )
    binding = validate_run_binding(
        run_dir=args.run_dir,
        selection_dir=args.selection_dir,
        model=model,
        training_manifest_loader=load_and_validate_manifest,
        runtime_code_loader=runtime_code_hashes,
        expected_model="NeuroVFM",
        expected_variant="canonical",
        runtime_migration=adapted_runtime_migration(
            model.model_name, lock.get("code_sha256", {})
        ),
        requested_amp=args.amp,
    )
    return {
        "frame": frame,
        "manifest": manifest,
        "registry": registry,
        "model": model,
        "binding": binding,
        "input_contract": input_contract,
    }


def runtime_for(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    binding = context["binding"]
    return load_adapted_model(
        model=context["model"],
        restore_checkpoint=restore_checkpoint,
        checkpoint_path=binding["checkpoint_path"],
        device_name=args.device,
        amp=binding["amp"],
        seed=binding["seed"],
        device_validator=validate_neurovfm_device,
    )


def command_cache(args: argparse.Namespace) -> None:
    """Materialize released StudyPreprocessor batches once for ON-Harmony."""

    frame, manifest = load_onharmony_manifest(args.filelist, args.metadata)
    selected = select_audit_records(
        frame, subject=args.subject, session=args.session
    )
    model = OnHarmonyNeuroVFMModel(
        NeuroVFMOptions(
            model_source=args.model_source,
            model_revision=args.model_revision,
            allow_download=args.allow_download,
            preprocessing_cache_root=args.cache_root,
            require_preprocessing_cache=True,
            memory_cache=False,
        ),
        args.cache_root,
    )
    missing_raw = [
        str(model.input_path(record))
        for record in selected.to_dict(orient="records")
        if not model.input_path(record).is_file()
    ]
    if missing_raw:
        raise FileNotFoundError(
            f"NeuroVFM has {len(missing_raw)} missing raw inputs; first 10: "
            f"{missing_raw[:10]}"
        )
    model.load_preprocessor()
    contract = model.preprocessing_cache_contract()
    release = model.release_provenance()
    code = runtime_code_hashes(model)
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{os.getpid()}"
    run_root = args.cache_root.expanduser().resolve() / "_runs"
    run_manifest = run_root / f"onharmony_cache_run_{run_id}.tsv"
    run_summary = run_root / f"onharmony_cache_run_{run_id}.json"

    print(
        f"NeuroVFM ON-Harmony StudyPreprocessor cache: n={len(selected)}; device=cpu",
        flush=True,
    )
    print(f"  cache root: {model.external_cache_root}", flush=True)
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
        except Exception as error:  # preserve the complete run ledger before failing
            result = {}
            status = "FAILED"
            message = f"{type(error).__name__}: {error}"
        rows.append(
            {
                "subject": str(record["subject"]),
                "session": str(record["session"]),
                "acquisition_id": str(record["acquisition_id"]),
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
            f"{record['acquisition_id']}: {message}",
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
        "operation": "cache_released_study_preprocessor_batches_for_onharmony",
        "label_blind": True,
        "outcome_labels_loaded": False,
        "cache_root": str(model.external_cache_root),
        "selected_acquisitions": int(len(selected)),
        "status_counts": status_counts,
        "elapsed_seconds": float(time.monotonic() - started),
        "cache_contract": contract,
        "on_harmony_manifest": manifest,
        "release": release,
        "code_sha256": code,
        "software": package_versions(),
        "runtime": {
            "device": "cpu",
            "implementation": "released neurovfm StudyPreprocessor",
            "encoder_loaded": False,
            "random_operations": False,
        },
        "run_manifest": str(run_manifest),
    }
    atomic_json(summary, run_summary)
    print(f"Run manifest: {run_manifest}", flush=True)
    print(f"Run summary : {run_summary}", flush=True)
    print(f"Summary     : {status_counts}", flush=True)
    failures = ledger.loc[ledger["status"] == "FAILED"]
    if not failures.empty:
        raise RuntimeError(f"NeuroVFM cache failed for {len(failures)} acquisition(s)")


def command_audit(args: argparse.Namespace) -> None:
    context = build_context(args)
    run_standard_audit(
        args,
        context=context,
        pipeline_version=PIPELINE_VERSION,
        script_path=SCRIPT_PATH,
        runtime_factory=lambda: runtime_for(args, context),
    )


def command_predict(args: argparse.Namespace) -> None:
    context = build_context(args)
    run_standard_prediction(
        args,
        context=context,
        pipeline_version=PIPELINE_VERSION,
        script_path=SCRIPT_PATH,
        runtime_factory=lambda: runtime_for(args, context),
        extra_audit=lambda runtime, predictions: {
            "preprocessing_cache_statistics": (
                context["model"].preprocessing_cache_statistics()
            )
        },
    )


def add_prediction_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--filelist", type=Path, default=DEFAULT_RAW_FILELIST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("bfloat16",))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    cache = commands.add_parser("cache")
    cache.add_argument("--filelist", type=Path, default=DEFAULT_RAW_FILELIST)
    cache.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    cache.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    cache.add_argument("--model-source", default=NEUROVFM_MODEL_REPO)
    cache.add_argument("--model-revision", default=NEUROVFM_MODEL_REVISION)
    cache.add_argument("--allow-download", action="store_true")
    cache.add_argument("--subject")
    cache.add_argument("--session")
    cache.add_argument("--overwrite", action="store_true")
    cache.set_defaults(func=command_cache)

    audit = commands.add_parser("audit")
    add_prediction_common(audit)
    audit.add_argument("--subject")
    audit.add_argument("--session")
    audit.add_argument("--check-input-files", action="store_true")
    audit.add_argument("--inspect-model", action="store_true")
    audit.add_argument("--smoke-forward", action="store_true")
    audit.set_defaults(func=command_audit)

    predict = commands.add_parser("predict")
    add_prediction_common(predict)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.set_defaults(func=command_predict)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
