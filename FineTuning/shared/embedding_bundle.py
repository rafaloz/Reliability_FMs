"""Transactional publication of an embedding table and its audit sidecar."""

from __future__ import annotations

import errno
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from FineTuning.shared.cohort import ALLOWED_SPLITS, sha256_file
from FineTuning.shared.metrics import validate_embedding_frame


PUBLICATION_PROTOCOL = "embedding_sidecar_commit_v1"
EMBEDDING_BUNDLE_SCHEMA_VERSION = "frozen_embedding_bundle_manifest_v1"
COMMITTED_PUBLICATION = {
    "protocol": PUBLICATION_PROTOCOL,
    "status": "COMMITTED",
    "sidecar_is_commit_marker": True,
}


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes when the backing filesystem supports it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            unsupported = {errno.EINVAL, errno.ENOTSUP}
            if hasattr(errno, "EOPNOTSUPP"):
                unsupported.add(errno.EOPNOTSUPP)
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def _is_sha256(value: object) -> bool:
    """Return whether a value is one lowercase or uppercase SHA-256 digest."""

    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _validate_nested_identity(
    field: str,
    observed: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """Bind a nested adapter or boundary description to the bundle identity."""

    expected = {
        key: payload[key]
        for key in ("model", "variant", "readout", "embedding_dim")
    }
    drift = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if drift:
        raise ValueError(
            f"Embedding sidecar semantic identity differs in {field}: {drift}"
        )


def _validate_semantic_payload(payload: Mapping[str, Any]) -> None:
    """Require a complete model/variant/readout identity before publication or use."""

    required = (
        "manifest_schema_version",
        "pipeline_version",
        "arm",
        "model",
        "variant",
        "readout",
        "split",
        "n_subjects",
        "embedding_dim",
        "artifact",
        "elapsed_seconds",
        "adapter",
        "frozen_boundary",
        "manifest",
        "release",
        "code_sha256",
        "software",
        "runtime",
        "internal_test_accessed",
        "internal_test_receipt",
        "on_harmony_accessed",
    )
    missing = [key for key in required if key not in payload]
    missing.extend(
        key
        for key in required
        if key != "internal_test_receipt"
        and key in payload
        and payload[key] in (None, "")
    )
    if missing:
        raise ValueError(f"Embedding sidecar lacks required semantic fields: {missing}")
    for key in ("pipeline_version", "model", "variant", "readout", "split"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"Embedding sidecar field {key!r} must be a non-empty string")
    if payload["manifest_schema_version"] != EMBEDDING_BUNDLE_SCHEMA_VERSION:
        raise ValueError("Embedding sidecar schema version differs")
    if payload["arm"] != "frozen_embedding_extraction":
        raise ValueError("Embedding sidecar arm differs")
    if payload["split"] not in {"train", "validation", "internal_test"}:
        raise ValueError("Embedding sidecar split is not recognized")
    if (
        type(payload["n_subjects"]) is not int
        or type(payload["embedding_dim"]) is not int
        or payload["n_subjects"] <= 0
        or payload["embedding_dim"] <= 0
    ):
        raise ValueError("Embedding sidecar counts must be strictly positive")
    if not isinstance(payload["artifact"], str) or not payload["artifact"].strip():
        raise ValueError("Embedding sidecar artifact path is invalid")
    try:
        elapsed_seconds = float(payload["elapsed_seconds"])
    except (TypeError, ValueError) as error:
        raise ValueError("Embedding sidecar elapsed time is invalid") from error
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("Embedding sidecar elapsed time is invalid")
    for key in ("adapter", "frozen_boundary", "release", "software", "runtime"):
        if not isinstance(payload[key], Mapping) or not payload[key]:
            raise ValueError(f"Embedding sidecar field {key!r} is incomplete")
    adapter = payload["adapter"]
    frozen_boundary = payload["frozen_boundary"]
    _validate_nested_identity("adapter", adapter, payload)
    _validate_nested_identity("frozen boundary", frozen_boundary, payload)
    if (
        type(frozen_boundary.get("encoder_total_parameters")) is not int
        or frozen_boundary["encoder_total_parameters"] <= 0
    ):
        raise ValueError("Embedding sidecar frozen boundary has an invalid encoder size")
    if (
        type(frozen_boundary.get("encoder_trainable_parameters")) is not int
        or frozen_boundary["encoder_trainable_parameters"] != 0
    ):
        raise ValueError("Embedding sidecar frozen boundary contains trainable parameters")
    trainable_fraction = frozen_boundary.get("encoder_trainable_fraction")
    if (
        type(trainable_fraction) not in (int, float)
        or not math.isfinite(float(trainable_fraction))
        or float(trainable_fraction) != 0.0
    ):
        raise ValueError("Embedding sidecar frozen boundary has a trainable fraction")
    for key in ("trainable_components", "trainable_parameter_names"):
        if frozen_boundary.get(key) != []:
            raise ValueError(f"Embedding sidecar frozen boundary has non-empty {key}")
    code_sha256 = payload["code_sha256"]
    if not isinstance(code_sha256, Mapping) or not code_sha256 or any(
        not isinstance(path, str) or not path.strip() or not _is_sha256(digest)
        for path, digest in code_sha256.items()
    ):
        raise ValueError("Embedding sidecar code hashes are incomplete")
    runtime = payload["runtime"]
    if runtime.get("input_hashes_verified") is not True:
        raise ValueError("Embedding sidecar did not verify its model inputs")
    if type(payload["internal_test_accessed"]) is not bool:
        raise ValueError("Embedding sidecar internal-test flag is invalid")
    expected_internal_test_access = payload["split"] == "internal_test"
    if payload["internal_test_accessed"] is not expected_internal_test_access:
        raise ValueError("Embedding sidecar internal-test flag differs from its split")
    if payload["on_harmony_accessed"] is not False:
        raise ValueError("Frozen embedding extraction must not access ON-Harmony")
    receipt = payload["internal_test_receipt"]
    if expected_internal_test_access:
        if not isinstance(receipt, Mapping):
            raise ValueError("Internal-test embedding sidecar lacks its reservation")
        receipt_path = receipt.get("path")
        attempt_id = receipt.get("attempt_id")
        if (
            not isinstance(receipt_path, str)
            or not receipt_path.strip()
            or not Path(receipt_path).expanduser().is_absolute()
            or not isinstance(attempt_id, str)
            or len(attempt_id) != 32
            or any(character not in "0123456789abcdef" for character in attempt_id)
        ):
            raise ValueError("Internal-test embedding sidecar has an invalid reservation")
    elif receipt is not None:
        raise ValueError("Development embedding sidecar contains an internal-test receipt")
    manifest = payload["manifest"]
    required_manifest_fields = (
        "manifest",
        "manifest_sha256",
        "n_subjects",
        "split_counts",
        "split_dataset_counts",
        "age_min",
        "age_max",
        "input_contract",
        "raw_input_files_checked",
        "missing_raw_input_files",
        "on_harmony_firewall",
    )
    if not isinstance(manifest, Mapping):
        raise ValueError("Embedding sidecar lacks the authoritative cohort manifest")
    missing_manifest = [key for key in required_manifest_fields if key not in manifest]
    if missing_manifest:
        raise ValueError(
            "Embedding sidecar cohort manifest is incomplete: "
            f"{missing_manifest}"
        )
    if (
        not isinstance(manifest["manifest"], str)
        or not manifest["manifest"].strip()
        or not Path(manifest["manifest"]).expanduser().is_absolute()
    ):
        raise ValueError("Embedding sidecar cohort manifest path is invalid")
    if not _is_sha256(manifest["manifest_sha256"]):
        raise ValueError("Embedding sidecar cohort manifest SHA-256 is invalid")
    if manifest["on_harmony_firewall"] != "PASS":
        raise ValueError("Embedding sidecar cohort firewall did not pass")
    if type(manifest["n_subjects"]) is not int or manifest["n_subjects"] <= 0:
        raise ValueError("Embedding sidecar cohort subject count is invalid")
    if not isinstance(manifest["split_counts"], Mapping) or any(
        not isinstance(key, str)
        or type(value) is not int
        or value < 0
        for key, value in manifest["split_counts"].items()
    ):
        raise ValueError("Embedding sidecar cohort split counts are invalid")
    if set(manifest["split_counts"]) != set(ALLOWED_SPLITS):
        raise ValueError("Embedding sidecar cohort does not contain exactly three splits")
    if sum(manifest["split_counts"].values()) != manifest["n_subjects"]:
        raise ValueError("Embedding sidecar cohort split counts do not sum to its total")
    split_dataset_counts = manifest["split_dataset_counts"]
    if not isinstance(split_dataset_counts, Mapping):
        raise ValueError("Embedding sidecar cohort dataset counts are invalid")
    if set(split_dataset_counts) != set(ALLOWED_SPLITS):
        raise ValueError("Embedding sidecar cohort split/dataset keys differ")
    for split, split_count in manifest["split_counts"].items():
        dataset_counts = split_dataset_counts.get(split)
        if not isinstance(dataset_counts, Mapping) or any(
            not isinstance(dataset, str)
            or not dataset.strip()
            or type(count) is not int
            or count <= 0
            for dataset, count in dataset_counts.items()
        ):
            raise ValueError("Embedding sidecar cohort dataset counts are invalid")
        if sum(dataset_counts.values()) != split_count:
            raise ValueError(
                "Embedding sidecar cohort dataset counts do not sum to their split"
            )
    age_min = manifest["age_min"]
    age_max = manifest["age_max"]
    if type(age_min) not in (int, float) or type(age_max) not in (int, float):
        raise ValueError("Embedding sidecar cohort age range is invalid")
    if not math.isfinite(age_min) or not math.isfinite(age_max) or age_min > age_max:
        raise ValueError("Embedding sidecar cohort age range is invalid")
    input_contract = manifest["input_contract"]
    permitted_contracts = {"required_offline_group", "required_runtime_condition"}
    if (
        not isinstance(input_contract, Mapping)
        or len(input_contract) != 1
        or not set(input_contract).issubset(permitted_contracts)
        or any(
            not isinstance(value, str) or not value.strip()
            for value in input_contract.values()
        )
    ):
        raise ValueError("Embedding sidecar cohort input contract is invalid")
    if adapter.get("input_contract") != input_contract:
        raise ValueError("Embedding sidecar adapter/cohort input contracts differ")
    if type(manifest["raw_input_files_checked"]) is not bool:
        raise ValueError("Embedding sidecar raw-input audit flag is invalid")
    if (
        type(manifest["missing_raw_input_files"]) is not int
        or manifest["missing_raw_input_files"] < 0
        or (
            manifest["raw_input_files_checked"]
            and manifest["missing_raw_input_files"] != 0
        )
    ):
        raise ValueError("Embedding sidecar raw-input audit count is invalid")
    if manifest["split_counts"].get(payload["split"]) != payload["n_subjects"]:
        raise ValueError("Embedding sidecar row count differs from its cohort split")


def ensure_new_embedding_bundle(
    output: Path | str, sidecar: Path | str
) -> Path:
    """Reject complete or interrupted bundles before expensive model execution."""

    destination = Path(output).resolve()
    audit = Path(sidecar).resolve()
    if destination == audit:
        raise ValueError("Embedding table and sidecar must use different paths")
    if destination.parent != audit.parent:
        raise ValueError("Embedding table and sidecar must share one directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() != audit.exists():
        raise RuntimeError(
            "Incomplete embedding bundle detected; table and audit sidecar must either "
            f"both exist or both be absent: {destination}, {audit}"
        )
    if destination.exists():
        raise FileExistsError(f"Embedding bundle already exists: {destination}, {audit}")
    return destination


def load_embedding_bundle_sidecar(
    output: Path | str,
    sidecar: Path | str,
    *,
    expected_pipeline_version: str,
    expected_model: str,
    expected_variant: str,
    expected_readout: str,
    expected_split: str,
    expected_dimension: int,
) -> dict[str, Any]:
    """Load a committed embedding sidecar and bind it to its table and pipeline."""

    expected_strings = {
        "pipeline_version": expected_pipeline_version,
        "model": expected_model,
        "variant": expected_variant,
        "readout": expected_readout,
    }
    invalid_strings = [
        key
        for key, value in expected_strings.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid_strings:
        raise ValueError(f"Invalid expected embedding identity fields: {invalid_strings}")
    if expected_split not in ALLOWED_SPLITS:
        raise ValueError(f"Invalid expected embedding split: {expected_split!r}")
    if type(expected_dimension) is not int or expected_dimension <= 0:
        raise ValueError(f"Invalid expected embedding dimension: {expected_dimension!r}")

    artifact = Path(output).resolve()
    audit = Path(sidecar).resolve()
    if artifact == audit:
        raise ValueError("Embedding table and sidecar must use different paths")
    if artifact.parent != audit.parent:
        raise ValueError("Embedding table and sidecar must share one directory")
    if not artifact.is_file() or not audit.is_file():
        raise FileNotFoundError(f"Expected both {artifact} and {audit}")
    artifact_sha256_before = sha256_file(artifact)
    audit_sha256_before = sha256_file(audit)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    audit_sha256_after = sha256_file(audit)
    artifact_sha256_after = sha256_file(artifact)
    if audit_sha256_after != audit_sha256_before:
        raise RuntimeError("Embedding sidecar changed while it was being read")
    if artifact_sha256_after != artifact_sha256_before:
        raise RuntimeError("Embedding table changed while its sidecar was being read")
    if not isinstance(payload, dict):
        raise TypeError(f"Embedding audit sidecar is not a JSON object: {audit}")
    _validate_semantic_payload(payload)
    if payload.get("publication") != COMMITTED_PUBLICATION:
        raise ValueError(f"Embedding sidecar is not a committed bundle: {audit}")
    if payload.get("pipeline_version") != expected_pipeline_version:
        raise ValueError(
            "Embedding sidecar pipeline version differs: "
            f"expected={expected_pipeline_version!r}, "
            f"observed={payload.get('pipeline_version')!r}"
        )
    expected_semantics = {
        "model": expected_model,
        "variant": expected_variant,
        "readout": expected_readout,
        "split": expected_split,
        "embedding_dim": expected_dimension,
    }
    semantic_drift = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected_semantics.items()
        if payload.get(key) != value
    }
    if semantic_drift:
        raise ValueError(f"Embedding sidecar semantic identity differs: {semantic_drift}")
    if Path(str(payload.get("artifact", ""))).expanduser().resolve() != artifact:
        raise ValueError(f"Embedding sidecar points to a different table: {artifact}")
    if payload.get("artifact_sha256") != artifact_sha256_after:
        raise ValueError(f"Embedding SHA-256 differs from its audit sidecar: {artifact}")
    return payload


def publish_embedding_bundle(
    frame: pd.DataFrame,
    output: Path | str,
    sidecar: Path | str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Stage and commit a TSV plus sidecar without overwriting existing artifacts.

    The table is linked first and the sidecar last. Therefore, the sidecar is the
    commit marker consumed by downstream readers. Catchable failures roll back both
    links; a hard interruption leaves a detectable incomplete bundle.
    """

    _validate_semantic_payload(payload)
    embedding_columns = validate_embedding_frame(
        frame,
        expected_model=payload["model"],
        expected_variant=payload["variant"],
    )
    if len(embedding_columns) != payload["embedding_dim"]:
        raise ValueError(
            "Embedding table dimension differs from its sidecar: "
            f"table={len(embedding_columns)}, sidecar={payload['embedding_dim']}"
        )
    observed_splits = set(frame["split"].astype(str))
    if observed_splits != {payload["split"]}:
        raise ValueError(
            "Embedding table split differs from its sidecar: "
            f"table={sorted(observed_splits)}, sidecar={payload['split']!r}"
        )
    if len(frame) != payload["n_subjects"]:
        raise ValueError(
            "Embedding table row count differs from its sidecar: "
            f"table={len(frame)}, sidecar={payload['n_subjects']}"
        )
    destination = ensure_new_embedding_bundle(output, sidecar)
    audit = Path(sidecar).resolve()
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.bundle.",
            dir=destination.parent,
        )
    )
    staged_table = staging / "embeddings.tsv"
    staged_audit = staging / "audit.json"
    table_published = False
    audit_published = False
    try:
        with staged_table.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, sep="\t", index=False)
            handle.flush()
            os.fsync(handle.fileno())

        committed = dict(payload)
        committed["artifact"] = str(destination)
        committed["artifact_sha256"] = sha256_file(staged_table)
        committed["publication"] = dict(COMMITTED_PUBLICATION)
        with staged_audit.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(committed, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        # Hard links provide atomic no-overwrite publication on the shared filesystem.
        os.link(staged_table, destination)
        table_published = True
        os.link(staged_audit, audit)
        audit_published = True
        _fsync_directory(destination.parent)
        published = load_embedding_bundle_sidecar(
            destination,
            audit,
            expected_pipeline_version=payload["pipeline_version"],
            expected_model=payload["model"],
            expected_variant=payload["variant"],
            expected_readout=payload["readout"],
            expected_split=payload["split"],
            expected_dimension=payload["embedding_dim"],
        )
        if published != json.loads(staged_audit.read_text(encoding="utf-8")):
            raise RuntimeError("Published embedding sidecar differs from its staged source")
        return published
    except BaseException:
        if audit_published:
            audit.unlink(missing_ok=True)
        if table_published:
            destination.unlink(missing_ok=True)
        if audit_published or table_published:
            _fsync_directory(destination.parent)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
