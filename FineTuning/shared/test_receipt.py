"""Exclusive, crash-visible receipts for one-time internal-test operations."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from FineTuning.shared.cohort import PROJECT_ROOT, sha256_file
from FineTuning.shared.metrics import regression_metrics


STARTED_STATUS = "INTERNAL_TEST_EVALUATION_STARTED"
COMPLETED_STATUS = "INTERNAL_TEST_EVALUATED"
EXTRACTION_STARTED_STATUS = "INTERNAL_TEST_EMBEDDING_EXTRACTION_STARTED"
EXTRACTION_COMPLETED_STATUS = "INTERNAL_TEST_EMBEDDINGS_EXTRACTED"
EMBEDDING_RECEIPT_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "brain_age"
    / "internal_test_receipts"
    / "frozen_embeddings"
)
RESULT_BUNDLE_PROTOCOL = "internal_test_result_directory_v2"
INTERNAL_TEST_PREDICTION_COLUMNS = (
    "dataset",
    "subject",
    "split",
    "age_true",
    "sex",
    "input_path",
    "age_pred",
    "error_years",
    "absolute_error_years",
)
INTERNAL_TEST_TEXT_COLUMNS = (
    "dataset",
    "subject",
    "split",
    "sex",
    "input_path",
)
RESULT_INPUT_FIELDS = (
    ("LOCKED.json", "training_lock_artifact", "training_lock_sha256"),
    ("Ridge model", "model_artifact", "model_artifact_sha256"),
    ("embedding table", "test_embeddings_artifact", "test_embeddings_sha256"),
    (
        "embedding sidecar",
        "test_embeddings_audit_artifact",
        "test_embeddings_audit_sha256",
    ),
)
RECEIPT_SYSTEM_FIELDS = frozenset(
    {
        "status",
        "operation",
        "attempt_id",
        "started_at_utc",
        "completed_at_utc",
        "internal_test_accessed",
    }
)
EVALUATION_STARTED_FIELDS = frozenset(
    {
        "pipeline_version",
        "arm",
        "model",
        "variant",
        "run_dir",
        "training_lock_artifact",
        "training_lock_sha256",
        "model_artifact",
        "model_artifact_sha256",
        "manifest_artifact",
        "manifest_sha256",
        "test_embeddings_artifact",
        "test_embeddings_audit_artifact",
        "output_dir",
        "on_harmony_accessed",
    }
)
EVALUATION_COMPLETION_FIELDS = frozenset(
    {
        "arm",
        "model",
        "variant",
        "training_lock_artifact",
        "training_lock_sha256",
        "test_embeddings_sha256",
        "test_embeddings_audit_sha256",
        "result",
        "result_sha256",
        "on_harmony_accessed",
    }
)
EXTRACTION_STARTED_FIELDS = frozenset(
    {
        "pipeline_version",
        "model",
        "variant",
        "readout",
        "embedding_dim",
        "split",
        "manifest_artifact",
        "manifest_sha256",
        "requested_artifact",
        "requested_sidecar",
        "requested_n_subjects",
        "on_harmony_accessed",
    }
)
EXTRACTION_COMPLETION_FIELDS = frozenset(
    {
        "artifact",
        "artifact_sha256",
        "sidecar",
        "sidecar_sha256",
        "n_subjects",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_receipt_component(value: str) -> str:
    """Accept one canonical filename component without lossy normalization."""

    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None
    ):
        raise ValueError(f"Invalid internal-test receipt label: {value!r}")
    return value


def _is_sha256(value: object) -> bool:
    """Return whether a value is one canonical lowercase SHA-256 digest."""

    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes when supported by the filesystem."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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


def _resolve_regular_file(label: str, value: Path | str) -> Path:
    """Resolve one declared input while rejecting relative or non-file paths."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Internal-test {label} path must be absolute: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Internal-test {label} is not a regular file: {resolved}"
        )
    return resolved


def _validate_predictions(predictions: Any) -> pd.DataFrame:
    """Validate the exact subject-level internal-test prediction schema."""

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("Internal-test predictions must be a pandas DataFrame")
    if predictions.empty:
        raise ValueError("Internal-test predictions are empty")
    observed_columns = tuple(predictions.columns)
    if observed_columns != INTERNAL_TEST_PREDICTION_COLUMNS:
        raise ValueError(
            "Internal-test prediction columns differ: "
            f"expected={INTERNAL_TEST_PREDICTION_COLUMNS}, observed={observed_columns}"
        )
    invalid_text = [
        column
        for column in INTERNAL_TEST_TEXT_COLUMNS
        if predictions[column].map(
            lambda value: not isinstance(value, str) or not value.strip()
        ).any()
    ]
    if invalid_text:
        raise ValueError(
            f"Internal-test predictions contain invalid text fields: {invalid_text}"
        )
    if set(predictions["split"]) != {"internal_test"}:
        raise ValueError("Internal-test predictions contain a non-test split")
    if predictions.duplicated(["dataset", "subject"]).any():
        raise ValueError("Internal-test predictions contain duplicate subject keys")
    if not predictions["input_path"].map(
        lambda value: Path(value).expanduser().is_absolute()
    ).all():
        raise ValueError("Internal-test prediction input paths must be absolute")
    firewall_values = predictions[["dataset", "subject", "input_path"]].map(
        lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    )
    if firewall_values.apply(
        lambda column: column.str.contains("onharmony", regex=False)
    ).any().any():
        raise ValueError("ON-Harmony firewall violation in internal-test predictions")

    numeric_columns = (
        "age_true",
        "age_pred",
        "error_years",
        "absolute_error_years",
    )
    numeric = predictions[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Internal-test predictions contain non-finite numeric values")
    expected_error = numeric["age_pred"].to_numpy() - numeric["age_true"].to_numpy()
    if not np.allclose(
        numeric["error_years"].to_numpy(), expected_error, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("Internal-test signed errors differ from prediction minus age")
    if not np.allclose(
        numeric["absolute_error_years"].to_numpy(),
        np.abs(expected_error),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Internal-test absolute errors differ from signed errors")
    return predictions


def _validate_result_payload(
    payload: Mapping[str, Any], predictions: pd.DataFrame
) -> None:
    """Bind the result record to its predictions, cohort, and frozen inputs."""

    required = {
        "pipeline_version",
        "arm",
        "model",
        "variant",
        "metrics",
        "manifest",
        "n_predictions",
        "n_subjects",
        "bias_correction",
        "internal_test_accessed",
        "on_harmony_accessed",
    }
    required.update(path_field for _, path_field, _ in RESULT_INPUT_FIELDS)
    required.update(hash_field for _, _, hash_field in RESULT_INPUT_FIELDS)
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Internal-test result payload lacks fields: {missing}")
    for field in ("pipeline_version", "model", "variant"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"Internal-test result field {field!r} is invalid")
    if payload["arm"] != "frozen":
        raise ValueError("Internal-test result is not from the frozen Ridge arm")
    if payload["bias_correction"] != "none":
        raise ValueError("Internal-test result applies an undeclared bias correction")
    if payload["internal_test_accessed"] is not True:
        raise ValueError("Internal-test result does not declare test access")
    if payload["on_harmony_accessed"] is not False:
        raise ValueError("Internal-test result declares ON-Harmony access")

    n_predictions = len(predictions)
    n_subjects = len(predictions[["dataset", "subject"]].drop_duplicates())
    if (
        type(payload["n_predictions"]) is not int
        or payload["n_predictions"] != n_predictions
    ):
        raise ValueError("Internal-test prediction count differs from its table")
    if type(payload["n_subjects"]) is not int or payload["n_subjects"] != n_subjects:
        raise ValueError("Internal-test subject count differs from its table")

    manifest = payload["manifest"]
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("on_harmony_firewall") != "PASS"
    ):
        raise ValueError("Internal-test result lacks a passing cohort firewall")
    if not _is_sha256(manifest.get("manifest_sha256")):
        raise ValueError("Internal-test cohort manifest SHA-256 is invalid")
    manifest_path = _resolve_regular_file(
        "cohort manifest", manifest.get("manifest", "")
    )
    if sha256_file(manifest_path) != manifest["manifest_sha256"]:
        raise RuntimeError("Internal-test cohort manifest changed before publication")
    split_counts = manifest.get("split_counts")
    if (
        not isinstance(split_counts, Mapping)
        or split_counts.get("internal_test") != n_predictions
    ):
        raise ValueError(
            "Internal-test prediction count differs from the cohort manifest"
        )

    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("Internal-test metrics are not a mapping")
    expected_metrics = regression_metrics(
        predictions["age_true"], predictions["age_pred"]
    )
    if (
        metrics.get("split") != "internal_test"
        or metrics.get("bias_correction") != "none"
    ):
        raise ValueError("Internal-test metrics have an invalid split or correction")
    if type(metrics.get("n")) is not int or metrics["n"] != n_predictions:
        raise ValueError("Internal-test metric count differs from its predictions")
    for field in ("mae_years", "rmse_years", "r2", "mean_error_years"):
        observed = metrics.get(field)
        if (
            type(observed) not in (int, float)
            or not np.isfinite(float(observed))
            or not np.isclose(
                float(observed), float(expected_metrics[field]), rtol=1e-12, atol=1e-12
            )
        ):
            raise ValueError(f"Internal-test metric {field!r} differs from predictions")

    for label, path_field, hash_field in RESULT_INPUT_FIELDS:
        if not isinstance(payload[path_field], str):
            raise ValueError(f"Internal-test {label} artifact path is invalid")
    verify_internal_test_input_integrity(
        lock_path=payload["training_lock_artifact"],
        expected_lock_sha256=payload["training_lock_sha256"],
        model_path=payload["model_artifact"],
        expected_model_sha256=payload["model_artifact_sha256"],
        embeddings_path=payload["test_embeddings_artifact"],
        expected_embeddings_sha256=payload["test_embeddings_sha256"],
        sidecar_path=payload["test_embeddings_audit_artifact"],
        expected_sidecar_sha256=payload["test_embeddings_audit_sha256"],
    )


def _reservation_path(destination: Path) -> Path:
    """Return the fixed sibling lock used to serialize result publication."""

    return destination.with_name(f".{destination.name}.internal-test.lock")


def _reserve_result_destination(destination: Path) -> Path:
    """Exclusively reserve a final directory across cooperating processes."""

    reservation = _reservation_path(destination)
    try:
        descriptor = os.open(
            reservation,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"Internal-test output path is already reserved: {destination}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "destination": str(destination),
                        "pid": os.getpid(),
                        "reserved_at_utc": _utc_now(),
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(destination.parent)
        return reservation
    except BaseException:
        reservation.unlink(missing_ok=True)
        raise


def _release_result_destination(reservation: Path) -> None:
    """Remove a publication reservation and persist that directory change."""

    reservation.unlink(missing_ok=True)
    _fsync_directory(reservation.parent)


def internal_test_embedding_receipt_path(model: str, variant: str) -> Path:
    """Return the project-wide one-time receipt for one frozen representation."""

    model_key = _safe_receipt_component(model)
    variant_key = _safe_receipt_component(variant)
    return EMBEDDING_RECEIPT_ROOT / f"{model_key}__{variant_key}.json"


def prepare_internal_test_output_dir(path: Path | str) -> Path:
    """Validate an absent final path without exposing an empty result directory."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Internal-test output path already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def verify_internal_test_input_integrity(
    *,
    lock_path: Path | str,
    expected_lock_sha256: str,
    model_path: Path | str,
    expected_model_sha256: str,
    embeddings_path: Path | str,
    expected_embeddings_sha256: str,
    sidecar_path: Path | str,
    expected_sidecar_sha256: str,
) -> None:
    """Reject drift in any frozen input immediately before result publication."""

    inputs = (
        ("LOCKED.json", lock_path, expected_lock_sha256),
        ("Ridge model", model_path, expected_model_sha256),
        ("embedding table", embeddings_path, expected_embeddings_sha256),
        ("embedding sidecar", sidecar_path, expected_sidecar_sha256),
    )
    resolved: list[Path] = []
    for label, path, expected_sha256 in inputs:
        if not _is_sha256(expected_sha256):
            raise ValueError(f"Internal-test {label} SHA-256 is invalid")
        artifact = _resolve_regular_file(label, path)
        resolved.append(artifact)
        if sha256_file(artifact) != expected_sha256:
            raise RuntimeError(
                f"Internal-test {label} changed before result publication"
            )
    if len(set(resolved)) != len(resolved):
        raise ValueError("Internal-test integrity inputs alias the same artifact")


def publish_internal_test_result_bundle(
    predictions: Any,
    output_dir: Path | str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and atomically publish one immutable internal-test result bundle."""

    if not isinstance(payload, Mapping):
        raise TypeError("Internal-test result payload must be a mapping")
    predictions = _validate_predictions(predictions)
    _validate_result_payload(payload, predictions)
    protected = {
        "predictions_artifact",
        "predictions_artifact_sha256",
        "publication",
    }
    collisions = sorted(protected & set(payload))
    if collisions:
        raise ValueError(
            f"Internal-test result payload overrides protected fields: {collisions}"
        )
    destination = prepare_internal_test_output_dir(output_dir)
    reservation = _reserve_result_destination(destination)
    staging: Path | None = None
    published = False

    try:
        if destination.exists():
            raise FileExistsError(
                f"Internal-test output path appeared during reservation: {destination}"
            )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.internal-test.",
                dir=destination.parent,
            )
        )
        staged_predictions = staging / "predictions_internal_test.tsv"
        staged_metrics = staging / "metrics_internal_test.json"
        with staged_predictions.open("w", encoding="utf-8", newline="") as handle:
            predictions.to_csv(handle, sep="\t", index=False)
            handle.flush()
            os.fsync(handle.fileno())

        committed = {
            **dict(payload),
            "predictions_artifact": str(
                destination / "predictions_internal_test.tsv"
            ),
            "predictions_artifact_sha256": sha256_file(staged_predictions),
            "publication": {
                "protocol": RESULT_BUNDLE_PROTOCOL,
                "status": "COMMITTED",
                "atomic_directory_promotion": True,
                "exclusive_destination_reservation": True,
                "post_publication_verified": True,
            },
        }
        with staged_metrics.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(committed, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        if destination.exists():
            raise FileExistsError(
                f"Internal-test output path appeared during evaluation: {destination}"
            )
        os.rename(staging, destination)
        published = True
        _fsync_directory(destination.parent)

        predictions_path = destination / staged_predictions.name
        metrics_path = destination / staged_metrics.name
        observed_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if observed_metrics != committed:
            raise RuntimeError("Published internal-test metrics failed verification")
        if sha256_file(predictions_path) != committed["predictions_artifact_sha256"]:
            raise RuntimeError(
                "Published internal-test predictions failed hash verification"
            )
        observed_predictions = pd.read_csv(predictions_path, sep="\t")
        _validate_predictions(observed_predictions)
        pd.testing.assert_frame_equal(
            observed_predictions.reset_index(drop=True),
            predictions.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        return observed_metrics
    except BaseException:
        if published:
            shutil.rmtree(destination, ignore_errors=True)
            _fsync_directory(destination.parent)
        elif staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        _release_result_destination(reservation)


def _validate_exact_fields(
    label: str, payload: Mapping[str, Any], expected: frozenset[str]
) -> None:
    """Reject missing and undeclared fields in one receipt transition."""

    observed = set(payload)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            f"Internal-test {label} fields differ: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _canonical_declared_path(label: str, value: object) -> Path:
    """Require an absolute, already canonical path without dereferencing its data."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Internal-test {label} path is invalid")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Internal-test {label} path must be absolute: {path}")
    resolved = path.resolve()
    if str(path) != str(resolved):
        raise ValueError(f"Internal-test {label} path is not canonical: {path}")
    return resolved


def _require_text_fields(
    label: str, payload: Mapping[str, Any], fields: tuple[str, ...]
) -> None:
    """Require non-empty string identity fields in a receipt payload."""

    invalid = [
        field
        for field in fields
        if not isinstance(payload.get(field), str) or not payload[field].strip()
    ]
    if invalid:
        raise ValueError(f"Internal-test {label} has invalid text fields: {invalid}")


def _verify_declared_file(
    label: str, path_value: object, expected_sha256: object
) -> Path:
    """Validate one canonical file path and bind it to a declared digest."""

    path = _canonical_declared_path(label, path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Internal-test {label} is not a regular file: {path}")
    if not _is_sha256(expected_sha256):
        raise ValueError(f"Internal-test {label} SHA-256 is invalid")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Internal-test {label} changed before receipt transition")
    return path


def _read_stable_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read one JSON object while detecting mutation during the read."""

    digest_before = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Internal-test {label} is invalid JSON: {path}") from error
    digest_after = sha256_file(path)
    if digest_after != digest_before:
        raise RuntimeError(f"Internal-test {label} changed while it was being read")
    if not isinstance(payload, dict):
        raise TypeError(f"Internal-test {label} is not a JSON object: {path}")
    return payload


def _validate_started_payload(
    payload: Mapping[str, Any], *, operation: str, receipt: Path
) -> dict[str, Any]:
    """Validate the immutable identity captured before internal-test access."""

    if not isinstance(payload, Mapping):
        raise TypeError("Internal-test STARTED payload must be a mapping")
    data = dict(payload)
    collisions = sorted(RECEIPT_SYSTEM_FIELDS & set(data))
    if collisions:
        raise ValueError(
            f"Internal-test STARTED payload overrides protected fields: {collisions}"
        )
    if operation == "evaluation":
        _validate_exact_fields("evaluation STARTED payload", data, EVALUATION_STARTED_FIELDS)
        _require_text_fields(
            "evaluation STARTED payload",
            data,
            ("pipeline_version", "model", "variant"),
        )
        if data["arm"] != "frozen" or data["on_harmony_accessed"] is not False:
            raise ValueError("Internal-test evaluation STARTED identity is invalid")
        run_dir = _canonical_declared_path("run directory", data["run_dir"])
        if not run_dir.is_dir() or receipt.parent != run_dir:
            raise ValueError("Internal-test receipt and frozen run directory differ")
        lock = _verify_declared_file(
            "LOCKED.json",
            data["training_lock_artifact"],
            data["training_lock_sha256"],
        )
        model = _verify_declared_file(
            "Ridge model",
            data["model_artifact"],
            data["model_artifact_sha256"],
        )
        manifest = _verify_declared_file(
            "cohort manifest",
            data["manifest_artifact"],
            data["manifest_sha256"],
        )
        if lock.parent != run_dir or model.parent != run_dir:
            raise ValueError("Frozen lock/model artifacts are outside their run directory")
        if len({lock, model, manifest}) != 3:
            raise ValueError("Internal-test STARTED inputs alias distinct artifacts")
        embeddings = _canonical_declared_path(
            "test embeddings", data["test_embeddings_artifact"]
        )
        sidecar = _canonical_declared_path(
            "test embedding sidecar", data["test_embeddings_audit_artifact"]
        )
        if embeddings == sidecar or embeddings.parent != sidecar.parent:
            raise ValueError("Test embeddings and sidecar paths are inconsistent")
        output_dir = _canonical_declared_path("output directory", data["output_dir"])
        if output_dir.exists():
            raise FileExistsError(
                f"Internal-test output path already exists: {output_dir}"
            )
    elif operation == "embedding extraction":
        _validate_exact_fields(
            "embedding-extraction STARTED payload",
            data,
            EXTRACTION_STARTED_FIELDS,
        )
        _require_text_fields(
            "embedding-extraction STARTED payload",
            data,
            ("pipeline_version", "model", "variant", "readout", "split"),
        )
        if data["split"] != "internal_test" or data["on_harmony_accessed"] is not False:
            raise ValueError("Internal-test extraction STARTED identity is invalid")
        if type(data["embedding_dim"]) is not int or data["embedding_dim"] <= 0:
            raise ValueError("Internal-test extraction embedding dimension is invalid")
        if (
            type(data["requested_n_subjects"]) is not int
            or data["requested_n_subjects"] <= 0
        ):
            raise ValueError("Internal-test extraction subject count is invalid")
        _verify_declared_file(
            "cohort manifest",
            data["manifest_artifact"],
            data["manifest_sha256"],
        )
        artifact = _canonical_declared_path(
            "requested embedding artifact", data["requested_artifact"]
        )
        sidecar = _canonical_declared_path(
            "requested embedding sidecar", data["requested_sidecar"]
        )
        if artifact == sidecar or artifact.parent != sidecar.parent:
            raise ValueError("Requested embedding artifact/sidecar paths are inconsistent")
        if artifact.exists() or sidecar.exists():
            raise FileExistsError("Requested internal-test embedding bundle already exists")
    else:
        raise ValueError(f"Unknown internal-test receipt operation: {operation!r}")
    return data


def _validate_evaluation_completion(
    completion: Mapping[str, Any], started: Mapping[str, Any]
) -> None:
    """Verify the complete frozen-test result and all artifacts it references."""

    verify_internal_test_input_integrity(
        lock_path=started["training_lock_artifact"],
        expected_lock_sha256=started["training_lock_sha256"],
        model_path=started["model_artifact"],
        expected_model_sha256=started["model_artifact_sha256"],
        embeddings_path=started["test_embeddings_artifact"],
        expected_embeddings_sha256=completion["test_embeddings_sha256"],
        sidecar_path=started["test_embeddings_audit_artifact"],
        expected_sidecar_sha256=completion["test_embeddings_audit_sha256"],
    )
    _verify_declared_file(
        "cohort manifest",
        started["manifest_artifact"],
        started["manifest_sha256"],
    )
    result_path = _verify_declared_file(
        "internal-test metrics", completion["result"], completion["result_sha256"]
    )
    output_dir = _canonical_declared_path("output directory", started["output_dir"])
    if result_path != output_dir / "metrics_internal_test.json":
        raise ValueError("Internal-test metrics path differs from the reserved output")
    result = _read_stable_json_object(result_path, "metrics result")
    predictions_path = _resolve_regular_file(
        "published predictions", result.get("predictions_artifact", "")
    )
    if predictions_path != output_dir / "predictions_internal_test.tsv":
        raise ValueError("Internal-test prediction path differs from the reserved output")
    predictions_sha256_before = sha256_file(predictions_path)
    predictions = pd.read_csv(predictions_path, sep="\t")
    predictions_sha256_after = sha256_file(predictions_path)
    if predictions_sha256_after != predictions_sha256_before:
        raise RuntimeError("Internal-test predictions changed while being read")
    if result.get("predictions_artifact_sha256") != predictions_sha256_after:
        raise RuntimeError("Internal-test prediction hash differs from its result record")
    _validate_predictions(predictions)
    _validate_result_payload(result, predictions)
    expected_publication = {
        "protocol": RESULT_BUNDLE_PROTOCOL,
        "status": "COMMITTED",
        "atomic_directory_promotion": True,
        "exclusive_destination_reservation": True,
        "post_publication_verified": True,
    }
    if result.get("publication") != expected_publication:
        raise ValueError("Internal-test result publication record is invalid")
    expected_identity = {
        "pipeline_version": started["pipeline_version"],
        "arm": started["arm"],
        "model": started["model"],
        "variant": started["variant"],
        "training_lock_artifact": started["training_lock_artifact"],
        "training_lock_sha256": started["training_lock_sha256"],
        "model_artifact": started["model_artifact"],
        "model_artifact_sha256": started["model_artifact_sha256"],
        "test_embeddings_artifact": started["test_embeddings_artifact"],
        "test_embeddings_sha256": completion["test_embeddings_sha256"],
        "test_embeddings_audit_artifact": started[
            "test_embeddings_audit_artifact"
        ],
        "test_embeddings_audit_sha256": completion[
            "test_embeddings_audit_sha256"
        ],
    }
    drift = {
        field: {"expected": value, "observed": result.get(field)}
        for field, value in expected_identity.items()
        if result.get(field) != value
    }
    if drift:
        raise ValueError(f"Internal-test result identity differs from receipt: {drift}")


def _validate_extraction_completion(
    completion: Mapping[str, Any], started: Mapping[str, Any], receipt: Path
) -> None:
    """Verify a committed embedding bundle before completing its receipt."""

    artifact = _verify_declared_file(
        "embedding artifact", completion["artifact"], completion["artifact_sha256"]
    )
    sidecar_path = _verify_declared_file(
        "embedding sidecar", completion["sidecar"], completion["sidecar_sha256"]
    )
    requested_artifact = _canonical_declared_path(
        "requested embedding artifact", started["requested_artifact"]
    )
    requested_sidecar = _canonical_declared_path(
        "requested embedding sidecar", started["requested_sidecar"]
    )
    if artifact != requested_artifact or sidecar_path != requested_sidecar:
        raise ValueError("Completed embedding bundle differs from its reservation")
    if (
        type(completion["n_subjects"]) is not int
        or completion["n_subjects"] != started["requested_n_subjects"]
    ):
        raise ValueError("Completed embedding subject count differs from its reservation")
    _verify_declared_file(
        "cohort manifest",
        started["manifest_artifact"],
        started["manifest_sha256"],
    )
    from FineTuning.shared.embedding_bundle import load_embedding_bundle_sidecar

    sidecar = load_embedding_bundle_sidecar(
        artifact,
        sidecar_path,
        expected_pipeline_version=started["pipeline_version"],
        expected_model=started["model"],
        expected_variant=started["variant"],
        expected_readout=started["readout"],
        expected_split="internal_test",
        expected_dimension=started["embedding_dim"],
    )
    if sidecar["n_subjects"] != completion["n_subjects"]:
        raise ValueError("Embedding sidecar subject count differs from its receipt")
    if sidecar["manifest"].get("manifest_sha256") != started["manifest_sha256"]:
        raise ValueError("Embedding sidecar manifest differs from its receipt")
    if sidecar.get("internal_test_receipt") != {
        "path": str(receipt),
        "attempt_id": started["attempt_id"],
    }:
        raise ValueError("Embedding sidecar reservation differs from its receipt")


def _validate_completion_payload(
    completion: Mapping[str, Any], *, started: Mapping[str, Any], receipt: Path
) -> dict[str, Any]:
    """Validate one operation-specific completion without allowing identity drift."""

    if not isinstance(completion, Mapping):
        raise TypeError("Internal-test completion payload must be a mapping")
    data = dict(completion)
    collisions = sorted(RECEIPT_SYSTEM_FIELDS & set(data))
    if collisions:
        raise ValueError(
            f"Internal-test completion overrides protected fields: {collisions}"
        )
    operation = started["operation"]
    expected_fields = (
        EVALUATION_COMPLETION_FIELDS
        if operation == "evaluation"
        else EXTRACTION_COMPLETION_FIELDS
    )
    _validate_exact_fields(f"{operation} completion", data, expected_fields)
    drift = {
        field: {"expected": started[field], "observed": value}
        for field, value in data.items()
        if field in started and started[field] != value
    }
    if drift:
        raise ValueError(f"Internal-test completion changes STARTED identity: {drift}")
    if operation == "evaluation":
        if data["arm"] != "frozen" or data["on_harmony_accessed"] is not False:
            raise ValueError("Internal-test evaluation completion identity is invalid")
        _require_text_fields(
            "evaluation completion", data, ("model", "variant", "result")
        )
        _validate_evaluation_completion(data, started)
    elif operation == "embedding extraction":
        _validate_extraction_completion(data, started, receipt)
    else:
        raise ValueError(f"Unknown internal-test receipt operation: {operation!r}")
    return data


def _completion_lock_path(receipt: Path) -> Path:
    """Return the fixed exclusive lock for one STARTED-to-COMPLETED transition."""

    return receipt.with_name(f".{receipt.name}.completion.lock")


def _reserve_completion(receipt: Path, attempt_id: str) -> Path:
    """Serialize completion attempts without modifying the STARTED receipt."""

    lock = _completion_lock_path(receipt)
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"Internal-test receipt completion is already reserved: {receipt}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "receipt": str(receipt),
                        "attempt_id": attempt_id,
                        "pid": os.getpid(),
                        "reserved_at_utc": _utc_now(),
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(receipt.parent)
        return lock
    except BaseException:
        lock.unlink(missing_ok=True)
        raise


def _release_completion(lock: Path) -> None:
    """Release and durably remove a receipt-completion lock."""

    lock.unlink(missing_ok=True)
    _fsync_directory(lock.parent)


def _reserve_receipt(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    started_status: str,
    operation: str,
) -> dict[str, Any]:
    """Atomically reserve one internal-test operation."""

    receipt = Path(path).resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    validated_payload = _validate_started_payload(
        payload, operation=operation, receipt=receipt
    )
    started = {
        **validated_payload,
        "status": started_status,
        "operation": operation,
        "attempt_id": uuid.uuid4().hex,
        "started_at_utc": _utc_now(),
        "internal_test_accessed": True,
    }
    serialized = json.dumps(started, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{receipt.name}.started.",
        dir=receipt.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, receipt)
        except FileExistsError as error:
            raise FileExistsError(
                f"Internal test already has an {operation} receipt: {receipt}"
            ) from error
        temporary.unlink()
        _fsync_directory(receipt.parent)
        observed = _read_stable_json_object(receipt, f"{operation} STARTED receipt")
        if observed != started:
            raise RuntimeError("Internal-test STARTED receipt failed verification")
        return observed
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(receipt.parent)


def _complete_receipt(
    path: Path | str,
    started: Mapping[str, Any],
    completion: Mapping[str, Any],
    *,
    started_status: str,
    completed_status: str,
    operation: str,
) -> dict[str, Any]:
    """Complete exactly the attempt that owns an existing STARTED receipt."""

    receipt = Path(path).resolve()
    if not isinstance(started, Mapping):
        raise TypeError("Internal-test STARTED receipt must be a mapping")
    started_copy = dict(started)
    expected_attempt = started_copy.get("attempt_id")
    if (
        not isinstance(expected_attempt, str)
        or re.fullmatch(r"[0-9a-f]{32}", expected_attempt) is None
    ):
        raise ValueError("Internal-test STARTED receipt has an invalid attempt_id")
    completion_lock = _reserve_completion(receipt, expected_attempt)
    temporary: Path | None = None
    try:
        try:
            observed = _read_stable_json_object(
                receipt, f"{operation} STARTED receipt"
            )
        except (FileNotFoundError, TypeError) as error:
            raise RuntimeError(
                f"Internal-test {operation} STARTED receipt is missing or invalid: "
                f"{receipt}"
            ) from error
        if (
            observed.get("status") != started_status
            or observed.get("operation") != operation
            or observed.get("attempt_id") != expected_attempt
            or observed != started_copy
        ):
            raise RuntimeError(
                "Internal-test receipt changed after exclusive reservation"
            )
        validated_completion = _validate_completion_payload(
            completion, started=observed, receipt=receipt
        )
        completed = {
            **observed,
            **validated_completion,
            "status": completed_status,
            "completed_at_utc": _utc_now(),
            "internal_test_accessed": True,
        }
        serialized = json.dumps(completed, indent=2, allow_nan=False) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{receipt.name}.completed.",
            dir=receipt.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        observed_before_replace = _read_stable_json_object(
            receipt, f"{operation} STARTED receipt"
        )
        if observed_before_replace != observed:
            raise RuntimeError("Internal-test receipt changed before completion")
        os.replace(temporary, receipt)
        temporary = None
        _fsync_directory(receipt.parent)
        persisted = _read_stable_json_object(
            receipt, f"{operation} COMPLETED receipt"
        )
        if persisted != completed:
            raise RuntimeError("Internal-test COMPLETED receipt failed verification")
        return persisted
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        _release_completion(completion_lock)


def reserve_internal_test_receipt(
    path: Path | str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically reserve one evaluation attempt before test artifacts are read."""

    return _reserve_receipt(
        path,
        payload,
        started_status=STARTED_STATUS,
        operation="evaluation",
    )


def complete_internal_test_receipt(
    path: Path | str,
    started: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    """Complete exactly the evaluation attempt that owns the STARTED receipt."""

    return _complete_receipt(
        path,
        started,
        completion,
        started_status=STARTED_STATUS,
        completed_status=COMPLETED_STATUS,
        operation="evaluation",
    )


def reserve_internal_test_embedding_receipt(
    path: Path | str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Reserve one frozen-embedding extraction before any test image is encoded."""

    return _reserve_receipt(
        path,
        payload,
        started_status=EXTRACTION_STARTED_STATUS,
        operation="embedding extraction",
    )


def complete_internal_test_embedding_receipt(
    path: Path | str,
    started: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    """Complete the frozen-embedding extraction that owns the STARTED receipt."""

    return _complete_receipt(
        path,
        started,
        completion,
        started_status=EXTRACTION_STARTED_STATUS,
        completed_status=EXTRACTION_COMPLETED_STATUS,
        operation="embedding extraction",
    )
