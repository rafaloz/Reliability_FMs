"""Common frozen-embedding linear brain-age regressor."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from FineTuning.shared.cohort import sha256_file
from FineTuning.shared.metrics import AgeStandardizer, regression_metrics
from FineTuning.shared.onharmony import (
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    FORBIDDEN_LABEL_COLUMNS,
    acquisition_from_path,
    load_onharmony_manifest,
    validate_prediction_frame,
)
from FineTuning.shared.runtime import (
    atomic_json,
    atomic_tsv,
    package_versions,
)


MODEL_FILENAME = "frozen_linear_model.npz"
ONHARMONY_EMBEDDING_MANIFEST_VERSION = "onharmony_embedding_manifest_v1"
RIDGE_ARTIFACT_ARRAYS = frozenset(
    {
        "feature_mean",
        "feature_scale",
        "coef",
        "intercept",
        "age_mean",
        "age_std",
        "alpha",
        "n_constant_dimensions",
    }
)


def add_frozen_onharmony_source_arguments(parser: Any) -> None:
    """Attach the label-blind ON-Harmony embedding-source CLI contract."""

    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--filelist", type=Path, default=DEFAULT_RAW_FILELIST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)


def add_frozen_onharmony_arguments(parser: Any) -> None:
    """Attach the common external-prediction CLI contract to a Ridge parser."""

    parser.add_argument("--run-dir", type=Path, required=True)
    add_frozen_onharmony_source_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)


def validate_frozen_embedding_contract(
    frame: pd.DataFrame,
    expected_manifest: pd.DataFrame,
    columns: Iterable[str],
    *,
    sidecar_dimension: object,
    expected_dimension: int,
    model_name: str,
) -> None:
    """Bind frozen features and sex metadata to the model and cohort contracts."""

    embedding_columns = tuple(columns)
    if (
        expected_dimension <= 0
        or len(embedding_columns) != expected_dimension
        or sidecar_dimension != expected_dimension
    ):
        raise ValueError(
            f"{model_name} embedding dimensionality drift: "
            f"columns={len(embedding_columns)}, sidecar={sidecar_dimension}, "
            f"expected={expected_dimension}"
        )

    observed_sex = frame["sex"].astype(str).reset_index(drop=True)
    expected_sex = expected_manifest["sex"].astype(str).reset_index(drop=True)
    if not observed_sex.equals(expected_sex):
        mismatch = observed_sex.ne(expected_sex)
        examples = [
            {
                "dataset": str(expected_manifest.iloc[index]["dataset"]),
                "subject": str(expected_manifest.iloc[index]["subject"]),
                "expected": str(expected_sex.iloc[index]),
                "observed": str(observed_sex.iloc[index]),
            }
            for index in mismatch[mismatch].index[:10]
        ]
        raise ValueError(f"Embedding sex differs from the authoritative manifest: {examples}")


def parse_ridge_alphas(value: str | Iterable[float]) -> list[float]:
    """Parse a strictly positive, unique alpha grid in deterministic order."""

    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            raise ValueError("--ridge-alphas cannot be empty")
        values = [float(piece) for piece in pieces]
    else:
        values = [float(item) for item in value]
    if not values or not np.isfinite(values).all() or any(item <= 0 for item in values):
        raise ValueError("Ridge alphas must be finite and strictly positive")
    if len(values) != len(set(values)):
        raise ValueError("Ridge alphas must be unique")
    return sorted(values)


def fit_feature_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit per-dimension train-only scaling and preserve constant dimensions."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError("Training features must be a finite 2D array")
    mean = array.mean(axis=0)
    scale = array.std(axis=0, ddof=0)
    constant = scale <= np.finfo(np.float64).eps
    scale[constant] = 1.0
    return mean, scale, int(constant.sum())


def transform_features(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply train-only feature scaling and reject dimension drift."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != mean.shape or scale.shape != mean.shape:
        raise ValueError(
            f"Feature shape mismatch: values={array.shape}, mean={mean.shape}, scale={scale.shape}"
        )
    transformed = (array - mean) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("Feature standardization produced non-finite values")
    return transformed


def fit_ridge_grid(
    train_x: np.ndarray,
    train_age: Iterable[float],
    validation_x: np.ndarray,
    validation_age: Iterable[float],
    alphas: Iterable[float],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Select a common linear Ridge head by uncorrected validation MAE."""

    try:
        from sklearn.linear_model import Ridge
    except ImportError as error:
        raise RuntimeError("The frozen arm requires scikit-learn") from error

    alpha_grid = parse_ridge_alphas(alphas)
    train_y = np.asarray(list(train_age), dtype=np.float64)
    validation_y = np.asarray(list(validation_age), dtype=np.float64)
    if train_x.shape[0] != train_y.size or validation_x.shape[0] != validation_y.size:
        raise ValueError("Feature and age row counts differ")
    if not np.isfinite(train_y).all() or not np.isfinite(validation_y).all():
        raise ValueError("Ages must be finite")

    feature_mean, feature_scale, constant_dimensions = fit_feature_standardizer(train_x)
    train_scaled = transform_features(train_x, feature_mean, feature_scale)
    validation_scaled = transform_features(validation_x, feature_mean, feature_scale)
    age_stats = AgeStandardizer.fit(train_y)
    train_z = np.asarray([age_stats.transform(value) for value in train_y], dtype=np.float64)

    candidates: list[dict[str, Any]] = []
    fitted: dict[float, Any] = {}
    predictions: dict[float, np.ndarray] = {}
    for alpha in alpha_grid:
        estimator = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
        estimator.fit(train_scaled, train_z)
        prediction_z = estimator.predict(validation_scaled)
        prediction_age = prediction_z * age_stats.std + age_stats.mean
        metrics = regression_metrics(validation_y, prediction_age)
        candidates.append({"alpha": alpha, **metrics})
        fitted[alpha] = estimator
        predictions[alpha] = prediction_age

    table = pd.DataFrame(candidates).sort_values(
        ["mae_years", "alpha"], ascending=[True, False], kind="stable"
    )
    best_alpha = float(table.iloc[0]["alpha"])
    best = fitted[best_alpha]
    artifact: dict[str, Any] = {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "coef": np.asarray(best.coef_, dtype=np.float64),
        "intercept": np.asarray([float(best.intercept_)], dtype=np.float64),
        "age_mean": np.asarray([age_stats.mean], dtype=np.float64),
        "age_std": np.asarray([age_stats.std], dtype=np.float64),
        "alpha": np.asarray([best_alpha], dtype=np.float64),
        "n_constant_dimensions": np.asarray([constant_dimensions], dtype=np.int64),
    }
    return artifact, table.reset_index(drop=True), predictions[best_alpha]


def predict_ridge(artifact: dict[str, Any], values: np.ndarray) -> np.ndarray:
    """Apply a saved frozen-arm model without importing scikit-learn."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Ridge prediction features must be a finite 2D array")
    validate_ridge_artifact(artifact, expected_dimension=array.shape[1])
    scaled = transform_features(array, artifact["feature_mean"], artifact["feature_scale"])
    prediction_z = scaled @ artifact["coef"] + float(artifact["intercept"][0])
    prediction = prediction_z * float(artifact["age_std"][0]) + float(artifact["age_mean"][0])
    if not np.isfinite(prediction).all():
        raise ValueError("Frozen linear model produced non-finite predictions")
    return prediction


def validate_ridge_artifact(
    artifact: Mapping[str, Any], *, expected_dimension: int | None = None
) -> int:
    """Validate the complete numerical and dimensional contract of one Ridge NPZ."""

    observed = set(artifact)
    missing = sorted(RIDGE_ARTIFACT_ARRAYS - observed)
    unexpected = sorted(observed - RIDGE_ARTIFACT_ARRAYS)
    if missing or unexpected:
        raise ValueError(
            "Frozen linear artifact array contract differs: "
            f"missing={missing}, unexpected={unexpected}"
        )

    vectors = {
        name: np.asarray(artifact[name])
        for name in ("feature_mean", "feature_scale", "coef")
    }
    dimension = int(vectors["feature_mean"].size)
    if dimension <= 0 or any(array.shape != (dimension,) for array in vectors.values()):
        shapes = {name: list(array.shape) for name, array in vectors.items()}
        raise ValueError(f"Frozen linear artifact vector shapes differ: {shapes}")
    if expected_dimension is not None and dimension != int(expected_dimension):
        raise ValueError(
            "Frozen linear artifact dimension differs: "
            f"observed={dimension}, expected={expected_dimension}"
        )

    scalars = {
        name: np.asarray(artifact[name])
        for name in ("intercept", "age_mean", "age_std", "alpha")
    }
    bad_scalar_shapes = {
        name: list(array.shape) for name, array in scalars.items() if array.shape != (1,)
    }
    if bad_scalar_shapes:
        raise ValueError(
            f"Frozen linear artifact scalar shapes differ: {bad_scalar_shapes}"
        )

    for name, array in {**vectors, **scalars}.items():
        try:
            finite = np.isfinite(array).all()
        except TypeError as error:
            raise ValueError(f"Frozen linear artifact array {name!r} is not numeric") from error
        if not finite:
            raise ValueError(f"Frozen linear artifact array {name!r} is non-finite")
    if np.any(vectors["feature_scale"] <= 0):
        raise ValueError("Frozen linear artifact feature_scale must be strictly positive")
    if float(scalars["age_std"][0]) <= 0:
        raise ValueError("Frozen linear artifact age_std must be strictly positive")
    if float(scalars["alpha"][0]) <= 0:
        raise ValueError("Frozen linear artifact alpha must be strictly positive")

    constant = np.asarray(artifact["n_constant_dimensions"])
    if constant.shape != (1,) or not np.issubdtype(constant.dtype, np.integer):
        raise ValueError(
            "Frozen linear artifact n_constant_dimensions must be one integer"
        )
    constant_count = int(constant[0])
    if constant_count < 0 or constant_count > dimension:
        raise ValueError(
            "Frozen linear artifact n_constant_dimensions is outside [0, dimension]"
        )
    return dimension


def atomic_npz(payload: dict[str, Any], path: Path | str) -> None:
    """Write a NumPy model artifact atomically without automatic suffix drift."""

    validate_ridge_artifact(payload)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Reopen with allow_pickle=False before promoting the artifact.
        validate_ridge_artifact(load_npz(temporary))
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_npz(path: Path | str) -> dict[str, np.ndarray]:
    """Load every array from a frozen linear artifact into memory."""

    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_locked_ridge_artifact(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_alpha: float,
    expected_dimension: int,
) -> dict[str, np.ndarray]:
    """Load a locked Ridge artifact while rejecting hash, shape, or value drift."""

    artifact_path = Path(path).resolve()
    sha256_before = sha256_file(artifact_path)
    if sha256_before != expected_sha256:
        raise ValueError("Frozen Ridge model hash differs from LOCKED.json")
    artifact = load_npz(artifact_path)
    sha256_after = sha256_file(artifact_path)
    if sha256_after != sha256_before:
        raise RuntimeError("Frozen Ridge model changed while it was being read")

    validate_ridge_artifact(artifact, expected_dimension=expected_dimension)
    alpha = np.asarray(artifact.get("alpha"), dtype=np.float64)
    if (
        alpha.shape != (1,)
        or not np.isfinite(alpha).all()
        or not np.isclose(float(alpha[0]), float(expected_alpha), rtol=0.0, atol=0.0)
    ):
        raise ValueError("Frozen Ridge alpha differs from LOCKED.json")
    return artifact


def _csv_header(path: Path) -> list[str]:
    """Read only a CSV header and reject ambiguous duplicate column names."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as error:
            raise ValueError(f"Empty ON-Harmony embedding CSV: {path}") from error
    if not header or len(header) != len(set(header)):
        raise ValueError(f"Invalid or duplicate ON-Harmony CSV columns: {path}")
    return header


def _feature_columns(
    header: Iterable[str], *, prefix: str, expected_dimension: int
) -> list[str]:
    """Require one complete, numerically ordered embedding stream."""

    expected = [f"{prefix}{index}" for index in range(expected_dimension)]
    observed = [column for column in header if column.startswith(prefix)]
    if set(observed) != set(expected) or len(observed) != expected_dimension:
        missing = sorted(set(expected) - set(observed))[:10]
        unexpected = sorted(set(observed) - set(expected))[:10]
        raise ValueError(
            "ON-Harmony embedding stream differs from the frozen model: "
            f"prefix={prefix!r}, expected={expected_dimension}, observed={len(observed)}, "
            f"missing={missing}, unexpected={unexpected}"
        )
    return expected


def _embedding_manifest_provenance(
    embeddings_path: Path,
    *,
    source_manifest: Path | str | None,
    artifact_sha256: str,
    n_rows: int,
    feature_prefix: str,
    expected_dimension: int,
    expected_pipeline_version: str,
    expected_model: str,
    expected_variant: str,
    expected_readout: str,
) -> dict[str, Any]:
    """Validate the mandatory semantic contract for one ON-Harmony embedding CSV."""

    if source_manifest is None:
        conventional = embeddings_path.with_name(
            f"{embeddings_path.stem}_manifest.json"
        )
        manifest_path = conventional
    else:
        manifest_path = Path(source_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "ON-Harmony embedding manifest is required: " f"{manifest_path}"
        )

    manifest_sha256_before = sha256_file(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256_after = sha256_file(manifest_path)
    if manifest_sha256_after != manifest_sha256_before:
        raise RuntimeError("ON-Harmony extractor manifest changed while it was being read")

    if not isinstance(payload, dict):
        raise TypeError(
            f"ON-Harmony extractor manifest is not a JSON object: {manifest_path}"
        )
    required = (
        "manifest_schema_version",
        "pipeline_version",
        "model",
        "variant",
        "readout",
        "feature_prefix",
        "embedding_dimensions",
        "extractor",
        "extractor_sha256",
        "output",
        "output_sha256",
        "n_rows",
    )
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise ValueError(
            f"ON-Harmony extractor manifest lacks required fields: {missing}"
        )
    expected_contract = {
        "manifest_schema_version": ONHARMONY_EMBEDDING_MANIFEST_VERSION,
        "pipeline_version": expected_pipeline_version,
        "model": expected_model,
        "variant": expected_variant,
        "readout": expected_readout,
        "feature_prefix": feature_prefix,
        "embedding_dimensions": expected_dimension,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected_contract.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"ON-Harmony extractor manifest semantic contract differs: {mismatches}"
        )
    extractor_sha256 = str(payload["extractor_sha256"])
    if len(extractor_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in extractor_sha256.lower()
    ):
        raise ValueError("ON-Harmony extractor manifest contains an invalid extractor hash")
    extractor_path = Path(str(payload["extractor"])).expanduser().resolve()
    if not extractor_path.is_file():
        raise FileNotFoundError(
            f"ON-Harmony extractor declared by its manifest is missing: {extractor_path}"
        )
    observed_extractor_sha256 = sha256_file(extractor_path)
    if observed_extractor_sha256 != extractor_sha256:
        raise ValueError("ON-Harmony extractor differs from its manifest")
    output_sha256 = str(payload["output_sha256"])
    if len(output_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in output_sha256.lower()
    ):
        raise ValueError("ON-Harmony extractor manifest contains an invalid output hash")
    if payload["output_sha256"] != artifact_sha256:
        raise ValueError("ON-Harmony embedding CSV differs from its extractor manifest")
    declared_output = Path(str(payload["output"])).expanduser().resolve()
    if declared_output != embeddings_path:
        raise ValueError("ON-Harmony extractor manifest points to a different output CSV")
    if type(payload["n_rows"]) is not int or payload["n_rows"] != n_rows:
        raise ValueError("ON-Harmony embedding row count differs from its extractor manifest")
    return {
        "status": "VALIDATED",
        "path": str(manifest_path),
        "sha256": manifest_sha256_after,
        **expected_contract,
        "extractor": str(extractor_path),
        "extractor_sha256": observed_extractor_sha256,
    }


def read_label_blind_onharmony_embeddings(
    path: Path | str,
    *,
    expected_manifest: pd.DataFrame,
    feature_prefix: str,
    expected_dimension: int,
    expected_pipeline_version: str,
    expected_model: str,
    expected_variant: str,
    expected_readout: str,
    source_manifest: Path | str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load identifiers and one embedding stream without materializing outcomes."""

    embeddings_path = Path(path).expanduser().resolve()
    if not embeddings_path.is_file() or embeddings_path.stat().st_size <= 0:
        raise FileNotFoundError(embeddings_path)
    header = _csv_header(embeddings_path)
    if "file_path" not in header:
        raise ValueError("ON-Harmony embedding CSV lacks file_path")
    columns = _feature_columns(
        header,
        prefix=feature_prefix,
        expected_dimension=expected_dimension,
    )
    artifact_sha256_before = sha256_file(embeddings_path)
    # usecols is the label firewall: outcomes and alternate readouts are not
    # materialized.
    loaded = pd.read_csv(
        embeddings_path,
        usecols=["file_path", *columns],
        dtype={"file_path": str},
    )
    artifact_sha256_after = sha256_file(embeddings_path)
    if artifact_sha256_after != artifact_sha256_before:
        raise RuntimeError("ON-Harmony embedding CSV changed while it was being read")

    acquisition_ids: list[str] = []
    for value in loaded["file_path"].tolist():
        acquisition_ids.append(acquisition_from_path(value)["acquisition_id"])
    if len(acquisition_ids) != len(set(acquisition_ids)):
        raise ValueError("ON-Harmony embedding CSV contains duplicate acquisitions")
    expected_ids = expected_manifest["acquisition_id"].astype(str).tolist()
    observed = set(acquisition_ids)
    expected = set(expected_ids)
    if observed != expected or len(acquisition_ids) != len(expected_ids):
        raise ValueError(
            "ON-Harmony embedding acquisitions differ from the canonical registry: "
            f"missing={sorted(expected - observed)[:10]}, "
            f"unexpected={sorted(observed - expected)[:10]}"
        )

    loaded = loaded.assign(acquisition_id=acquisition_ids).set_index(
        "acquisition_id", drop=True
    )
    ordered = loaded.loc[expected_ids, columns]
    values = ordered.to_numpy(dtype=np.float64)
    if values.shape != (len(expected_ids), expected_dimension):
        raise ValueError(f"Unexpected ON-Harmony embedding matrix shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("ON-Harmony embedding CSV contains non-finite feature values")

    extractor_manifest = _embedding_manifest_provenance(
        embeddings_path,
        source_manifest=source_manifest,
        artifact_sha256=artifact_sha256_after,
        n_rows=len(loaded),
        feature_prefix=feature_prefix,
        expected_dimension=expected_dimension,
        expected_pipeline_version=expected_pipeline_version,
        expected_model=expected_model,
        expected_variant=expected_variant,
        expected_readout=expected_readout,
    )
    provenance = {
        "artifact": str(embeddings_path),
        "artifact_sha256": artifact_sha256_after,
        "n_rows": int(len(loaded)),
        "feature_prefix": feature_prefix,
        "embedding_dim": expected_dimension,
        "source_outcome_columns_present": sorted(
            FORBIDDEN_LABEL_COLUMNS & set(header)
        ),
        "columns_loaded": ["file_path", *columns],
        "outcome_columns_loaded": [],
        "extractor_manifest": extractor_manifest,
    }
    return values, provenance


def audit_frozen_onharmony_source(
    *,
    embeddings_path: Path | str,
    model_name: str,
    variant: str,
    expected_dimension: int,
    feature_prefix: str,
    extractor_pipeline_version: str,
    readout: str,
    filelist: Path | str = DEFAULT_RAW_FILELIST,
    metadata: Path | str = DEFAULT_METADATA,
    source_manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Validate one complete label-blind embedding source without fitting Ridge."""

    frame, manifest = load_onharmony_manifest(filelist, metadata)
    values, provenance = read_label_blind_onharmony_embeddings(
        embeddings_path,
        expected_manifest=frame,
        feature_prefix=feature_prefix,
        expected_dimension=expected_dimension,
        expected_pipeline_version=extractor_pipeline_version,
        expected_model=model_name,
        expected_variant=variant,
        expected_readout=readout,
        source_manifest=source_manifest,
    )
    result = {
        "status": "ON_HARMONY_EMBEDDINGS_VALIDATED",
        "arm": "frozen",
        "model": model_name,
        "variant": variant,
        "readout": readout,
        "label_blind": True,
        "outcome_labels_loaded": False,
        "embedding_matrix_shape": list(values.shape),
        "on_harmony_manifest": manifest,
        "on_harmony_embeddings": provenance,
    }
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def validate_frozen_onharmony_lock(
    run_dir: Path | str,
    *,
    expected_pipeline_version: str,
    expected_model: str,
    expected_variant: str,
    expected_readout: str,
    expected_dimension: int,
    expected_training_manifest_sha256: str,
    current_fit_code_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], Path, Path, dict[str, str]]:
    """Bind external prediction to one untouched validation-only Ridge run."""

    run_dir = Path(run_dir).expanduser().resolve()
    lock_path = run_dir / "LOCKED.json"
    model_path = run_dir / MODEL_FILENAME
    if not lock_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Expected {lock_path} and {model_path}")
    lock_sha256_before = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_sha256_after = sha256_file(lock_path)
    if lock_sha256_after != lock_sha256_before:
        raise RuntimeError("Frozen Ridge LOCKED.json changed while it was being read")
    if not isinstance(lock, dict):
        raise TypeError(f"Frozen Ridge lock is not a JSON object: {lock_path}")
    expected = {
        "status": "LOCKED_AFTER_TRAIN_VALIDATION",
        "pipeline_version": expected_pipeline_version,
        "arm": "frozen",
        "model": expected_model,
        "variant": expected_variant,
        "readout": expected_readout,
        "internal_test_accessed": False,
        "on_harmony_accessed": False,
        "embedding_dim": expected_dimension,
    }
    mismatches = {
        key: {"expected": value, "observed": lock.get(key)}
        for key, value in expected.items()
        if lock.get(key) != value
    }
    selected_alpha = lock.get("selected_alpha")
    if (
        isinstance(selected_alpha, bool)
        or not isinstance(selected_alpha, (int, float))
        or not np.isfinite(float(selected_alpha))
        or float(selected_alpha) <= 0.0
    ):
        mismatches["selected_alpha"] = {
            "expected": "one finite positive number",
            "observed": selected_alpha,
        }
    training_manifest = lock.get("manifest")
    observed_manifest_sha256 = (
        training_manifest.get("manifest_sha256")
        if isinstance(training_manifest, Mapping)
        else None
    )
    if observed_manifest_sha256 != expected_training_manifest_sha256:
        mismatches["manifest.manifest_sha256"] = {
            "expected": expected_training_manifest_sha256,
            "observed": observed_manifest_sha256,
        }
    if mismatches:
        raise ValueError(f"Invalid validation-only frozen lock: {mismatches}")
    model_sha256 = sha256_file(model_path)
    if lock.get("model_artifact_sha256") != model_sha256:
        raise ValueError("Frozen Ridge model hash differs from LOCKED.json")
    if lock.get("fit_code_sha256") != dict(current_fit_code_sha256):
        raise ValueError("Runtime Ridge code differs from the locked frozen model")
    integrity = {
        "training_lock_sha256": lock_sha256_after,
        "model_artifact_sha256": model_sha256,
    }
    return lock, model_path, lock_path, integrity


def _prepare_staged_output_dir(path: Path | str) -> tuple[Path, Path]:
    """Reserve a sibling staging directory for all-or-nothing publication."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {destination}")
        destination.rmdir()
    staging = destination.with_name(
        f".{destination.name}.staging.{os.getpid()}.{time.time_ns()}"
    )
    staging.mkdir(exist_ok=False)
    return destination, staging


def _verify_frozen_prediction_inputs(
    *,
    lock_path: Path,
    model_path: Path,
    expected_lock_sha256: str,
    expected_model_sha256: str,
    embedding_provenance: Mapping[str, Any],
) -> None:
    """Reject any locked-model or embedding drift before publication."""

    if sha256_file(lock_path) != expected_lock_sha256:
        raise RuntimeError("Frozen Ridge LOCKED.json changed during ON-Harmony prediction")
    if sha256_file(model_path) != expected_model_sha256:
        raise RuntimeError("Frozen Ridge model changed during ON-Harmony prediction")
    embeddings_path = Path(str(embedding_provenance["artifact"]))
    if sha256_file(embeddings_path) != embedding_provenance["artifact_sha256"]:
        raise RuntimeError("ON-Harmony embedding CSV changed during prediction")
    manifest = embedding_provenance["extractor_manifest"]
    manifest_path = Path(str(manifest["path"]))
    if sha256_file(manifest_path) != manifest["sha256"]:
        raise RuntimeError("ON-Harmony embedding manifest changed during prediction")
    extractor_path = Path(str(manifest["extractor"]))
    if sha256_file(extractor_path) != manifest["extractor_sha256"]:
        raise RuntimeError("ON-Harmony embedding extractor changed during prediction")


def run_frozen_onharmony_prediction(
    *,
    run_dir: Path | str,
    embeddings_path: Path | str,
    output_dir: Path | str,
    model_name: str,
    variant: str,
    expected_dimension: int,
    feature_prefix: str,
    extractor_pipeline_version: str,
    readout: str,
    pipeline_version: str,
    expected_training_manifest_sha256: str,
    current_fit_code_sha256: Mapping[str, str],
    filelist: Path | str = DEFAULT_RAW_FILELIST,
    metadata: Path | str = DEFAULT_METADATA,
    source_manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Apply one locked frozen Ridge head to the canonical ON-Harmony embeddings."""

    lock, model_path, lock_path, integrity = validate_frozen_onharmony_lock(
        run_dir,
        expected_pipeline_version=pipeline_version,
        expected_model=model_name,
        expected_variant=variant,
        expected_readout=readout,
        expected_dimension=expected_dimension,
        expected_training_manifest_sha256=expected_training_manifest_sha256,
        current_fit_code_sha256=current_fit_code_sha256,
    )
    frame, manifest = load_onharmony_manifest(filelist, metadata)
    values, embedding_provenance = read_label_blind_onharmony_embeddings(
        embeddings_path,
        expected_manifest=frame,
        feature_prefix=feature_prefix,
        expected_dimension=expected_dimension,
        expected_pipeline_version=extractor_pipeline_version,
        expected_model=model_name,
        expected_variant=variant,
        expected_readout=readout,
        source_manifest=source_manifest,
    )
    artifact = load_locked_ridge_artifact(
        model_path,
        expected_sha256=integrity["model_artifact_sha256"],
        expected_alpha=float(lock["selected_alpha"]),
        expected_dimension=expected_dimension,
    )
    _verify_frozen_prediction_inputs(
        lock_path=lock_path,
        model_path=model_path,
        expected_lock_sha256=integrity["training_lock_sha256"],
        expected_model_sha256=integrity["model_artifact_sha256"],
        embedding_provenance=embedding_provenance,
    )
    prediction = predict_ridge(artifact, values)

    metadata_columns = [
        "subject",
        "session",
        "acquisition_id",
        "cohort",
        "scanner_code",
        "scanner_name",
        "vendor",
        "scanner_type",
        "scanner_typeloc",
        "scan_session_num",
    ]
    predictions = frame[metadata_columns].copy()
    predictions.insert(0, "model", model_name)
    predictions.insert(1, "variant", variant)
    predictions.insert(2, "arm", "frozen")
    predictions.insert(3, "ridge_alpha", float(artifact["alpha"][0]))
    predictions.insert(4, "ridge_model_sha256", lock["model_artifact_sha256"])
    predictions["age_pred"] = prediction
    validate_prediction_frame(predictions, frame)

    destination, staging = _prepare_staged_output_dir(output_dir)
    predictions_path = destination / "predictions.tsv"
    staged_predictions_path = staging / predictions_path.name
    try:
        atomic_tsv(predictions, staged_predictions_path)
        audit = {
            "pipeline_version": f"{pipeline_version}_frozen_onharmony_v1",
            "status": "ON_HARMONY_PREDICTED",
            "arm": "frozen",
            "model": model_name,
            "variant": variant,
            "label_blind": True,
            "outcome_labels_loaded": False,
            "bias_correction": "none",
            "n_predictions": int(len(predictions)),
            "n_subjects": int(predictions["subject"].nunique()),
            "ridge_run": str(model_path.parent),
            "ridge_model": str(model_path),
            "ridge_model_sha256": integrity["model_artifact_sha256"],
            "ridge_alpha": float(artifact["alpha"][0]),
            "training_lock": str(lock_path),
            "training_lock_sha256": integrity["training_lock_sha256"],
            "training_manifest": lock.get("manifest"),
            "on_harmony_manifest": manifest,
            "on_harmony_embeddings": embedding_provenance,
            "fit_code_sha256": dict(current_fit_code_sha256),
            "software": package_versions(),
            "predictions_artifact": str(predictions_path),
            "predictions_artifact_sha256": sha256_file(staged_predictions_path),
            "internal_test_accessed": False,
            "on_harmony_accessed": True,
        }
        atomic_json(audit, staging / "prediction_audit.json")
        _verify_frozen_prediction_inputs(
            lock_path=lock_path,
            model_path=model_path,
            expected_lock_sha256=integrity["training_lock_sha256"],
            expected_model_sha256=integrity["model_artifact_sha256"],
            embedding_provenance=embedding_provenance,
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"Saved {len(predictions)} label-free frozen predictions: {predictions_path}\n"
        f"SHA-256: {audit['predictions_artifact_sha256']}",
        flush=True,
    )
    return audit
