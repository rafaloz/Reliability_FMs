"""Embedding validation, age standardization, and regression metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EMBEDDING_METADATA_COLUMNS = (
    "dataset",
    "subject",
    "split",
    "age",
    "sex",
    "input_path",
    "model",
    "variant",
)
TEXT_EMBEDDING_METADATA_COLUMNS = (
    "dataset",
    "subject",
    "split",
    "sex",
    "input_path",
    "model",
    "variant",
)


@dataclass(frozen=True)
class AgeStandardizer:
    """Train-only target standardization used by neural age heads."""

    mean: float
    std: float

    @classmethod
    def fit(cls, values: Iterable[float]) -> "AgeStandardizer":
        array = np.asarray(list(values), dtype=np.float64)
        if array.size < 2 or not np.isfinite(array).all():
            raise ValueError("Cannot fit age standardizer to invalid values")
        std = float(array.std(ddof=0))
        if std <= 0:
            raise ValueError("Training ages have zero variance")
        return cls(mean=float(array.mean()), std=std)

    def transform(self, value: float) -> float:
        return (float(value) - self.mean) / self.std

    def inverse(self, value: float) -> float:
        return float(value) * self.std + self.mean


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    """Return contiguous emb_0...emb_N columns and reject malformed tables."""

    columns = [column for column in frame.columns if str(column).startswith("emb_")]
    if not columns:
        raise ValueError("Embedding table has no emb_* columns")
    try:
        ordered = sorted(columns, key=lambda column: int(str(column).split("_", 1)[1]))
    except ValueError as error:
        raise ValueError("Embedding column suffixes must be integers") from error
    if ordered != [f"emb_{index}" for index in range(len(ordered))]:
        raise ValueError("Embedding columns must be contiguous from emb_0")
    return ordered


def validate_embedding_frame(
    frame: pd.DataFrame,
    *,
    expected_manifest: pd.DataFrame | None = None,
    expected_model: str | None = None,
    expected_variant: str | None = None,
) -> list[str]:
    """Validate one extracted embedding artifact and optionally match its subjects."""

    missing = sorted(set(EMBEDDING_METADATA_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Embedding table lacks metadata columns: {missing}")
    if frame.empty:
        raise ValueError("Embedding table is empty")
    invalid_text_columns = [
        column
        for column in TEXT_EMBEDDING_METADATA_COLUMNS
        if frame[column].map(
            lambda value: not isinstance(value, str) or not value.strip()
        ).any()
    ]
    if invalid_text_columns:
        raise ValueError(
            "Embedding table contains invalid text metadata in columns: "
            f"{invalid_text_columns}"
        )
    if frame.duplicated(["dataset", "subject"]).any():
        raise ValueError("Embedding table contains duplicate dataset/subject keys")
    if expected_model is not None and set(frame["model"].astype(str)) != {expected_model}:
        raise ValueError(f"Embedding table is not exclusively model {expected_model!r}")
    if expected_variant is not None and set(frame["variant"].astype(str)) != {expected_variant}:
        raise ValueError(f"Embedding table is not exclusively variant {expected_variant!r}")

    columns = embedding_columns(frame)
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Embedding table contains non-finite feature values")
    ages = pd.to_numeric(frame["age"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(ages).all():
        raise ValueError("Embedding table contains non-finite ages")

    if expected_manifest is not None:
        keys = ["dataset", "subject", "split"]
        observed = frame[keys].astype(str).reset_index(drop=True)
        expected = expected_manifest[keys].astype(str).reset_index(drop=True)
        if not observed.equals(expected):
            raise ValueError("Embedding rows/order do not exactly match the selected manifest")
        expected_ages = expected_manifest["age"].to_numpy(dtype=np.float64)
        if not np.array_equal(ages, expected_ages):
            raise ValueError("Embedding ages differ from the authoritative manifest")
        observed_sex = frame["sex"].astype(str).reset_index(drop=True)
        expected_sex = expected_manifest["sex"].astype(str).reset_index(drop=True)
        if not observed_sex.equals(expected_sex):
            raise ValueError("Embedding sex labels differ from the authoritative manifest")
    return columns


def regression_metrics(
    truth: Iterable[float], prediction: Iterable[float]
) -> dict[str, float | int]:
    """Calculate uncorrected subject-level regression metrics in years."""

    true = np.asarray(list(truth), dtype=np.float64)
    pred = np.asarray(list(prediction), dtype=np.float64)
    if true.shape != pred.shape or true.ndim != 1 or true.size == 0:
        raise ValueError("truth and prediction must be nonempty, equally sized vectors")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require finite values")
    residual = pred - true
    ss_total = float(np.sum((true - true.mean()) ** 2))
    return {
        "n": int(true.size),
        "mae_years": float(np.mean(np.abs(residual))),
        "rmse_years": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / ss_total) if ss_total > 0 else float("nan"),
        "mean_error_years": float(np.mean(residual)),
    }
