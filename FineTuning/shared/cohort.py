#!/usr/bin/env python3
"""Immutable cohort, split, input-contract, and ON-Harmony firewall.

This module deliberately has no model or PyTorch dependency. It validates the
development cohort before any checkpoint is loaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINETUNING_ROOT = PROJECT_ROOT / "preprocessings" / "FineTunningDataset"
DEFAULT_MANIFEST = (
    FINETUNING_ROOT
    / "brainage_splits"
    / "brainage_definitive_subject_manifest.tsv"
)

MANIFEST_SHA256 = "b2ee2dc7e06349ef1ac550da2061274ee724d2b121ddd652b37c1907ae8bc212"
SPLIT_SEED = 20260811
ALLOWED_SPLITS = ("train", "validation", "internal_test")
EXPECTED_SPLIT_DATASET_COUNTS = {
    "train": {"CamCAN": 521, "MR-ART": 118, "NIMH-RV": 118, "NeuroCog": 239, "WAND": 134},
    "validation": {"CamCAN": 65, "MR-ART": 15, "NIMH-RV": 14, "NeuroCog": 30, "WAND": 17},
    "internal_test": {"CamCAN": 64, "MR-ART": 15, "NIMH-RV": 14, "NeuroCog": 30, "WAND": 17},
}
REQUIRED_COLUMNS = {
    "dataset",
    "subject",
    "raw_t1w_path",
    "age",
    "sex",
    "split",
    "split_seed",
    "inclusion_status",
    "required_offline_groups",
    "raw_runtime_conditions",
}

def sha256_file(path: Path | str) -> str:
    """Return a streaming SHA-256 digest without loading large files in RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_for_firewall(value: object) -> str:
    """Collapse punctuation and case so ON-Harmony aliases cannot evade checks."""

    return "".join(character for character in str(value).lower() if character.isalnum())


def _semicolon_values(value: object) -> set[str]:
    """Parse a manifest cell containing semicolon-separated input contracts."""

    return {part.strip() for part in str(value).split(";") if part.strip()}


def _observed_split_dataset_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Represent split-by-dataset counts in a stable JSON-friendly mapping."""

    observed: dict[str, dict[str, int]] = {}
    for split in ALLOWED_SPLITS:
        counts = frame.loc[frame["split"] == split, "dataset"].value_counts().sort_index()
        observed[split] = {str(name): int(count) for name, count in counts.items()}
    return observed


def load_and_validate_manifest(
    path: Path | str = DEFAULT_MANIFEST,
    *,
    expected_sha256: str | None = MANIFEST_SHA256,
    expected_counts: Mapping[str, Mapping[str, int]] | None = EXPECTED_SPLIT_DATASET_COUNTS,
    required_offline_group: str | None = None,
    required_runtime_condition: str | None = None,
    check_raw_input_files: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the cohort and reject split drift, duplicates, or input-contract drift."""

    if required_offline_group and required_runtime_condition:
        raise ValueError("A model cannot require both offline and raw-runtime input contracts")

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    digest = sha256_file(manifest_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Manifest SHA-256 drift: expected {expected_sha256}, observed {digest}: "
            f"{manifest_path}"
        )

    frame = pd.read_csv(manifest_path, sep="\t")
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Manifest lacks required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Manifest is empty")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        bad = frame[list(REQUIRED_COLUMNS)].isna().sum()
        raise ValueError(f"Manifest contains missing required values: {bad[bad > 0].to_dict()}")
    if frame.duplicated(["dataset", "subject"]).any():
        raise ValueError("Manifest contains duplicate dataset/subject keys")
    if frame["raw_t1w_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate raw_t1w_path values")

    observed_splits = set(frame["split"].astype(str))
    if observed_splits != set(ALLOWED_SPLITS):
        raise ValueError(f"Unexpected split set: {sorted(observed_splits)}")
    if set(pd.to_numeric(frame["split_seed"], errors="raise").astype(int)) != {SPLIT_SEED}:
        raise ValueError(f"Every row must use split_seed={SPLIT_SEED}")
    if set(frame["inclusion_status"].astype(str)) != {"INCLUDED"}:
        raise ValueError("The definitive manifest may contain only INCLUDED subjects")

    input_contract: dict[str, str] = {}
    if required_offline_group:
        permitted = frame["required_offline_groups"].map(_semicolon_values)
        if not permitted.map(lambda values: required_offline_group in values).all():
            raise ValueError(
                f"Every row must permit offline input group {required_offline_group!r}"
            )
        input_contract["required_offline_group"] = required_offline_group
    if required_runtime_condition:
        permitted = frame["raw_runtime_conditions"].map(_semicolon_values)
        if not permitted.map(lambda values: required_runtime_condition in values).all():
            raise ValueError(
                f"Every row must permit raw runtime condition {required_runtime_condition!r}"
            )
        input_contract["required_runtime_condition"] = required_runtime_condition

    ages = pd.to_numeric(frame["age"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(ages).all():
        raise ValueError("Manifest contains non-finite ages")
    frame = frame.copy()
    frame["age"] = ages
    frame["dataset"] = frame["dataset"].astype(str)
    frame["subject"] = frame["subject"].astype(str)
    frame["split"] = frame["split"].astype(str)

    firewall_values = frame[["dataset", "subject", "raw_t1w_path"]].astype(str)
    if firewall_values.map(_normalise_for_firewall).apply(
        lambda column: column.str.contains("onharmony", regex=False)
    ).any().any():
        raise ValueError("ON-Harmony firewall violation in the development manifest")

    observed_counts = _observed_split_dataset_counts(frame)
    expected_counts_plain = (
        {split: dict(counts) for split, counts in expected_counts.items()}
        if expected_counts is not None
        else None
    )
    if expected_counts_plain is not None and observed_counts != expected_counts_plain:
        raise ValueError(
            "Frozen split-by-dataset counts changed: "
            f"expected={expected_counts_plain}, observed={observed_counts}"
        )

    missing_paths: list[str] = []
    if check_raw_input_files:
        missing_paths = [
            value for value in frame["raw_t1w_path"].astype(str) if not Path(value).is_file()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"Manifest has {len(missing_paths)} missing raw inputs; first 10: "
                f"{missing_paths[:10]}"
            )

    summary: dict[str, object] = {
        "manifest": str(manifest_path),
        "manifest_sha256": digest,
        "n_subjects": int(len(frame)),
        "split_counts": {
            key: int(value) for key, value in frame["split"].value_counts().items()
        },
        "split_dataset_counts": observed_counts,
        "age_min": float(ages.min()),
        "age_max": float(ages.max()),
        "input_contract": input_contract,
        "raw_input_files_checked": bool(check_raw_input_files),
        "missing_raw_input_files": len(missing_paths),
        "on_harmony_firewall": "PASS",
    }
    return frame, summary


def frame_for_splits(frame: pd.DataFrame, splits: Iterable[str]) -> pd.DataFrame:
    """Return selected splits in manifest order after validating their names."""

    selected = tuple(splits)
    if not selected or not set(selected).issubset(ALLOWED_SPLITS):
        raise ValueError(f"Invalid split selection: {selected}")
    return frame.loc[frame["split"].isin(selected)].reset_index(drop=True)
