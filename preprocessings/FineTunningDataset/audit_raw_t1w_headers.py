#!/usr/bin/env python3
"""Audit raw brain-age T1w headers for incomplete physical coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "preprocessings/FineTunningDataset/brainage_splits"
    / "brainage_definitive_subject_manifest.tsv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "preprocessings/FineTunningDataset/brainage_splits"
)
DEFAULT_MINIMUM_FOV_MM = 120.0
DEFAULT_MINIMUM_RELATIVE_FOV = 0.70
DEFAULT_MAXIMUM_SPACING_MISMATCH_MM = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-fov-mm", type=float, default=DEFAULT_MINIMUM_FOV_MM)
    parser.add_argument(
        "--minimum-relative-fov",
        type=float,
        default=DEFAULT_MINIMUM_RELATIVE_FOV,
    )
    parser.add_argument(
        "--maximum-spacing-mismatch-mm",
        type=float,
        default=DEFAULT_MAXIMUM_SPACING_MISMATCH_MM,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"dataset", "subject", "split", "raw_t1w_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest is empty or lacks required columns: {path}")
    return rows


def inspect_header(row: dict[str, str]) -> dict[str, Any]:
    path = Path(row["raw_t1w_path"])
    image = nib.load(str(path))
    shape = tuple(int(value) for value in image.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI, got shape {shape}")

    header_spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("NIfTI affine is invalid or non-finite")
    affine_spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (
        header_spacing.shape != (3,)
        or not np.isfinite(header_spacing).all()
        or not np.isfinite(affine_spacing).all()
        or np.any(header_spacing <= 0)
        or np.any(affine_spacing <= 0)
    ):
        raise ValueError("NIfTI voxel spacing is invalid")

    fov = np.asarray(shape, dtype=np.float64) * affine_spacing
    sorted_fov = np.sort(fov)
    return {
        "dataset": row["dataset"],
        "subject": row["subject"],
        "split": row["split"],
        "raw_t1w_path": str(path),
        "shape": "x".join(str(value) for value in shape),
        "header_spacing_mm": "x".join(f"{value:.6g}" for value in header_spacing),
        "affine_spacing_mm": "x".join(f"{value:.6g}" for value in affine_spacing),
        "fov_mm": "x".join(f"{value:.3f}" for value in fov),
        "minimum_fov_mm": float(sorted_fov[0]),
        "middle_fov_mm": float(sorted_fov[1]),
        "maximum_fov_mm": float(sorted_fov[2]),
        "spacing_mismatch_mm": float(np.max(np.abs(header_spacing - affine_spacing))),
        "relative_minimum_fov": None,
        "flags": "",
        "error": "",
    }


def atomic_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.minimum_fov_mm <= 0:
        raise ValueError("--minimum-fov-mm must be positive")
    if not 0 < args.minimum_relative_fov <= 1:
        raise ValueError("--minimum-relative-fov must be in (0, 1]")

    manifest = args.manifest.resolve()
    source = read_manifest(manifest)
    inspected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in source:
        try:
            inspected.append(inspect_header(row))
        except Exception as error:  # noqa: BLE001 - preserve every failed identity
            errors.append(
                {
                    "dataset": row["dataset"],
                    "subject": row["subject"],
                    "split": row["split"],
                    "raw_t1w_path": row["raw_t1w_path"],
                    "shape": "",
                    "header_spacing_mm": "",
                    "affine_spacing_mm": "",
                    "fov_mm": "",
                    "minimum_fov_mm": "",
                    "middle_fov_mm": "",
                    "maximum_fov_mm": "",
                    "spacing_mismatch_mm": "",
                    "relative_minimum_fov": "",
                    "flags": "HEADER_ERROR",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inspected:
        by_dataset[str(row["dataset"])].append(row)

    dataset_summary: dict[str, dict[str, Any]] = {}
    for dataset, rows in sorted(by_dataset.items()):
        values = np.asarray([row["minimum_fov_mm"] for row in rows], dtype=np.float64)
        median = float(np.median(values))
        dataset_summary[dataset] = {
            "n": int(values.size),
            "minimum_fov_mm": float(values.min()),
            "p01_fov_mm": float(np.quantile(values, 0.01)),
            "p05_fov_mm": float(np.quantile(values, 0.05)),
            "median_fov_mm": median,
            "maximum_fov_mm": float(values.max()),
        }
        for row in rows:
            relative = float(row["minimum_fov_mm"] / median)
            flags: list[str] = []
            if row["minimum_fov_mm"] < args.minimum_fov_mm:
                flags.append("FOV_BELOW_ABSOLUTE_MINIMUM")
            if relative < args.minimum_relative_fov:
                flags.append("FOV_BELOW_DATASET_RELATIVE_MINIMUM")
            if row["spacing_mismatch_mm"] > args.maximum_spacing_mismatch_mm:
                flags.append("HEADER_AFFINE_SPACING_MISMATCH")
            row["relative_minimum_fov"] = relative
            row["flags"] = ";".join(flags)

    fields = [
        "dataset",
        "subject",
        "split",
        "raw_t1w_path",
        "shape",
        "header_spacing_mm",
        "affine_spacing_mm",
        "fov_mm",
        "minimum_fov_mm",
        "middle_fov_mm",
        "maximum_fov_mm",
        "spacing_mismatch_mm",
        "relative_minimum_fov",
        "flags",
        "error",
    ]
    rows = sorted(
        inspected + errors,
        key=lambda row: (str(row["dataset"]), str(row["subject"])),
    )
    flagged = [row for row in rows if row["flags"]]
    status = "PASS" if len(rows) == len(source) and not flagged else "FAIL"

    output_tsv = args.output_dir / "brainage_raw_t1w_header_qc.tsv"
    output_json = args.output_dir / "brainage_raw_t1w_header_qc.json"
    atomic_tsv(output_tsv, rows, fields)
    summary = {
        "status": status,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "thresholds": {
            "minimum_fov_mm": args.minimum_fov_mm,
            "minimum_relative_fov": args.minimum_relative_fov,
            "maximum_spacing_mismatch_mm": args.maximum_spacing_mismatch_mm,
        },
        "n_manifest_rows": len(source),
        "n_headers_read": len(inspected),
        "n_header_errors": len(errors),
        "n_flagged": len(flagged),
        "flag_counts": dict(Counter(flag for row in flagged for flag in row["flags"].split(";"))),
        "dataset_summary": dataset_summary,
        "flagged_cases": [
            {
                key: row[key]
                for key in (
                    "dataset",
                    "subject",
                    "split",
                    "raw_t1w_path",
                    "shape",
                    "fov_mm",
                    "minimum_fov_mm",
                    "relative_minimum_fov",
                    "flags",
                    "error",
                )
            }
            for row in flagged
        ],
        "outputs": {"tsv": str(output_tsv), "json": str(output_json)},
    }
    atomic_json(output_json, summary)

    print(f"Status: {status}")
    print(f"Headers: {len(inspected)}/{len(source)}; flagged={len(flagged)}; errors={len(errors)}")
    for dataset, values in dataset_summary.items():
        print(
            f"  {dataset}: n={values['n']} minimum={values['minimum_fov_mm']:.1f} "
            f"median={values['median_fov_mm']:.1f} mm"
        )
    for row in flagged:
        print(
            f"  FLAG {row['dataset']}/{row['subject']}: shape={row['shape']} "
            f"fov={row['fov_mm']} flags={row['flags']} {row['error']}"
        )
    print(f"TSV:  {output_tsv}")
    print(f"JSON: {output_json}")
    if status != "PASS":
        raise RuntimeError("Raw T1w header QC failed; inspect the emitted TSV/JSON")


if __name__ == "__main__":
    main()
