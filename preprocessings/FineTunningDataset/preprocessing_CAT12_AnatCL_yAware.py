#!/usr/bin/env python3
"""Create the shared CAT12 input used by AnatCL and y-Aware.

The expensive part of both pipelines is identical: raw T1w -> CAT12 VBM
with Dartel/IXI555 -> modulated normalized grey-matter (mwp1) at
121 x 145 x 121 voxels and 1.5 mm isotropic resolution. CAT12 is therefore run
once, but the script materializes the two model-specific outputs using the same
formats as the ON-Harmony pipelines:

* ``preproc_AnatCL/<dataset>/<subject>/<subject>_anatcl_ready.nii.gz``:
  the full CAT12 mwp1 NIfTI. The AnatCL loader centre-crops Y from 145 to 128
  without intensity normalization.
* ``preproc_yAware/<dataset>/<subject>/<subject>_yaware_ready.npy``:
  the full 121 x 145 x 121 volume after whole-volume z-scoring, with a leading
  channel axis.

The paper evidence is intentionally distinguished from release evidence:

* AnatCL section 4.1 reports CAT12, nonlinear MNI registration, modulated GM,
  1.5 mm and 121 x 145 x 121. Its release expects 121 x 128 x 121; the listed
  Normalize(mean=0, std=1) is an identity transform.
* y-Aware section 3 reports nonlinear MNI registration, GM extraction, 1.5 mm
  and 121 x 145 x 121. Its release applies a whole-array z-score and expects
  (1, 121, 145, 121).
* The explicit Dartel/IXI555 choice comes from the historical brainprep/CAT12
  inference recipe used to generate the released inputs, not from either paper
  text alone.

The definitive brain-age manifest is the default input. Outputs are separated
by dataset and subject to avoid cross-dataset identifier collisions. CAT12
intermediates, MATLAB logs, CAT12 reports and per-subject QC are retained.

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
FINETUNING_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = FINETUNING_ROOT.parents[1]

DEFAULT_MANIFEST = (
    FINETUNING_ROOT
    / "brainage_splits"
    / "brainage_definitive_subject_manifest.tsv"
)
DEFAULT_ANATCL_OUTPUT_ROOT = FINETUNING_ROOT / "preproc_AnatCL"
DEFAULT_YAWARE_OUTPUT_ROOT = FINETUNING_ROOT / "preproc_yAware"
DEFAULT_LOG_ROOT = FINETUNING_ROOT / "logs_preprocs" / "AnatCL_yAware_shared"
DEFAULT_TEMPLATE = PROJECT_ROOT / "templates" / "Template_1_IXI555_MNI152.nii"

PIPELINE_VERSION = "cat12-ixi555-anatcl-yaware-dual-output-v2"
EXPECTED_SHAPE = (121, 145, 121)
EXPECTED_ZOOMS = (1.5, 1.5, 1.5)
ANATCL_SHAPE = (1, 121, 128, 121)
YAWARE_SHAPE = (1, 121, 145, 121)
YAWARE_ZSCORE_EPS = 1e-8
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")

REQUIRED_MANIFEST_COLUMNS = {
    "dataset",
    "subject",
    "raw_t1w_path",
    "age",
    "sex",
    "split",
    "inclusion_status",
    "required_offline_groups",
}


CAT12_BATCH_TEMPLATE = r"""
% Shared CAT12 VBM batch for AnatCL and y-Aware (auto-generated).
matlabbatch{{1}}.spm.tools.cat.estwrite.data = {{'{input_file}'}};
matlabbatch{{1}}.spm.tools.cat.estwrite.nproc = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.tpm = {{fullfile(spm('dir'),'tpm','TPM.nii')}};
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.affreg = 'mni';
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.registration.dartel.darteltpm = {{'{template_path}'}};
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.vox = 1.5;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.native = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.mod = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.native = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.mod = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.dartel = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.surface = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.ROImenu.noROI = struct([]);
spm('defaults','fmri');
spm_jobman('run', matlabbatch);
""".strip() + "\n"


CONTRACT_PAYLOAD = {
    "pipeline_version": PIPELINE_VERSION,
    "expected_shape": EXPECTED_SHAPE,
    "expected_zooms_mm": EXPECTED_ZOOMS,
    "cat12_batch": CAT12_BATCH_TEMPLATE,
    "anatcl_view": {
        "materialized_output": "full mwp1 NIfTI named *_anatcl_ready.nii.gz",
        "shape": ANATCL_SHAPE,
        "operation": "centre crop Y[8:136] from shared GM; add channel",
        "normalization": "none (release Normalize(mean=0,std=1) is identity)",
    },
    "yaware_view": {
        "materialized_output": "z-scored channel-first NumPy array named *_yaware_ready.npy",
        "shape": YAWARE_SHAPE,
        "operation": "whole-volume z-score; add channel",
        "epsilon": YAWARE_ZSCORE_EPS,
    },
}
PIPELINE_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(CONTRACT_PAYLOAD, sort_keys=True).encode("utf-8")
).hexdigest()


# Return a timezone-aware UTC timestamp for QC records and run manifests.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Compute a file's SHA-256 digest in chunks for provenance and integrity checks.
# Chunked reading avoids loading large images or templates fully into memory.
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Write JSON through a temporary file and atomically replace the destination.
# This prevents interrupted jobs from leaving a partial QC or summary file.
def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


# Write a tabular run manifest through a temporary file before replacing the target.
# The atomic replacement keeps cohort-level status records complete after failures.
def atomic_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


# Escape single quotes so filesystem paths can be embedded safely in MATLAB strings.
def matlab_quote(path: Path) -> str:
    return str(path).replace("'", "''")


# Return the final log lines for concise diagnostics when MATLAB or CAT12 fails.
# An unavailable log is represented explicitly instead of masking the original error.
def tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<log unavailable>"
    return "\n".join(content[-lines:])


# Record the raw input path, size, and modification time used by resume validation.
# This inexpensive fingerprint detects replacement of the source image between runs.
def raw_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


# Derive the complete six-file Dartel template set from Template_1 and require all files.
# This prevents CAT12 from running with a partial or ambiguously named IXI555 template set.
def template_set(template_1: Path) -> list[Path]:
    match = re.fullmatch(r"Template_1_(.+)\.nii", template_1.name)
    if not match:
        raise ValueError(
            "--template must name the first Dartel template as "
            "Template_1_<name>.nii so Template_2..6 can be verified"
        )
    suffix = match.group(1)
    paths = [template_1.with_name(f"Template_{index}_{suffix}.nii") for index in range(1, 7)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Dartel template files: " + ", ".join(missing))
    return paths


# Capture path, size, and SHA-256 provenance for every Dartel template file.
# The resulting records are stored in QC and checked when existing outputs are reused.
def template_provenance(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


# Load the definitive TSV manifest and enforce its schema, inclusion status, and uniqueness.
# Dataset and subject identifiers are also constrained to safe output-path components.
def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Definitive subject manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    keys: set[tuple[str, str]] = set()
    paths: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        dataset = row["dataset"].strip()
        subject = row["subject"].strip()
        raw_path = row["raw_t1w_path"].strip()
        if not SAFE_COMPONENT.fullmatch(dataset) or not SAFE_COMPONENT.fullmatch(subject):
            raise ValueError(
                f"Unsafe dataset/subject component at line {line_number}: "
                f"dataset={dataset!r}, subject={subject!r}"
            )
        key = (dataset, subject)
        if key in keys:
            raise ValueError(f"Duplicate dataset/subject at line {line_number}: {key}")
        if raw_path in paths:
            raise ValueError(f"Duplicate raw_t1w_path at line {line_number}: {raw_path}")
        if row["inclusion_status"].strip().upper() != "INCLUDED":
            raise ValueError(
                f"Non-included subject present in definitive manifest at line {line_number}: {key}"
            )
        groups = {value.strip() for value in row["required_offline_groups"].split(";")}
        if "cat12_ixi555_gm" not in groups:
            raise ValueError(f"{key} does not request cat12_ixi555_gm")
        row["dataset"] = dataset
        row["subject"] = subject
        row["raw_t1w_path"] = raw_path
        keys.add(key)
        paths.add(raw_path)
    return rows


# Apply optional dataset, split, subject, and row-limit selectors to the manifest.
# Unknown selectors and empty selections fail early rather than launching the wrong cohort.
def select_rows(
    rows: list[dict[str, str]],
    datasets: list[str] | None,
    splits: list[str] | None,
    subjects: list[str] | None,
    limit: int | None,
) -> list[dict[str, str]]:
    selected = rows
    if datasets:
        wanted = set(datasets)
        unknown = wanted - {row["dataset"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown --dataset values: {sorted(unknown)}")
        selected = [row for row in selected if row["dataset"] in wanted]
    if splits:
        wanted = set(splits)
        selected = [row for row in selected if row["split"] in wanted]
    if subjects:
        wanted = set(subjects)
        selected = [row for row in selected if row["subject"] in wanted]
        unknown = wanted - {row["subject"] for row in selected}
        if unknown:
            raise ValueError(f"Unknown --subject values after filtering: {sorted(unknown)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No subjects remain after applying selectors")
    return selected


# Construct deterministic, dataset-scoped output directories and filenames for one subject.
# The QC record is colocated with the shared AnatCL NIfTI that generated both model views.
def output_paths(
    anatcl_output_root: Path,
    yaware_output_root: Path,
    dataset: str,
    subject: str,
) -> tuple[Path, Path, Path, Path, Path]:
    anatcl_subject_dir = anatcl_output_root / dataset / subject
    yaware_subject_dir = yaware_output_root / dataset / subject
    anatcl_output = anatcl_subject_dir / f"{subject}_anatcl_ready.nii.gz"
    yaware_output = yaware_subject_dir / f"{subject}_yaware_ready.npy"
    qc = anatcl_subject_dir / "preprocessing_qc.json"
    return anatcl_subject_dir, yaware_subject_dir, anatcl_output, yaware_output, qc


# Load a CAT12 modulated GM NIfTI and enforce the shared geometry and numerical contract.
# Return the image, float32 voxels, and summary statistics used by downstream QC.
def validate_shared_gm(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Shared CAT12 GM output not found: {path}")
    image = nib.load(str(path))
    shape = tuple(int(value) for value in image.shape)
    if shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape for {path}: {shape}, expected {EXPECTED_SHAPE}")
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    if not np.allclose(zooms, EXPECTED_ZOOMS, rtol=0.0, atol=1e-3):
        raise ValueError(f"Unexpected zooms for {path}: {zooms}, expected {EXPECTED_ZOOMS}")
    affine = np.asarray(image.affine)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"Invalid affine in {path}")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-8:
        raise ValueError(f"Degenerate affine in {path}")

    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite voxels in {path}")
    standard_deviation = float(np.std(data))
    if standard_deviation <= 0.0:
        raise ValueError(f"Zero-variance GM image: {path}")
    stats = {
        "shape": list(shape),
        "zooms_mm": [float(value) for value in zooms],
        "dtype_loaded": str(data.dtype),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
        "mean": float(np.mean(data)),
        "standard_deviation": standard_deviation,
        "nonzero_voxels": int(np.count_nonzero(data)),
        "affine_determinant": float(np.linalg.det(affine[:3, :3])),
    }
    return image, data, stats


# Validate the materialized y-Aware array's shape, dtype, finiteness, and z-score statistics.
# Return the checked array and compact statistics for the subject-level QC record.
def validate_yaware_output(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"y-Aware output not found: {path}")
    data = np.load(path, allow_pickle=False)
    if data.shape != YAWARE_SHAPE:
        raise ValueError(f"Unexpected y-Aware shape for {path}: {data.shape}, expected {YAWARE_SHAPE}")
    if data.dtype != np.float32:
        raise ValueError(f"Unexpected y-Aware dtype for {path}: {data.dtype}, expected float32")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in y-Aware output: {path}")
    stats = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
        "mean": float(np.mean(data)),
        "standard_deviation": float(np.std(data)),
    }
    if abs(stats["mean"]) > 2e-5 or abs(stats["standard_deviation"] - 1.0) > 2e-5:
        raise ValueError(
            f"Unexpected y-Aware normalization for {path}: "
            f"mean={stats['mean']}, std={stats['standard_deviation']}"
        )
    return data, stats


# Derive AnatCL's channel-first 121 x 128 x 121 centre crop from the shared GM volume.
# Intensities remain CAT12-scaled because the released AnatCL input is not re-z-scored.
def make_anatcl_input(data: np.ndarray) -> np.ndarray:
    """Return one AnatCL sample as [C, X, Y, Z] without z-scoring."""
    array = np.asarray(data, dtype=np.float32)
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(f"AnatCL shared input must be {EXPECTED_SHAPE}, got {array.shape}")
    start = (array.shape[1] - ANATCL_SHAPE[2]) // 2
    cropped = array[:, start : start + ANATCL_SHAPE[2], :]
    output = np.ascontiguousarray(cropped[np.newaxis, ...], dtype=np.float32)
    if output.shape != ANATCL_SHAPE or not np.isfinite(output).all():
        raise ValueError(f"Invalid AnatCL runtime view: shape={output.shape}")
    return output


# Derive y-Aware's channel-first input by z-scoring the complete shared GM volume.
# The operation mirrors the released inference normalization and rejects invalid variance.
def make_yaware_input(data: np.ndarray, eps: float = YAWARE_ZSCORE_EPS) -> np.ndarray:
    """Return one y-Aware sample as [C, X, Y, Z], matching release Normalize."""
    array = np.asarray(data, dtype=np.float32)
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(f"y-Aware shared input must be {EXPECTED_SHAPE}, got {array.shape}")
    mean = np.mean(array)
    standard_deviation = np.std(array)
    if not np.isfinite(mean) or not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Cannot z-score a non-finite or zero-variance y-Aware input")
    normalized = (array - mean) / (standard_deviation + eps)
    output = np.ascontiguousarray(normalized[np.newaxis, ...], dtype=np.float32)
    if output.shape != YAWARE_SHAPE or not np.isfinite(output).all():
        raise ValueError(f"Invalid y-Aware runtime view: shape={output.shape}")
    return output


# Load and validate a shared GM NIfTI before constructing the AnatCL runtime view.
# Extractors can use this helper without duplicating preprocessing assumptions.
def load_anatcl_input(path: str | Path) -> np.ndarray:
    """Load a shared NIfTI and derive the AnatCL runtime view."""
    _, data, _ = validate_shared_gm(Path(path))
    return make_anatcl_input(data)


# Load and validate a shared GM NIfTI before constructing the y-Aware runtime view.
# Extractors can use this helper to reproduce the materialized normalization exactly.
def load_yaware_input(path: str | Path) -> np.ndarray:
    """Load a shared NIfTI and derive the y-Aware runtime view."""
    _, data, _ = validate_shared_gm(Path(path))
    return make_yaware_input(data)


# Build both model-specific runtime views to confirm that one shared GM volume supports them.
# Return their shapes and intensity summaries for provenance and QC reporting.
def validate_views(data: np.ndarray) -> dict[str, Any]:
    anatcl = make_anatcl_input(data)
    yaware = make_yaware_input(data)
    return {
        "anatcl": {
            "shape": list(anatcl.shape),
            "normalization": "none",
            "crop_y": [8, 136],
            "mean": float(np.mean(anatcl)),
            "standard_deviation": float(np.std(anatcl)),
        },
        "yaware": {
            "shape": list(yaware.shape),
            "normalization": "whole-volume z-score",
            "epsilon": YAWARE_ZSCORE_EPS,
            "mean": float(np.mean(yaware)),
            "standard_deviation": float(np.std(yaware)),
        },
    }


# Locate the CAT12 report log and require explicit confirmation of Dartel registration.
# The matching line is retained as evidence that the intended registration branch ran.
def assert_dartel_in_catlog(work_dir: Path) -> tuple[Path, str]:
    candidates = list(work_dir.glob("report/catlog_*.txt"))
    candidates.extend(work_dir.glob("catlog_*.txt"))
    candidates.extend(work_dir.glob("**/catlog_*.txt"))
    unique = {path.resolve(): path for path in candidates}
    if not unique:
        raise RuntimeError(f"CAT12 catlog_*.txt not found under {work_dir}")
    catlog = max(unique.values(), key=lambda path: path.stat().st_mtime_ns)
    text = catlog.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "dartel registration" in line.lower():
            return catlog, line.strip()
    observed = [line.strip() for line in text.splitlines() if "registration" in line.lower()]
    raise RuntimeError(
        f"CAT12 did not confirm Dartel Registration in {catlog}; "
        f"observed={observed[:20]}"
    )


# Locate CAT12's modulated normalized GM output, preferring its deterministic filename.
# Ambiguous fallback matches are rejected so the wrong tissue image cannot be selected.
def find_modulated_gm(work_dir: Path) -> Path:
    expected = work_dir / "mri" / "mwp1input_T1w.nii"
    if expected.is_file():
        return expected
    candidates = sorted((work_dir / "mri").glob("mwp1*.nii"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one CAT12 mwp1 output under {work_dir / 'mri'}, "
            f"found {[str(path) for path in candidates]}"
        )
    return candidates[0]


# Materialize the shared GM NIfTI through a temporary file and atomic replacement.
# This avoids exposing a truncated image if saving is interrupted.
def save_nifti_atomic(image: nib.spatialimages.SpatialImage, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.nii.gz")
    try:
        nib.save(image, str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


# Materialize the y-Aware NumPy array through a temporary file and atomic replacement.
# Pickling is disabled so the output remains a plain, reproducible numerical array.
def save_npy_atomic(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.npy")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


# Resolve either an explicit MATLAB executable or a command available on PATH.
# Fail before cohort processing if the selected executable is missing or not executable.
def resolve_matlab(command: str) -> str:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or "/" in command:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FileNotFoundError(f"MATLAB executable is not executable: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"MATLAB executable not found on PATH: {command}")
    return resolved


# Check whether existing outputs still match their raw input, templates, contract, and hashes.
# This is the provenance gate used before validated files are skipped or reused.
def qc_matches(
    qc_path: Path,
    anatcl_output: Path,
    yaware_output: Path,
    raw: Path,
    templates: list[dict[str, Any]],
) -> bool:
    if not qc_path.is_file() or not anatcl_output.is_file() or not yaware_output.is_file():
        return False
    try:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        qc.get("status") == "PASS"
        and qc.get("pipeline_contract_sha256") == PIPELINE_CONTRACT_SHA256
        and qc.get("dartel_templates") == templates
        and qc.get("raw_input") == raw_fingerprint(raw)
        and qc.get("anatcl_output_sha256") == sha256_file(anatcl_output)
        and qc.get("yaware_output_sha256") == sha256_file(yaware_output)
    )


# Run or validate the complete shared CAT12 workflow for one manifest subject.
# The function materializes both model inputs, records QC, and converts failures into run records.
def process_subject(
    row: dict[str, str],
    anatcl_output_root: Path,
    yaware_output_root: Path,
    log_root: Path,
    matlab: str,
    template_1: Path,
    templates: list[dict[str, Any]],
    run_id: str,
    force: bool,
    validate_existing: bool,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    dataset = row["dataset"]
    subject = row["subject"]
    raw = Path(row["raw_t1w_path"])
    (
        anatcl_subject_dir,
        yaware_subject_dir,
        anatcl_output,
        yaware_output,
        qc_path,
    ) = output_paths(anatcl_output_root, yaware_output_root, dataset, subject)
    subject_log_dir = log_root / dataset
    subject_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = subject_log_dir / f"{subject}_{run_id}.log"
    historical_qc = subject_log_dir / f"{subject}_{run_id}_qc.json"
    started = utc_now()
    record: dict[str, Any] = {
        "dataset": dataset,
        "subject": subject,
        "split": row["split"],
        "age": row["age"],
        "sex": row["sex"],
        "raw_t1w_path": str(raw),
        "anatcl_output_path": str(anatcl_output),
        "yaware_output_path": str(yaware_output),
        "qc_path": str(qc_path),
        "log_path": str(log_path),
        "started_utc": started,
        "finished_utc": "",
        "status": "",
        "message": "",
    }
    archived_work_dir: Path | None = None

    try:
        if not raw.is_file():
            raise FileNotFoundError(f"Raw T1w input not found: {raw}")

        if validate_existing:
            _, data, _ = validate_shared_gm(anatcl_output)
            validate_views(data)
            saved_yaware, _ = validate_yaware_output(yaware_output)
            expected_yaware = make_yaware_input(data)
            if not np.array_equal(saved_yaware, expected_yaware):
                raise RuntimeError("Stored y-Aware array differs from the view derived from AnatCL mwp1")
            if not qc_matches(qc_path, anatcl_output, yaware_output, raw, templates):
                raise RuntimeError(f"Outputs are valid but provenance QC is stale or missing: {qc_path}")
            record.update(status="VALIDATED", message="Both outputs and provenance QC validated")
            return record

        if (anatcl_output.exists() or yaware_output.exists()) and not force:
            _, data, _ = validate_shared_gm(anatcl_output)
            validate_views(data)
            saved_yaware, _ = validate_yaware_output(yaware_output)
            if not np.array_equal(saved_yaware, make_yaware_input(data)):
                raise RuntimeError("Existing y-Aware array differs from the AnatCL mwp1 source")
            if not qc_matches(qc_path, anatcl_output, yaware_output, raw, templates):
                raise RuntimeError(
                    "Existing output pair has stale/missing provenance. "
                    "Inspect it and rerun with --force if replacement is intended."
                )
            record.update(status="SKIPPED", message="Current validated AnatCL/y-Aware outputs already exist")
            return record

        anatcl_subject_dir.mkdir(parents=True, exist_ok=True)
        yaware_subject_dir.mkdir(parents=True, exist_ok=True)
        work_dir = anatcl_subject_dir / "cat12_work"
        if work_dir.is_dir() and any(work_dir.iterdir()):
            archived_work_dir = anatcl_subject_dir / f"cat12_work_interrupted_{run_id}"
            if archived_work_dir.exists():
                raise FileExistsError(
                    f"Interrupted-work archive already exists: {archived_work_dir}"
                )
            os.replace(work_dir, archived_work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        input_copy = work_dir / "input_T1w.nii"
        input_image = nib.load(str(raw))
        nib.save(input_image, str(input_copy))

        batch_file = work_dir / f"cat12_batch_{run_id}.m"
        batch_file.write_text(
            CAT12_BATCH_TEMPLATE.format(
                input_file=matlab_quote(input_copy),
                template_path=matlab_quote(template_1),
            ),
            encoding="utf-8",
        )
        command = [
            matlab,
            "-nodisplay",
            "-nosplash",
            "-batch",
            f"run('{matlab_quote(batch_file)}')",
        ]
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"pipeline_version={PIPELINE_VERSION}\n")
            log_handle.write(f"pipeline_contract_sha256={PIPELINE_CONTRACT_SHA256}\n")
            log_handle.write(f"dataset={dataset}\nsubject={subject}\nraw={raw}\n")
            log_handle.write(f"template={template_1}\n")
            log_handle.write(f"anatcl_output={anatcl_output}\n")
            log_handle.write(f"yaware_output={yaware_output}\n")
            log_handle.write("command=" + " ".join(command) + "\n\n")
            log_handle.flush()
            completed = subprocess.run(
                command,
                cwd=work_dir,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"MATLAB/CAT12 exited with code {completed.returncode}. Log tail:\n{tail(log_path)}"
            )

        catlog, dartel_line = assert_dartel_in_catlog(work_dir)
        gm_path = find_modulated_gm(work_dir)
        gm_image, data, stats = validate_shared_gm(gm_path)
        views = validate_views(data)
        save_nifti_atomic(gm_image, anatcl_output)
        _, final_data, final_stats = validate_shared_gm(anatcl_output)
        final_views = validate_views(final_data)
        if views["anatcl"]["shape"] != final_views["anatcl"]["shape"]:
            raise RuntimeError("AnatCL view changed while materializing the mwp1 NIfTI")
        if views["yaware"]["shape"] != final_views["yaware"]["shape"]:
            raise RuntimeError("y-Aware view changed while materializing the mwp1 NIfTI")

        expected_yaware = make_yaware_input(final_data)
        save_npy_atomic(expected_yaware, yaware_output)
        saved_yaware, yaware_stats = validate_yaware_output(yaware_output)
        if not np.array_equal(saved_yaware, expected_yaware):
            raise RuntimeError("Saved y-Aware array is not bit-exact to the derived runtime view")

        qc = {
            "status": "PASS",
            "created_utc": utc_now(),
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_contract_sha256": PIPELINE_CONTRACT_SHA256,
            "script_path": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "dataset": dataset,
            "subject": subject,
            "split": row["split"],
            "raw_input": raw_fingerprint(raw),
            "template_1_sha256": templates[0]["sha256"],
            "dartel_templates": templates,
            "matlab_executable": matlab,
            "cat12_batch_file": str(batch_file),
            "cat12_log": str(catlog),
            "dartel_confirmation": dartel_line,
            "cat12_mwp1_source": str(gm_path),
            "archived_interrupted_work_dir": (
                str(archived_work_dir) if archived_work_dir is not None else None
            ),
            "anatcl_output": str(anatcl_output),
            "anatcl_output_sha256": sha256_file(anatcl_output),
            "anatcl_output_statistics": final_stats,
            "yaware_output": str(yaware_output),
            "yaware_output_sha256": sha256_file(yaware_output),
            "yaware_output_statistics": yaware_stats,
            "runtime_views": final_views,
        }
        atomic_json(qc_path, qc)
        atomic_json(historical_qc, qc)
        record.update(
            status="PASS",
            message=(
                f"{tuple(stats['shape'])} @ {tuple(stats['zooms_mm'])} mm; "
                "Dartel confirmed; AnatCL NIfTI and y-Aware NPY saved"
            ),
        )
    except Exception as error:  # Continue the cohort while retaining a full failure log.
        record.update(status="FAILED", message=f"{type(error).__name__}: {error}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n=== PIPELINE FAILURE ===\n")
            handle.write(traceback.format_exc())
    finally:
        record["finished_utc"] = utc_now()
    return record


# Aggregate subject records into durable TSV and JSON run-level artifacts.
# The JSON also captures inputs, template provenance, arguments, and status counts.
def write_run_artifacts(
    records: list[dict[str, Any]],
    log_root: Path,
    run_id: str,
    manifest_path: Path,
    anatcl_output_root: Path,
    yaware_output_root: Path,
    templates: list[dict[str, Any]],
    arguments: dict[str, Any],
) -> tuple[Path, Path]:
    fields = [
        "dataset",
        "subject",
        "split",
        "age",
        "sex",
        "raw_t1w_path",
        "anatcl_output_path",
        "yaware_output_path",
        "qc_path",
        "log_path",
        "status",
        "message",
        "started_utc",
        "finished_utc",
    ]
    tsv_path = log_root / f"_anatcl_yaware_run_{run_id}.tsv"
    json_path = log_root / f"_anatcl_yaware_run_{run_id}.json"
    atomic_tsv(tsv_path, records, fields)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    atomic_json(
        json_path,
        {
            "run_id": run_id,
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_contract_sha256": PIPELINE_CONTRACT_SHA256,
            "script_path": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "definitive_manifest": str(manifest_path),
            "definitive_manifest_sha256": sha256_file(manifest_path),
            "anatcl_output_root": str(anatcl_output_root),
            "yaware_output_root": str(yaware_output_root),
            "dartel_templates": templates,
            "arguments": arguments,
            "counts": counts,
            "records_tsv": str(tsv_path),
        },
    )
    return tsv_path, json_path


# Define the command-line interface for cohort selection, execution, and validation modes.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared CAT12/Dartel IXI555 GM preprocessing for AnatCL and y-Aware"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--anatcl-output-root", type=Path, default=DEFAULT_ANATCL_OUTPUT_ROOT
    )
    parser.add_argument(
        "--yaware-output-root", type=Path, default=DEFAULT_YAWARE_OUTPUT_ROOT
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--matlab-cmd",
        default=os.environ.get("MATLAB_CMD", "matlab"),
        help="MATLAB executable or $MATLAB_CMD (default: matlab on PATH)",
    )
    parser.add_argument("--dataset", action="append", help="Dataset selector; repeatable")
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "validation", "internal_test"],
        help="Split selector; repeatable",
    )
    parser.add_argument("--subject", action="append", help="Subject selector; repeatable")
    parser.add_argument("--limit", type=int, help="Process only the first N selected rows")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent MATLAB jobs (default: 1)")
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=0.0,
        help="Per-subject MATLAB timeout; 0 disables it",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing outputs")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate outputs, both runtime views and provenance without running CAT12",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the selected plan only")
    return parser


# Validate arguments, resolve inputs, dispatch subject jobs, and write the final run summary.
# A nonzero exit status is returned when any selected subject fails.
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.timeout_minutes < 0:
        parser.error("--timeout-minutes cannot be negative")
    if args.force and args.validate_existing:
        parser.error("--force and --validate-existing are mutually exclusive")

    manifest_path = args.manifest.expanduser().resolve()
    anatcl_output_root = args.anatcl_output_root.expanduser().resolve()
    yaware_output_root = args.yaware_output_root.expanduser().resolve()
    log_root = args.log_root.expanduser().resolve()
    template_1 = args.template.expanduser().resolve()
    rows = load_manifest(manifest_path)
    selected = select_rows(rows, args.dataset, args.split, args.subject, args.limit)

    print("Shared CAT12 preprocessing for AnatCL + y-Aware")
    print(f"  manifest : {manifest_path}")
    print(f"  selected : {len(selected)}/{len(rows)}")
    print(f"  AnatCL   : {anatcl_output_root}")
    print(f"  y-Aware  : {yaware_output_root}")
    print(f"  logs     : {log_root}")
    print(f"  template : {template_1}")
    print(f"  workers  : {args.workers}")
    print(f"  contract : {PIPELINE_CONTRACT_SHA256}")
    print("  views    : AnatCL=crop Y 145->128, no z-score; y-Aware=full-volume z-score")

    if args.dry_run:
        for index, row in enumerate(selected, start=1):
            _, _, anatcl_output, yaware_output, _ = output_paths(
                anatcl_output_root, yaware_output_root, row["dataset"], row["subject"]
            )
            print(
                f"  [{index}/{len(selected)}] {row['dataset']}/{row['subject']} "
                f"{row['split']}\n"
                f"      AnatCL  -> {anatcl_output}\n"
                f"      y-Aware -> {yaware_output}"
            )
        print("Dry run: no files were created or modified.")
        return

    templates_paths = template_set(template_1)
    templates = template_provenance(templates_paths)
    matlab = "<validation-only>" if args.validate_existing else resolve_matlab(args.matlab_cmd)
    timeout_seconds = args.timeout_minutes * 60.0 if args.timeout_minutes else None
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    anatcl_output_root.mkdir(parents=True, exist_ok=True)
    yaware_output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_subject,
                row,
                anatcl_output_root,
                yaware_output_root,
                log_root,
                matlab,
                template_1,
                templates,
                run_id,
                args.force,
                args.validate_existing,
                timeout_seconds,
            ): row
            for row in selected
        }
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            record = future.result()
            records.append(record)
            print(
                f"  [{completed_count}/{len(selected)}] {record['status']:<9} "
                f"{record['dataset']}/{record['subject']}: {record['message']}"
            )

    order = {(row["dataset"], row["subject"]): index for index, row in enumerate(selected)}
    records.sort(key=lambda record: order[(record["dataset"], record["subject"])])
    arguments = {
        "dataset": args.dataset or [],
        "split": args.split or [],
        "subject": args.subject or [],
        "limit": args.limit,
        "workers": args.workers,
        "timeout_minutes": args.timeout_minutes,
        "force": args.force,
        "validate_existing": args.validate_existing,
    }
    tsv_path, json_path = write_run_artifacts(
        records,
        log_root,
        run_id,
        manifest_path,
        anatcl_output_root,
        yaware_output_root,
        templates,
        arguments,
    )
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print(f"Run manifest: {tsv_path}")
    print(f"Run summary : {json_path}")
    print(f"Summary     : {counts}")
    if counts.get("FAILED", 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
