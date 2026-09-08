#!/usr/bin/env python3
"""Audit all seven ON-Harmony model-input and embedding pipelines.

The audit validates the five materialized preprocessing pipelines, every
canonical embedding CSV, the available extraction manifests, and exact cohort
agreement across all seven models. Statistical result files are deliberately
outside this script's scope and are checked by the analysis audit.

Use ``--full-data`` to read every materialized final array and reject any NaN
or Inf values. Without it, NIfTI/NumPy headers, logs, QC sidecars, hashes, CSV
contents, and cohort identities are still checked.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
ON_HARMONY_RELATIVE = Path("preprocessings") / "OnHarmonyDataset"
STEM_RE = re.compile(r"(sub-[A-Za-z0-9]+_ses-[A-Za-z0-9]+_T1w)")

BSF_PIPELINE_VERSION = "brainsegfounder-ukb-gdcoff-onharmony-v4"
BSF_CONTRACT_SHA256 = (
    "08f983cc00900d7c57015eb8482eb763f90bd2eeb9bd960aa06563020dc7e30b"
)


@dataclass(frozen=True)
class MaterializedSpec:
    """Expected output, log, and optional sidecar contract for one model."""

    name: str
    directory: str
    filelist: str
    suffix: str
    shape: tuple[int, ...]
    zooms: tuple[float, float, float] | None
    log_directory: Path
    log_prefix: str
    kind: str = "nifti"
    required_log_fragments: tuple[str, ...] = ("All done.",)
    require_dartel_log: bool = False
    qc_kind: str | None = None


@dataclass(frozen=True)
class EmbeddingFileSpec:
    """One CSV and its exact coordinate groups."""

    filename: str
    feature_groups: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ExtractionSpec:
    """Canonical extraction contract for one model."""

    name: str
    filelist: str
    files: tuple[EmbeddingFileSpec, ...]
    manifest: str | None = None
    raw_inputs: bool = False


MATERIALIZED_SPECS = (
    MaterializedSpec(
        "3D-Neuro-SimCLR",
        "preproc_SimCLR",
        "filelist_3DNeuroSimCLR.txt",
        "_simclr_ready.nii.gz",
        (193, 229, 193),
        (1.0, 1.0, 1.0),
        Path("logs_preprocs"),
        "simclr_",
        required_log_fragments=("All done.", "exit_code: 0"),
    ),
    MaterializedSpec(
        "AnatCL",
        "preproc_AnatCL",
        "filelist_AnatCL.txt",
        "_anatcl_ready.nii.gz",
        (121, 145, 121),
        (1.5, 1.5, 1.5),
        Path("logs_preprocs"),
        "anatcl_corrected_",
        require_dartel_log=True,
    ),
    MaterializedSpec(
        "BrainIAC",
        "preproc_BrainIAC",
        "filelist_BrainIAC.txt",
        "_brainiac_ready.nii.gz",
        (96, 96, 96),
        (1.0, 1.0, 1.0),
        Path("logs_preprocs"),
        "brainiac_",
        required_log_fragments=("All done.", "exit_code: 0"),
        qc_kind="brainiac",
    ),
    MaterializedSpec(
        "BrainSegFounder",
        "preproc_BrainSegFounder",
        "filelist_BrainSegFounder.txt",
        "_brainsegfounder_ready.nii.gz",
        (96, 96, 96),
        (2.0, 2.0, 2.0),
        Path("logs_preprocs") / "BrainSegFounder",
        "brainsegfounder_",
        required_log_fragments=("All done.", "exit_code: 0"),
        qc_kind="brainsegfounder",
    ),
    MaterializedSpec(
        "y-Aware",
        "preproc_yAware",
        "filelist_yAware.txt",
        "_yaware_ready.npy",
        (1, 121, 145, 121),
        None,
        Path("logs_preprocs"),
        "yaware_corrected_",
        kind="npy",
        require_dartel_log=True,
    ),
)


EXTRACTION_SPECS = (
    ExtractionSpec(
        "3D-Neuro-SimCLR",
        "filelist_3DNeuroSimCLR.txt",
        (EmbeddingFileSpec("embeddings_3DNeuroSimCLR.csv", (("emb_", 512),)),),
        manifest="embeddings_3DNeuroSimCLR_manifest.json",
    ),
    ExtractionSpec(
        "AnatCL",
        "filelist_AnatCL.txt",
        tuple(
            EmbeddingFileSpec(f"embeddings_AnatCL_fold{fold}.csv", (("emb_", 512),))
            for fold in range(5)
        ),
    ),
    ExtractionSpec(
        "BrainIAC",
        "filelist_BrainIAC.txt",
        (EmbeddingFileSpec("embeddings_BrainIAC.csv", (("emb_", 768),)),),
    ),
    ExtractionSpec(
        "BrainSegFounder",
        "filelist_BrainSegFounder.txt",
        (EmbeddingFileSpec("embeddings_BrainSegFounder.csv", (("emb_", 768),)),),
        manifest="embeddings_BrainSegFounder_manifest.json",
    ),
    ExtractionSpec(
        "Curia-2",
        "filelist_Curia2.txt",
        (
            EmbeddingFileSpec(
                "embeddings_Curia2.csv",
                (
                    ("cls_all_", 768),
                    ("patch_all_", 768),
                    ("cls_core_", 768),
                    ("patch_core_", 768),
                ),
            ),
        ),
        manifest="embeddings_Curia2_manifest.json",
        raw_inputs=True,
    ),
    ExtractionSpec(
        "NeuroVFM",
        "filelist_NeuroVFM.txt",
        (EmbeddingFileSpec("embeddings_NeuroVFM.csv", (("emb_", 768),)),),
        raw_inputs=True,
    ),
    ExtractionSpec(
        "y-Aware",
        "filelist_yAware.txt",
        (EmbeddingFileSpec("embeddings_yAware.csv", (("emb_", 1024),)),),
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse the read-only audit options."""
    parser = argparse.ArgumentParser(
        description="Validate all canonical ON-Harmony inputs and embeddings"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--expected-sessions", type=int, default=165)
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Read every final array and require all voxel values to be finite",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "JSON report path (default: OnHarmonyDataset/logs_preprocs/"
            "_preprocessing_qc_<time>.json)"
        ),
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write a JSON report (useful for a read-only mounted copy)",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish the report only after its complete JSON has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_filelist(path: Path) -> list[str]:
    """Read a nonempty ordered registry while preserving its exact strings."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    if not entries:
        raise ValueError(f"Empty filelist: {path}")
    return entries


def stem_from_text(value: object) -> str:
    """Recover the canonical subject-session T1w key from a path or label."""
    match = STEM_RE.search(str(value))
    if match is None:
        raise ValueError(f"Cannot recover ON-Harmony acquisition stem from {value!r}")
    return match.group(1)


def validate_registry(
    filelist: Path,
    expected_sessions: int,
) -> tuple[list[str], list[str], list[str]]:
    """Validate registry size and uniqueness and return entries and stems."""
    failures: list[str] = []
    try:
        entries = read_filelist(filelist)
    except Exception as error:
        return [], [], [f"cannot read filelist {filelist}: {error}"]

    stems: list[str] = []
    for entry in entries:
        try:
            stems.append(stem_from_text(entry))
        except ValueError as error:
            failures.append(str(error))
    if len(entries) != expected_sessions:
        failures.append(
            f"filelist has {len(entries)} entries; expected {expected_sessions}"
        )
    if len(set(entries)) != len(entries):
        failures.append("filelist contains duplicate paths")
    if len(set(stems)) != len(stems):
        failures.append("filelist contains duplicate acquisition stems")
    return entries, stems, failures


def finite_array(path: Path, kind: str) -> bool:
    """Check all values in a final model input without dtype conversion."""
    if kind == "npy":
        data = np.load(path, mmap_mode="r")
    else:
        data = np.asanyarray(nib.load(str(path)).dataobj)
    return bool(np.isfinite(data).all())


def validate_brainiac_qc(
    subject_dir: Path,
    stem: str,
    output: Path,
) -> list[str]:
    """Accept the retained historical or current BrainIAC QC schema."""
    failures: list[str] = []
    qc_json = subject_dir / f"{stem}_qc.json"
    qc_png = subject_dir / f"{stem}_qc.png"
    if not qc_json.is_file() or not qc_png.is_file() or qc_png.stat().st_size == 0:
        return [f"missing BrainIAC QC JSON/PNG for {stem}"]
    try:
        payload = json.loads(qc_json.read_text(encoding="utf-8"))
    except Exception as error:
        return [f"cannot read BrainIAC QC JSON for {stem}: {error}"]

    if str(payload.get("status", "")).upper() != "PASS":
        failures.append(f"BrainIAC QC status is not PASS for {stem}")
    if payload.get("brainiac_shape") != [96, 96, 96]:
        failures.append(f"BrainIAC QC has the wrong final shape for {stem}")
    for key in ("shape_ok", "norm_ok", "brain_ok"):
        if payload.get(key) is not True:
            failures.append(f"BrainIAC QC does not confirm {key} for {stem}")
    hashes = payload.get("output_sha256")
    if isinstance(hashes, dict) and hashes.get("ready") != sha256_file(output):
        failures.append(f"BrainIAC ready-output hash drift for {stem}")
    return failures


def validate_brainsegfounder_qc(
    subject_dir: Path,
    stem: str,
    output: Path,
) -> list[str]:
    """Validate the current BSF scientific contract and final-output hash."""
    failures: list[str] = []
    qc_json = subject_dir / "preprocessing_qc.json"
    if not qc_json.is_file():
        return [f"missing BrainSegFounder preprocessing_qc.json for {stem}"]
    try:
        payload = json.loads(qc_json.read_text(encoding="utf-8"))
    except Exception as error:
        return [f"cannot read BrainSegFounder QC JSON for {stem}: {error}"]

    if payload.get("status") != "PASS":
        failures.append(f"BrainSegFounder QC status is not PASS for {stem}")
    if payload.get("pipeline_version") != BSF_PIPELINE_VERSION:
        failures.append(f"BrainSegFounder pipeline-version drift for {stem}")
    if payload.get("pipeline_contract_sha256") != BSF_CONTRACT_SHA256:
        failures.append(f"BrainSegFounder preprocessing-contract drift for {stem}")
    model_ready = payload.get("model_ready_output", {})
    if model_ready.get("shape") != [96, 96, 96]:
        failures.append(f"BrainSegFounder QC has the wrong final shape for {stem}")
    observed_spacing = model_ready.get("spacing_mm")
    if observed_spacing is None or not np.allclose(
        observed_spacing, [2.0, 2.0, 2.0], rtol=0.0, atol=1e-5
    ):
        failures.append(f"BrainSegFounder QC has the wrong spacing for {stem}")
    hashes = payload.get("output_sha256", {})
    if hashes.get("ready") != sha256_file(output):
        failures.append(f"BrainSegFounder ready-output hash drift for {stem}")
    return failures


def validate_qc_sidecar(
    spec: MaterializedSpec,
    subject_dir: Path,
    stem: str,
    output: Path,
) -> list[str]:
    """Dispatch the model-specific QC-sidecar contract."""
    if spec.qc_kind == "brainiac":
        return validate_brainiac_qc(subject_dir, stem, output)
    if spec.qc_kind == "brainsegfounder":
        return validate_brainsegfounder_qc(subject_dir, stem, output)
    return []


def check_materialized_pipeline(
    root: Path,
    spec: MaterializedSpec,
    expected_sessions: int,
    full_data: bool,
) -> tuple[dict[str, Any], set[str]]:
    """Audit one materialized input pipeline against its current layout."""
    failures: list[str] = []
    warnings: list[str] = []
    filelist = root / "DATA" / "filelists" / spec.filelist
    entries, stems, registry_failures = validate_registry(filelist, expected_sessions)
    failures.extend(registry_failures)

    on_harmony = root / ON_HARMONY_RELATIVE
    output_root = on_harmony / spec.directory
    log_root = on_harmony / spec.log_directory
    n_outputs = 0
    n_headers_ok = 0
    n_finite_ok = 0
    n_logs_ok = 0
    n_dartel_ok = 0
    n_qc_ok = 0

    for index, stem in enumerate(stems, start=1):
        subject_dir = output_root / stem
        output = subject_dir / f"{stem}{spec.suffix}"
        if not output.is_file() or output.stat().st_size < 1024:
            failures.append(f"missing or undersized output: {output}")
            continue
        n_outputs += 1

        try:
            if spec.kind == "npy":
                observed_shape = tuple(np.load(output, mmap_mode="r").shape)
                observed_zooms = None
            else:
                image = nib.load(str(output))
                observed_shape = tuple(image.shape)
                observed_zooms = tuple(
                    round(float(value), 5)
                    for value in image.header.get_zooms()[:3]
                )
            if observed_shape != spec.shape:
                failures.append(
                    f"wrong shape for {stem}: {observed_shape}; expected {spec.shape}"
                )
            elif spec.zooms is not None and not np.allclose(
                observed_zooms, spec.zooms, rtol=0.0, atol=1e-5
            ):
                failures.append(
                    f"wrong voxel size for {stem}: {observed_zooms}; "
                    f"expected {spec.zooms}"
                )
            else:
                n_headers_ok += 1
        except Exception as error:
            failures.append(f"cannot read {output}: {type(error).__name__}: {error}")
            continue

        if full_data:
            try:
                if finite_array(output, spec.kind):
                    n_finite_ok += 1
                else:
                    failures.append(f"NaN or Inf values in {output}")
            except Exception as error:
                failures.append(
                    f"cannot check voxel values in {output}: "
                    f"{type(error).__name__}: {error}"
                )

        log_path = log_root / f"{spec.log_prefix}{stem}.log"
        if not log_path.is_file() or log_path.stat().st_size == 0:
            failures.append(f"missing or empty pipeline log: {log_path}")
        else:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            missing = [
                fragment for fragment in spec.required_log_fragments
                if fragment not in text
            ]
            if missing:
                failures.append(
                    f"pipeline log for {stem} lacks markers: {', '.join(missing)}"
                )
            else:
                n_logs_ok += 1

        if spec.require_dartel_log:
            matches = sorted(
                (subject_dir / "cat12_work" / "report").glob("catlog_*.txt")
            )
            if not matches:
                failures.append(f"missing CAT12 runtime log for {stem}")
            else:
                cat_text = matches[-1].read_text(errors="replace").lower()
                if "dartel registration" in cat_text:
                    n_dartel_ok += 1
                else:
                    failures.append(
                        f"CAT12 runtime log does not confirm Dartel for {stem}"
                    )

        if spec.qc_kind is not None:
            qc_failures = validate_qc_sidecar(spec, subject_dir, stem, output)
            failures.extend(qc_failures)
            if not qc_failures:
                n_qc_ok += 1

        if index % 25 == 0 or index == len(stems):
            print(f"  {spec.name}: checked {index}/{len(stems)}", flush=True)

    discovered: set[str] = set()
    if output_root.is_dir():
        for directory in output_root.iterdir():
            if directory.is_dir() and (
                directory / f"{directory.name}{spec.suffix}"
            ).is_file():
                discovered.add(directory.name)
    unexpected = sorted(discovered - set(stems))
    if unexpected:
        warnings.append(
            f"{len(unexpected)} final outputs are not listed; first: "
            + ", ".join(unexpected[:5])
        )

    report = {
        "pipeline": spec.name,
        "pipeline_type": "materialized_preprocessing",
        "status": "PASS" if not failures else "FAIL",
        "filelist": str(filelist),
        "filelist_sha256": sha256_file(filelist) if filelist.is_file() else None,
        "output_root": str(output_root),
        "log_root": str(log_root),
        "n_filelist": len(entries),
        "n_unique_stems": len(set(stems)),
        "n_outputs": n_outputs,
        "n_headers_ok": n_headers_ok,
        "n_finite_ok": n_finite_ok if full_data else None,
        "n_logs_ok": n_logs_ok,
        "n_dartel_logs_ok": n_dartel_ok if spec.require_dartel_log else None,
        "n_qc_sidecars_ok": n_qc_ok if spec.qc_kind is not None else None,
        "full_data_checked": full_data,
        "failures": failures,
        "warnings": warnings,
    }
    return report, set(stems)


def embedding_path(root: Path, filename: str) -> Path:
    """Resolve the canonical embedding directory with legacy fallback."""
    candidates = (
        root / "DATA" / "embeddings" / filename,
        root / "DATA" / filename,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def frame_stems(frame: pd.DataFrame, path: Path) -> list[str]:
    """Recover ordered acquisition identities from any supported CSV schema."""
    if "stem" in frame:
        values = frame["stem"].astype(str)
    elif "file_path" in frame:
        values = frame["file_path"].astype(str)
    elif {"subject_id", "session_id"}.issubset(frame.columns):
        values = (
            frame["subject_id"].astype(str)
            + "_"
            + frame["session_id"].astype(str)
            + "_T1w"
        )
    else:
        raise ValueError(f"{path}: no columns from which to recover acquisition stems")
    return [stem_from_text(value) for value in values]


def validate_embedding_csv(
    path: Path,
    spec: EmbeddingFileSpec,
    expected_sessions: int,
    expected_stems: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Validate rows, coordinate columns, finite values, and cohort identity."""
    failures: list[str] = []
    if not path.is_file():
        return ({"path": str(path), "status": "FAIL"}, [f"missing CSV: {path}"])
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        return (
            {"path": str(path), "status": "FAIL"},
            [f"cannot read {path}: {type(error).__name__}: {error}"],
        )

    if len(frame) != expected_sessions:
        failures.append(
            f"{path.name}: {len(frame)} rows; expected {expected_sessions}"
        )
    feature_columns: list[str] = []
    for prefix, dimensions in spec.feature_groups:
        expected = [f"{prefix}{index}" for index in range(dimensions)]
        observed = [column for column in frame if column.startswith(prefix)]
        if observed != expected:
            failures.append(
                f"{path.name}: {prefix} columns are incomplete or out of order "
                f"({len(observed)}/{dimensions})"
            )
        feature_columns.extend(observed)
    if feature_columns and not bool(
        np.isfinite(frame[feature_columns].to_numpy(dtype=np.float64)).all()
    ):
        failures.append(f"{path.name}: embedding matrix contains NaN or Inf")

    stems: list[str] = []
    try:
        stems = frame_stems(frame, path)
    except Exception as error:
        failures.append(str(error))
    if len(set(stems)) != len(stems):
        failures.append(f"{path.name}: duplicate acquisition stems")
    if stems and set(stems) != expected_stems:
        failures.append(f"{path.name}: cohort differs from its input filelist")

    if path.name == "embeddings_Curia2.csv":
        required = {"stem", "n_all_slices", "n_core_slices"}
        if not required.issubset(frame.columns):
            failures.append("Curia-2 CSV lacks slice-count metadata")
        elif not bool(
            (frame["n_all_slices"] > 0).all()
            and (frame["n_core_slices"] > 0).all()
            and (frame["n_core_slices"] <= frame["n_all_slices"]).all()
        ):
            failures.append("Curia-2 CSV has invalid slice counts")
    if path.name == "embeddings_NeuroVFM.csv":
        required = {"subject_id", "session_id", "scanner", "n_tokens"}
        if not required.issubset(frame.columns):
            failures.append("NeuroVFM CSV lacks token or identifier metadata")
        elif not bool((frame["n_tokens"] > 0).all()):
            failures.append("NeuroVFM CSV has invalid token counts")

    record = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "features": len(feature_columns),
        "feature_groups": [
            {"prefix": prefix, "dimensions": dimensions}
            for prefix, dimensions in spec.feature_groups
        ],
        "n_unique_stems": len(set(stems)),
        "status": "PASS" if not failures else "FAIL",
    }
    return record, failures


def validate_extraction_manifest(
    manifest_path: Path,
    csv_record: dict[str, Any],
    filelist: Path,
    expected_sessions: int,
) -> list[str]:
    """Bind a release-style extraction manifest to its filelist and CSV."""
    if not manifest_path.is_file():
        return [f"missing extraction manifest: {manifest_path}"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        return [f"cannot read extraction manifest {manifest_path}: {error}"]

    failures: list[str] = []
    if payload.get("n_rows") != expected_sessions:
        failures.append(f"{manifest_path.name}: wrong n_rows")
    if payload.get("output_sha256") != csv_record.get("sha256"):
        failures.append(f"{manifest_path.name}: output CSV hash drift")
    registry = payload.get("registry", {})
    expected_filelist_hash = sha256_file(filelist)
    if registry.get("input_list_sha256") != expected_filelist_hash:
        failures.append(f"{manifest_path.name}: input filelist hash drift")
    output_name = Path(str(payload.get("output", ""))).name
    if output_name != Path(str(csv_record.get("path", ""))).name:
        failures.append(f"{manifest_path.name}: output filename mismatch")
    return failures


def check_extraction_pipeline(
    root: Path,
    spec: ExtractionSpec,
    expected_sessions: int,
) -> tuple[dict[str, Any], set[str]]:
    """Audit all canonical embedding components for one model."""
    failures: list[str] = []
    warnings: list[str] = []
    filelist = root / "DATA" / "filelists" / spec.filelist
    entries, stems, registry_failures = validate_registry(filelist, expected_sessions)
    failures.extend(registry_failures)
    expected_stems = set(stems)

    n_inputs_present: int | None = None
    if spec.raw_inputs:
        n_inputs_present = sum(Path(entry).is_file() for entry in entries)
        if n_inputs_present == 0 and entries:
            warnings.append(
                "raw /datos/BD inputs are not visible from this mounted copy; "
                "run on the cluster to stat them"
            )
        elif n_inputs_present != len(entries):
            failures.append(
                f"only {n_inputs_present}/{len(entries)} raw inputs are accessible"
            )

    components: list[dict[str, Any]] = []
    for file_spec in spec.files:
        csv_path = embedding_path(root, file_spec.filename)
        record, component_failures = validate_embedding_csv(
            csv_path, file_spec, expected_sessions, expected_stems
        )
        components.append(record)
        failures.extend(component_failures)

    manifest_record: dict[str, Any] | None = None
    if spec.manifest is not None:
        manifest_path = embedding_path(root, spec.manifest)
        manifest_failures = validate_extraction_manifest(
            manifest_path, components[0], filelist, expected_sessions
        )
        failures.extend(manifest_failures)
        manifest_record = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
            "status": "PASS" if not manifest_failures else "FAIL",
        }

    report = {
        "pipeline": spec.name,
        "pipeline_type": "embedding_extraction",
        "status": "PASS" if not failures else "FAIL",
        "filelist": str(filelist),
        "filelist_sha256": sha256_file(filelist) if filelist.is_file() else None,
        "n_filelist": len(entries),
        "n_unique_stems": len(expected_stems),
        "n_raw_inputs_present": n_inputs_present,
        "components": components,
        "manifest": manifest_record,
        "failures": failures,
        "warnings": warnings,
    }
    return report, expected_stems


def main() -> None:
    """Run every check, compare cohorts, and optionally publish one report."""
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    if args.expected_sessions < 1:
        raise SystemExit("--expected-sessions must be positive")
    if not (root / ON_HARMONY_RELATIVE).is_dir():
        raise SystemExit(f"Invalid project root or missing OnHarmonyDataset: {root}")

    print(f"Seven-model ON-Harmony QC: {root}")
    print(f"Expected sessions: {args.expected_sessions}")
    print(f"Full voxel-data check: {args.full_data}")

    preprocessing_reports: list[dict[str, Any]] = []
    extraction_reports: list[dict[str, Any]] = []
    stem_sets: dict[str, set[str]] = {}

    for spec in MATERIALIZED_SPECS:
        print(f"\n=== {spec.name}: preprocessing ===", flush=True)
        report, stems = check_materialized_pipeline(
            root, spec, args.expected_sessions, args.full_data
        )
        preprocessing_reports.append(report)
        stem_sets[f"preprocessing:{spec.name}"] = stems

    for spec in EXTRACTION_SPECS:
        print(f"\n=== {spec.name}: embedding extraction ===", flush=True)
        report, stems = check_extraction_pipeline(
            root, spec, args.expected_sessions
        )
        extraction_reports.append(report)
        stem_sets[f"extraction:{spec.name}"] = stems

    reference_name = "extraction:3D-Neuro-SimCLR"
    reference = stem_sets[reference_name]
    cross_pipeline_failures: list[dict[str, Any]] = []
    for name, stems in stem_sets.items():
        if stems != reference:
            cross_pipeline_failures.append({
                "pipeline": name,
                "missing_vs_reference": sorted(reference - stems),
                "extra_vs_reference": sorted(stems - reference),
            })

    passed = (
        all(report["status"] == "PASS" for report in preprocessing_reports)
        and all(report["status"] == "PASS" for report in extraction_reports)
        and not cross_pipeline_failures
    )
    payload = {
        "schema_version": 3,
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": str(root),
        "auditor": str(SCRIPT_PATH),
        "auditor_sha256": sha256_file(SCRIPT_PATH),
        "expected_sessions": args.expected_sessions,
        "full_data_checked": args.full_data,
        "status": "PASS" if passed else "FAIL",
        "preprocessing": preprocessing_reports,
        "embedding_extraction": extraction_reports,
        "cross_pipeline_stem_check": (
            "PASS" if not cross_pipeline_failures else "FAIL"
        ),
        "cross_pipeline_failures": cross_pipeline_failures,
        "scope_note": (
            "Statistical result provenance is intentionally audited outside "
            "the preprocessing and extraction QC."
        ),
    }

    if not args.no_report:
        output = args.output
        if output is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = (
                root / ON_HARMONY_RELATIVE / "logs_preprocs"
                / f"_preprocessing_qc_{stamp}.json"
            )
        output = output.expanduser().resolve()
        atomic_json(output, payload)
        print(f"\nReport: {output}")

    print(f"Overall: {payload['status']}")
    for report in preprocessing_reports:
        print(
            f"  {report['pipeline']} preprocessing: {report['status']} "
            f"outputs={report['n_outputs']}/{args.expected_sessions} "
            f"headers={report['n_headers_ok']}/{args.expected_sessions} "
            f"logs={report['n_logs_ok']}/{args.expected_sessions}"
        )
    for report in extraction_reports:
        component_status = sum(
            component.get("status") == "PASS"
            for component in report["components"]
        )
        print(
            f"  {report['pipeline']} embeddings: {report['status']} "
            f"components={component_status}/{len(report['components'])}"
        )
    for report in (*preprocessing_reports, *extraction_reports):
        for failure in report["failures"][:10]:
            print(f"    FAIL [{report['pipeline']}]: {failure}")
        if len(report["failures"]) > 10:
            print(
                f"    ... {len(report['failures']) - 10} additional failures "
                f"for {report['pipeline']}"
            )
        for warning in report["warnings"]:
            print(f"    WARN [{report['pipeline']}]: {warning}")
    for failure in cross_pipeline_failures:
        print(f"    FAIL [cohort alignment]: {failure['pipeline']}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
