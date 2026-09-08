#!/usr/bin/env python3
"""Build retrospective provenance for the 165 retained AnatCL inputs.

The historical Python wrapper was deleted after preprocessing. The scientific
MATLAB batch, CAT12 runtime log, copied CAT12 input, wrapper log, and final
NIfTI remain available per acquisition. This auditor validates and hashes those
material artifacts without claiming that the current reference preprocessor
was the literal historical generator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import nibabel as nib


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
ONHARMONY_ROOT = Path("preprocessings") / "OnHarmonyDataset"
EXPECTED_ACQUISITIONS = 165
EXPECTED_SHAPE = (121, 145, 121)
EXPECTED_ZOOMS = (1.5, 1.5, 1.5)
EXPECTED_TEMPLATE_SHA256 = (
    "d681851db70c3e8619eb923b8a34a110962cf2daf6f8657fb5ba5d5469c6af9a"
)
HISTORICAL_GENERATOR = (
    "/datos/work/rnavgon/ON-Harmony_Preproc/preprocessings/"
    "preprocessing_AnatCL_corrected.py"
)
STEM_RE = re.compile(r"(sub-[A-Za-z0-9]+_ses-[A-Za-z0-9]+_T1w)")
TEMPLATE_RE = re.compile(
    r"registration\.dartel\.darteltpm\s*=\s*\{'([^']+)'\};"
)
REQUIRED_BATCH_FRAGMENTS = (
    "spm.tools.cat.estwrite.opts.affreg = 'mni';",
    "spm.tools.cat.estwrite.extopts.registration.dartel.darteltpm",
    "spm.tools.cat.estwrite.extopts.vox = 1.5;",
    "spm.tools.cat.estwrite.output.GM.mod = 1;",
)
CASE_COLUMNS = (
    "acquisition_id",
    "raw_t1w_path",
    "raw_t1w_sha256",
    "cat12_input_path",
    "cat12_input_sha256",
    "cat12_batch_path",
    "cat12_batch_sha256",
    "declared_template_path",
    "cat12_log_path",
    "cat12_log_sha256",
    "dartel_confirmation",
    "cat12_version",
    "wrapper_log_path",
    "wrapper_log_sha256",
    "output_path",
    "output_sha256",
    "shape",
    "zooms_mm",
    "status",
    "failures",
)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of one material artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using a stable serialization."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def acquisition_id(value: str | Path) -> str:
    """Extract the canonical subject-session T1w identifier."""

    match = STEM_RE.search(str(value))
    if match is None:
        raise ValueError(f"Cannot parse AnatCL acquisition from {value!r}")
    return match.group(1)


def read_registry(path: Path, expected_count: int) -> list[str]:
    """Read one ordered, unique registry with the expected cohort size."""

    if not path.is_file():
        raise FileNotFoundError(path)
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != expected_count or len(values) != len(set(values)):
        raise ValueError(
            f"Expected {expected_count} unique entries in {path}, found {len(values)}"
        )
    stems = [acquisition_id(value) for value in values]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Registry contains duplicate acquisition IDs: {path}")
    return values


def relative_or_absolute(path: Path, project_root: Path) -> str:
    """Store project artifacts relocatably and external artifacts absolutely."""

    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def require_one(paths: list[Path], label: str) -> Path:
    """Require exactly one retained artifact for a scientific role."""

    if len(paths) != 1:
        raise ValueError(f"Expected one {label}, found {len(paths)}: {paths[:5]}")
    return paths[0]


def inspect_batch(text: str) -> tuple[str | None, list[str]]:
    """Validate the retained MATLAB batch and return its declared template."""

    failures = [
        f"missing batch fragment: {fragment}"
        for fragment in REQUIRED_BATCH_FRAGMENTS
        if fragment not in text
    ]
    match = TEMPLATE_RE.search(text)
    template = match.group(1) if match else None
    if template is None:
        failures.append("cannot recover Dartel template from CAT12 batch")
    elif Path(template).name != "Template_1_IXI555_MNI152.nii":
        failures.append(f"unexpected Dartel template declaration: {template}")
    return template, failures


def inspect_catlog(text: str) -> tuple[str | None, str | None, list[str]]:
    """Extract CAT12 version and the material Dartel confirmation line."""

    dartel = next(
        (line.strip() for line in text.splitlines() if "dartel registration" in line.lower()),
        None,
    )
    version = next(
        (line.strip() for line in text.splitlines() if re.search(r"CAT12\.[0-9]", line)),
        None,
    )
    failures = []
    if dartel is None:
        failures.append("CAT12 runtime log does not confirm Dartel registration")
    if version is None:
        failures.append("CAT12 runtime log does not identify the CAT12 version")
    return dartel, version, failures


def inspect_wrapper_log(text: str, stem: str, output_name: str) -> list[str]:
    """Require a final successful wrapper marker for the intended acquisition."""

    required_suffix = f"/{stem}/{output_name}"
    done = [
        line.strip().removeprefix("Done:").strip()
        for line in text.splitlines()
        if line.strip().startswith("Done:")
    ]
    failures = []
    if "All done." not in text:
        failures.append("wrapper log lacks final 'All done.' marker")
    if not any(value.endswith(required_suffix) for value in done):
        failures.append(f"wrapper log lacks Done marker for {output_name}")
    return failures


def inspect_nifti(path: Path) -> tuple[tuple[int, ...], tuple[float, ...], list[str]]:
    """Validate the geometry of the exact NIfTI consumed by AnatCL."""

    image = nib.load(str(path))
    shape = tuple(int(value) for value in image.shape[:3])
    zooms = tuple(round(float(value), 5) for value in image.header.get_zooms()[:3])
    failures = []
    if shape != EXPECTED_SHAPE:
        failures.append(f"unexpected output shape: {shape}")
    if zooms != EXPECTED_ZOOMS:
        failures.append(f"unexpected output spacing: {zooms}")
    return shape, zooms, failures


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one JSON artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    """Publish the ordered case-level provenance table atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=CASE_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit_case(
    *,
    stem: str,
    raw_path: Path | None,
    project_root: Path,
    preproc_root: Path,
    wrapper_log_root: Path,
    hash_raw: bool,
) -> dict[str, Any]:
    """Validate and hash all retained provenance for one acquisition."""

    subject_dir = preproc_root / stem
    output = subject_dir / f"{stem}_anatcl_ready.nii.gz"
    work_dir = subject_dir / "cat12_work"
    batch = work_dir / "cat12_batch.m"
    wrapper_log = wrapper_log_root / f"anatcl_corrected_{stem}.log"
    failures: list[str] = []

    try:
        catlog = require_one(
            sorted((work_dir / "report").glob("catlog_*.txt")), "CAT12 runtime log"
        )
    except ValueError as error:
        failures.append(str(error))
        catlog = work_dir / "report" / f"catlog_{stem}.txt"
    try:
        cat12_input = require_one(
            sorted(path for path in work_dir.glob(f"{stem}.nii") if path.is_file()),
            "copied CAT12 input",
        )
    except ValueError as error:
        failures.append(str(error))
        cat12_input = work_dir / f"{stem}.nii"

    required = {
        "output": output,
        "cat12_batch": batch,
        "cat12_log": catlog,
        "wrapper_log": wrapper_log,
        "cat12_input": cat12_input,
    }
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty {label}: {path}")

    declared_template = None
    dartel = None
    cat12_version = None
    shape: tuple[int, ...] = ()
    zooms: tuple[float, ...] = ()
    if batch.is_file():
        declared_template, found = inspect_batch(batch.read_text(errors="replace"))
        failures.extend(found)
    if catlog.is_file():
        dartel, cat12_version, found = inspect_catlog(
            catlog.read_text(errors="replace")
        )
        failures.extend(found)
    if wrapper_log.is_file():
        failures.extend(
            inspect_wrapper_log(
                wrapper_log.read_text(errors="replace"), stem, output.name
            )
        )
    if output.is_file():
        try:
            shape, zooms, found = inspect_nifti(output)
            failures.extend(found)
        except Exception as error:
            failures.append(f"cannot read output NIfTI: {type(error).__name__}: {error}")

    def digest(path: Path) -> str:
        return sha256_file(path) if path.is_file() else ""

    raw_sha = ""
    if raw_path is not None and hash_raw and raw_path.is_file():
        raw_sha = sha256_file(raw_path)

    return {
        "acquisition_id": stem,
        "raw_t1w_path": str(raw_path) if raw_path is not None else "",
        "raw_t1w_sha256": raw_sha,
        "cat12_input_path": relative_or_absolute(cat12_input, project_root),
        "cat12_input_sha256": digest(cat12_input),
        "cat12_batch_path": relative_or_absolute(batch, project_root),
        "cat12_batch_sha256": digest(batch),
        "declared_template_path": declared_template or "",
        "cat12_log_path": relative_or_absolute(catlog, project_root),
        "cat12_log_sha256": digest(catlog),
        "dartel_confirmation": dartel or "",
        "cat12_version": cat12_version or "",
        "wrapper_log_path": relative_or_absolute(wrapper_log, project_root),
        "wrapper_log_sha256": digest(wrapper_log),
        "output_path": relative_or_absolute(output, project_root),
        "output_sha256": digest(output),
        "shape": "x".join(str(value) for value in shape),
        "zooms_mm": "x".join(str(value) for value in zooms),
        "status": "PASS" if not failures else "FAIL",
        "failures": " | ".join(failures),
    }


def parse_args() -> argparse.Namespace:
    """Parse the retrospective, read-only audit interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_ACQUISITIONS)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel read workers; useful when artifacts are accessed through SSHFS",
    )
    parser.add_argument(
        "--hash-raw",
        action="store_true",
        help="Hash canonical raw T1w files when the raw data mount is available",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Build and publish the case-level table and cohort-level audit."""

    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    root = args.project_root.expanduser().resolve()
    onharmony = root / ONHARMONY_ROOT
    filelist = root / "DATA" / "filelists" / "filelist_AnatCL.txt"
    raw_filelist = root / "DATA" / "filelists" / "filelist_BrainIAC_raw.txt"
    entries = read_registry(filelist, args.expected_count)
    raw_entries = read_registry(raw_filelist, args.expected_count)
    raw_by_stem = {acquisition_id(value): Path(value) for value in raw_entries}
    stems = [acquisition_id(value) for value in entries]
    if set(stems) != set(raw_by_stem):
        raise ValueError("AnatCL and canonical raw registries contain different acquisitions")

    preproc_root = onharmony / "preproc_AnatCL"
    wrapper_log_root = onharmony / "logs_preprocs"

    def inspect(stem: str) -> dict[str, Any]:
        return audit_case(
            stem=stem,
            raw_path=raw_by_stem[stem],
            project_root=root,
            preproc_root=preproc_root,
            wrapper_log_root=wrapper_log_root,
            hash_raw=args.hash_raw,
        )

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, record in enumerate(executor.map(inspect, stems), start=1):
            records.append(record)
            if index % 10 == 0 or index == len(stems):
                print(f"AnatCL provenance: {index}/{len(stems)}", flush=True)

    reference = PROJECT_ROOT / ONHARMONY_ROOT / "preprocessing_AnatCL.py"
    template = PROJECT_ROOT / "templates" / "Template_1_IXI555_MNI152.nii"
    template_sha = sha256_file(template) if template.is_file() else None
    failures = [record for record in records if record["status"] != "PASS"]
    reference_failures = []
    if template_sha != EXPECTED_TEMPLATE_SHA256:
        reference_failures.append(
            "The canonical IXI555 template differs from its expected SHA-256"
        )
    raw_hashes = sum(bool(record["raw_t1w_sha256"]) for record in records)
    output_dir = args.output_dir.expanduser().resolve()
    cases_path = output_dir / "anatcl_preprocessing_case_provenance.tsv"
    manifest_path = output_dir / "anatcl_preprocessing_provenance.json"
    atomic_tsv(cases_path, records)

    payload = {
        "schema_version": "anatcl_onharmony_retrospective_provenance_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures and not reference_failures else "FAIL",
        "scope": "retained ON-Harmony AnatCL preprocessing artifacts",
        "artifact_project_root": str(root),
        "historical_generator": {
            "declared_path": HISTORICAL_GENERATOR,
            "source_available": False,
            "sha256": None,
            "note": (
                "The historical Python wrapper was deleted. It is not represented "
                "by the reference implementation hash."
            ),
        },
        "reference_implementation": {
            "path": relative_or_absolute(reference, PROJECT_ROOT),
            "sha256": sha256_file(reference),
            "literal_historical_generator": False,
        },
        "template_reference": {
            "path": relative_or_absolute(template, PROJECT_ROOT),
            "sha256": template_sha,
            "expected_sha256": EXPECTED_TEMPLATE_SHA256,
            "identity_matches_expected": template_sha == EXPECTED_TEMPLATE_SHA256,
            "historical_file_hash_recorded_at_execution": False,
        },
        "registry": {
            "path": relative_or_absolute(filelist, root),
            "sha256": sha256_file(filelist),
            "raw_registry_path": relative_or_absolute(raw_filelist, root),
            "raw_registry_sha256": sha256_file(raw_filelist),
            "n_acquisitions": len(records),
            "ordered_acquisition_ids_sha256": canonical_sha256(stems),
        },
        "evidence": {
            "case_table": cases_path.name,
            "case_table_sha256": sha256_file(cases_path),
            "case_records_sha256": canonical_sha256(records),
            "cases_passed": len(records) - len(failures),
            "cases_failed": len(failures),
            "raw_files_hashed": raw_hashes,
            "cat12_inputs_hashed": sum(bool(r["cat12_input_sha256"]) for r in records),
            "batches_hashed": sum(bool(r["cat12_batch_sha256"]) for r in records),
            "catlogs_hashed": sum(bool(r["cat12_log_sha256"]) for r in records),
            "wrapper_logs_hashed": sum(bool(r["wrapper_log_sha256"]) for r in records),
            "outputs_hashed": sum(bool(r["output_sha256"]) for r in records),
        },
        "limitations": [
            "The exact historical Python wrapper source and SHA-256 are unavailable.",
            "The executed batches declare the historical template path but do not record its SHA-256.",
            "Hashes establish artifact identity and integrity, not scientific validity.",
        ],
        "failure_examples": [record for record in failures[:10]],
        "reference_failures": reference_failures,
        "auditor": {
            "path": relative_or_absolute(SCRIPT_PATH, PROJECT_ROOT),
            "sha256": sha256_file(SCRIPT_PATH),
        },
    }
    atomic_json(manifest_path, payload)
    print(f"Cases: {cases_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Status: {payload['status']}")
    if failures:
        raise SystemExit(f"AnatCL provenance failed for {len(failures)} acquisitions")


if __name__ == "__main__":
    main()
