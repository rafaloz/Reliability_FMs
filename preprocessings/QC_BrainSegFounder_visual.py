#!/usr/bin/env python3
"""Build and finalize a label-blind visual QC for BrainSegFounder inputs.

The report covers the definitive brain-age cohort and ON-Harmony. It validates
the preprocessing QC and hashes before exposing one montage per consumed
input. Automatic priority only orders the human inspection; it is not a
quality score and never excludes a case.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.BrainSegFounder.model import (  # noqa: E402
    COMPATIBLE_PREPROCESSING_CONTRACTS,
    load_and_validate_manifest,
)
from FineTuning.BrainSegFounder.predict_onharmony import (  # noqa: E402
    ALLOWED_GENERATOR_HASHES,
    INPUT_CONTRACT_SHA256,
    INPUT_CONTRACT_VERSION,
)
from FineTuning.shared.cohort import sha256_file  # noqa: E402
from FineTuning.shared.onharmony import (  # noqa: E402
    DEFAULT_METADATA,
    DEFAULT_RAW_FILELIST,
    load_onharmony_manifest,
)


EXPECTED_FINE_TUNING = 1411
EXPECTED_ONHARMONY = 165
MONTAGE_VERSION = "brainsegfounder_visual_qc_v2"
DECISIONS = {"PASS", "REVIEW", "FAIL", "UNREVIEWED"}
METRIC_COLUMNS = (
    "ready_nonzero_fraction",
    "ready_nonzero_mean",
    "ready_nonzero_std",
    "ready_center_offset",
    "mni_nonzero_fraction",
    "mni_center_offset",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    scope: str
    group: str
    dataset: str
    subject: str
    split: str
    acquisition_id: str
    raw_path: Path
    directory: Path
    stem: str

    def paths(self) -> dict[str, Path]:
        return {
            "reoriented_t1": self.directory / f"{self.stem}_ukb_reoriented.nii.gz",
            "fov_brain": self.directory / f"{self.stem}_ukb_fov_brain.nii.gz",
            "reduced_t1": self.directory / f"{self.stem}_ukb_T1.nii.gz",
            "fnirt_brain": self.directory / f"{self.stem}_fnirt_brain.nii.gz",
            "ready": self.directory / f"{self.stem}_brainsegfounder_ready.nii.gz",
            "qc": self.directory / "preprocessing_qc.json",
            "qc_png": self.directory / "preprocessing_qc.png",
            "log": self.directory / "preprocessing.log",
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, sep="\t", index=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_file(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise FileNotFoundError(candidate)
    return candidate


def load_fine_tuning_cases(
    manifest_path: Path, preproc_root: Path, expected: int
) -> tuple[list[Case], dict[str, Any]]:
    frame, manifest = load_and_validate_manifest(manifest_path)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} fine-tuning cases, found {len(frame)}")
    cases: list[Case] = []
    for record in frame.to_dict(orient="records"):
        dataset = str(record["dataset"])
        subject = str(record["subject"])
        cases.append(
            Case(
                case_id=f"finetuning::{dataset}::{subject}",
                scope="finetuning",
                group=f"finetuning::{dataset}",
                dataset=dataset,
                subject=subject,
                split=str(record["split"]),
                acquisition_id="",
                raw_path=Path(str(record["raw_t1w_path"])).expanduser().resolve(),
                directory=preproc_root.expanduser().resolve() / dataset / subject,
                stem=subject,
            )
        )
    return cases, manifest


def load_onharmony_cases(
    filelist: Path, metadata: Path, preproc_root: Path, expected: int
) -> tuple[list[Case], dict[str, Any]]:
    frame, manifest = load_onharmony_manifest(
        filelist, metadata, expected_acquisitions=expected, expected_subjects=20
    )
    cases: list[Case] = []
    root = preproc_root.expanduser().resolve()
    for record in frame.to_dict(orient="records"):
        acquisition = str(record["acquisition_id"])
        cases.append(
            Case(
                case_id=f"onharmony::{acquisition}",
                scope="onharmony",
                group=f"onharmony::{record['scanner_code']}",
                dataset="ON-Harmony",
                subject=str(record["subject"]),
                split="external",
                acquisition_id=acquisition,
                raw_path=Path(str(record["raw_t1w_path"])).expanduser().resolve(),
                directory=root / acquisition,
                stem=acquisition,
            )
        )
    return cases, manifest


def validate_qc(case: Case, paths: Mapping[str, Path]) -> dict[str, Any]:
    qc_path = require_file(paths["qc"])
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if qc.get("status") != "PASS":
        raise ValueError(f"Preprocessing status is not PASS: {qc_path}")

    if case.scope == "finetuning":
        expected = {
            "dataset": case.dataset,
            "subject": case.subject,
            "split": case.split,
        }
        mismatches = {
            key: {"expected": value, "observed": qc.get(key)}
            for key, value in expected.items()
            if qc.get(key) != value
        }
        version = qc.get("pipeline_version")
        contract = qc.get("pipeline_contract_sha256")
        if COMPATIBLE_PREPROCESSING_CONTRACTS.get(version) != contract:
            mismatches["contract"] = {
                "expected": COMPATIBLE_PREPROCESSING_CONTRACTS,
                "observed": {"version": version, "sha256": contract},
            }
        raw = qc.get("raw_input", {})
        if Path(str(raw.get("path", ""))).expanduser().resolve() != case.raw_path:
            mismatches["raw_input.path"] = {
                "expected": str(case.raw_path),
                "observed": raw.get("path"),
            }
        if mismatches:
            raise ValueError(f"Fine-tuning QC mismatch: {mismatches}")
        required_hashes = (
            "reoriented_t1",
            "fov_brain",
            "reduced_t1",
            "fnirt_brain",
            "ready",
            "qc_png",
        )
    else:
        expected = {
            "pipeline_version": INPUT_CONTRACT_VERSION,
            "pipeline_contract_sha256": INPUT_CONTRACT_SHA256,
        }
        mismatches = {
            key: {"expected": value, "observed": qc.get(key)}
            for key, value in expected.items()
            if qc.get(key) != value
        }
        if qc.get("script_sha256") not in ALLOWED_GENERATOR_HASHES:
            mismatches["script_sha256"] = {
                "expected_one_of": sorted(ALLOWED_GENERATOR_HASHES),
                "observed": qc.get("script_sha256"),
            }
        raw = qc.get("raw_input", {})
        raw_path = Path(str(raw.get("path", ""))).expanduser().resolve()
        if raw_path != case.raw_path:
            mismatches["raw_input.path"] = {
                "expected": str(case.raw_path),
                "observed": raw.get("path"),
            }
        if mismatches:
            raise ValueError(f"ON-Harmony QC mismatch: {mismatches}")
        log = require_file(paths["log"]).read_text(encoding="utf-8", errors="replace")
        required_fragments = (
            f"pipeline_version={INPUT_CONTRACT_VERSION}",
            f"contract={INPUT_CONTRACT_SHA256}",
            "/bin/fnirt ",
            "/bin/convertwarp ",
            "/bin/applywarp ",
        )
        missing = [fragment for fragment in required_fragments if fragment not in log]
        if missing:
            raise ValueError(f"ON-Harmony preprocessing log lacks {missing}")
        required_hashes = (
            "reoriented_t1",
            "fov_brain",
            "reduced_t1",
            "fnirt_brain",
            "ready",
        )

    hashes = qc.get("output_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"QC lacks output_sha256: {qc_path}")
    for key in required_hashes:
        path = require_file(paths[key])
        if hashes.get(key) != sha256_file(path):
            raise ValueError(f"{key} differs from preprocessing QC: {path}")
    return qc


def reference_brain_from_qc(qc: Mapping[str, Any]) -> tuple[Path, str]:
    assets = qc.get("mni_assets") or qc.get("assets")
    if not isinstance(assets, Mapping) or not isinstance(assets.get("brain"), Mapping):
        raise ValueError("Preprocessing QC lacks the MNI brain reference")
    brain = assets["brain"]
    path = require_file(Path(str(brain.get("path", ""))).expanduser().resolve())
    expected_hash = str(brain.get("sha256", ""))
    observed_hash = sha256_file(path)
    if not expected_hash or observed_hash != expected_hash:
        raise ValueError(f"MNI brain reference differs from preprocessing QC: {path}")
    return path, observed_hash


def volume_metrics(path: Path, *, expected_shape: tuple[int, int, int] | None = None) -> dict[str, float]:
    image = nib.load(str(require_file(path)))
    if image.ndim != 3:
        raise ValueError(f"Expected one 3D volume, found {image.shape}: {path}")
    if expected_shape is not None and tuple(image.shape) != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, found {image.shape}: {path}")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite volume: {path}")
    mask = np.abs(data) > 1e-8
    count = int(mask.sum())
    if count < 1000:
        raise ValueError(f"Too few nonzero voxels ({count}): {path}")
    coordinates = np.argwhere(mask)
    center = coordinates.mean(axis=0)
    target = (np.asarray(data.shape, dtype=float) - 1.0) / 2.0
    normalized_offset = np.linalg.norm((center - target) / np.asarray(data.shape))
    values = data[mask].astype(np.float64)
    return {
        "nonzero_fraction": float(mask.mean()),
        "nonzero_mean": float(values.mean()),
        "nonzero_std": float(values.std()),
        "center_offset": float(normalized_offset),
    }


def display_limits(volume: np.ndarray) -> tuple[float, float]:
    values = volume[np.isfinite(volume) & (np.abs(volume) > 1e-8)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(values.min()), float(values.max() + 1e-6)
    return float(low), float(high)


def orthogonal_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = tuple(size // 2 for size in volume.shape)
    return (
        volume[center[0], :, :],
        volume[:, center[1], :],
        volume[:, :, center[2]],
    )


def reference_edge_mask(section: np.ndarray) -> np.ndarray:
    finite = np.isfinite(section)
    support = finite & (np.abs(section) > 1e-8)
    if int(support.sum()) < 100:
        raise ValueError("MNI reference slice has too little foreground")
    low, high = display_limits(section)
    normalized = np.clip((section.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    gradient_y, gradient_x = np.gradient(normalized)
    magnitude = np.hypot(gradient_x, gradient_y)
    threshold = float(np.percentile(magnitude[support], 90.0))
    return support & (magnitude >= threshold) & (magnitude > 0.0)


def render_montage(
    paths: Mapping[str, Path],
    reference_brain_path: Path,
    destination: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    registered_image = nib.load(str(paths["fnirt_brain"]))
    reference_image = nib.load(str(reference_brain_path))
    if registered_image.shape != reference_image.shape or not np.allclose(
        registered_image.affine, reference_image.affine, rtol=0.0, atol=1e-5
    ):
        raise ValueError("Registered brain and MNI reference do not share an exact grid")
    reference = np.asarray(reference_image.dataobj)
    volumes = [
        (
            "1. Reoriented T1w | skull retained",
            np.asarray(nib.load(str(paths["reoriented_t1"])).dataobj),
            None,
        ),
        (
            "2. FOV crop + BET | brain only",
            np.asarray(nib.load(str(paths["fov_brain"])).dataobj),
            None,
        ),
        (
            "3. Registration ROI | skull retained",
            np.asarray(nib.load(str(paths["reduced_t1"])).dataobj),
            None,
        ),
        (
            "4. MNI registration | orange = template edges",
            np.asarray(registered_image.dataobj),
            reference,
        ),
        (
            "5. Model input | 96^3 brain only",
            np.asarray(nib.load(str(paths["ready"])).dataobj),
            None,
        ),
    ]
    figure, axes = plt.subplots(len(volumes), 3, figsize=(9, 12))
    for row, (label, volume, overlay_reference) in enumerate(volumes):
        if volume.ndim != 3 or not np.isfinite(volume).all():
            raise ValueError(f"Invalid montage volume for {label}: {volume.shape}")
        low, high = display_limits(volume)
        reference_sections = (
            orthogonal_slices(overlay_reference)
            if overlay_reference is not None
            else (None, None, None)
        )
        for column, (section, reference_section) in enumerate(
            zip(orthogonal_slices(volume), reference_sections)
        ):
            axes[row, column].imshow(np.rot90(section), cmap="gray", vmin=low, vmax=high)
            if reference_section is not None:
                edge = np.rot90(reference_edge_mask(reference_section))
                overlay = np.zeros((*edge.shape, 4), dtype=np.float32)
                overlay[edge] = (0.91, 0.36, 0.12, 0.85)
                axes[row, column].imshow(overlay)
            axes[row, column].set_axis_off()
            if column == 0:
                axes[row, column].set_title(label, loc="left", fontsize=9)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100, bbox_inches="tight")
    plt.close(figure)


def safe_image_name(case: Case) -> str:
    digest = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:12]
    stem = case.acquisition_id or f"{case.dataset}_{case.subject}"
    normalized = "".join(character if character.isalnum() else "_" for character in stem)
    return f"{normalized}_{digest}.png"


def inspect_case(case: Case, output_dir: Path) -> dict[str, Any]:
    paths = case.paths()
    qc = validate_qc(case, paths)
    reference_brain_path, reference_brain_sha256 = reference_brain_from_qc(qc)
    ready = volume_metrics(paths["ready"], expected_shape=(96, 96, 96))
    registered = volume_metrics(paths["fnirt_brain"])
    image_name = safe_image_name(case)
    destination = output_dir / "images" / image_name
    display_id = case.acquisition_id or f"{case.dataset}/{case.subject}"
    render_montage(
        paths,
        reference_brain_path,
        destination,
        f"{display_id} - BrainSegFounder QC",
    )
    row: dict[str, Any] = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(case).items()},
        "image_relpath": f"images/{image_name}",
        "image_sha256": sha256_file(destination),
        "qc_path": str(paths["qc"]),
        "qc_sha256": sha256_file(paths["qc"]),
        "ready_path": str(paths["ready"]),
        "ready_sha256": sha256_file(paths["ready"]),
        "fnirt_brain_path": str(paths["fnirt_brain"]),
        "fnirt_brain_sha256": sha256_file(paths["fnirt_brain"]),
        "reference_brain_path": str(reference_brain_path),
        "reference_brain_sha256": reference_brain_sha256,
        "montage_version": MONTAGE_VERSION,
        "pipeline_version": str(qc.get("pipeline_version")),
        "pipeline_contract_sha256": str(qc.get("pipeline_contract_sha256")),
        "registration_recovery_used": bool(
            qc.get("registration_recovery", {}).get("used", False)
        ),
        "processing_status": "PASS",
        "processing_error": "",
        "ready_nonzero_fraction": ready["nonzero_fraction"],
        "ready_nonzero_mean": ready["nonzero_mean"],
        "ready_nonzero_std": ready["nonzero_std"],
        "ready_center_offset": ready["center_offset"],
        "mni_nonzero_fraction": registered["nonzero_fraction"],
        "mni_center_offset": registered["center_offset"],
    }
    return row


def robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    deviation = (numeric - median).abs()
    mad = deviation.median()
    if not np.isfinite(mad) or mad <= 0:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return 0.6744897501960817 * deviation / mad


def assign_priority(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for metric in METRIC_COLUMNS:
        result[f"z_{metric}"] = 0.0
    for _, indices in result.groupby("group", sort=False).groups.items():
        for metric in METRIC_COLUMNS:
            result.loc[indices, f"z_{metric}"] = robust_z(result.loc[indices, metric])
    z_columns = [f"z_{metric}" for metric in METRIC_COLUMNS]
    result["automatic_outlier_count"] = (result[z_columns] >= 3.5).sum(axis=1)
    result["automatic_max_robust_z"] = result[z_columns].max(axis=1)
    priority = np.full(len(result), "ROUTINE", dtype=object)
    review = (
        (result["automatic_max_robust_z"] >= 3.5)
        | result["registration_recovery_used"].astype(bool)
    )
    high = (
        (result["automatic_max_robust_z"] >= 5.0)
        | (result["automatic_outlier_count"] >= 2)
    )
    priority[review.to_numpy()] = "REVIEW"
    priority[high.to_numpy()] = "HIGH"
    result["automatic_priority"] = priority
    result["decision"] = "UNREVIEWED"
    result["reviewer_notes"] = ""
    return result


def report_html(frame: pd.DataFrame, created: str) -> str:
    columns = [
        "case_id",
        "scope",
        "group",
        "dataset",
        "subject",
        "split",
        "acquisition_id",
        "image_relpath",
        "automatic_priority",
        "automatic_max_robust_z",
        "automatic_outlier_count",
        "registration_recovery_used",
        "ready_nonzero_fraction",
        "ready_nonzero_std",
        "mni_nonzero_fraction",
        "decision",
        "reviewer_notes",
    ]
    records = frame[columns].to_dict(orient="records")
    payload = json.dumps(records, separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BrainSegFounder visual QC</title><style>
:root{{--bg:#f3f5f6;--paper:#fff;--ink:#172126;--muted:#5d6a70;--line:#ccd5d9;--teal:#118b83;--orange:#d86d2f;--red:#b93636}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 Arial,sans-serif}}
header{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px}}
.bar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}select,input,button{{height:34px;border:1px solid #aebbc0;background:#fff;padding:0 9px}}button{{cursor:pointer}}button.primary{{background:var(--teal);color:#fff;border-color:var(--teal)}}
.note{{margin:10px 0 0;color:var(--muted)}}main{{padding:14px;max-width:1500px;margin:auto}}.count{{font-weight:700;margin:0 0 10px}}
.card{{background:var(--paper);border:1px solid var(--line);display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:12px;padding:12px;margin-bottom:12px}}
.card.active{{outline:3px solid var(--teal)}}img{{display:block;width:100%;height:auto;background:#111}}h2{{font-size:17px;margin:0 0 8px;word-break:break-word}}.meta{{color:var(--muted);margin-bottom:10px;word-break:break-word}}.metrics{{font-family:monospace;font-size:12px;white-space:pre-wrap}}.decision{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:12px 0}}.decision button.selected{{color:#fff}}.pass.selected{{background:var(--teal)}}.review.selected{{background:var(--orange)}}.fail.selected{{background:var(--red)}}textarea{{width:100%;min-height:90px;padding:7px}}.badge{{font-weight:700}}@media(max-width:850px){{.card{{grid-template-columns:1fr}}}}
</style></head><body><header><div class="bar">
<select id="scope"><option value="ALL">All scopes</option><option value="finetuning">Fine-tuning</option><option value="onharmony">ON-Harmony</option></select>
<select id="priority"><option value="ALL">All priorities</option><option>HIGH</option><option>REVIEW</option><option>ROUTINE</option></select>
<select id="decision"><option value="ALL">All decisions</option><option>UNREVIEWED</option><option>PASS</option><option>REVIEW</option><option>FAIL</option></select>
<input id="search" placeholder="subject, dataset, scanner"><button id="previous" title="Previous visible case">Previous</button><button id="next" title="Next visible case">Next</button><button id="export" class="primary">Export TSV</button></div>
<p class="note"><b>Human review is authoritative.</b> Rows: 1 reoriented T1w with skull; 2 cropped BET brain; 3 registration ROI with skull because standard_space_roi applies the crop to the raw T1w; 4 registered brain with edges from the exact MNI reference in orange; 5 final 96^3 model input. Automatic priority only orders cases; it is not a quality score and never excludes a scan. Keys: arrows move, 1/P=PASS, 2/R=REVIEW, 3/F=FAIL. No age, diagnosis, prediction, BAG or error was loaded. Created {html.escape(created)}.</p></header><main><p class="count" id="count"></p><div id="root"></div></main>
<script>const rows={payload};const key='brainsegfounder-qc-v2';let saved={{}};let storageAvailable=true;let visible=[];let active=0;
try{{saved=JSON.parse(localStorage.getItem(key)||'{{}}');}}catch(error){{storageAvailable=false;saved={{}};}}
for(const row of rows){{if(saved[row.case_id]){{row.decision=saved[row.case_id].decision||'UNREVIEWED';row.reviewer_notes=saved[row.case_id].reviewer_notes||'';}}}}
function esc(v){{return String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function persist(row){{saved[row.case_id]={{decision:row.decision,reviewer_notes:row.reviewer_notes}};if(storageAvailable){{try{{localStorage.setItem(key,JSON.stringify(saved));}}catch(error){{storageAvailable=false;}}}}}}
function selectDecision(row,value){{row.decision=value;persist(row);filter();}}
function render(){{const root=document.getElementById('root');root.innerHTML='';visible.forEach((row,index)=>{{const card=document.createElement('section');card.className='card'+(index===active?' active':'');card.dataset.index=index;card.innerHTML=`<div><img loading="lazy" src="${{esc(row.image_relpath)}}"></div><div><h2>${{esc(row.subject)}} ${{esc(row.acquisition_id)}}</h2><div class="meta">${{esc(row.scope)}} | ${{esc(row.dataset)}} | ${{esc(row.split)}} | ${{esc(row.group)}}<br><span class="badge">Priority: ${{esc(row.automatic_priority)}}</span></div><div class="metrics">max robust z: ${{Number(row.automatic_max_robust_z).toFixed(2)}}\noutlier metrics: ${{row.automatic_outlier_count}}\nregistration recovery: ${{row.registration_recovery_used}}\nready nonzero: ${{Number(row.ready_nonzero_fraction).toFixed(3)}}\nready std: ${{Number(row.ready_nonzero_std).toFixed(3)}}\nMNI nonzero: ${{Number(row.mni_nonzero_fraction).toFixed(3)}}</div><div class="decision"><button class="pass ${{row.decision==='PASS'?'selected':''}}">PASS</button><button class="review ${{row.decision==='REVIEW'?'selected':''}}">REVIEW</button><button class="fail ${{row.decision==='FAIL'?'selected':''}}">FAIL</button></div><textarea placeholder="Optional notes">${{esc(row.reviewer_notes)}}</textarea></div>`;const buttons=card.querySelectorAll('.decision button');buttons[0].onclick=()=>selectDecision(row,'PASS');buttons[1].onclick=()=>selectDecision(row,'REVIEW');buttons[2].onclick=()=>selectDecision(row,'FAIL');card.querySelector('textarea').onchange=e=>{{row.reviewer_notes=e.target.value;persist(row);}};card.onclick=()=>{{active=index;highlight();}};root.appendChild(card);}});document.getElementById('count').textContent=`${{visible.length}} visible / ${{rows.length}} total`;highlight();}}
function highlight(){{const cards=[...document.querySelectorAll('.card')];cards.forEach((c,i)=>c.classList.toggle('active',i===active));if(cards[active])cards[active].scrollIntoView({{block:'nearest'}});}}
function filter(){{const scope=document.getElementById('scope').value,priority=document.getElementById('priority').value,decision=document.getElementById('decision').value,q=document.getElementById('search').value.toLowerCase();visible=rows.filter(r=>(scope==='ALL'||r.scope===scope)&&(priority==='ALL'||r.automatic_priority===priority)&&(decision==='ALL'||r.decision===decision)&&JSON.stringify(r).toLowerCase().includes(q));active=Math.min(active,Math.max(0,visible.length-1));render();}}
function move(delta){{if(!visible.length)return;active=(active+delta+visible.length)%visible.length;highlight();}}
function decide(value){{if(visible[active])selectDecision(visible[active],value);}}
function exportTSV(){{const columns=['case_id','scope','group','dataset','subject','split','acquisition_id','automatic_priority','decision','reviewer_notes'];const clean=v=>String(v??'').replace(/[\\t\\r\\n]+/g,' ');const lines=[columns.join('\\t'),...rows.map(r=>columns.map(c=>clean(r[c])).join('\\t'))];const blob=new Blob([lines.join('\\n')+'\\n'],{{type:'text/tab-separated-values'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='qc_review.tsv';a.click();URL.revokeObjectURL(a.href);}}
function keyboard(e){{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;if(e.key==='ArrowRight'||e.key==='ArrowDown'){{e.preventDefault();move(1);}}else if(e.key==='ArrowLeft'||e.key==='ArrowUp'){{e.preventDefault();move(-1);}}else if(e.key==='1'||e.key.toLowerCase()==='p')decide('PASS');else if(e.key==='2'||e.key.toLowerCase()==='r')decide('REVIEW');else if(e.key==='3'||e.key.toLowerCase()==='f')decide('FAIL');}}
['scope','priority','decision'].forEach(id=>document.getElementById(id).addEventListener('change',filter));document.getElementById('search').addEventListener('input',filter);document.getElementById('previous').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);document.getElementById('export').onclick=exportTSV;document.addEventListener('keydown',keyboard);filter();</script></body></html>"""


def validate_report_images(frame: pd.DataFrame, report_dir: Path) -> None:
    for record in frame.to_dict(orient="records"):
        image_path = report_dir / str(record["image_relpath"])
        if sha256_file(image_path) != record["image_sha256"]:
            raise RuntimeError(f"Report image differs from metrics: {image_path}")


def require_current_montage(frame: pd.DataFrame, source: Mapping[str, Any]) -> None:
    if source.get("montage_version") != MONTAGE_VERSION:
        raise ValueError("QC report uses an obsolete montage; rebuild it before review")
    if "montage_version" not in frame.columns or set(frame["montage_version"]) != {
        MONTAGE_VERSION
    }:
        raise ValueError("QC metrics and report manifest use different montage versions")


def command_build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"QC output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fine_tuning, fine_tuning_manifest = load_fine_tuning_cases(
        args.manifest, args.fine_tuning_root, args.expected_fine_tuning
    )
    onharmony, onharmony_manifest = load_onharmony_cases(
        args.filelist,
        args.metadata,
        args.onharmony_root,
        args.expected_onharmony,
    )
    cases = [*fine_tuning, *onharmony]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Visual QC case identifiers are not unique")

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        rows.append(inspect_case(case, output_dir))
        if index % 25 == 0 or index == len(cases):
            print(f"  [{index}/{len(cases)}] {case.case_id}", flush=True)
    frame = assign_priority(pd.DataFrame(rows))
    metrics_path = output_dir / "qc_metrics.tsv"
    source_path = output_dir / "qc_source_manifest.json"
    report_path = output_dir / "index.html"
    atomic_tsv(metrics_path, frame)
    created = utc_now()
    source = {
        "status": "QC_REPORT_BUILT",
        "created_utc": created,
        "label_blind": True,
        "outcome_labels_loaded": False,
        "n_cases": len(frame),
        "n_fine_tuning": len(fine_tuning),
        "n_onharmony": len(onharmony),
        "fine_tuning_manifest": fine_tuning_manifest,
        "onharmony_manifest": onharmony_manifest,
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "ordered_case_ids_sha256": canonical_json_hash(frame["case_id"].tolist()),
        "montage_version": MONTAGE_VERSION,
        "montage_rows": [
            "reoriented T1w with skull",
            "FOV crop plus BET brain",
            "registration ROI with skull",
            "registered brain with exact MNI-reference edges",
            "final 96^3 model input",
        ],
        "automatic_priority_is_not_a_quality_score": True,
        "automatic_priority_never_excludes": True,
    }
    atomic_json(source_path, source)
    atomic_text(report_path, report_html(frame, created))
    validate_report_images(frame, output_dir)
    print(json.dumps(source, indent=2, allow_nan=False), flush=True)
    print(f"Report: {report_path}", flush=True)


def command_rerender(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = report_dir / "qc_metrics.tsv"
    source_path = report_dir / "qc_source_manifest.json"
    frame = pd.read_csv(metrics_path, sep="\t", keep_default_na=False)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("QC metrics differ from their source manifest")
    require_current_montage(frame, source)
    validate_report_images(frame, report_dir)
    atomic_text(report_dir / "index.html", report_html(frame, str(source["created_utc"])))
    print(f"Report: {report_dir / 'index.html'}")


def command_finalize(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = report_dir / "qc_metrics.tsv"
    source_path = report_dir / "qc_source_manifest.json"
    review_path = require_file(args.review_tsv)
    metrics = pd.read_csv(metrics_path, sep="\t", keep_default_na=False)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("QC metrics differ from their source manifest")
    require_current_montage(metrics, source)
    review = pd.read_csv(review_path, sep="\t", keep_default_na=False)
    required = {"case_id", "decision", "reviewer_notes"}
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError(f"Review TSV lacks columns: {missing}")
    if review["case_id"].duplicated().any():
        raise ValueError("Review TSV contains duplicate case_id values")
    if set(review["case_id"]) != set(metrics["case_id"]):
        raise ValueError("Review TSV does not contain exactly the report cases")
    invalid = sorted(set(review["decision"]) - DECISIONS)
    if invalid:
        raise ValueError(f"Invalid QC decisions: {invalid}")
    joined = metrics.drop(columns=["decision", "reviewer_notes"]).merge(
        review[["case_id", "decision", "reviewer_notes"]],
        on="case_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if joined["case_id"].tolist() != metrics["case_id"].tolist():
        raise RuntimeError("Final review order differs from QC metrics")
    decisions = joined["decision"].fillna("UNREVIEWED")
    if not args.allow_incomplete and (decisions == "UNREVIEWED").any():
        raise ValueError(
            f"Review is incomplete: {(decisions == 'UNREVIEWED').sum()} cases"
        )
    completed_path = report_dir / "qc_review_completed.tsv"
    summary_path = report_dir / "qc_final_summary.json"
    atomic_tsv(completed_path, joined)
    summary = {
        "status": (
            "HUMAN_REVIEW_COMPLETED"
            if (decisions != "UNREVIEWED").all()
            else "HUMAN_REVIEW_PARTIAL"
        ),
        "created_utc": utc_now(),
        "label_blind": True,
        "n_cases": len(joined),
        "n_reviewed": int((decisions != "UNREVIEWED").sum()),
        "decision_counts": {
            str(key): int(value) for key, value in decisions.value_counts().items()
        },
        "metrics_sha256": sha256_file(metrics_path),
        "source_manifest_sha256": sha256_file(source_path),
        "review_source": str(review_path),
        "review_source_sha256": sha256_file(review_path),
        "completed_review": str(completed_path),
        "completed_review_sha256": sha256_file(completed_path),
        "automatic_priority_is_not_an_exclusion_decision": True,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "preprocessings/FineTunningDataset/brainage_splits"
            / "brainage_definitive_subject_manifest.tsv"
        ),
    )
    build.add_argument(
        "--fine-tuning-root",
        type=Path,
        default=PROJECT_ROOT / "preprocessings/FineTunningDataset/preproc_BrainSegFounder",
    )
    build.add_argument("--filelist", type=Path, default=DEFAULT_RAW_FILELIST)
    build.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    build.add_argument(
        "--onharmony-root",
        type=Path,
        default=PROJECT_ROOT / "preprocessings/OnHarmonyDataset/preproc_BrainSegFounder",
    )
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--expected-fine-tuning", type=int, default=EXPECTED_FINE_TUNING)
    build.add_argument("--expected-onharmony", type=int, default=EXPECTED_ONHARMONY)
    build.set_defaults(func=command_build)
    rerender = commands.add_parser("rerender")
    rerender.add_argument("--report-dir", type=Path, required=True)
    rerender.set_defaults(func=command_rerender)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--report-dir", type=Path, required=True)
    finalize.add_argument("--review-tsv", type=Path, required=True)
    finalize.add_argument("--allow-incomplete", action="store_true")
    finalize.set_defaults(func=command_finalize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
