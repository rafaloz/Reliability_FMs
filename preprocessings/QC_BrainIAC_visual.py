#!/usr/bin/env python3
"""Build and freeze a label-blind visual QC review of BrainIAC inputs.

The BrainIAC preprocessors already create one orthogonal QC PNG per case. This
script validates those PNGs and their preprocessing records, orders cases using
transparent automated flags, and builds a keyboard-driven HTML review. The
automated priority is not a quality score and never excludes a case.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_FINE_TUNING = 1411
EXPECTED_ONHARMONY = 165
EXPECTED_TOTAL = EXPECTED_FINE_TUNING + EXPECTED_ONHARMONY
MANIFEST_SHA256 = "b2ee2dc7e06349ef1ac550da2061274ee724d2b121ddd652b37c1907ae8bc212"
OFFLINE_GROUP = "preproc_brainiac"
FT_PIPELINE_VERSION = "brainiac-release-aligned-finetuning-v1"
FT_CONTRACT_SHA256 = "2eea43f6031355e2ff4d131c549637ac633bff195989e2d0dec30b5499f231ca"
ONH_PIPELINE_VERSION = "brainiac-release-aligned-onharmony-v1"
ONH_CONTRACT_SHA256 = "d7315a8346a9fb900ba8392048d39f0ec99f77bb482eddb84025de247ce85dd2"
ONH_CURRENT_GENERATOR_SHA256 = (
    "06124069330ae294271c74b7ca288f5f11b8e53cfc3dde20b2f583aaf580d1a0"
)
ONH_HISTORICAL_GENERATOR_SHA256 = (
    "f298f8f0efab28e1b6a07746775c6b1ef5a37c16a4c9dbc3df707bdd52d8bdbb"
)
ALLOWED_ONH_GENERATORS = {
    ONH_CURRENT_GENERATOR_SHA256,
    ONH_HISTORICAL_GENERATOR_SHA256,
}
EXPECTED_READY_SHAPE = [96, 96, 96]
REVIEW_DECISIONS = ("PASS", "REVIEW", "FAIL")
ACQUISITION_RE = re.compile(
    r"(?P<acquisition>sub-(?P<subject>[0-9]+)_ses-"
    r"(?P<session>[A-Za-z0-9]+)_T1w)"
)
LOG_GENERATOR_RE = re.compile(
    r"^script\s*:\s*([0-9a-f]{64})\s*$", flags=re.MULTILINE
)
OUTLIER_METRICS = (
    "brain_nonzero_fraction",
    "ready_nonzero_fraction",
    "ready_mean_nonzero",
    "ready_std_nonzero",
    "registration_metric",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    scope: str
    group: str
    dataset: str
    subject: str
    session: str
    acquisition_id: str
    split: str
    ready_path: Path
    brain_path: Path
    qc_path: Path
    qc_png_path: Path
    log_path: Path | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def parse_acquisition(value: str | Path) -> tuple[str, str, str]:
    match = ACQUISITION_RE.search(Path(value).name)
    if match is None:
        raise ValueError(f"Cannot parse ON-Harmony acquisition from {value}")
    return (
        str(match.group("acquisition")),
        f"sub-{match.group('subject')}",
        str(match.group("session")),
    )


def scanner_from_session(session: str) -> str:
    return re.sub(r"\d{3}$", "", session)


def load_fine_tuning_cases(
    manifest_path: Path,
    preproc_root: Path,
    expected: int | None,
    *,
    expected_manifest_sha256: str | None = MANIFEST_SHA256,
) -> list[Case]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError(
            "Definitive manifest SHA-256 drift: "
            f"expected {expected_manifest_sha256}, observed {manifest_sha}"
        )
    columns = {
        "dataset",
        "subject",
        "split",
        "inclusion_status",
        "required_offline_groups",
    }
    frame = pd.read_csv(
        manifest_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=sorted(columns),
    )
    frame = frame.loc[frame["inclusion_status"].eq("INCLUDED")].copy()
    if expected is not None and len(frame) != expected:
        raise ValueError(f"Expected {expected} fine-tuning cases, found {len(frame)}")
    if frame.duplicated(["dataset", "subject"]).any():
        raise ValueError("Fine-tuning manifest contains duplicate subjects")
    permitted = frame["required_offline_groups"].str.split(";").apply(
        lambda values: OFFLINE_GROUP in values
    )
    if not permitted.all():
        raise ValueError("Fine-tuning manifest contains cases without BrainIAC permission")

    cases: list[Case] = []
    for row in frame.to_dict(orient="records"):
        root = preproc_root / row["dataset"] / row["subject"]
        cases.append(
            Case(
                case_id=f"brain_age::{row['dataset']}::{row['subject']}",
                scope="brain_age",
                group=f"brain_age::{row['dataset']}",
                dataset=str(row["dataset"]),
                subject=str(row["subject"]),
                session="",
                acquisition_id="",
                split=str(row["split"]),
                ready_path=root / f"{row['subject']}_brainiac_ready.nii.gz",
                brain_path=root / f"{row['subject']}_brain.nii.gz",
                qc_path=root / "preprocessing_qc.json",
                qc_png_path=root / "preprocessing_qc.png",
            )
        )
    return cases


def load_onharmony_cases(
    input_list: Path,
    preproc_root: Path,
    log_root: Path,
    expected: int | None,
) -> list[Case]:
    if not input_list.is_file():
        raise FileNotFoundError(input_list)
    listed = [line.strip() for line in input_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if expected is not None and len(listed) != expected:
        raise ValueError(f"Expected {expected} ON-Harmony cases, found {len(listed)}")
    acquisitions = [parse_acquisition(value) for value in listed]
    if len({item[0] for item in acquisitions}) != len(acquisitions):
        raise ValueError("ON-Harmony input list contains duplicate acquisitions")

    cases: list[Case] = []
    for acquisition, subject, session in acquisitions:
        root = preproc_root / acquisition
        cases.append(
            Case(
                case_id=f"onharmony::{acquisition}",
                scope="onharmony",
                group=f"onharmony::{scanner_from_session(session)}",
                dataset="ON-Harmony",
                subject=subject,
                session=session,
                acquisition_id=acquisition,
                split="external",
                ready_path=root / f"{acquisition}_brainiac_ready.nii.gz",
                brain_path=root / f"{acquisition}_brain.nii.gz",
                qc_path=root / f"{acquisition}_qc.json",
                qc_png_path=root / f"{acquisition}_qc.png",
                log_path=log_root / f"brainiac_{acquisition}.log",
            )
        )
    return cases


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label}: expected {expected!r}, observed {observed!r}")


def _validate_png(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    if width < 600 or height < 400:
        raise ValueError(f"QC montage is unexpectedly small: {width}x{height}")
    return int(width), int(height)


def _validate_fine_tuning_qc(case: Case, qc: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    expected = {
        "status": "PASS",
        "pipeline_version": FT_PIPELINE_VERSION,
        "pipeline_contract_sha256": FT_CONTRACT_SHA256,
        "dataset": case.dataset,
        "subject": case.subject,
        "split": case.split,
    }
    for key, value in expected.items():
        _require_equal(qc.get(key), value, f"Fine-tuning QC field {key}")
    ready = qc.get("model_ready_output", {})
    _require_equal(ready.get("shape"), EXPECTED_READY_SHAPE, "BrainIAC ready shape")
    _require_equal(ready.get("dtype_on_disk"), "float32", "BrainIAC ready dtype")
    output_sha = qc.get("output_sha256", {})
    for key in ("ready", "brain", "qc_png"):
        _require_equal(output_sha.get(key), hashes[key], f"Fine-tuning {key} SHA-256")
    spatial = qc.get("spatial_outputs", {})
    return {
        "provenance_schema": "current",
        "preprocessing_generator_sha256": str(qc.get("script_sha256", "")),
        "brain_nonzero_fraction": float(spatial["brain_nonzero_fraction"]),
        "ready_nonzero_fraction": float(ready["nonzero_fraction"]),
        "ready_mean_nonzero": float(ready["mean_nonzero"]),
        "ready_std_nonzero": float(ready["standard_deviation_nonzero"]),
        "registration_metric": float(qc["registration"]["metric_value"]),
    }


def _validate_onharmony_qc(case: Case, qc: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    expected = {
        "status": "PASS",
        "brainiac_shape": EXPECTED_READY_SHAPE,
        "shape_ok": True,
        "norm_ok": True,
        "brain_ok": True,
    }
    for key, value in expected.items():
        _require_equal(qc.get(key), value, f"ON-Harmony QC field {key}")
    if case.acquisition_id not in Path(str(qc.get("raw_input", ""))).name:
        raise ValueError("ON-Harmony QC raw input does not identify the acquisition")
    if case.log_path is None:
        raise RuntimeError("ON-Harmony case has no preprocessing log")
    log = case.log_path.read_text(encoding="utf-8", errors="replace")
    generator_match = LOG_GENERATOR_RE.search(log)
    generator = generator_match.group(1) if generator_match else None
    if generator not in ALLOWED_ONH_GENERATORS:
        raise ValueError(f"Unrecognized BrainIAC preprocessing generator: {generator}")
    required_log = (
        "[Preprocessing 4-5/5] Resample 96",
        "status=PASS",
        "All done.",
        "exit_code: 0",
    )
    missing = [fragment for fragment in required_log if fragment not in log]
    if missing:
        raise ValueError(f"Incomplete ON-Harmony BrainIAC log: {missing}")

    provenance_schema = "historical"
    if "pipeline_contract_sha256" in qc:
        provenance_schema = "current"
        _require_equal(qc.get("pipeline_version"), ONH_PIPELINE_VERSION, "ON-Harmony pipeline")
        _require_equal(
            qc.get("pipeline_contract_sha256"), ONH_CONTRACT_SHA256, "ON-Harmony contract"
        )
        _require_equal(generator, ONH_CURRENT_GENERATOR_SHA256, "ON-Harmony generator")
        output_sha = qc.get("output_sha256", {})
        for key in ("ready", "brain", "qc_png"):
            _require_equal(output_sha.get(key), hashes[key], f"ON-Harmony {key} SHA-256")
    else:
        _require_equal(generator, ONH_HISTORICAL_GENERATOR_SHA256, "Historical generator")

    brain_shape = qc.get("brain_shape")
    brain_voxels = float(qc["brain_nonzero_voxels"])
    brain_fraction = brain_voxels / float(np.prod(np.asarray(brain_shape, dtype=int)))
    return {
        "provenance_schema": provenance_schema,
        "preprocessing_generator_sha256": str(generator),
        "brain_nonzero_fraction": brain_fraction,
        "ready_nonzero_fraction": float(qc["brainiac_nonzero_fraction"]),
        "ready_mean_nonzero": float(qc["brainiac_mean_nonzero"]),
        "ready_std_nonzero": float(qc["brainiac_std_nonzero"]),
        "registration_metric": math.nan,
    }


def process_case(case: Case, report_dir: Path) -> dict[str, Any]:
    required = [case.ready_path, case.brain_path, case.qc_path, case.qc_png_path]
    if case.log_path is not None:
        required.append(case.log_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing BrainIAC artifacts: {missing}")

    qc_sha_before = sha256_file(case.qc_path)
    hashes = {
        "ready": sha256_file(case.ready_path),
        "brain": sha256_file(case.brain_path),
        "qc_png": sha256_file(case.qc_png_path),
    }
    if case.log_path is not None:
        hashes["log"] = sha256_file(case.log_path)
    png_width, png_height = _validate_png(case.qc_png_path)
    qc = json.loads(case.qc_path.read_text(encoding="utf-8"))
    details = (
        _validate_fine_tuning_qc(case, qc, hashes)
        if case.scope == "brain_age"
        else _validate_onharmony_qc(case, qc, hashes)
    )
    qc_sha_after = sha256_file(case.qc_path)
    if qc_sha_before != qc_sha_after:
        raise RuntimeError("BrainIAC QC changed while it was being validated")

    return {
        **asdict(case),
        **details,
        "ready_path": str(case.ready_path),
        "brain_path": str(case.brain_path),
        "qc_path": str(case.qc_path),
        "qc_png_path": str(case.qc_png_path),
        "log_path": str(case.log_path) if case.log_path else "",
        "ready_sha256": hashes["ready"],
        "brain_sha256": hashes["brain"],
        "qc_sha256": qc_sha_after,
        "qc_png_sha256": hashes["qc_png"],
        "log_sha256": hashes.get("log", ""),
        "qc_png_width": png_width,
        "qc_png_height": png_height,
        "qc_png_file": os.path.relpath(case.qc_png_path, report_dir),
        "status": "PASS",
        "error": "",
    }


def failed_row(case: Case, error: BaseException, report_dir: Path) -> dict[str, Any]:
    return {
        **asdict(case),
        "ready_path": str(case.ready_path),
        "brain_path": str(case.brain_path),
        "qc_path": str(case.qc_path),
        "qc_png_path": str(case.qc_png_path),
        "log_path": str(case.log_path) if case.log_path else "",
        "qc_png_file": (
            os.path.relpath(case.qc_png_path, report_dir) if case.qc_png_path.is_file() else ""
        ),
        "status": "FAIL",
        "error": f"{type(error).__name__}: {error}",
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return result
    center = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - center)))
    if mad > 0:
        result[finite] = 0.67448975 * (values[finite] - center) / mad
        return result
    q1, q3 = np.percentile(values[finite], [25, 75])
    scale = float(q3 - q1)
    if scale > 0:
        result[finite] = (values[finite] - center) / (scale / 1.3489795)
    return result


def add_priorities(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for metric in OUTLIER_METRICS:
        if metric not in result:
            result[metric] = math.nan
        result[f"robust_z_{metric}"] = 0.0
    valid = result["status"].eq("PASS")
    for _, indices in result.loc[valid].groupby("group", sort=True).groups.items():
        for metric in OUTLIER_METRICS:
            values = result.loc[indices, metric].to_numpy(float)
            result.loc[indices, f"robust_z_{metric}"] = robust_z(values)

    priorities: list[str] = []
    reasons: list[str] = []
    for _, row in result.iterrows():
        row_reasons: list[str] = []
        if row["status"] != "PASS":
            priorities.append("HIGH")
            reasons.append(str(row["error"]))
            continue
        if abs(float(row["ready_mean_nonzero"])) > 0.01:
            row_reasons.append("nonzero mean differs from zero")
        if abs(float(row["ready_std_nonzero"]) - 1.0) > 0.01:
            row_reasons.append("nonzero standard deviation differs from one")
        for metric in OUTLIER_METRICS:
            z_value = float(row[f"robust_z_{metric}"])
            if math.isfinite(z_value) and abs(z_value) >= 5.0:
                row_reasons.append(f"{metric} robust z={z_value:.1f}")
        priorities.append("HIGH" if len(row_reasons) >= 2 else "REVIEW" if row_reasons else "ROUTINE")
        reasons.append("; ".join(row_reasons))
    result["auto_priority"] = priorities
    result["auto_reasons"] = reasons
    order = pd.Categorical(result["auto_priority"], ["HIGH", "REVIEW", "ROUTINE"], ordered=True)
    return (
        result.assign(_priority=order)
        .sort_values(["_priority", "scope", "group", "case_id"], kind="stable")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


def public_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "case_id",
        "scope",
        "group",
        "dataset",
        "subject",
        "session",
        "acquisition_id",
        "split",
        "qc_png_file",
        "auto_priority",
        "auto_reasons",
        "brain_nonzero_fraction",
        "ready_nonzero_fraction",
        "ready_mean_nonzero",
        "ready_std_nonzero",
    ]
    records = frame.copy()
    for column in columns:
        if column not in records:
            records[column] = "" if column not in OUTLIER_METRICS else math.nan
    records = records[columns].replace({np.nan: None})
    return records.to_dict(orient="records")


def report_html(frame: pd.DataFrame, created: str) -> str:
    payload = json.dumps(public_records(frame), separators=(",", ":"), allow_nan=False)
    report_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BrainIAC visual QC</title>
<style>
body{{margin:0;background:#f3f5f6;color:#17212b;font:14px Arial,sans-serif}}header{{position:sticky;top:0;z-index:2;background:#214c56;color:white;padding:11px 18px}}h1{{font-size:20px;margin:0 0 7px;letter-spacing:0}}.tools{{display:flex;gap:7px;flex-wrap:wrap;align-items:center}}button,select,input{{font:inherit;padding:6px 8px;border:1px solid #9aa8ad;background:white;color:#17212b}}main{{padding:14px}}.note{{max-width:920px;margin:0 auto 12px;line-height:1.4}}.card{{background:white;border:1px solid #c8d0d3;margin:0 auto 14px;max-width:880px;padding:10px;scroll-margin-top:100px}}.card img{{width:100%;height:auto;display:block;background:#111;object-fit:contain}}.meta{{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}}.priority-HIGH{{color:#a51d18;font-weight:bold}}.priority-REVIEW{{color:#a05a00;font-weight:bold}}.priority-ROUTINE{{color:#22663c;font-weight:bold}}.decision button{{margin-right:6px}}.selected{{outline:3px solid #d97a31}}.muted{{color:#52616b;line-height:1.4}}.hidden{{display:none}}#counter{{font-weight:bold}}
</style></head><body>
<header><h1>BrainIAC visual QC</h1><div class="tools"><span id="counter"></span>
<select id="scope"><option value="ALL">All cohorts</option><option value="brain_age">Brain age</option><option value="onharmony">ON-Harmony</option></select>
<select id="priority"><option value="ALL">All priorities</option><option>HIGH</option><option>REVIEW</option><option>ROUTINE</option></select>
<select id="decision"><option value="ALL">All decisions</option><option value="UNREVIEWED">Unreviewed</option><option>PASS</option><option>REVIEW</option><option>FAIL</option></select>
<input id="search" placeholder="subject, scanner, dataset"><button id="export">Export review TSV</button></div></header>
<main><p class="note"><b>Human review is authoritative.</b> Automatic priority only orders cases using preprocessing checks and robust within-group outliers. It is not a quality score and never excludes a scan. The montage shows registered anatomy, HD-BET brain extraction and the 96-cube consumed by BrainIAC. No age, diagnosis, prediction, BAG or error was loaded. Created {html.escape(created)}.</p><div id="root"></div></main>
<script>
const CASES={payload};const KEY='brainiac-qc-{report_id}';let decisions=JSON.parse(localStorage.getItem(KEY)||'{{}}');let activeId=null;
function esc(x){{return String(x??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function fmt(x,d=3){{return x===null||x===''?'NA':Number(x).toFixed(d);}}
function visibleCards(){{return [...document.querySelectorAll('.card:not(.hidden)')];}}
function activate(el,scroll=true){{if(!el)return;document.querySelectorAll('.card.selected').forEach(item=>item.classList.remove('selected'));el.classList.add('selected');activeId=el.dataset.id;if(scroll)el.scrollIntoView({{block:'center'}});}}
function move(step){{let cards=visibleCards();if(!cards.length)return;let index=cards.findIndex(el=>el.dataset.id===activeId);if(index<0)index=step>0?-1:0;index=Math.max(0,Math.min(cards.length-1,index+step));activate(cards[index]);}}
function save(){{localStorage.setItem(KEY,JSON.stringify(decisions));filter();}}
function choose(id,value){{let before=visibleCards(),index=before.findIndex(el=>el.dataset.id===id);decisions[id]={{decision:value,reviewed_at_utc:new Date().toISOString()}};save();let after=visibleCards(),current=after.find(el=>el.dataset.id===id),fallback=after[Math.min(Math.max(index,0),after.length-1)];activate(current||fallback);}}
function decide(value){{let cards=visibleCards(),target=cards.find(el=>el.dataset.id===activeId)||cards[0];if(target)choose(target.dataset.id,value);}}
function card(c){{let d=decisions[c.case_id]?.decision||'UNREVIEWED',image=c.qc_png_file?`<img loading="lazy" src="${{esc(c.qc_png_file)}}" alt="QC ${{esc(c.case_id)}}">`:'<p>QC image unavailable</p>';return `<article class="card" tabindex="0" data-id="${{esc(c.case_id)}}" data-scope="${{c.scope}}" data-priority="${{c.auto_priority}}" data-decision="${{d}}">${{image}}<div class="meta"><b>${{esc(c.case_id)}}</b><span class="priority-${{c.auto_priority}}">${{c.auto_priority}}</span><span>${{esc(c.group)}}</span><span>decision: <b>${{d}}</b></span></div><div class="muted">brain support=${{fmt(100*c.brain_nonzero_fraction,2)}}%; ready support=${{fmt(100*c.ready_nonzero_fraction,2)}}%; ready mean=${{fmt(c.ready_mean_nonzero,4)}}; ready SD=${{fmt(c.ready_std_nonzero,4)}}<br>${{esc(c.auto_reasons||'No automatic flag')}}</div><p class="decision"><button title="P or 1" onclick="choose('${{esc(c.case_id)}}','PASS')">PASS</button><button title="R or 2" onclick="choose('${{esc(c.case_id)}}','REVIEW')">REVIEW</button><button title="F or 3" onclick="choose('${{esc(c.case_id)}}','FAIL')">FAIL</button></p></article>`;}}
function render(){{let root=document.getElementById('root');root.innerHTML=CASES.map(card).join('');root.querySelectorAll('.card').forEach(el=>{{el.addEventListener('click',event=>{{if(!event.target.closest('button'))activate(el,false);}});el.addEventListener('focus',()=>activate(el,false));}});filter();}}
function filter(){{let s=document.getElementById('scope').value,p=document.getElementById('priority').value,d=document.getElementById('decision').value,q=document.getElementById('search').value.toLowerCase(),n=0;document.querySelectorAll('.card').forEach(el=>{{let current=decisions[el.dataset.id]?.decision||'UNREVIEWED';el.dataset.decision=current;let ok=(s==='ALL'||el.dataset.scope===s)&&(p==='ALL'||el.dataset.priority===p)&&(d==='ALL'||current===d)&&(!q||el.dataset.id.toLowerCase().includes(q)||el.textContent.toLowerCase().includes(q));el.classList.toggle('hidden',!ok);if(ok)n++;let b=el.querySelector('.meta span:last-child b');if(b)b.textContent=current;}});document.getElementById('counter').textContent=`${{n}} shown / ${{CASES.length}} total / ${{Object.keys(decisions).length}} reviewed`;let active=[...document.querySelectorAll('.card')].find(el=>el.dataset.id===activeId);if(!active||active.classList.contains('hidden')){{let first=visibleCards()[0];if(first)activate(first,false);else activeId=null;}}}}
function exportTSV(){{let rows=['case_id\\tdecision\\treviewed_at_utc'];CASES.forEach(c=>{{let d=decisions[c.case_id];if(d)rows.push([c.case_id,d.decision,d.reviewed_at_utc].join('\\t'));}});let blob=new Blob([rows.join('\\n')+'\\n'],{{type:'text/tab-separated-values'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='qc_review.tsv';a.click();URL.revokeObjectURL(a.href);}}
function keyboard(event){{if(event.defaultPrevented||event.altKey||event.ctrlKey||event.metaKey||['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;let key=event.key.toLowerCase(),handled=true;if(key==='arrowdown'||key==='j')move(1);else if(key==='arrowup'||key==='k')move(-1);else if(key==='p'||key==='1')decide('PASS');else if(key==='r'||key==='2')decide('REVIEW');else if(key==='f'||key==='3')decide('FAIL');else handled=false;if(handled)event.preventDefault();}}
['scope','priority','decision','search'].forEach(id=>document.getElementById(id).addEventListener(id==='search'?'input':'change',filter));document.getElementById('export').onclick=exportTSV;document.addEventListener('keydown',keyboard);render();
</script></body></html>"""


def command_build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    running = output_dir / "RUNNING.json"
    atomic_json(running, {"status": "RUNNING", "started_utc": utc_now()})

    expected_ft = None if args.allow_noncanonical_counts else EXPECTED_FINE_TUNING
    expected_onh = None if args.allow_noncanonical_counts else EXPECTED_ONHARMONY
    manifest_path = args.manifest.expanduser().resolve()
    input_list = args.onharmony_input_list.expanduser().resolve()
    cases = load_fine_tuning_cases(
        manifest_path,
        args.fine_tuning_root.expanduser().resolve(),
        expected_ft,
    ) + load_onharmony_cases(
        input_list,
        args.onharmony_root.expanduser().resolve(),
        args.onharmony_log_root.expanduser().resolve(),
        expected_onh,
    )
    if not args.allow_noncanonical_counts and len(cases) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} total cases, found {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate BrainIAC QC case identifiers")

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_case, case, output_dir): case for case in cases}
        completed = 0
        for future in as_completed(futures):
            case = futures[future]
            try:
                rows.append(future.result())
            except BaseException as error:
                rows.append(failed_row(case, error, output_dir))
            completed += 1
            if completed % 25 == 0 or completed == len(cases):
                print(f"[{completed}/{len(cases)}]", flush=True)

    frame = add_priorities(pd.DataFrame(rows))
    metrics_path = output_dir / "qc_metrics.tsv"
    review_path = output_dir / "qc_review.tsv"
    report_path = output_dir / "index.html"
    created = utc_now()
    atomic_tsv(metrics_path, frame)
    atomic_tsv(review_path, pd.DataFrame(columns=["case_id", "decision", "reviewed_at_utc"]))
    atomic_text(report_path, report_html(frame, created))
    source_manifest = {
        "schema_version": "brainiac_visual_qc_sources_v1",
        "label_blind": True,
        "outcomes_loaded": [],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "onharmony_input_list": str(input_list),
        "onharmony_input_list_sha256": sha256_file(input_list),
        "fine_tuning_root": str(args.fine_tuning_root.expanduser().resolve()),
        "onharmony_root": str(args.onharmony_root.expanduser().resolve()),
        "onharmony_log_root": str(args.onharmony_log_root.expanduser().resolve()),
    }
    source_manifest_path = output_dir / "qc_source_manifest.json"
    atomic_json(source_manifest_path, source_manifest)
    summary = {
        "status": "COMPLETED",
        "created_utc": created,
        "label_blind": True,
        "outcomes_loaded": [],
        "n_cases": int(len(frame)),
        "scope_counts": {str(key): int(value) for key, value in frame["scope"].value_counts().items()},
        "automatic_priority_counts": {
            str(key): int(value) for key, value in frame["auto_priority"].value_counts().items()
        },
        "processing_failures": int(frame["status"].ne("PASS").sum()),
        "metrics_sha256": sha256_file(metrics_path),
        "report_sha256": sha256_file(report_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "human_review_required": True,
        "automatic_priority_is_not_a_quality_score": True,
    }
    atomic_json(output_dir / "qc_summary.json", summary)
    running.unlink(missing_ok=True)
    print(f"Report: {report_path}")
    print(f"Metrics: {metrics_path}")


def command_rerender(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = report_dir / "qc_metrics.tsv"
    summary_path = report_dir / "qc_summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Expected {metrics_path} and {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sha256_file(metrics_path) != summary.get("metrics_sha256"):
        raise ValueError("QC metrics SHA-256 differs from qc_summary.json")
    frame = pd.read_csv(metrics_path, sep="\t", dtype={"case_id": str})
    if len(frame) != int(summary.get("n_cases", -1)):
        raise ValueError("QC metric count differs from qc_summary.json")
    report_path = report_dir / "index.html"
    atomic_text(report_path, report_html(frame, str(summary["created_utc"])))
    summary["report_sha256"] = sha256_file(report_path)
    summary["report_rebuilt_utc"] = utc_now()
    summary["script_sha256"] = sha256_file(SCRIPT_PATH)
    atomic_json(summary_path, summary)
    print(f"Report rebuilt: {report_path}")


def command_finalize(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = report_dir / "qc_metrics.tsv"
    summary_path = report_dir / "qc_summary.json"
    review_path = args.review_tsv.expanduser().resolve()
    if not metrics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Expected {metrics_path} and {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sha256_file(metrics_path) != summary.get("metrics_sha256"):
        raise ValueError("QC metrics SHA-256 differs from qc_summary.json")
    completed_path = report_dir / "qc_review_completed.tsv"
    final_path = report_dir / "qc_final_summary.json"
    if completed_path.exists() or final_path.exists():
        raise FileExistsError("Final BrainIAC QC artifacts already exist")

    metrics = pd.read_csv(metrics_path, sep="\t", dtype={"case_id": str})
    review = pd.read_csv(review_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"case_id", "decision", "reviewed_at_utc"}
    if not required.issubset(review.columns):
        raise ValueError(f"Review TSV lacks columns: {sorted(required - set(review.columns))}")
    if review["case_id"].duplicated().any():
        raise ValueError("Review TSV contains duplicate case IDs")
    if not set(review["decision"]).issubset(REVIEW_DECISIONS):
        raise ValueError("Review TSV contains an invalid decision")
    if review["reviewed_at_utc"].eq("").any():
        raise ValueError("Review TSV contains an empty review timestamp")
    unknown = set(review["case_id"]) - set(metrics["case_id"])
    if unknown:
        raise ValueError(f"Review TSV contains unknown cases: {sorted(unknown)[:10]}")
    if not args.allow_incomplete and set(review["case_id"]) != set(metrics["case_id"]):
        raise ValueError(f"Incomplete review: {len(review)}/{len(metrics)} cases")

    joined = metrics.merge(review[list(required)], on="case_id", how="left")
    decisions = joined["decision"].fillna("UNREVIEWED")
    atomic_tsv(completed_path, joined)
    final = {
        "status": (
            "HUMAN_REVIEW_COMPLETED"
            if (decisions != "UNREVIEWED").all()
            else "HUMAN_REVIEW_PARTIAL"
        ),
        "created_utc": utc_now(),
        "label_blind": True,
        "n_cases": int(len(joined)),
        "n_reviewed": int((decisions != "UNREVIEWED").sum()),
        "decision_counts": {str(key): int(value) for key, value in decisions.value_counts().items()},
        "metrics_sha256": sha256_file(metrics_path),
        "review_source": str(review_path),
        "review_source_sha256": sha256_file(review_path),
        "completed_review_sha256": sha256_file(completed_path),
        "automatic_priority_is_not_an_exclusion_decision": True,
    }
    atomic_json(final_path, final)
    print(json.dumps(final, indent=2, allow_nan=False))


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
        default=PROJECT_ROOT / "preprocessings/FineTunningDataset/preproc_BrainIAC",
    )
    build.add_argument(
        "--onharmony-input-list",
        type=Path,
        default=PROJECT_ROOT / "DATA/filelists/filelist_BrainIAC.txt",
    )
    build.add_argument(
        "--onharmony-root",
        type=Path,
        default=PROJECT_ROOT / "preprocessings/OnHarmonyDataset/preproc_BrainIAC",
    )
    build.add_argument(
        "--onharmony-log-root",
        type=Path,
        default=PROJECT_ROOT / "preprocessings/OnHarmonyDataset/logs_preprocs",
    )
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--allow-noncanonical-counts", action="store_true")
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
    if getattr(args, "workers", 1) < 1:
        raise ValueError("--workers must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
