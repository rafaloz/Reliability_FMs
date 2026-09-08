#!/usr/bin/env python3
"""Build a label-blind visual QC report for 3D-Neuro-SimCLR inputs.

Each montage overlays the registered binary mask on the WhiteStripe-normalized
image consumed by the model. Automated metrics only prioritise review; they do
not assign exclusions. Human PASS/REVIEW/FAIL decisions can be exported from
the browser and frozen with the ``finalize`` command.

The montages contain sensitive MRI data and are not publication artifacts.
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
import shutil
import tempfile
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_SHAPE = (193, 229, 193)
EXPECTED_ZOOMS = (1.0, 1.0, 1.0)
EXPECTED_FINE_TUNING = 1411
EXPECTED_ONHARMONY = 165
EXPECTED_TOTAL = EXPECTED_FINE_TUNING + EXPECTED_ONHARMONY
OFFLINE_GROUP = "preproc_3d_neuro_simclr"
ACQUISITION_RE = re.compile(
    r"(?P<acquisition>sub-(?P<subject>[0-9]+)_ses-"
    r"(?P<session>[A-Za-z0-9]+)_T1w)"
)
REVIEW_DECISIONS = ("PASS", "REVIEW", "FAIL")
METRICS_FOR_OUTLIERS = (
    "mask_volume_ml",
    "mask_support_fraction",
    "mask_largest_component_fraction",
    "mask_com_distance_voxels",
    "cropped_mask_retention_fraction",
    "inside_intensity_std",
    "inside_intensity_iqr",
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
    image_path: Path
    mask_path: Path


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


def parse_acquisition(path: str | Path) -> tuple[str, str, str]:
    match = ACQUISITION_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse ON-Harmony acquisition from {path}")
    acquisition = str(match.group("acquisition"))
    subject = f"sub-{match.group('subject')}"
    session = str(match.group("session"))
    return acquisition, subject, session


def scanner_from_session(session: str) -> str:
    return re.sub(r"\d{3}$", "", session)


def load_fine_tuning_cases(
    manifest_path: Path, preproc_root: Path, expected: int | None
) -> list[Case]:
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
        raise ValueError("Fine-tuning manifest contains cases without SimCLR permission")
    result: list[Case] = []
    for row in frame.to_dict(orient="records"):
        root = preproc_root / row["dataset"] / row["subject"]
        result.append(
            Case(
                case_id=f"brain_age::{row['dataset']}::{row['subject']}",
                scope="brain_age",
                group=f"brain_age::{row['dataset']}",
                dataset=row["dataset"],
                subject=row["subject"],
                session="",
                acquisition_id="",
                split=row["split"],
                image_path=root / f"{row['subject']}_simclr_ready.nii.gz",
                mask_path=root / f"{row['subject']}_mask_mni.nii.gz",
            )
        )
    return result


def load_onharmony_cases(input_list: Path, expected: int | None) -> list[Case]:
    paths = [
        Path(line.strip()).expanduser().resolve()
        for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if expected is not None and len(paths) != expected:
        raise ValueError(f"Expected {expected} ON-Harmony cases, found {len(paths)}")
    if len(paths) != len(set(paths)):
        raise ValueError("ON-Harmony input list contains duplicate paths")
    result: list[Case] = []
    for image_path in paths:
        acquisition, subject, session = parse_acquisition(image_path)
        expected_suffix = "_simclr_ready.nii.gz"
        if not image_path.name.endswith(expected_suffix):
            raise ValueError(f"Unexpected SimCLR input name: {image_path.name}")
        mask_path = image_path.with_name(
            image_path.name.removesuffix(expected_suffix) + "_mask_mni.nii.gz"
        )
        scanner = scanner_from_session(session)
        result.append(
            Case(
                case_id=f"onharmony::{acquisition}",
                scope="onharmony",
                group=f"onharmony::{scanner}",
                dataset="ON-Harmony",
                subject=subject,
                session=session,
                acquisition_id=acquisition,
                split="external",
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return result


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


def crop_mask(mask: np.ndarray) -> np.ndarray:
    transposed = np.transpose(mask[np.newaxis, ...], (0, 3, 2, 1))
    cropped = transposed[:, 10:160, 19:211, 1:]
    if cropped.shape != (1, 150, 192, 192):
        raise RuntimeError(f"Unexpected cropped mask shape: {cropped.shape}")
    return cropped[0]


def image_metrics(data: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    foreground = data[mask]
    if foreground.size == 0:
        raise ValueError("Empty registered brain mask")
    labels, components = ndimage.label(mask)
    component_sizes = np.bincount(labels.ravel())[1:]
    largest = int(component_sizes.max()) if components else 0
    coordinates = np.argwhere(mask)
    center = coordinates.mean(axis=0)
    template_center = (np.asarray(mask.shape, dtype=float) - 1.0) / 2.0
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    boundary[:, :, [0, -1]] = True
    cropped = crop_mask(mask)
    quantiles = np.percentile(foreground, [1, 25, 50, 75, 99])
    return {
        "mask_voxels": int(mask.sum()),
        "mask_volume_ml": float(mask.sum() / 1000.0),
        "mask_support_fraction": float(mask.mean()),
        "mask_components": int(components),
        "mask_largest_component_fraction": float(largest / mask.sum()),
        "mask_boundary_fraction": float((mask & boundary).sum() / mask.sum()),
        "mask_com_x": float(center[0]),
        "mask_com_y": float(center[1]),
        "mask_com_z": float(center[2]),
        "mask_com_distance_voxels": float(np.linalg.norm(center - template_center)),
        "cropped_mask_retention_fraction": float(cropped.sum() / mask.sum()),
        "cropped_mask_support_fraction": float(cropped.mean()),
        "inside_intensity_mean": float(foreground.mean()),
        "inside_intensity_std": float(foreground.std()),
        "inside_intensity_p01": float(quantiles[0]),
        "inside_intensity_p25": float(quantiles[1]),
        "inside_intensity_median": float(quantiles[2]),
        "inside_intensity_p75": float(quantiles[3]),
        "inside_intensity_p99": float(quantiles[4]),
        "inside_intensity_iqr": float(quantiles[3] - quantiles[1]),
    }


def scale_slice(values: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def panel(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    axis: int,
    index: int,
    low: float,
    high: float,
    size: int = 240,
) -> Image.Image:
    values = np.take(data, index, axis=axis)
    mask_slice = np.take(mask, index, axis=axis)
    values = np.rot90(values)
    mask_slice = np.rot90(mask_slice)
    gray = scale_slice(values, low, high)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    contour = mask_slice ^ ndimage.binary_erosion(mask_slice)
    rgb[contour] = np.asarray([241, 102, 56], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def render_montage(case: Case, data: np.ndarray, mask: np.ndarray, destination: Path) -> None:
    foreground = data[mask]
    low, high = np.percentile(foreground, [1, 99])
    center = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
    views = [
        ("sagittal", 0, int(center[0])),
        ("coronal", 1, int(center[1])),
        ("axial", 2, int(center[2])),
    ]
    width = 3 * 240
    canvas = Image.new("RGB", (width, 285), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{case.case_id} | orange = registered mask boundary"
    draw.text((10, 8), title, fill=(20, 28, 36), font=ImageFont.load_default())
    for offset, (label, axis, index) in enumerate(views):
        view = panel(data, mask, axis=axis, index=index, low=float(low), high=float(high))
        x = offset * 240
        canvas.paste(view, (x, 35))
        draw.text((x + 8, 263), f"{label} [{index}]", fill=(20, 28, 36))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        canvas.save(handle, format="JPEG", quality=88, optimize=True)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def process_case(case: Case, montage_dir: Path) -> dict[str, Any]:
    if not case.image_path.is_file() or not case.mask_path.is_file():
        raise FileNotFoundError(f"Missing image/mask for {case.case_id}")
    image_sha = sha256_file(case.image_path)
    mask_sha = sha256_file(case.mask_path)
    image = nib.load(str(case.image_path))
    mask_image = nib.load(str(case.mask_path))
    if tuple(image.shape) != EXPECTED_SHAPE or tuple(mask_image.shape) != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected grid for {case.case_id}")
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    if not np.allclose(zooms, EXPECTED_ZOOMS, rtol=0.0, atol=1e-6):
        raise ValueError(f"Unexpected spacing for {case.case_id}: {zooms}")
    if not np.allclose(image.affine, mask_image.affine, rtol=0.0, atol=1e-6):
        raise ValueError(f"Image/mask affine mismatch for {case.case_id}")
    data = image.get_fdata(dtype=np.float32)
    mask_values = mask_image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not np.isfinite(mask_values).all():
        raise ValueError(f"Non-finite image/mask for {case.case_id}")
    binary = np.isclose(mask_values, 0.0, atol=1e-6, rtol=0.0) | np.isclose(
        mask_values, 1.0, atol=1e-6, rtol=0.0
    )
    if not binary.all():
        raise ValueError(f"Non-binary registered mask for {case.case_id}")
    mask = mask_values > 0.5
    montage_path = montage_dir / f"{hashlib.sha256(case.case_id.encode()).hexdigest()[:20]}.jpg"
    render_montage(case, data, mask, montage_path)
    return {
        **asdict(case),
        **image_metrics(data, mask),
        "image_path": str(case.image_path),
        "mask_path": str(case.mask_path),
        "image_sha256": image_sha,
        "mask_sha256": mask_sha,
        "montage_path": str(montage_path),
        "montage_file": f"montages/{montage_path.name}",
        "montage_sha256": sha256_file(montage_path),
        "status": "PASS",
        "error": "",
    }


def failed_row(case: Case, error: BaseException) -> dict[str, Any]:
    return {
        **asdict(case),
        "image_path": str(case.image_path),
        "mask_path": str(case.mask_path),
        "status": "FAIL",
        "error": f"{type(error).__name__}: {error}",
        "montage_file": "",
    }


def add_priorities(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for metric in METRICS_FOR_OUTLIERS:
        result[f"robust_z_{metric}"] = 0.0
    valid = result["status"].eq("PASS")
    for _, indices in result.loc[valid].groupby("group", sort=True).groups.items():
        for metric in METRICS_FOR_OUTLIERS:
            values = result.loc[indices, metric].to_numpy(float)
            result.loc[indices, f"robust_z_{metric}"] = robust_z(values)

    priorities: list[str] = []
    reasons: list[str] = []
    for _, row in result.iterrows():
        row_reasons: list[str] = []
        hard = False
        if row["status"] != "PASS":
            hard = True
            row_reasons.append(str(row["error"]))
        else:
            checks = (
                (row["mask_largest_component_fraction"] < 0.98, "fragmented mask"),
                (row["mask_boundary_fraction"] > 0.005, "mask touches grid boundary"),
                (row["cropped_mask_retention_fraction"] < 0.95, "crop removes mask"),
            )
            for condition, label in checks:
                if bool(condition):
                    hard = True
                    row_reasons.append(label)
            extreme = []
            for metric in METRICS_FOR_OUTLIERS:
                z = float(row[f"robust_z_{metric}"])
                if abs(z) >= 5.0:
                    extreme.append(f"{metric} z={z:.1f}")
            row_reasons.extend(extreme)
        if hard or len(row_reasons) >= 2:
            priorities.append("HIGH")
        elif row_reasons:
            priorities.append("REVIEW")
        else:
            priorities.append("ROUTINE")
        reasons.append("; ".join(row_reasons))
    result["auto_priority"] = priorities
    result["auto_reasons"] = reasons
    order = pd.Categorical(result["auto_priority"], ["HIGH", "REVIEW", "ROUTINE"], ordered=True)
    result = result.assign(_priority=order).sort_values(
        ["_priority", "scope", "group", "case_id"], kind="stable"
    ).drop(columns="_priority").reset_index(drop=True)
    return result


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
        "montage_file",
        "auto_priority",
        "auto_reasons",
        "mask_volume_ml",
        "mask_largest_component_fraction",
        "mask_boundary_fraction",
        "cropped_mask_retention_fraction",
    ]
    records = frame[columns].copy()
    return json.loads(records.to_json(orient="records"))


def report_html(frame: pd.DataFrame, created: str) -> str:
    payload = json.dumps(public_records(frame), separators=(",", ":"), allow_nan=False)
    report_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D-Neuro-SimCLR visual QC</title>
<style>
body{{margin:0;background:#f4f6f7;color:#17212b;font:14px Arial,sans-serif}}header{{position:sticky;top:0;z-index:2;background:#17324d;color:white;padding:12px 18px}}h1{{font-size:20px;margin:0 0 7px}}.tools{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}button,select,input{{font:inherit;padding:6px 9px}}main{{padding:14px}}.note{{max-width:1100px;margin:0 0 12px;line-height:1.4}}.card{{background:white;border:1px solid #cbd3d8;margin:0 auto 14px;max-width:850px;padding:10px;scroll-margin-top:100px}}.card img{{width:100%;display:block;background:black}}.meta{{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}}.priority-HIGH{{color:#a51d18;font-weight:bold}}.priority-REVIEW{{color:#9a5b00;font-weight:bold}}.priority-ROUTINE{{color:#22663c;font-weight:bold}}.decision button{{margin-right:6px}}.selected{{outline:3px solid #1f6f78}}.muted{{color:#52616b}}.hidden{{display:none}}#counter{{font-weight:bold}}
</style></head><body>
<header><h1>3D-Neuro-SimCLR visual QC</h1><div class="tools">
<span id="counter"></span><select id="scope"><option value="ALL">All cohorts</option><option value="brain_age">Brain age</option><option value="onharmony">ON-Harmony</option></select>
<select id="priority"><option value="ALL">All priorities</option><option>HIGH</option><option>REVIEW</option><option>ROUTINE</option></select>
<select id="decision"><option value="ALL">All decisions</option><option value="UNREVIEWED">Unreviewed</option><option>PASS</option><option>REVIEW</option><option>FAIL</option></select>
<input id="search" placeholder="subject, scanner, dataset"><button id="export">Export review TSV</button></div></header>
<main><p class="note"><b>Human review is authoritative.</b> Automatic priority only orders cases using transparent mask/image checks and robust within-group outliers. It is not a quality score and never excludes a scan. Orange is the registered mask boundary. No age, diagnosis, prediction, BAG or error was loaded. Created {html.escape(created)}.</p><div id="root"></div></main>
<script>
const CASES={payload}; const KEY='simclr-qc-{report_id}'; let decisions=JSON.parse(localStorage.getItem(KEY)||'{{}}'); let activeId=null;
function esc(x){{return String(x??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function save(){{localStorage.setItem(KEY,JSON.stringify(decisions));filter();}}
function visibleCards(){{return [...document.querySelectorAll('.card:not(.hidden)')];}}
function activate(el,scroll=true){{if(!el)return;document.querySelectorAll('.card.selected').forEach(item=>item.classList.remove('selected'));el.classList.add('selected');activeId=el.dataset.id;if(scroll)el.scrollIntoView({{block:'center'}});}}
function move(step){{let cards=visibleCards();if(!cards.length)return;let index=cards.findIndex(el=>el.dataset.id===activeId);if(index<0)index=step>0?-1:0;index=Math.max(0,Math.min(cards.length-1,index+step));activate(cards[index]);}}
function choose(id,value){{let before=visibleCards(),index=before.findIndex(el=>el.dataset.id===id);decisions[id]={{decision:value,reviewed_at_utc:new Date().toISOString()}};save();let after=visibleCards(),current=after.find(el=>el.dataset.id===id),fallback=after[Math.min(Math.max(index,0),after.length-1)];activate(current||fallback);}}
function decide(value){{let cards=visibleCards(),target=cards.find(el=>el.dataset.id===activeId)||cards[0];if(target)choose(target.dataset.id,value);}}
function card(c){{let d=decisions[c.case_id]?.decision||'UNREVIEWED';return `<article class="card" tabindex="0" data-id="${{esc(c.case_id)}}" data-scope="${{c.scope}}" data-priority="${{c.auto_priority}}" data-decision="${{d}}"><img loading="lazy" src="${{esc(c.montage_file)}}"><div class="meta"><b>${{esc(c.case_id)}}</b><span class="priority-${{c.auto_priority}}">${{c.auto_priority}}</span><span>${{esc(c.group)}}</span><span>decision: <b>${{d}}</b></span></div><div class="muted">mask=${{c.mask_volume_ml.toFixed(1)}} ml; largest component=${{(100*c.mask_largest_component_fraction).toFixed(2)}}%; crop retained=${{(100*c.cropped_mask_retention_fraction).toFixed(2)}}%; boundary=${{(100*c.mask_boundary_fraction).toFixed(3)}}%<br>${{esc(c.auto_reasons||'No automatic flag')}}</div><p class="decision"><button title="P or 1" aria-keyshortcuts="P 1" onclick="choose('${{esc(c.case_id)}}','PASS')">PASS</button><button title="R or 2" aria-keyshortcuts="R 2" onclick="choose('${{esc(c.case_id)}}','REVIEW')">REVIEW</button><button title="F or 3" aria-keyshortcuts="F 3" onclick="choose('${{esc(c.case_id)}}','FAIL')">FAIL</button></p></article>`;}}
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
    cases = load_fine_tuning_cases(
        args.manifest.expanduser().resolve(),
        args.fine_tuning_root.expanduser().resolve(),
        expected_ft,
    ) + load_onharmony_cases(args.onharmony_input_list.expanduser().resolve(), expected_onh)
    if not args.allow_noncanonical_counts and len(cases) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} total cases, found {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate QC case identifiers")

    montage_dir = output_dir / "montages"
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_case, case, montage_dir): case for case in cases}
        completed = 0
        for future in as_completed(futures):
            case = futures[future]
            try:
                rows.append(future.result())
            except BaseException as error:
                rows.append(failed_row(case, error))
            completed += 1
            if completed % 25 == 0 or completed == len(cases):
                print(f"[{completed}/{len(cases)}]", flush=True)
    frame = add_priorities(pd.DataFrame(rows))
    metrics_path = output_dir / "qc_metrics.tsv"
    review_path = output_dir / "qc_review.tsv"
    report_path = output_dir / "index.html"
    created = utc_now()
    atomic_tsv(metrics_path, frame)
    atomic_tsv(
        review_path,
        pd.DataFrame(columns=["case_id", "decision", "reviewed_at_utc"]),
    )
    atomic_text(report_path, report_html(frame, created))
    summary = {
        "status": "COMPLETED",
        "created_utc": created,
        "label_blind": True,
        "outcomes_loaded": [],
        "n_cases": int(len(frame)),
        "scope_counts": {str(k): int(v) for k, v in frame["scope"].value_counts().items()},
        "automatic_priority_counts": {
            str(k): int(v) for k, v in frame["auto_priority"].value_counts().items()
        },
        "processing_failures": int(frame["status"].ne("PASS").sum()),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "human_review_required": True,
    }
    atomic_json(output_dir / "qc_summary.json", summary)
    running.unlink(missing_ok=True)
    print(f"Report: {report_path}")
    print(f"Metrics: {metrics_path}")


def command_rerender(args: argparse.Namespace) -> None:
    """Rebuild the HTML shell without recomputing QC metrics or montages."""
    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = report_dir / "qc_metrics.tsv"
    summary_path = report_dir / "qc_summary.json"
    report_path = report_dir / "index.html"
    for path in (metrics_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETED":
        raise ValueError("QC report is not in COMPLETED state")
    observed_metrics_sha256 = sha256_file(metrics_path)
    if observed_metrics_sha256 != summary.get("metrics_sha256"):
        raise ValueError("QC metrics SHA-256 differs from qc_summary.json")

    metrics = pd.read_csv(metrics_path, sep="\t", dtype={"case_id": str})
    if len(metrics) != int(summary.get("n_cases", -1)):
        raise ValueError("QC metric count differs from qc_summary.json")
    for relative_path in metrics["montage_file"]:
        montage_path = (report_dir / str(relative_path)).resolve()
        if not montage_path.is_relative_to(report_dir) or not montage_path.is_file():
            raise FileNotFoundError(f"Missing or invalid montage: {relative_path}")

    created = str(summary["created_utc"])
    atomic_text(report_path, report_html(metrics, created))
    summary.update(
        {
            "report_sha256": sha256_file(report_path),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "report_rebuilt_utc": utc_now(),
            "report_rebuild_reason": (
                "Corrected HTML JavaScript escaping; QC metrics and montages unchanged."
            ),
        }
    )
    atomic_json(summary_path, summary)
    print(f"Report rebuilt: {report_path}")
    print(f"Cases preserved: {len(metrics)}")


def command_finalize(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = report_dir / "qc_metrics.tsv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    review_path = args.review_tsv.expanduser().resolve()
    metrics = pd.read_csv(metrics_path, sep="\t", dtype={"case_id": str})
    review = pd.read_csv(review_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"case_id", "decision", "reviewed_at_utc"}
    if not required.issubset(review.columns):
        raise ValueError(f"Review TSV lacks columns: {sorted(required - set(review.columns))}")
    if review["case_id"].duplicated().any():
        raise ValueError("Review TSV contains duplicate case IDs")
    if not set(review["decision"]).issubset(REVIEW_DECISIONS):
        raise ValueError("Review TSV contains an invalid decision")
    unknown = set(review["case_id"]) - set(metrics["case_id"])
    if unknown:
        raise ValueError(f"Review TSV contains unknown cases: {sorted(unknown)[:10]}")
    if not args.allow_incomplete and set(review["case_id"]) != set(metrics["case_id"]):
        raise ValueError(
            f"Incomplete review: {len(review)}/{len(metrics)} cases. "
            "Pass --allow-incomplete only for an explicitly partial audit."
        )
    joined = metrics.merge(review[list(required)], on="case_id", how="left")
    decisions = joined["decision"].fillna("UNREVIEWED")
    final = {
        "status": "HUMAN_REVIEW_COMPLETED" if (decisions != "UNREVIEWED").all() else "HUMAN_REVIEW_PARTIAL",
        "created_utc": utc_now(),
        "label_blind": True,
        "n_cases": int(len(joined)),
        "n_reviewed": int((decisions != "UNREVIEWED").sum()),
        "decision_counts": {str(k): int(v) for k, v in decisions.value_counts().items()},
        "metrics_sha256": sha256_file(metrics_path),
        "review_source": str(review_path),
        "review_source_sha256": sha256_file(review_path),
        "automatic_priority_is_not_an_exclusion_decision": True,
    }
    atomic_tsv(report_dir / "qc_review_completed.tsv", joined)
    final["completed_review_sha256"] = sha256_file(report_dir / "qc_review_completed.tsv")
    atomic_json(report_dir / "qc_final_summary.json", final)
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
        default=PROJECT_ROOT / "preprocessings/FineTunningDataset/preproc_3DNeuroSimCLR",
    )
    build.add_argument(
        "--onharmony-input-list",
        type=Path,
        default=PROJECT_ROOT / "DATA/filelists/filelist_3DNeuroSimCLR.txt",
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
