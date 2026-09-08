#!/usr/bin/env python3
"""Build a label-blind visual QC report for the shared CAT12 GM pipeline.

The report covers the CAT12 output consumed by AnatCL and used to derive the
y-Aware input. Automated image metrics only prioritise visual review; they do
not make exclusion decisions. Human decisions are stored in the browser and
can be exported as a TSV, then frozen with the ``finalize`` subcommand.
The ``rescore`` subcommand can add or refresh the transparent 0-10 automatic
quality score in an existing, not-yet-finalized report without reopening the
images or regenerating montages.

The generated montages contain native T1w slices and must be treated as
sensitive imaging data. They are not publication or GitHub artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from typing import Any, Iterable
from xml.etree import ElementTree

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_GM_SHAPE = (121, 145, 121)
EXPECTED_GM_ZOOMS = (1.5, 1.5, 1.5)
EXPECTED_FT_CASES = 1411
EXPECTED_ONH_CASES = 165
GM_THRESHOLD = 0.10
DOWNSAMPLE_STEP = 2
ROBUST_OUTLIER_Z = 3.5
ROBUST_SEVERE_Z = 5.0
QUALITY_SCORE_VERSION = "cat12_siqr_inverted_quality_score_v2"
QUALITY_FULL_PENALTY_Z = 5.0
ROBUST_METRIC_DIRECTIONS = {
    "gm_template_ncc": "low",
    "gm_quartic_mean_z": "high",
    "gm_volume_ml": "both",
    "gm_support_fraction": "both",
    "gm_edge_mass_fraction": "high",
    "gm_lr_asymmetry": "high",
    "gm_com_distance_mm": "high",
    "cat12_siqr": "high",
    "cat12_ncr": "high",
    "cat12_icr": "high",
    "within_subject_ncc": "low",
}
REVIEW_COMPONENTS = {
    "registration": {
        "weight": 0.35,
        "metrics": ("gm_template_ncc", "gm_quartic_mean_z", "gm_com_distance_mm"),
    },
    "morphology": {
        "weight": 0.25,
        "metrics": (
            "gm_volume_ml",
            "gm_support_fraction",
            "gm_edge_mass_fraction",
            "gm_lr_asymmetry",
        ),
    },
    "cat12": {
        "weight": 0.30,
        "metrics": ("cat12_siqr", "cat12_ncr", "cat12_icr"),
    },
    "repeatability": {
        "weight": 0.10,
        "metrics": ("within_subject_ncc",),
        "sources": ("onharmony",),
    },
}
ACQUISITION_RE = re.compile(
    r"(?P<acquisition>sub-[A-Za-z0-9]+_ses-(?P<session>[A-Za-z0-9]+)_T1w)"
)
SUBJECT_RE = re.compile(r"^(?P<subject>sub-[A-Za-z0-9]+)_ses-")
DECISIONS = ("PASS", "REVIEW", "FAIL")
REVIEW_COLUMNS = (
    "case_id",
    "source",
    "dataset",
    "subject",
    "session",
    "acquisition_id",
    "decision",
    "reasons",
    "notes",
    "rater",
    "reviewed_at_utc",
    "raw_sha256",
    "gm_sha256",
    "montage_sha256",
)


@dataclass(frozen=True)
class Case:
    """One pseudonymised scan and the two images required for visual QC."""

    case_id: str
    source: str
    group: str
    dataset: str
    subject: str
    session: str
    acquisition_id: str
    split: str
    raw_path: Path
    gm_path: Path
    qc_path: Path | None = None
    expected_raw_sha256: str = ""
    expected_gm_sha256: str = ""
    provenance_status: str = "UNKNOWN"


def utc_now() -> str:
    """Return a stable UTC timestamp for audit records."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    """Publish one text artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    """Publish JSON with finite numeric values only."""

    atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    """Publish one TSV atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, sep="\t", index=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def acquisition_id(value: str | Path) -> str:
    """Extract the canonical ON-Harmony acquisition identifier."""

    match = ACQUISITION_RE.search(str(value))
    if match is None:
        raise ValueError(f"Cannot parse acquisition ID from {value!r}")
    return match.group("acquisition")


def scanner_from_session(session: str) -> str:
    """Remove the three-digit sequence suffix from an ON-Harmony session."""

    return re.sub(r"\d{3}$", "", session)


def read_lines(path: Path) -> list[str]:
    """Read a nonempty, ordered, duplicate-free file registry."""

    if not path.is_file():
        raise FileNotFoundError(path)
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"Empty file registry: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate paths in file registry: {path}")
    return values


def load_ft_cases(
    manifest_path: Path,
    preproc_root: Path,
    *,
    expected_cases: int | None = EXPECTED_FT_CASES,
) -> list[Case]:
    """Load only label-blind fields from the definitive fine-tuning manifest."""

    required = {
        "dataset", "subject", "raw_t1w_path", "split", "inclusion_status",
        "required_offline_groups",
    }
    header = set(pd.read_csv(manifest_path, sep="\t", nrows=0).columns)
    missing = required - header
    if missing:
        raise ValueError(f"Fine-tuning manifest lacks columns: {sorted(missing)}")
    frame = pd.read_csv(
        manifest_path,
        sep="\t",
        dtype=str,
        usecols=sorted(required),
        keep_default_na=False,
    )
    frame = frame.loc[frame["inclusion_status"].eq("INCLUDED")].copy()
    if frame.empty:
        raise ValueError("Fine-tuning manifest has no INCLUDED cases")
    if expected_cases is not None and len(frame) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} fine-tuning cases, found {len(frame)}"
        )
    if frame.duplicated(["dataset", "subject"]).any():
        raise ValueError("Fine-tuning manifest contains duplicate dataset/subject keys")
    if not frame["required_offline_groups"].str.split(";").apply(
        lambda values: "cat12_ixi555_gm" in values
    ).all():
        raise ValueError("Some fine-tuning rows do not permit cat12_ixi555_gm")

    cases = []
    for row in frame.to_dict(orient="records"):
        dataset = row["dataset"]
        subject = row["subject"]
        root = preproc_root / dataset / subject
        cases.append(
            Case(
                case_id=f"finetuning::{dataset}::{subject}",
                source="finetuning",
                group=dataset,
                dataset=dataset,
                subject=subject,
                session="",
                acquisition_id=subject,
                split=row["split"],
                raw_path=Path(row["raw_t1w_path"]),
                gm_path=root / f"{subject}_anatcl_ready.nii.gz",
                qc_path=root / "preprocessing_qc.json",
            )
        )
    return cases


def _read_onh_provenance(path: Path | None) -> dict[str, dict[str, str]]:
    """Read the safe fields needed to bind historical ON-Harmony artifacts."""

    if path is None or not path.is_file():
        return {}
    wanted = {
        "acquisition_id", "raw_t1w_sha256", "output_sha256", "status",
    }
    header = set(pd.read_csv(path, sep="\t", nrows=0).columns)
    if not wanted.issubset(header):
        raise ValueError(f"ON-Harmony provenance lacks columns: {sorted(wanted - header)}")
    frame = pd.read_csv(path, sep="\t", dtype=str, usecols=sorted(wanted)).fillna("")
    if frame["acquisition_id"].duplicated().any():
        raise ValueError("Duplicate acquisition IDs in ON-Harmony provenance")
    return frame.set_index("acquisition_id").to_dict(orient="index")


def load_onh_cases(
    raw_filelist: Path,
    gm_filelist: Path,
    provenance_path: Path | None,
    *,
    expected_cases: int | None = EXPECTED_ONH_CASES,
) -> list[Case]:
    """Pair raw and CAT12 paths by acquisition ID without loading outcomes."""

    raw_values = read_lines(raw_filelist)
    gm_values = read_lines(gm_filelist)
    if expected_cases is not None:
        if len(raw_values) != expected_cases or len(gm_values) != expected_cases:
            raise ValueError(
                f"Expected {expected_cases} ON-Harmony paths; "
                f"raw={len(raw_values)}, GM={len(gm_values)}"
            )
    raw_by_id = {acquisition_id(value): Path(value) for value in raw_values}
    gm_by_id = {acquisition_id(value): Path(value) for value in gm_values}
    if len(raw_by_id) != len(raw_values) or len(gm_by_id) != len(gm_values):
        raise ValueError("Duplicate ON-Harmony acquisition IDs")
    if set(raw_by_id) != set(gm_by_id):
        raise ValueError("Raw and CAT12 ON-Harmony registries contain different acquisitions")

    provenance = _read_onh_provenance(provenance_path)
    cases = []
    for stem in [acquisition_id(value) for value in raw_values]:
        match = ACQUISITION_RE.fullmatch(stem)
        subject_match = SUBJECT_RE.match(stem)
        if match is None or subject_match is None:
            raise ValueError(f"Malformed acquisition ID: {stem}")
        session = match.group("session")
        details = provenance.get(stem, {})
        cases.append(
            Case(
                case_id=f"onharmony::{stem}",
                source="onharmony",
                group=scanner_from_session(session),
                dataset="ON-Harmony",
                subject=subject_match.group("subject"),
                session=f"ses-{session}",
                acquisition_id=stem,
                split="external",
                raw_path=raw_by_id[stem],
                gm_path=gm_by_id[stem],
                expected_raw_sha256=details.get("raw_t1w_sha256", ""),
                expected_gm_sha256=details.get("output_sha256", ""),
                provenance_status=details.get("status", "MISSING"),
            )
        )
    return cases


def select_cases(cases: list[Case], limit_per_source: int | None) -> list[Case]:
    """Select a deterministic pilot while preserving both source cohorts."""

    if limit_per_source is None:
        return cases
    if limit_per_source <= 0:
        raise ValueError("--limit-per-source must be positive")
    selected: list[Case] = []
    for source in sorted({case.source for case in cases}):
        source_cases = [case for case in cases if case.source == source]
        if len(source_cases) <= limit_per_source:
            selected.extend(source_cases)
            continue
        positions = np.linspace(0, len(source_cases) - 1, limit_per_source, dtype=int)
        selected.extend(source_cases[int(position)] for position in positions)
    return selected


def _shape_text(shape: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _zooms_text(zooms: Iterable[float]) -> str:
    return "x".join(f"{float(value):.5g}" for value in zooms)


def _safe_float(value: float | np.floating[Any]) -> float:
    result = float(value)
    return result if math.isfinite(result) else 0.0


def raw_metrics(data: np.ndarray) -> dict[str, float]:
    """Compute simple native-T1 metrics that are robust to absolute scaling."""

    array = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(array)
    fraction = float(finite.mean())
    values = array[finite]
    if values.size == 0:
        raise ValueError("Raw T1w contains no finite voxels")
    q01, q50, q99 = np.quantile(values, [0.01, 0.50, 0.99])
    dynamic = float(q99 - q01)
    threshold = float(q01 + 0.05 * dynamic)
    return {
        "raw_finite_fraction": fraction,
        "raw_q01": float(q01),
        "raw_q50": float(q50),
        "raw_q99": float(q99),
        "raw_robust_dynamic_range": dynamic,
        "raw_foreground_fraction": float(np.mean(array > threshold)),
    }


def gm_metrics(data: np.ndarray, affine: np.ndarray) -> dict[str, float]:
    """Compute geometry, occupancy and symmetry metrics for normalized CAT12 GM."""

    array = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.all():
        raise ValueError("CAT12 GM contains non-finite voxels")
    positive = np.clip(array, 0.0, None)
    total = float(positive.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("CAT12 GM has no positive mass")
    centre_voxel = np.empty(3, dtype=np.float64)
    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        axis_mass = positive.sum(axis=other_axes, dtype=np.float64)
        centre_voxel[axis] = float(
            np.dot(np.arange(array.shape[axis], dtype=np.float64), axis_mass) / total
        )
    centre_world = np.asarray(affine, dtype=np.float64) @ np.r_[centre_voxel, 1.0]
    edge = np.zeros(array.shape, dtype=bool)
    edge_width = 3
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, edge_width)
        upper[axis] = slice(-edge_width, None)
        edge[tuple(lower)] = True
        edge[tuple(upper)] = True
    mirrored = np.flip(positive, axis=0)
    symmetry_denominator = 2.0 * float(np.mean(positive, dtype=np.float64))
    asymmetry = float(np.mean(np.abs(positive - mirrored), dtype=np.float64))
    if symmetry_denominator > 0.0:
        asymmetry /= symmetry_denominator
    voxel_volume = abs(float(np.linalg.det(np.asarray(affine)[:3, :3])))
    return {
        "gm_finite_fraction": 1.0,
        "gm_negative_fraction": float(np.mean(array < 0.0)),
        "gm_mean": float(np.mean(array, dtype=np.float64)),
        "gm_std": float(np.std(array, dtype=np.float64)),
        "gm_q99": float(np.quantile(array, 0.99)),
        "gm_support_fraction": float(np.mean(array > GM_THRESHOLD)),
        "gm_edge_mass_fraction": float(positive[edge].sum(dtype=np.float64) / total),
        "gm_lr_asymmetry": asymmetry,
        "gm_volume_ml": float(total * voxel_volume / 1000.0),
        "gm_com_x_mm": float(centre_world[0]),
        "gm_com_y_mm": float(centre_world[1]),
        "gm_com_z_mm": float(centre_world[2]),
    }


def _xml_float(section: ElementTree.Element | None, names: Iterable[str]) -> float:
    """Read the first finite scalar from alternative CAT12 XML fields."""

    if section is None:
        return float("nan")
    for name in names:
        node = section.find(f".//{name}")
        if node is None or node.text is None:
            continue
        try:
            value = float(node.text.strip())
        except ValueError:
            continue
        if math.isfinite(value):
            return value
    return float("nan")


def load_cat12_quality(gm_path: Path) -> dict[str, Any]:
    """Load official CAT12 quality ratings when the retained XML is available."""

    candidates = sorted((gm_path.parent / "cat12_work" / "report").glob("cat_*.xml"))
    if not candidates:
        candidates = sorted((gm_path.parent / "report").glob("cat_*.xml"))
    if len(candidates) != 1:
        return {
            "cat12_xml_status": "MISSING" if not candidates else "AMBIGUOUS",
            "cat12_xml_path": "",
            "cat12_xml_sha256": "",
            "cat12_siqr": float("nan"),
            "cat12_ncr": float("nan"),
            "cat12_icr": float("nan"),
            "cat12_resolution_rating": float("nan"),
        }
    path = candidates[0]
    try:
        root = ElementTree.parse(path).getroot()
        ratings = root.find(".//qualityratings")
        return {
            "cat12_xml_status": "PASS",
            "cat12_xml_path": str(path),
            "cat12_xml_sha256": sha256_file(path),
            "cat12_siqr": _xml_float(ratings, ("SIQR", "IQR")),
            "cat12_ncr": _xml_float(ratings, ("NCR",)),
            "cat12_icr": _xml_float(ratings, ("ICR",)),
            "cat12_resolution_rating": _xml_float(ratings, ("res_RMS",)),
        }
    except (OSError, ElementTree.ParseError) as error:
        return {
            "cat12_xml_status": f"INVALID: {type(error).__name__}",
            "cat12_xml_path": str(path),
            "cat12_xml_sha256": "",
            "cat12_siqr": float("nan"),
            "cat12_ncr": float("nan"),
            "cat12_icr": float("nan"),
            "cat12_resolution_rating": float("nan"),
        }


def robust_z_scores(values: np.ndarray) -> np.ndarray:
    """Return median/MAD z-scores, with a standard-deviation fallback."""

    array = np.asarray(values, dtype=np.float64)
    result = np.full(array.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(array)
    if finite.sum() < 3:
        return result
    observed = array[finite]
    median = float(np.median(observed))
    scale = 1.4826 * float(np.median(np.abs(observed - median)))
    if scale <= np.finfo(np.float64).eps:
        scale = float(np.std(observed, ddof=1))
    if scale <= np.finfo(np.float64).eps:
        result[finite] = 0.0
    else:
        result[finite] = (observed - median) / scale
    return result


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation between two finite vectors."""

    a = np.asarray(left, dtype=np.float32).ravel()
    b = np.asarray(right, dtype=np.float32).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 2:
        return float("nan")
    a = a[finite] - float(np.mean(a[finite]))
    b = b[finite] - float(np.mean(b[finite]))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")


def _normalise_u8(array: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        lower, upper = 0.0, 1.0
    clipped = np.clip((np.asarray(array, dtype=np.float32) - lower) / (upper - lower), 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)


def _slice(array: np.ndarray, axis: int, fraction: float) -> np.ndarray:
    index = int(round((array.shape[axis] - 1) * fraction))
    return np.rot90(np.take(array, index, axis=axis))


def _letterbox(image: Image.Image, size: int = 150) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (5, 8, 12))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def _render_raw_tile(array: np.ndarray, axis: int, fraction: float, lo: float, hi: float) -> Image.Image:
    pixels = _normalise_u8(_slice(array, axis, fraction), lo, hi)
    return _letterbox(Image.fromarray(pixels, mode="L").convert("RGB"))


def _render_gm_tile(
    gm: np.ndarray,
    template: np.ndarray,
    axis: int,
    fraction: float,
    gm_hi: float,
    template_lo: float,
    template_hi: float,
) -> Image.Image:
    background = _normalise_u8(_slice(template, axis, fraction), template_lo, template_hi)
    overlay = np.asarray(_slice(gm, axis, fraction), dtype=np.float32)
    alpha = np.clip(overlay / max(gm_hi, 1e-6), 0.0, 1.0) * 0.82
    alpha[overlay < 0.03] = 0.0
    rgb = np.repeat(background[..., None], 3, axis=2).astype(np.float32)
    colour = np.array([14.0, 210.0, 190.0], dtype=np.float32)
    rgb = rgb * (1.0 - alpha[..., None]) + colour * alpha[..., None]
    return _letterbox(Image.fromarray(np.round(rgb).astype(np.uint8), mode="RGB"))


def placeholder_tile(message: str) -> Image.Image:
    """Return a visible placeholder for a missing or invalid image."""

    image = Image.new("RGB", (150, 150), (72, 18, 24))
    draw = ImageDraw.Draw(image)
    words = message[:90].split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 20:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    draw.multiline_text((8, 12), "\n".join(lines[:8]), fill=(255, 220, 220), spacing=3)
    return image


def render_montage(
    raw: np.ndarray | None,
    gm: np.ndarray | None,
    template: np.ndarray,
    output: Path,
    errors: list[str],
) -> None:
    """Render native T1w and normalized-GM views into one compact JPEG."""

    axes = ((0, "Sag"), (1, "Cor"), (2, "Ax"))
    fractions = (0.30, 0.50, 0.70)
    tile = 150
    label_width = 90
    header_height = 28
    row_gap = 10
    canvas = Image.new(
        "RGB", (label_width + 9 * tile, header_height + 2 * tile + row_gap), (5, 8, 12)
    )
    draw = ImageDraw.Draw(canvas)
    for axis_index, (_, axis_label) in enumerate(axes):
        for fraction_index, fraction in enumerate(fractions):
            x = label_width + (axis_index * 3 + fraction_index) * tile
            draw.text((x + 5, 8), f"{axis_label} {int(fraction * 100)}%", fill=(220, 226, 232))
    draw.text((8, header_height + 66), "RAW T1w", fill=(255, 255, 255))
    draw.text((8, header_height + tile + row_gap + 60), "CAT12 GM", fill=(14, 210, 190))

    if raw is not None:
        finite = raw[np.isfinite(raw)]
        lo, hi = np.quantile(finite, [0.01, 0.995]) if finite.size else (0.0, 1.0)
    else:
        lo, hi = 0.0, 1.0
    if gm is not None:
        gm_positive = gm[gm > 0.0]
        gm_hi = float(np.quantile(gm_positive, 0.995)) if gm_positive.size else 1.0
    else:
        gm_hi = 1.0
    template_finite = template[np.isfinite(template)]
    template_lo, template_hi = np.quantile(template_finite, [0.01, 0.995])

    for axis_index, (axis, _) in enumerate(axes):
        for fraction_index, fraction in enumerate(fractions):
            column = axis_index * 3 + fraction_index
            x = label_width + column * tile
            raw_tile = (
                _render_raw_tile(raw, axis, fraction, float(lo), float(hi))
                if raw is not None
                else placeholder_tile("RAW unavailable")
            )
            gm_tile = (
                _render_gm_tile(
                    gm, template, axis, fraction, gm_hi,
                    float(template_lo), float(template_hi),
                )
                if gm is not None
                else placeholder_tile("GM unavailable")
            )
            canvas.paste(raw_tile, (x, header_height))
            canvas.paste(gm_tile, (x, header_height + tile + row_gap))

    if errors:
        draw.rectangle((0, 0, canvas.width - 1, canvas.height - 1), outline=(240, 70, 78), width=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".jpg", dir=output.parent, prefix=f".{output.stem}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        canvas.save(temporary, format="JPEG", quality=91, optimize=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _load_ft_provenance(case: Case) -> tuple[str, str, str, list[str]]:
    """Validate the identity fields of a fine-tuning preprocessing QC."""

    errors: list[str] = []
    if case.qc_path is None or not case.qc_path.is_file():
        return "", "", "MISSING", [f"missing preprocessing QC: {case.qc_path}"]
    try:
        payload = json.loads(case.qc_path.read_text())
    except Exception as error:
        return "", "", "INVALID", [f"cannot read preprocessing QC: {error}"]
    expected = {
        "status": "PASS",
        "dataset": case.dataset,
        "subject": case.subject,
        "split": case.split,
    }
    for key, value in expected.items():
        if str(payload.get(key, "")) != value:
            errors.append(f"QC {key}={payload.get(key)!r}, expected {value!r}")
    raw_details = payload.get("raw_input", {})
    raw_sha = str(raw_details.get("sha256", "")) if isinstance(raw_details, dict) else ""
    gm_sha = str(payload.get("anatcl_output_sha256", ""))
    if not gm_sha:
        errors.append("QC lacks CAT12 GM SHA-256")
    return raw_sha, gm_sha, "PASS" if not errors else "INVALID", errors


def inspect_case(
    case: Case,
    template: np.ndarray,
    montage_dir: Path,
    *,
    verify_hashes: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    """Inspect one case, render its montage and return a downsampled GM volume."""

    record: dict[str, Any] = {
        "case_id": case.case_id,
        "source": case.source,
        "group": case.group,
        "dataset": case.dataset,
        "subject": case.subject,
        "session": case.session,
        "acquisition_id": case.acquisition_id,
        "split": case.split,
        "raw_path": str(case.raw_path),
        "gm_path": str(case.gm_path),
    }
    errors: list[str] = []
    warnings: list[str] = []
    raw_sha = case.expected_raw_sha256
    gm_sha = case.expected_gm_sha256
    provenance_status = case.provenance_status
    if case.source == "finetuning":
        raw_sha, gm_sha, provenance_status, provenance_errors = _load_ft_provenance(case)
        errors.extend(provenance_errors)
    elif provenance_status != "PASS":
        errors.append(f"ON-Harmony provenance status is {provenance_status}")
    record["provenance_status"] = provenance_status

    raw: np.ndarray | None = None
    gm: np.ndarray | None = None
    raw_image: nib.spatialimages.SpatialImage | None = None
    gm_image: nib.spatialimages.SpatialImage | None = None
    try:
        if not case.raw_path.is_file():
            raise FileNotFoundError(case.raw_path)
        raw_image = nib.load(str(case.raw_path))
        if len(raw_image.shape) == 4 and raw_image.shape[3] == 1:
            raw_image = nib.funcs.squeeze_image(raw_image)
        if len(raw_image.shape) != 3:
            raise ValueError(f"raw T1w must be 3D, got {raw_image.shape}")
        canonical_raw = nib.as_closest_canonical(raw_image)
        raw = canonical_raw.get_fdata(dtype=np.float32)
        record.update(raw_metrics(raw))
        record["raw_shape"] = _shape_text(raw_image.shape)
        record["raw_zooms_mm"] = _zooms_text(raw_image.header.get_zooms()[:3])
        record["raw_orientation"] = "".join(nib.aff2axcodes(raw_image.affine))
        determinant = float(np.linalg.det(np.asarray(raw_image.affine)[:3, :3]))
        record["raw_affine_determinant"] = determinant
        if abs(determinant) < 1e-8:
            errors.append("raw affine is degenerate")
        if record["raw_finite_fraction"] < 1.0:
            errors.append("raw T1w contains NaN or Inf")
    except Exception as error:
        errors.append(f"raw load failed: {type(error).__name__}: {error}")

    downsample_shape = tuple((value + DOWNSAMPLE_STEP - 1) // DOWNSAMPLE_STEP for value in EXPECTED_GM_SHAPE)
    downsample = np.full(downsample_shape, np.nan, dtype=np.float16)
    try:
        if not case.gm_path.is_file():
            raise FileNotFoundError(case.gm_path)
        gm_image = nib.load(str(case.gm_path))
        if tuple(int(value) for value in gm_image.shape) != EXPECTED_GM_SHAPE:
            raise ValueError(
                f"GM shape {gm_image.shape}, expected {EXPECTED_GM_SHAPE}"
            )
        zooms = tuple(float(value) for value in gm_image.header.get_zooms()[:3])
        if not np.allclose(zooms, EXPECTED_GM_ZOOMS, rtol=0.0, atol=1e-3):
            raise ValueError(f"GM zooms {zooms}, expected {EXPECTED_GM_ZOOMS}")
        canonical_gm = nib.as_closest_canonical(gm_image)
        if tuple(int(value) for value in canonical_gm.shape) != EXPECTED_GM_SHAPE:
            raise ValueError(
                f"canonical GM shape {canonical_gm.shape}, expected {EXPECTED_GM_SHAPE}; "
                "orientation is inconsistent with the fixed CAT12 grid"
            )
        gm = canonical_gm.get_fdata(dtype=np.float32)
        record.update(gm_metrics(gm, canonical_gm.affine))
        record.update(load_cat12_quality(case.gm_path))
        record["gm_shape"] = _shape_text(gm_image.shape)
        record["gm_zooms_mm"] = _zooms_text(zooms)
        record["gm_orientation"] = "".join(nib.aff2axcodes(gm_image.affine))
        record["gm_affine_determinant"] = float(
            np.linalg.det(np.asarray(gm_image.affine)[:3, :3])
        )
        downsample = np.ascontiguousarray(
            gm[::DOWNSAMPLE_STEP, ::DOWNSAMPLE_STEP, ::DOWNSAMPLE_STEP],
            dtype=np.float16,
        )
        record["gm_loaded"] = True
    except Exception as error:
        record["gm_loaded"] = False
        record.update(load_cat12_quality(case.gm_path))
        errors.append(f"GM load failed: {type(error).__name__}: {error}")

    hash_check = "NOT_RUN"
    if verify_hashes and raw is not None and gm is not None:
        try:
            observed_raw = sha256_file(case.raw_path)
            observed_gm = sha256_file(case.gm_path)
            if raw_sha and observed_raw != raw_sha:
                errors.append("raw T1w SHA-256 differs from provenance")
            if gm_sha and observed_gm != gm_sha:
                errors.append("CAT12 GM SHA-256 differs from provenance")
            if not raw_sha:
                raw_sha = observed_raw
                warnings.append("raw SHA had no prior provenance and was computed now")
            if not gm_sha:
                gm_sha = observed_gm
                warnings.append("GM SHA had no prior provenance and was computed now")
            hash_check = "PASS" if not any("SHA-256 differs" in item for item in errors) else "FAIL"
        except Exception as error:
            hash_check = "FAIL"
            errors.append(f"hash verification failed: {type(error).__name__}: {error}")
    record["hash_check"] = hash_check
    record["raw_sha256"] = raw_sha
    record["gm_sha256"] = gm_sha

    safe_name = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:20] + ".jpg"
    montage_path = montage_dir / safe_name
    render_montage(raw, gm, template, montage_path, errors)
    record["montage"] = str(Path("montages") / safe_name)
    record["montage_sha256"] = sha256_file(montage_path)
    record["technical_status"] = "FAIL" if errors else ("WARN" if warnings else "PASS")
    record["technical_errors"] = "; ".join(errors)
    record["technical_warnings"] = "; ".join(warnings)
    return record, downsample


def quality_score_contract() -> dict[str, Any]:
    """Describe the CAT12-derived score and separate review-priority metrics."""

    return {
        "version": QUALITY_SCORE_VERSION,
        "range": [0.0, 10.0],
        "higher_is_better": True,
        "role": "display_CAT12_SIQR_on_an_intuitive_higher_is_better_scale",
        "exclusion_decision": False,
        "source_measure": "CAT12 SIQR numeric quality grade",
        "source_measure_range": [0.5, 10.5],
        "source_measure_direction": "lower_is_better",
        "formula": "clip(10.5 - CAT12_SIQR, 0, 10)",
        "missing_value": "NA when no valid CAT12 SIQR is available",
        "technical_failure": "score forced to 0",
        "reference": {
            "title": "CAT manual - CAT report and quality control",
            "url": "https://neuro-jena.github.io/cat12-help/#qc",
        },
        "interpretation": (
            "This is a linear re-expression of CAT12 SIQR, not a new classifier or "
            "automatic exclusion threshold. Human visual review remains authoritative."
        ),
        "review_priority": {
            "role": "prioritisation_only",
            "reference_population": (
                "robust z-scores calculated within FineTuning dataset or "
                "ON-Harmony scanner group"
            ),
            "metric_penalty": (
                "Only deviations in the adverse direction are penalised; "
                "penalty = clip(adverse_robust_z / 5, 0, 1)."
            ),
            "components": {
                name: {
                    "weight": details["weight"],
                    "metrics": list(details["metrics"]),
                    "sources": list(
                        details.get("sources", ("finetuning", "onharmony"))
                    ),
                }
                for name, details in REVIEW_COMPONENTS.items()
            },
            "caveat": (
                "Relative deviations can reflect anatomy, age, pathology, scanner, or "
                "preprocessing; they indicate what to inspect, not what to reject."
            ),
        },
    }


def _metric_penalty(robust_z: Any, direction: str) -> float:
    """Map one adverse robust z-score to a bounded 0-1 penalty."""

    try:
        value = float(robust_z)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(value):
        return float("nan")
    if direction == "low":
        adverse = -value
    elif direction == "high":
        adverse = value
    elif direction == "both":
        adverse = abs(value)
    else:
        raise ValueError(f"Unknown quality-score direction: {direction}")
    return float(np.clip(max(0.0, adverse) / QUALITY_FULL_PENALTY_Z, 0.0, 1.0))


def _finite_float_or_nan(value: Any) -> float:
    """Convert TSV/object values to a finite float, otherwise return NaN."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return converted if math.isfinite(converted) else float("nan")


def _review_component_score(row: pd.Series, metrics: Iterable[str]) -> float:
    """Return one 0-10 relative-typicality component for review triage."""

    penalties = [
        _metric_penalty(
            row.get(f"{metric}_robust_z", np.nan),
            ROBUST_METRIC_DIRECTIONS[metric],
        )
        for metric in metrics
    ]
    finite = [value for value in penalties if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(10.0 * (1.0 - np.mean(finite)))


def add_automatic_assessment(frame: pd.DataFrame) -> pd.DataFrame:
    """Add CAT12 quality score plus separate relative review-priority flags."""

    result = frame.copy()
    stale_columns = [
        column
        for column in result.columns
        if column.startswith("auto_quality_")
        or column.startswith("auto_review_")
        or column == "auto_typicality_score"
    ]
    if stale_columns:
        result = result.drop(columns=stale_columns)
    priorities: list[str] = []
    flag_texts: list[str] = []
    flag_counts: list[int] = []
    quality_scores: list[float] = []
    typicality_scores: list[float] = []
    coverages: list[float] = []
    component_values: dict[str, list[float]] = {
        name: [] for name in REVIEW_COMPONENTS
    }

    for _, row in result.iterrows():
        flags: list[str] = []
        severe = False
        for metric, direction in ROBUST_METRIC_DIRECTIONS.items():
            value = _finite_float_or_nan(row.get(f"{metric}_robust_z", np.nan))
            if not math.isfinite(value):
                continue
            flagged = (
                (direction == "high" and value > ROBUST_OUTLIER_Z)
                or (direction == "low" and value < -ROBUST_OUTLIER_Z)
                or (direction == "both" and abs(value) > ROBUST_OUTLIER_Z)
            )
            if flagged:
                flags.append(metric)
                severe = severe or abs(value) > ROBUST_SEVERE_Z
        siqr = _finite_float_or_nan(row.get("cat12_siqr", np.nan))
        if math.isfinite(siqr) and siqr >= 5.0:
            flags.append("cat12_siqr_critical")
            severe = severe or siqr >= 7.0
        negative_fraction = _finite_float_or_nan(
            row.get("gm_negative_fraction", 0.0)
        )
        if not math.isfinite(negative_fraction):
            negative_fraction = 0.0
        if negative_fraction > 1e-5:
            flags.append("negative_gm_values")
        technical_failure = row.get("technical_status") == "FAIL"
        if technical_failure:
            flags.insert(0, "technical_failure")
        flag_count = len(flags)
        if technical_failure or severe or flag_count >= 2:
            priority = "HIGH"
        elif flag_count == 1:
            priority = "REVIEW"
        else:
            priority = "LOW"

        observed_weight = 0.0
        weighted_score = 0.0
        source = str(row.get("source", ""))
        for name, details in REVIEW_COMPONENTS.items():
            sources = details.get("sources", ("finetuning", "onharmony"))
            if source not in sources:
                component_values[name].append(float("nan"))
                continue
            weight = float(details["weight"])
            component = _review_component_score(row, details["metrics"])
            component_values[name].append(component)
            if math.isfinite(component):
                observed_weight += weight
                weighted_score += weight * component
        typicality = (
            weighted_score / observed_weight
            if observed_weight > 0.0
            else float("nan")
        )
        quality = (
            float(np.clip(10.5 - siqr, 0.0, 10.0))
            if math.isfinite(siqr)
            else float("nan")
        )
        coverage = 1.0 if math.isfinite(siqr) else 0.0
        if technical_failure:
            quality = 0.0

        priorities.append(priority)
        flag_texts.append(";".join(flags))
        flag_counts.append(flag_count)
        quality_scores.append(
            round(quality, 2) if math.isfinite(quality) else float("nan")
        )
        typicality_scores.append(
            round(typicality, 2) if math.isfinite(typicality) else float("nan")
        )
        coverages.append(coverage)

    result["auto_priority"] = priorities
    result["auto_flags"] = flag_texts
    result["auto_score"] = flag_counts
    result["auto_quality_score"] = quality_scores
    result["auto_quality_coverage"] = coverages
    result["auto_quality_source"] = "CAT12_SIQR"
    result["auto_typicality_score"] = typicality_scores
    for name, values in component_values.items():
        result[f"auto_review_{name}"] = [
            round(float(value), 2) if math.isfinite(value) else np.nan
            for value in values
        ]
    return result


def add_group_metrics(frame: pd.DataFrame, downsampled: np.ndarray) -> pd.DataFrame:
    """Add CAT-style homogeneity measures and robust within-group outlier flags."""

    result = frame.copy()
    result["gm_template_ncc"] = np.nan
    result["gm_quartic_mean_z"] = np.nan
    result["gm_com_distance_mm"] = np.nan
    result["within_subject_ncc"] = np.nan
    group_keys = result["source"].astype(str) + "::" + result["group"].astype(str)

    for group_key in sorted(group_keys.unique()):
        indices = np.flatnonzero(group_keys.eq(group_key).to_numpy() & result["gm_loaded"].to_numpy(dtype=bool))
        if len(indices) < 3:
            continue
        volumes = downsampled[indices].astype(np.float32)
        mean = np.mean(volumes, axis=0, dtype=np.float32)
        std = np.std(volumes, axis=0, dtype=np.float32)
        correlation_mask = mean > 0.02
        z_mask = correlation_mask & (std > 0.005)
        for local_index, frame_index in enumerate(indices):
            volume = volumes[local_index]
            if correlation_mask.sum() > 10:
                result.loc[frame_index, "gm_template_ncc"] = normalized_correlation(
                    volume[correlation_mask], mean[correlation_mask]
                )
            if z_mask.sum() > 10:
                z = (
                    (volume[z_mask] - mean[z_mask]) / std[z_mask]
                ).astype(np.float64)
                result.loc[frame_index, "gm_quartic_mean_z"] = float(
                    np.mean(np.square(np.square(np.clip(z, -50.0, 50.0)))) ** 0.25
                )
        com_columns = ["gm_com_x_mm", "gm_com_y_mm", "gm_com_z_mm"]
        centres = result.loc[indices, com_columns].to_numpy(dtype=np.float64)
        median_centre = np.median(centres, axis=0)
        result.loc[indices, "gm_com_distance_mm"] = np.linalg.norm(
            centres - median_centre, axis=1
        )

    onh = result.index[result["source"].eq("onharmony") & result["gm_loaded"]]
    for _, subject_indices in result.loc[onh].groupby("subject").groups.items():
        indices = np.asarray(list(subject_indices), dtype=int)
        if len(indices) < 2:
            continue
        volumes = downsampled[indices].astype(np.float32)
        volume_sum = np.sum(volumes, axis=0, dtype=np.float32)
        for local_index, frame_index in enumerate(indices):
            reference = (volume_sum - volumes[local_index]) / float(len(indices) - 1)
            mask = reference > 0.02
            if mask.sum() <= 10:
                continue
            result.loc[frame_index, "within_subject_ncc"] = normalized_correlation(
                volumes[local_index][mask], reference[mask]
            )

    for metric in (
        name for name in ROBUST_METRIC_DIRECTIONS if name != "within_subject_ncc"
    ):
        result[f"{metric}_robust_z"] = np.nan
        for group_key in sorted(group_keys.unique()):
            mask = group_keys.eq(group_key)
            values = pd.to_numeric(result.loc[mask, metric], errors="coerce").to_numpy()
            result.loc[mask, f"{metric}_robust_z"] = robust_z_scores(values)
    if result["within_subject_ncc"].notna().sum() >= 3:
        result["within_subject_ncc_robust_z"] = np.nan
        onh_mask = result["source"].eq("onharmony")
        result.loc[onh_mask, "within_subject_ncc_robust_z"] = robust_z_scores(
            pd.to_numeric(result.loc[onh_mask, "within_subject_ncc"], errors="coerce").to_numpy()
        )
    else:
        result["within_subject_ncc_robust_z"] = np.nan

    return add_automatic_assessment(result)


def _display_number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def _case_payload(row: pd.Series) -> dict[str, Any]:
    """Return only fields safe to embed in the label-blind browser report."""

    quality = _finite_float_or_nan(row.get("auto_quality_score", np.nan))
    return {
        "case_id": row["case_id"],
        "source": row["source"],
        "group": row["group"],
        "dataset": row["dataset"],
        "subject": row["subject"],
        "session": row["session"],
        "acquisition_id": row["acquisition_id"],
        "split": row["split"],
        "auto_priority": row["auto_priority"],
        "auto_score": int(row["auto_score"]),
        "auto_quality_score": quality if math.isfinite(quality) else None,
        "auto_quality_coverage": float(row["auto_quality_coverage"]),
        "auto_quality_source": row.get("auto_quality_source", "CAT12_SIQR"),
        "auto_flags": row["auto_flags"],
        "technical_status": row["technical_status"],
        "technical_errors": row["technical_errors"],
        "montage": row["montage"],
        "raw_sha256": row["raw_sha256"],
        "gm_sha256": row["gm_sha256"],
        "montage_sha256": row["montage_sha256"],
        "metrics": {
            "CAT12 SIQR": _display_number(row.get("cat12_siqr"), 2),
            "GM volume (ml)": _display_number(row.get("gm_volume_ml"), 1),
            "Group NCC": _display_number(row.get("gm_template_ncc"), 4),
            "Quartic mean z": _display_number(row.get("gm_quartic_mean_z"), 3),
            "COM distance (mm)": _display_number(row.get("gm_com_distance_mm"), 2),
            "Edge mass": _display_number(row.get("gm_edge_mass_fraction"), 4),
            "L/R asymmetry": _display_number(row.get("gm_lr_asymmetry"), 3),
            "Within-subject NCC": _display_number(row.get("within_subject_ncc"), 4),
            "Relative registration typicality": _display_number(row.get("auto_review_registration"), 1),
            "Relative morphology typicality": _display_number(row.get("auto_review_morphology"), 1),
            "Relative CAT12 typicality": _display_number(row.get("auto_review_cat12"), 1),
            "Relative repeatability typicality": _display_number(row.get("auto_review_repeatability"), 1),
        },
    }


def build_report_html(frame: pd.DataFrame, *, created_utc: str) -> str:
    """Build the interactive, dependency-free visual review report."""

    priority_order = {"HIGH": 0, "REVIEW": 1, "LOW": 2}
    ordered = frame.assign(
        _priority=frame["auto_priority"].map(priority_order),
    ).sort_values(
        ["_priority", "auto_quality_score", "auto_score", "source", "group", "case_id"],
        ascending=[True, True, False, True, True, True],
    )
    cases = [_case_payload(row) for _, row in ordered.iterrows()]
    report_id = hashlib.sha256(
        "\n".join(
            f"{case['case_id']}\t{case['montage_sha256']}" for case in cases
        ).encode("utf-8")
    ).hexdigest()[:16]
    counts = frame.groupby(["source", "group", "auto_priority"], dropna=False).size().reset_index(name="n")
    summary_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.source))}</td>"
        f"<td>{html.escape(str(row.group))}</td>"
        f"<td>{html.escape(str(row.auto_priority))}</td>"
        f"<td>{int(row.n)}</td>"
        "</tr>"
        for row in counts.itertuples(index=False)
    )
    score_summary = (
        frame.assign(
            auto_quality_score=pd.to_numeric(frame["auto_quality_score"], errors="coerce")
        )
        .groupby(["source", "group"], dropna=False)["auto_quality_score"]
        .agg(["count", "min", "median", "max"])
        .reset_index()
    )
    score_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.source))}</td>"
        f"<td>{html.escape(str(row.group))}</td>"
        f"<td>{int(row.count)}</td>"
        f"<td>{float(row.min):.2f}</td>"
        f"<td>{float(row.median):.2f}</td>"
        f"<td>{float(row.max):.2f}</td>"
        "</tr>"
        for row in score_summary.itertuples(index=False)
    )
    cases_json = json.dumps(cases, ensure_ascii=True, allow_nan=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAT12 visual QC</title>
<style>
:root{{--bg:#f5f6f7;--ink:#182026;--muted:#65727c;--line:#ccd3d8;--teal:#0b8f85;--orange:#d97706;--red:#b4232c;--panel:#fff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.35 Arial,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid var(--line);padding:12px 18px}}
h1{{font-size:21px;margin:0 0 4px}} .warning{{color:var(--red);font-weight:700}} .muted{{color:var(--muted)}}
.toolbar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}} input,select,button{{font:inherit}}
input,select{{border:1px solid #aeb8bf;background:white;padding:7px}} button{{border:1px solid #89959d;background:white;padding:7px 10px;cursor:pointer}}
button:hover{{background:#edf1f3}} .primary{{background:var(--teal);color:white;border-color:var(--teal)}}
main{{padding:16px;max-width:1540px;margin:auto}} details{{background:white;border:1px solid var(--line);padding:10px;margin-bottom:14px}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #e4e8eb}}
.case{{background:var(--panel);border:1px solid var(--line);margin:0 0 13px;padding:10px;scroll-margin-top:145px}}
.case.current{{outline:3px solid #3273a8}} .case-head{{display:flex;justify-content:space-between;gap:12px;align-items:start}}
.identity{{font-weight:700;font-size:17px}} .badge{{display:inline-block;padding:3px 7px;border-radius:3px;font-weight:700;font-size:12px}}
.HIGH{{background:#fde4e6;color:#8f1720}} .REVIEW{{background:#ffedcc;color:#8a4b00}} .LOW{{background:#dff3ef;color:#07675f}}
.quality{{display:inline-block;padding:4px 8px;border-radius:3px;font-weight:800;font-size:13px;margin-right:4px}}
.QGOOD{{background:#0b8f85;color:white}} .QCHECK{{background:#d97706;color:white}} .QLOW{{background:#b4232c;color:white}} .QNA{{background:#65727c;color:white}}
.montage{{width:100%;height:auto;margin-top:8px;background:#05080c;display:block}}
.metrics{{display:flex;gap:13px;flex-wrap:wrap;color:#34434d;font-size:13px;margin:7px 0}} .metric b{{color:#111}}
.review{{display:grid;grid-template-columns:auto auto minmax(180px,1fr);gap:8px;align-items:center}}
.decisions{{display:flex;gap:5px}} .decisions button.active[data-value=PASS]{{background:#167a56;color:white}}
.decisions button.active[data-value=REVIEW]{{background:#d97706;color:white}} .decisions button.active[data-value=FAIL]{{background:#b4232c;color:white}}
.note{{width:100%}} .flags{{color:var(--red);font-size:13px}} .hidden{{display:none!important}}
@media(max-width:800px){{.review{{grid-template-columns:1fr}} header{{position:static}} .case{{scroll-margin-top:10px}}}}
</style>
</head>
<body>
<header>
  <h1>CAT12 visual QC: AnatCL / y-Aware</h1>
  <div><span class="warning">SENSITIVE IMAGING DATA - DO NOT PUBLISH.</span> Created {html.escape(created_utc)}.</div>
  <div class="muted">The 0-10 score is the official CAT12 SIQR grade inverted so higher is better. Separate group-relative flags prioritise inspection. Human PASS/REVIEW/FAIL is the QC decision. No age, prediction, BAG, diagnosis or outcome is loaded into this report.</div>
  <div class="toolbar">
    <input id="search" placeholder="Search subject/session/group">
    <select id="priority"><option value="">All priorities</option><option>HIGH</option><option>REVIEW</option><option>LOW</option></select>
    <select id="quality"><option value="">All scores</option><option value="7.5">Score ≤ 7.5</option><option value="8">Score ≤ 8</option><option value="8.5">Score ≤ 8.5</option></select>
    <select id="decision"><option value="">All decisions</option><option value="UNREVIEWED">UNREVIEWED</option><option>PASS</option><option>REVIEW</option><option>FAIL</option></select>
    <input id="rater" placeholder="Rater">
    <button id="export" class="primary">Export qc_review.tsv</button>
    <span id="progress"></span>
  </div>
</header>
<main>
<details>
<summary>Automated triage summary and interpretation</summary>
<p>The turquoise overlay is the subject's modulated normalized GM on the fixed IXI555 Dartel template. Native T1w and CAT12 GM are shown at 30%, 50% and 70% in sagittal, coronal and axial orientations.</p>
<p><b>Inspect:</b> coverage and native artifacts; gross orientation; skull stripping/tissue assignment; registration to template; missing cortex, cerebellum or brain stem; extracranial GM. CAT12 SIQR, group NCC, CAT-style quartic mean voxelwise z, GM volume, centre-of-mass displacement, edge mass, asymmetry and leave-one-out within-participant NCC only order suspicious cases. Anatomy and scanner effects can also produce outliers.</p>
<p><b>Automatic CAT12 quality score (0-10):</b> the official SIQR numerical grade runs from 0.5 (best) to 10.5 (worst). The displayed score is <code>clip(10.5 - SIQR, 0, 10)</code>, so higher is better. It is a re-expression of CAT12 SIQR, not a new classifier or an automatic exclusion threshold.</p>
<p><b>Review priority and flags are separate:</b> group-relative NCC, quartic z, morphology, CAT12 ratings and ON-Harmony repeatability identify atypical cases. A HIGH priority can reflect anatomy, age, pathology or scanner differences even when preprocessing is visually acceptable. It means inspect first, not reject.</p>
<table><thead><tr><th>Source</th><th>Group</th><th>N</th><th>Minimum</th><th>Median</th><th>Maximum</th></tr></thead><tbody>{score_rows}</tbody></table>
<table><thead><tr><th>Source</th><th>Group</th><th>Priority</th><th>N</th></tr></thead><tbody>{summary_rows}</tbody></table>
</details>
<div id="cases"></div>
</main>
<script>
const CASES={cases_json};
const REASONS=['','raw_artifact','incomplete_coverage','orientation','segmentation','registration','extracranial_tissue','other'];
const KEY='cat12-qc-review-v1-{report_id}';
let state=JSON.parse(localStorage.getItem(KEY)||'{{}}');
let current=0;
const root=document.getElementById('cases');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function saved(c){{return state[c.case_id]||{{decision:'UNREVIEWED',reasons:'',notes:'',rater:'',reviewed_at_utc:''}}}}
function card(c,i){{const s=saved(c),hasq=Number.isFinite(c.auto_quality_score),qclass=!hasq?'QNA':(c.auto_quality_score<5.5?'QLOW':(c.auto_quality_score<7.5?'QCHECK':'QGOOD')),qtext=hasq?`CAT12 QUALITY ${{c.auto_quality_score.toFixed(1)}}/10`:'CAT12 QUALITY NA';const metrics=Object.entries(c.metrics).map(([k,v])=>`<span class="metric"><b>${{esc(k)}}:</b> ${{esc(v)}}</span>`).join('');
return `<article class="case" id="case-${{i}}" data-index="${{i}}" data-priority="${{c.auto_priority}}" data-search="${{esc((c.case_id+' '+c.group+' '+c.subject+' '+c.session).toLowerCase())}}">
<div class="case-head"><div><div class="identity">${{i+1}}/${{CASES.length}} · ${{esc(c.acquisition_id)}}</div><div class="muted">${{esc(c.source)}} · ${{esc(c.group)}} · ${{esc(c.split)}}</div></div><div><span class="quality ${{qclass}}">${{qtext}}</span><span class="badge ${{c.auto_priority}}">REVIEW ${{c.auto_priority}}</span> <span class="badge ${{c.technical_status==='PASS'?'LOW':'HIGH'}}">TECH ${{c.technical_status}}</span></div></div>
${{c.auto_flags?`<div class="flags">Flags: ${{esc(c.auto_flags)}}</div>`:''}}${{c.technical_errors?`<div class="flags">${{esc(c.technical_errors)}}</div>`:''}}
<img class="montage" loading="lazy" src="${{esc(c.montage)}}" alt="QC montage for ${{esc(c.acquisition_id)}}">
<div class="metrics">${{metrics}}</div>
<div class="review"><div class="decisions">${{['PASS','REVIEW','FAIL'].map(v=>`<button data-value="${{v}}" class="${{s.decision===v?'active':''}}">${{v}} <u>${{v[0]}}</u></button>`).join('')}} <button data-value="UNREVIEWED">Clear</button></div>
<select class="reason">${{REASONS.map(v=>`<option ${{s.reasons===v?'selected':''}}>${{v}}</option>`).join('')}}</select><input class="note" value="${{esc(s.notes)}}" placeholder="Optional note"></div></article>`}}
function render(){{root.innerHTML=CASES.map(card).join('');bind();filter();setCurrent(Math.min(current,CASES.length-1));}}
function persist(){{localStorage.setItem(KEY,JSON.stringify(state));updateProgress()}}
function setDecision(i,value){{const c=CASES[i],old=saved(c);state[c.case_id]={{...old,decision:value,rater:document.getElementById('rater').value||old.rater,reviewed_at_utc:value==='UNREVIEWED'?'':new Date().toISOString()}};persist();const el=document.getElementById('case-'+i);if(el)el.querySelectorAll('.decisions button').forEach(b=>b.classList.toggle('active',b.dataset.value===value&&value!=='UNREVIEWED'));filter();setCurrent(i)}}
function bind(){{document.querySelectorAll('.case').forEach(el=>{{const i=+el.dataset.index;el.querySelectorAll('.decisions button').forEach(b=>b.onclick=()=>setDecision(i,b.dataset.value));el.querySelector('.reason').onchange=e=>{{state[CASES[i].case_id]={{...saved(CASES[i]),reasons:e.target.value}};persist()}};el.querySelector('.note').onchange=e=>{{state[CASES[i].case_id]={{...saved(CASES[i]),notes:e.target.value}};persist()}}}})}}
function visible(){{return [...document.querySelectorAll('.case:not(.hidden)')]}}
function setCurrent(i){{document.querySelectorAll('.case').forEach(x=>x.classList.remove('current'));const el=document.getElementById('case-'+i);if(el&&!el.classList.contains('hidden')){{current=i;el.classList.add('current')}}}}
function move(delta){{const list=visible(),pos=list.findIndex(x=>+x.dataset.index===current),next=list[Math.max(0,Math.min(list.length-1,(pos<0?0:pos)+delta))];if(next){{setCurrent(+next.dataset.index);next.scrollIntoView({{block:'center'}})}}}}
function filter(){{const q=document.getElementById('search').value.toLowerCase(),p=document.getElementById('priority').value,qualityFilter=document.getElementById('quality').value,maxq=Number(qualityFilter),d=document.getElementById('decision').value;document.querySelectorAll('.case').forEach(el=>{{const c=CASES[+el.dataset.index],s=saved(c),qualityMismatch=qualityFilter&&(!Number.isFinite(c.auto_quality_score)||c.auto_quality_score>maxq);el.classList.toggle('hidden',!!((q&&!el.dataset.search.includes(q))||(p&&c.auto_priority!==p)||qualityMismatch||(d&&s.decision!==d)))}});updateProgress()}}
function updateProgress(){{const reviewed=CASES.filter(c=>saved(c).decision!=='UNREVIEWED').length;document.getElementById('progress').textContent=`${{reviewed}} / ${{CASES.length}} reviewed`}}
function tsv(value){{return String(value??'').replace(/[\\t\\r\\n]+/g,' ')}}
function exportReview(){{const rater=document.getElementById('rater').value.trim();if(!rater){{alert('Enter the rater name before export.');return}}const cols={json.dumps(list(REVIEW_COLUMNS))};const lines=[cols.join('\\t')];CASES.forEach(c=>{{const s=saved(c);lines.push([c.case_id,c.source,c.dataset,c.subject,c.session,c.acquisition_id,s.decision,s.reasons,s.notes,s.rater||rater,s.reviewed_at_utc,c.raw_sha256,c.gm_sha256,c.montage_sha256].map(tsv).join('\\t'))}});const blob=new Blob([lines.join('\\n')+'\\n'],{{type:'text/tab-separated-values'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='qc_review.tsv';a.click();URL.revokeObjectURL(a.href)}}
['search','priority','quality','decision'].forEach(id=>document.getElementById(id).oninput=filter);document.getElementById('export').onclick=exportReview;document.getElementById('rater').value=localStorage.getItem(KEY+'-rater')||'';document.getElementById('rater').onchange=e=>localStorage.setItem(KEY+'-rater',e.target.value);
document.addEventListener('keydown',e=>{{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='j'||e.key==='ArrowDown')move(1);if(e.key==='k'||e.key==='ArrowUp')move(-1);if(e.key==='p')setDecision(current,'PASS');if(e.key==='r')setDecision(current,'REVIEW');if(e.key==='f')setDecision(current,'FAIL');if(e.key==='u')setDecision(current,'UNREVIEWED')}});
render();
</script>
</body></html>"""


def initial_review_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the blank, hash-bound human review registry."""

    review = frame[
        [
            "case_id", "source", "dataset", "subject", "session", "acquisition_id",
            "raw_sha256", "gm_sha256", "montage_sha256",
        ]
    ].copy()
    review["decision"] = "UNREVIEWED"
    review["reasons"] = ""
    review["notes"] = ""
    review["rater"] = ""
    review["reviewed_at_utc"] = ""
    return review[list(REVIEW_COLUMNS)]


def automatic_quality_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-safe descriptive statistics for the 0-10 score."""

    all_values = pd.to_numeric(
        frame["auto_quality_score"], errors="coerce"
    ).to_numpy()
    values = all_values[np.isfinite(all_values)]
    if len(values) == 0:
        raise ValueError("Automatic quality scoring produced no finite values")
    return {
        "n": int(len(values)),
        "n_missing": int(len(all_values) - len(values)),
        "min": float(np.min(values)),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "bands": {
            "0_to_lt_5_5": int(np.sum(values < 5.5)),
            "5_5_to_lt_7_5": int(
                np.sum((values >= 5.5) & (values < 7.5))
            ),
            "7_5_to_lt_8_5": int(
                np.sum((values >= 7.5) & (values < 8.5))
            ),
            "8_5_to_10": int(np.sum(values >= 8.5)),
        },
    }


def load_template(path: Path) -> np.ndarray:
    """Load the fixed IXI555 Dartel template used for overlay context."""

    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.shape != EXPECTED_GM_SHAPE:
        raise ValueError(
            f"Template shape {data.shape}, expected {EXPECTED_GM_SHAPE}"
        )
    if not np.isfinite(data).all():
        raise ValueError("Template contains non-finite voxels")
    return data


def ensure_new_output_dir(path: Path) -> None:
    """Reject accidental mixing with a previous visual-QC run."""

    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def command_build(args: argparse.Namespace) -> None:
    """Generate metrics, montages, review template and interactive HTML."""

    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ensure_new_output_dir(output_dir)
    template_path = (
        args.template or project_root / "templates/Template_1_IXI555_MNI152.nii"
    ).expanduser().resolve()
    template = load_template(template_path)
    cases: list[Case] = []
    registries: dict[str, str] = {}
    if "finetuning" in args.sources:
        manifest = (
            args.ft_manifest
            or project_root
            / "preprocessings/FineTunningDataset/brainage_splits/brainage_definitive_subject_manifest.tsv"
        ).expanduser().resolve()
        preproc_root = (
            args.ft_preproc_root
            or project_root / "preprocessings/FineTunningDataset/preproc_AnatCL"
        ).expanduser().resolve()
        cases.extend(
            load_ft_cases(
                manifest, preproc_root,
                expected_cases=None if args.allow_noncanonical_count else EXPECTED_FT_CASES,
            )
        )
        registries["fine_tuning_manifest"] = str(manifest)
    if "onharmony" in args.sources:
        raw_filelist = (
            args.onh_raw_filelist
            or project_root / "DATA/filelists/filelist_BrainIAC_raw.txt"
        ).expanduser().resolve()
        gm_filelist = (
            args.onh_gm_filelist
            or project_root / "DATA/filelists/filelist_AnatCL.txt"
        ).expanduser().resolve()
        provenance = (
            args.onh_provenance
            or project_root
            / "ExcelsChecks/paper_actual/audits/AnatCL/anatcl_preprocessing_case_provenance.tsv"
        ).expanduser().resolve()
        cases.extend(
            load_onh_cases(
                raw_filelist, gm_filelist, provenance,
                expected_cases=None if args.allow_noncanonical_count else EXPECTED_ONH_CASES,
            )
        )
        registries["onh_raw_filelist"] = str(raw_filelist)
        registries["onh_gm_filelist"] = str(gm_filelist)
        if provenance is not None:
            registries["onh_provenance"] = str(provenance)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs across selected cohorts")
    cases = select_cases(cases, args.limit_per_source)
    if not cases:
        raise ValueError("No cases selected")

    started = utc_now()
    atomic_json(
        output_dir / "RUNNING.json",
        {
            "status": "RUNNING",
            "started_utc": started,
            "n_cases": len(cases),
            "sources": args.sources,
            "label_blind": True,
            "sensitive_imaging_data": True,
            "script_sha256": sha256_file(SCRIPT_PATH),
        },
    )
    montage_dir = output_dir / "montages"
    downsample_shape = tuple(
        (value + DOWNSAMPLE_STEP - 1) // DOWNSAMPLE_STEP for value in EXPECTED_GM_SHAPE
    )
    downsampled = np.full((len(cases), *downsample_shape), np.nan, dtype=np.float16)
    records: list[dict[str, Any]] = []

    def worker(case: Case) -> tuple[dict[str, Any], np.ndarray]:
        return inspect_case(
            case, template, montage_dir, verify_hashes=args.verify_hashes
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (record, volume) in enumerate(executor.map(worker, cases), start=1):
            records.append(record)
            downsampled[index - 1] = volume
            if index == 1 or index % 25 == 0 or index == len(cases):
                print(f"[{index}/{len(cases)}] {record['case_id']} -> {record['technical_status']}", flush=True)

    frame = add_group_metrics(pd.DataFrame(records), downsampled)
    metrics_path = output_dir / "qc_metrics.tsv"
    review_path = output_dir / "qc_review.tsv"
    html_path = output_dir / "index.html"
    atomic_tsv(metrics_path, frame)
    atomic_tsv(review_path, initial_review_frame(frame))
    completed = utc_now()
    atomic_text(html_path, build_report_html(frame, created_utc=completed))
    registry_hashes = {
        key: sha256_file(Path(value)) for key, value in registries.items() if Path(value).is_file()
    }
    counts = {str(key): int(value) for key, value in frame["auto_priority"].value_counts().items()}
    summary = {
        "status": "COMPLETE",
        "started_utc": started,
        "completed_utc": completed,
        "n_cases": len(frame),
        "sources": sorted(frame["source"].unique().tolist()),
        "auto_priority_counts": counts,
        "auto_quality_score_summary": automatic_quality_summary(frame),
        "quality_score_contract": quality_score_contract(),
        "technical_failure_count": int(frame["technical_status"].eq("FAIL").sum()),
        "cat12_xml_status_counts": {
            str(key): int(value)
            for key, value in frame["cat12_xml_status"].value_counts().items()
        },
        "label_blind": True,
        "manifest_columns_loaded": {
            "finetuning": [
                "dataset", "subject", "raw_t1w_path", "split",
                "inclusion_status", "required_offline_groups",
            ],
            "onharmony": ["raw_path", "processed_path", "acquisition_id"],
        },
        "outcomes_not_loaded": ["age", "sex", "diagnosis", "age_pred", "BAG", "error"],
        "automated_metrics_are_exclusion_decisions": False,
        "human_visual_review_required": True,
        "sensitive_imaging_data": True,
        "metric_contract": {
            "gm_threshold": GM_THRESHOLD,
            "downsample_step": DOWNSAMPLE_STEP,
            "robust_outlier_z": ROBUST_OUTLIER_Z,
            "robust_severe_z": ROBUST_SEVERE_Z,
            "grouping": "fine-tuning by dataset; ON-Harmony by scanner",
            "quartic_mean_z": "CAT-style approximation on 3-mm downsampled unsmoothed GM",
            "cat12_siqr": "official CAT12 XML rating when retained; lower is better",
            "within_subject_ncc": "leave-one-acquisition-out reference for ON-Harmony",
        },
        "registries": registries,
        "registry_sha256": registry_hashes,
        "template": str(template_path),
        "template_sha256": sha256_file(template_path),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "outputs": {
            "metrics": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "blank_review": str(review_path),
            "blank_review_sha256": sha256_file(review_path),
            "html": str(html_path),
            "html_sha256": sha256_file(html_path),
        },
    }
    atomic_json(output_dir / "qc_summary.json", summary)
    (output_dir / "RUNNING.json").unlink()
    print(f"Report: {html_path}")
    print(f"Metrics: {metrics_path}")
    print("Automated triage is complete; human visual decisions remain pending.")


def command_rescore(args: argparse.Namespace) -> None:
    """Add the current 0-10 score to one completed, unfinalized build."""

    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = report_dir / "qc_metrics.tsv"
    review_path = report_dir / "qc_review.tsv"
    html_path = report_dir / "index.html"
    summary_path = report_dir / "qc_summary.json"
    required = (metrics_path, review_path, html_path, summary_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete QC build; missing: {missing}")
    if (report_dir / "RUNNING.json").exists():
        raise RuntimeError("Cannot rescore while the build is still running")
    if (report_dir / "qc_final_summary.json").exists() or (report_dir / "qc_final_report.html").exists():
        raise FileExistsError("Cannot rescore a finalized human review")

    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "COMPLETE":
        raise ValueError(f"QC build is not COMPLETE: {summary.get('status')!r}")
    recorded_metrics_sha = summary.get("outputs", {}).get("metrics_sha256")
    if recorded_metrics_sha and sha256_file(metrics_path) != recorded_metrics_sha:
        raise ValueError("qc_metrics.tsv differs from qc_summary.json")
    recorded_review_sha = summary.get("outputs", {}).get("blank_review_sha256")
    if recorded_review_sha and sha256_file(review_path) != recorded_review_sha:
        raise ValueError("qc_review.tsv differs from the blank review recorded by the build")

    frame = pd.read_csv(
        metrics_path,
        sep="\t",
        low_memory=False,
        keep_default_na=False,
    )
    expected_z = {
        f"{metric}_robust_z" for metric in ROBUST_METRIC_DIRECTIONS
    }
    missing_columns = expected_z - set(frame.columns)
    if missing_columns:
        raise ValueError(f"qc_metrics.tsv lacks score inputs: {sorted(missing_columns)}")
    rescored = add_automatic_assessment(frame)
    rescored_utc = utc_now()
    atomic_tsv(metrics_path, rescored)
    atomic_text(html_path, build_report_html(rescored, created_utc=rescored_utc))

    summary["rescored_utc"] = rescored_utc
    summary["script"] = str(SCRIPT_PATH)
    summary["script_sha256"] = sha256_file(SCRIPT_PATH)
    summary["auto_priority_counts"] = {
        str(key): int(value)
        for key, value in rescored["auto_priority"].value_counts().items()
    }
    summary["auto_quality_score_summary"] = automatic_quality_summary(rescored)
    summary["quality_score_contract"] = quality_score_contract()
    summary["outputs"]["metrics_sha256"] = sha256_file(metrics_path)
    summary["outputs"]["html_sha256"] = sha256_file(html_path)
    summary["outputs"]["blank_review_sha256"] = sha256_file(review_path)
    atomic_json(summary_path, summary)
    print(f"Rescored metrics: {metrics_path}")
    print(f"Updated report: {html_path}")
    print("No NIfTI or montage was reopened; human decisions remain pending.")


def _validate_review(metrics: pd.DataFrame, review: pd.DataFrame, allow_incomplete: bool) -> pd.DataFrame:
    missing = set(REVIEW_COLUMNS) - set(review.columns)
    if missing:
        raise ValueError(f"Review TSV lacks columns: {sorted(missing)}")
    if review["case_id"].duplicated().any():
        raise ValueError("Review TSV contains duplicate case IDs")
    if set(review["case_id"]) != set(metrics["case_id"]):
        raise ValueError("Review TSV case IDs do not exactly match qc_metrics.tsv")
    joined = metrics.merge(review[list(REVIEW_COLUMNS)], on="case_id", suffixes=("", "_review"), validate="one_to_one")
    for column in ("source", "dataset", "subject", "session", "acquisition_id", "raw_sha256", "gm_sha256", "montage_sha256"):
        if not joined[column].fillna("").astype(str).equals(joined[f"{column}_review"].fillna("").astype(str)):
            raise ValueError(f"Review identity drift for {column}")
    decisions = joined["decision"].fillna("").astype(str).str.upper()
    allowed = set(DECISIONS) | ({"UNREVIEWED"} if allow_incomplete else set())
    invalid = sorted(set(decisions) - allowed)
    if invalid:
        raise ValueError(f"Invalid or incomplete review decisions: {invalid}")
    if not allow_incomplete and decisions.eq("UNREVIEWED").any():
        raise ValueError("Review is incomplete; UNREVIEWED cases remain")
    joined["decision"] = decisions
    decided = joined["decision"].isin(DECISIONS)
    if (joined.loc[decided, "rater"].fillna("").str.strip() == "").any():
        raise ValueError("Every decided case must identify a rater")
    if (joined.loc[decided, "reviewed_at_utc"].fillna("").str.strip() == "").any():
        raise ValueError("Every decided case must have reviewed_at_utc")
    return joined


def final_report_html(joined: pd.DataFrame, summary: dict[str, Any]) -> str:
    """Build a compact frozen report after all human decisions are imported."""

    counts = joined.groupby(["source", "group", "decision"]).size().reset_index(name="n")
    rows = "".join(
        f"<tr><td>{html.escape(str(row.source))}</td><td>{html.escape(str(row.group))}</td>"
        f"<td>{html.escape(str(row.decision))}</td><td>{int(row.n)}</td></tr>"
        for row in counts.itertuples(index=False)
    )
    exceptions = joined.loc[~joined["decision"].eq("PASS")]
    exception_rows = "".join(
        f"<tr><td>{html.escape(str(row.case_id))}</td><td>{html.escape(str(row.decision))}</td>"
        f"<td>{html.escape(str(row.reasons))}</td><td>{html.escape(str(row.notes))}</td>"
        f"<td>{html.escape(str(row.auto_priority))}</td></tr>"
        for row in exceptions.itertuples(index=False)
    ) or "<tr><td colspan='5'>None</td></tr>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>CAT12 final QC</title>
<style>body{{font:15px/1.45 Arial,sans-serif;max-width:1100px;margin:35px auto;color:#182026}}h1,h2{{margin-bottom:6px}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{padding:7px;border-bottom:1px solid #ccd3d8;text-align:left}}.warning{{color:#b4232c;font-weight:bold}}code{{font-size:12px}}</style></head><body>
<h1>CAT12 visual QC - frozen human review</h1><p class="warning">SENSITIVE STUDY RECORD - NOT FOR PUBLICATION.</p>
<p>Completed {html.escape(str(summary['completed_utc']))}. Automated metrics were used only to prioritise visual inspection. Final decisions are human ratings.</p>
<h2>Decision counts</h2><table><thead><tr><th>Source</th><th>Group</th><th>Decision</th><th>N</th></tr></thead><tbody>{rows}</tbody></table>
<h2>REVIEW / FAIL cases</h2><table><thead><tr><th>Case</th><th>Decision</th><th>Reason</th><th>Note</th><th>Auto priority</th></tr></thead><tbody>{exception_rows}</tbody></table>
<h2>Audit binding</h2><pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=True))}</pre>
</body></html>"""


def command_finalize(args: argparse.Namespace) -> None:
    """Validate an exported review TSV and freeze the human QC summary."""

    report_dir = args.report_dir.expanduser().resolve()
    metrics_path = report_dir / "qc_metrics.tsv"
    review_path = args.review_tsv.expanduser().resolve()
    if not metrics_path.is_file() or not review_path.is_file():
        raise FileNotFoundError(f"Expected {metrics_path} and {review_path}")
    output_json = report_dir / "qc_final_summary.json"
    output_html = report_dir / "qc_final_report.html"
    if output_json.exists() or output_html.exists():
        raise FileExistsError("Final QC report already exists")
    metrics = pd.read_csv(metrics_path, sep="\t", dtype=str, keep_default_na=False)
    review = pd.read_csv(review_path, sep="\t", dtype=str, keep_default_na=False)
    joined = _validate_review(metrics, review, args.allow_incomplete)
    completed = utc_now()
    summary = {
        "status": "COMPLETE" if not joined["decision"].eq("UNREVIEWED").any() else "INCOMPLETE",
        "completed_utc": completed,
        "n_cases": len(joined),
        "decision_counts": {
            str(key): int(value) for key, value in joined["decision"].value_counts().items()
        },
        "n_review_or_fail": int(joined["decision"].isin(["REVIEW", "FAIL"]).sum()),
        "raters": sorted(value for value in joined["rater"].unique() if value),
        "label_blind": True,
        "automated_metrics_are_exclusion_decisions": False,
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "review": str(review_path),
        "review_sha256": sha256_file(review_path),
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    atomic_json(output_json, summary)
    atomic_text(output_html, final_report_html(joined, summary))
    print(f"Final summary: {output_json}")
    print(f"Final report: {output_html}")


def parse_args() -> argparse.Namespace:
    """Parse build/rescore/finalize commands with explicit paths."""

    parser = argparse.ArgumentParser(
        description="Generate and freeze label-blind CAT12 visual QC reports"
    )
    commands = parser.add_subparsers(required=True)
    build = commands.add_parser("build", help="Generate montages and interactive report")
    build.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    build.add_argument(
        "--sources", nargs="+", choices=("finetuning", "onharmony"),
        default=["finetuning", "onharmony"],
    )
    build.add_argument(
        "--ft-manifest", type=Path,
    )
    build.add_argument(
        "--ft-preproc-root", type=Path,
    )
    build.add_argument(
        "--onh-raw-filelist", type=Path,
    )
    build.add_argument(
        "--onh-gm-filelist", type=Path,
    )
    build.add_argument(
        "--onh-provenance", type=Path,
    )
    build.add_argument(
        "--template", type=Path,
    )
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--workers", type=int, default=4)
    build.add_argument(
        "--verify-hashes", action="store_true",
        help="Recompute raw and GM SHA-256 (slower; existing provenance is used otherwise)",
    )
    build.add_argument(
        "--limit-per-source", type=int,
        help="Deterministic pilot with at most N cases from each selected source",
    )
    build.add_argument(
        "--allow-noncanonical-count", action="store_true",
        help="Permit a reduced/synthetic registry; never use for the definitive run",
    )
    build.set_defaults(func=command_build)

    rescore = commands.add_parser(
        "rescore",
        help="Add/refresh the 0-10 automatic score without reopening images",
    )
    rescore.add_argument("--report-dir", type=Path, required=True)
    rescore.set_defaults(func=command_rescore)

    finalize = commands.add_parser("finalize", help="Freeze an exported human review")
    finalize.add_argument("--report-dir", type=Path, required=True)
    finalize.add_argument("--review-tsv", type=Path, required=True)
    finalize.add_argument("--allow-incomplete", action="store_true")
    finalize.set_defaults(func=command_finalize)
    args = parser.parse_args()
    if getattr(args, "workers", 1) <= 0:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
