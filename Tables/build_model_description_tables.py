#!/usr/bin/env python3
"""Build the two standalone model-description tables (CSV and LaTeX)."""

from __future__ import annotations

import csv
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent

PRETRAINING_ROWS = [
    {
        "model_reference": "BrainIAC (Tak et al., 2026)",
        "architecture": "3D ViT-B/16",
        "pretraining_data": "32,015 MRI scans",
        "modalities": "T1w, T2w, FLAIR, T1CE",
        "strategy": "SimCLR contrastive SSL",
        "dimension": "768",
    },
    {
        "model_reference": "3D-Neuro-SimCLR (Kaczmarek et al., 2025)",
        "architecture": "3D ResNet-18",
        "pretraining_data": "44,958 scans; 18,759 patients",
        "modalities": "T1w",
        "strategy": "SimCLR contrastive SSL",
        "dimension": "512",
    },
    {
        "model_reference": "AnatCL (Barbano et al., 2026)",
        "architecture": "3D ResNet-18",
        "pretraining_data": "3,984 scans",
        "modalities": "T1w-derived GM maps",
        "strategy": "Anatomy-aware and age-guided contrastive learning",
        "dimension": "512",
    },
    {
        "model_reference": "y-Aware (Dufumier et al., 2021)",
        "architecture": "3D DenseNet-121",
        "pretraining_data": "10,420 scans; 7,764 participants",
        "modalities": "T1w-derived GM maps",
        "strategy": "Age-guided y-Aware InfoNCE",
        "dimension": "1,024",
    },
    {
        "model_reference": "BrainSegFounder (Cox et al., 2024)",
        "architecture": "3D SwinUNETR",
        "pretraining_data": "82,800 volumes; 41,400 participants",
        "modalities": "T1w and T2-FLAIR",
        "strategy": "Reconstruction, rotation prediction and contrastive SSL",
        "dimension": "768",
    },
    {
        "model_reference": "Curia-2 (Saporta et al., 2026)",
        "architecture": "2D ViT-B/16",
        "pretraining_data": ">200 million 2D images",
        "modalities": "CT and MRI",
        "strategy": "Large-scale slice-based SSL",
        "dimension": "768",
    },
    {
        "model_reference": "NeuroVFM (Kondepudi et al., 2026)",
        "architecture": "3D ViT-B",
        "pretraining_data": "5.24 million volumes; 566,915 studies",
        "modalities": "CT and MRI",
        "strategy": "Vol-JEPA latent predictive SSL",
        "dimension": "768",
    },
]

IMPLEMENTATION_ROWS = [
    {
        "model": "AnatCL",
        "evaluated_input": "CAT12/Dartel modulated GM; center crop 121x128x121",
        "checkpoint": "Five corrected anatcl-g3 CV folds",
        "canonical_readout": "Backbone global average pooling; fold0 primary, other folds retained for sensitivity",
        "sensitivity": "Input normalization and registration checks",
        "match": "Matches corrected release",
    },
    {
        "model": "y-Aware",
        "evaluated_input": "CAT12/Dartel GM, 121x145x121; per-volume z-score",
        "checkpoint": "Released 3D DenseNet checkpoint",
        "canonical_readout": "Backbone global average pooling",
        "sensitivity": "Preprocessing logs verified",
        "match": "Matches paper and release",
    },
    {
        "model": "3D-Neuro-SimCLR",
        "evaluated_input": "TurboPrep; 150x192x192 crop; masked z-score",
        "checkpoint": "Released 3D ResNet-18 checkpoint",
        "canonical_readout": "Backbone global average pooling",
        "sensitivity": "Negative-intensity safeguard audited",
        "match": "Matches paper and repository",
    },
    {
        "model": "BrainIAC",
        "evaluated_input": "N4, rigid MNI registration, HD-BET, 96^3, z-score",
        "checkpoint": "Released ViT-B checkpoint",
        "canonical_readout": "Token at position 0 (first patch; classification=False)",
        "sensitivity": "Mean pool, last patch and max pool",
        "match": "Matches released code; paper does not fully specify readout",
    },
    {
        "model": "BrainSegFounder",
        "evaluated_input": "BET, FLIRT/FNIRT, MNI mask; crop, z-score, resize 96^3",
        "checkpoint": "Released Stage-1 SwinUNETR checkpoint",
        "canonical_readout": "Global mean of deepest 3x3x3 bottleneck",
        "sensitivity": "Six intensity/spatial/readout configurations",
        "match": "Registration matches paper; embedding readout is deployment-defined",
    },
    {
        "model": "Curia-2",
        "evaluated_input": "Raw T1w, PL orientation; float tensor; slices with >=5% content",
        "checkpoint": "Public Curia-2 Base checkpoint",
        "canonical_readout": "CLS token averaged over all retained slices (cls_all)",
        "sensitivity": "patch_all, cls_core and patch_core",
        "match": "Volume averaging matches paper; slice filter/readout documented",
    },
    {
        "model": "NeuroVFM",
        "evaluated_input": "Official StudyPreprocessor; RPI, 1x1x4 mm, percentile scaling",
        "checkpoint": "Public NeuroVFM ViT-B checkpoint",
        "canonical_readout": "Mean over all variable-length tokens",
        "sensitivity": "Checkpoint/cosine smoke test; token counts recorded",
        "match": "Released inference matched; task-agnostic pooling is study-defined",
    },
]


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        ">=": r"$\geq$",
        ">": r"$>$",
    }
    text = str(value)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(
    path: Path,
    rows: list[dict[str, str]],
    headers: list[str],
    keys: list[str],
    column_spec: str,
) -> None:
    with path.open("w") as handle:
        handle.write("% Standalone generated table; not yet inserted into the manuscript.\n")
        handle.write("\\noindent\\resizebox{\\textwidth}{!}{%\n")
        handle.write(f"\\begin{{tabular}}{{{column_spec}}}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(headers) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(
                " & ".join(latex_escape(row[key]) for key in keys) + " \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}%\n")
        handle.write("}\n")


def main() -> None:
    if len(PRETRAINING_ROWS) != 7 or len(IMPLEMENTATION_ROWS) != 7:
        raise RuntimeError("Both model tables must contain exactly seven FMs")
    if {row["model_reference"].split(" (")[0] for row in PRETRAINING_ROWS} != {
        row["model"] for row in IMPLEMENTATION_ROWS
    }:
        raise RuntimeError("Model rosters differ between description tables")

    write_csv(OUTPUT_DIR / "table_models_pretraining.csv", PRETRAINING_ROWS)
    write_latex(
        OUTPUT_DIR / "table_models_pretraining.tex",
        PRETRAINING_ROWS,
        ["Model and reference", "Architecture", "Pretraining data", "Modalities",
         "Pretraining strategy", "Dim."],
        ["model_reference", "architecture", "pretraining_data", "modalities",
         "strategy", "dimension"],
        "lllllr",
    )
    write_csv(OUTPUT_DIR / "table_models_implementation.csv", IMPLEMENTATION_ROWS)
    write_latex(
        OUTPUT_DIR / "table_models_implementation.tex",
        IMPLEMENTATION_ROWS,
        ["Model", "Evaluated input", "Checkpoint", "Canonical readout",
         "Sensitivity analyses", "Agreement"],
        ["model", "evaluated_input", "checkpoint", "canonical_readout",
         "sensitivity", "match"],
        "llllll",
    )
    print("Generated two seven-model description tables (CSV and LaTeX).")


if __name__ == "__main__":
    main()
