#!/usr/bin/env python3
"""
Reliability analysis for FreeSurfer morphometric baseline (DKT atlas).

Reads pre-extracted IDPs from the ON-Harmony all_idps.csv.
Uses FreeSurfer DKT atlas features: cortical thickness (ThickAvg),
surface area (SurfArea), gray matter volume (GrayVol) for 31 DKT regions
× 2 hemispheres, plus 17 subcortical volumes from aseg.

Input:  all_idps.csv from ON-Harmony dataset
Output: ../Results/results_FreeSurfer.json

Usage:
  python analyse_FreeSurfer.py [--input /path/to/all_idps.csv]
                               [--output_dir ../Results]
                               [--n_boot 1000]
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils_reliability import run_full_analysis


def load_freesurfer_idps(csv_path):
    """
    Load ON-Harmony all_idps.csv and extract FreeSurfer DKT + aseg columns.
    Maps metadata columns to the standard names used by the analysis pipeline.

    Returns
    -------
    df : DataFrame with columns [subject_id, scanner, session_id, feat_0..feat_N]
    feat_cols : list of feature column names
    """
    raw = pd.read_csv(csv_path)

    # Map metadata
    df = pd.DataFrame()
    df["subject_id"] = raw["subject"].astype(str)
    df["scanner"] = raw["scanner_typeloc"].astype(str)
    df["session_id"] = raw["session"].astype(str)

    # Select FreeSurfer DKT atlas columns (ThickAvg, SurfArea, GrayVol)
    dkt_cols = [c for c in raw.columns
                if "freesurfer_" in c and "DKTatlas" in c
                and any(m in c for m in ["ThickAvg", "SurfArea", "GrayVol"])]

    # Select FreeSurfer aseg subcortical volumes
    aseg_cols = [c for c in raw.columns
                 if c.startswith("freesurfer_aseg_Volume_mm3")]

    feat_source_cols = sorted(dkt_cols) + sorted(aseg_cols)

    # Extract features as a block (avoids fragmented DataFrame)
    feat_data = raw[feat_source_cols].apply(pd.to_numeric, errors="coerce")
    feat_cols = [f"feat_{i}" for i in range(len(feat_source_cols))]
    feat_data.columns = feat_cols
    df = pd.concat([df, feat_data], axis=1)

    # Drop rows where all features are NaN
    df = df.dropna(subset=feat_cols, how="all").reset_index(drop=True)

    # Store original feature names for reference
    df.attrs["feature_names"] = {f"feat_{i}": col for i, col in enumerate(feat_source_cols)}

    print(f"FreeSurfer IDPs: {len(dkt_cols)} DKT cortical + {len(aseg_cols)} aseg subcortical "
          f"= {len(feat_cols)} total features")
    print(f"  {df['subject_id'].nunique()} subjects, {df['scanner'].nunique()} scanners, "
          f"{len(df)} observations")

    return df, feat_cols


def main():
    parser = argparse.ArgumentParser(description="Reliability analysis — FreeSurfer (DKT)")
    parser.add_argument("--input", type=str,
                        default="/media/rafa/DATA_4TB/all_idps.csv",
                        help="Path to ON-Harmony all_idps.csv")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "Results"))
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    df, feat_cols = load_freesurfer_idps(args.input)

    run_full_analysis(df, feat_cols, model_name="FreeSurfer",
                      output_dir=args.output_dir, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
