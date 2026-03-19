#!/usr/bin/env python3
"""
Reliability analysis for ComBat-harmonized FreeSurfer morphometric features.

Applies neuroCombat to remove scanner effects from FreeSurfer DKT + aseg
features, then runs the same reliability pipeline as the raw FreeSurfer
analysis. The harmonized results serve as an upper-bound reference:
"best achievable reliability when scanner effects are explicitly removed."

Input:  all_idps.csv from ON-Harmony dataset + participants.tsv (for covariates)
Output: ../Results/results_FreeSurfer_harmonized.json

Usage:
  python analyse_FreeSurfer_harmonized.py [--input /path/to/all_idps.csv]
                                          [--participants ../DATA/participants.tsv]
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
    Returns raw DataFrame with metadata + features (before harmonization).
    """
    raw = pd.read_csv(csv_path)

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

    feat_data = raw[feat_source_cols].apply(pd.to_numeric, errors="coerce")
    feat_cols = [f"feat_{i}" for i in range(len(feat_source_cols))]
    feat_data.columns = feat_cols
    df = pd.concat([df, feat_data], axis=1)

    df = df.dropna(subset=feat_cols, how="all").reset_index(drop=True)
    df.attrs["feature_names"] = {f"feat_{i}": col
                                  for i, col in enumerate(feat_source_cols)}

    # Remove degenerate features: zero/near-zero variance OR high NaN fraction.
    # The 4 thalamus aseg features have ~50% NaN (missing in many sessions),
    # which causes a singular design matrix in ComBat.
    degenerate = [c for c in feat_cols
                  if df[c].std() < 1e-10 or df[c].isna().mean() > 0.1]
    if degenerate:
        print(f"  Removing {len(degenerate)} degenerate (high-NaN or zero-variance) features")
        feat_cols = [c for c in feat_cols if c not in degenerate]
        df = df[["subject_id", "scanner", "session_id"] + feat_cols].copy()

    print(f"FreeSurfer IDPs: {len(dkt_cols)} DKT cortical + {len(aseg_cols)} "
          f"aseg subcortical = {len(feat_source_cols)} total, "
          f"{len(feat_cols)} after removing degenerate")
    print(f"  {df['subject_id'].nunique()} subjects, "
          f"{df['scanner'].nunique()} scanners, {len(df)} observations")

    return df, feat_cols


def apply_combat(df, feat_cols, participants_path):
    """
    Apply neuroCombat to harmonize FreeSurfer features across scanners.

    Parameters
    ----------
    df : DataFrame with subject_id, scanner, session_id, and feature columns
    feat_cols : list of feature column names
    participants_path : path to participants.tsv (for age/sex covariates)

    Returns
    -------
    df_harmonized : DataFrame with harmonized features (same structure as input)
    """
    from neuroCombat import neuroCombat

    # Load biological covariates
    participants = pd.read_csv(participants_path, sep="\t")
    participants["subject_id"] = participants["participant_id"].str.replace(
        "sub-", "_", regex=False
    )
    participants["sex_num"] = (participants["sex"] == "M").astype(int)

    # Merge covariates into df
    df_merged = df.merge(
        participants[["subject_id", "age", "sex_num"]],
        on="subject_id",
        how="left",
    )

    # Check for missing covariates
    n_missing = df_merged[["age", "sex_num"]].isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} rows missing age/sex covariates")

    # Prepare inputs for neuroCombat
    # data: (features × samples) matrix
    data = df_merged[feat_cols].values.T  # (D, N)

    # Covariates DataFrame (batch + biological)
    covars = pd.DataFrame({
        "batch": df_merged["scanner"].values,
        "age": df_merged["age"].values,
        "sex": df_merged["sex_num"].values,
    })

    print(f"\n  Running neuroCombat...")
    print(f"    Batch variable: scanner ({covars['batch'].nunique()} levels)")
    print(f"    Biological covariates: age, sex")
    print(f"    Data matrix: {data.shape[0]} features × {data.shape[1]} samples")

    # Run ComBat
    combat_result = neuroCombat(
        dat=data,
        covars=covars,
        batch_col="batch",
    )

    # Extract harmonized data
    data_harmonized = combat_result["data"]  # (D, N)

    # Create harmonized DataFrame
    df_harmonized = df.copy()
    df_harmonized[feat_cols] = data_harmonized.T

    print(f"    ComBat harmonization complete.")

    # Quick sanity check: scanner variance before/after
    from utils_reliability import variance_decomposition
    var_before = variance_decomposition(df, feat_cols)
    var_after = variance_decomposition(df_harmonized, feat_cols)
    print(f"\n  Scanner variance: {np.nanmean(var_before['pct_scanner']):.1f}% "
          f"→ {np.nanmean(var_after['pct_scanner']):.1f}%")
    print(f"  Subject variance: {np.nanmean(var_before['pct_subject']):.1f}% "
          f"→ {np.nanmean(var_after['pct_subject']):.1f}%")

    return df_harmonized


def main():
    parser = argparse.ArgumentParser(
        description="Reliability analysis — ComBat-harmonized FreeSurfer (DKT)"
    )
    parser.add_argument("--input", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "DATA",
                                             "FreeSurfer_IDPs_ON-Harmony.csv"),
                        help="Path to ON-Harmony all_idps.csv")
    parser.add_argument("--participants", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "DATA", "participants.tsv"),
                        help="Path to participants.tsv")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "Results"))
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    # Load raw features
    df, feat_cols = load_freesurfer_idps(args.input)

    # Apply ComBat harmonization
    df_harmonized = apply_combat(df, feat_cols, args.participants)

    # Run full reliability analysis on harmonized features
    run_full_analysis(df_harmonized, feat_cols,
                      model_name="FreeSurfer_harmonized",
                      output_dir=args.output_dir, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
