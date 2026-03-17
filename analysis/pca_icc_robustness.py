#!/usr/bin/env python3
"""
PCA-reduced ICC robustness check.

Reduces all models to the same number of principal components, then
recomputes between-scanner ICC(2,1) to verify that the reliability
ranking is not driven by differences in embedding dimensionality.

Output: ../Results/pca_icc_robustness.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from utils_reliability import compute_icc_between_scanner, icc_summary

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "Results")

# Models and their CSV files
MODELS = {
    "AnatCL":          "embeddings_AnatCL_fold0.csv",
    "FreeSurfer":      "FreeSurfer_IDPs_ON-Harmony.csv",
    "y-Aware":         "embeddings_yAware.csv",
    "BrainIAC":        "embeddings_BrainIAC.csv",
    "3D-Neuro-SimCLR": "embeddings_3DNeuroSimCLR.csv",
    "BrainSegFounder": "embeddings_BrainSegFounder.csv",
}

# Fixed PC counts to test
PC_COUNTS = [10, 20]


def load_model_data(model_name, csv_file):
    """Load embedding CSV and return df, emb_cols."""
    path = os.path.join(DATA_DIR, csv_file)
    raw = pd.read_csv(path)

    if model_name == "FreeSurfer":
        # Replicate the DKT + aseg selection from analyse_FreeSurfer.py
        df = pd.DataFrame()
        df["subject_id"] = raw["subject"].astype(str)
        df["scanner"] = raw["scanner_typeloc"].astype(str)
        df["session_id"] = raw["session"].astype(str)

        # DKT atlas: ThickAvg, SurfArea, GrayVol
        dkt_cols = [c for c in raw.columns
                    if "freesurfer_" in c and "DKTatlas" in c
                    and any(m in c for m in ["ThickAvg", "SurfArea", "GrayVol"])]
        # aseg subcortical volumes
        aseg_cols = [c for c in raw.columns
                     if c.startswith("freesurfer_aseg_Volume_mm3")]
        feat_source_cols = sorted(dkt_cols) + sorted(aseg_cols)

        feat_data = raw[feat_source_cols].apply(pd.to_numeric, errors="coerce")
        emb_cols = [f"feat_{i}" for i in range(len(feat_source_cols))]
        feat_data.columns = emb_cols
        df = pd.concat([df, feat_data], axis=1)
        df = df.dropna(subset=emb_cols, how="all").reset_index(drop=True)

        # Drop zero-variance features
        emb_cols = [c for c in emb_cols if df[c].std() > 0]
        return df, emb_cols
    else:
        emb_cols = [c for c in raw.columns if c.startswith("emb_")]
        return raw, emb_cols


def pca_reduce_and_icc(df, emb_cols, n_components, model_name):
    """Apply PCA to embeddings, then compute between-scanner ICC(2,1)."""
    X = df[emb_cols].values.copy()

    # Impute NaN with column median (for FreeSurfer)
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = np.isnan(col)
        if nans.any():
            X[nans, j] = np.nanmedian(col)

    # Fit PCA
    n_comp = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    # Build dataframe with PCA columns
    pc_cols = [f"pc_{i}" for i in range(n_comp)]
    df_pca = df[["subject_id", "scanner"]].copy()
    for i, col_name in enumerate(pc_cols):
        df_pca[col_name] = X_pca[:, i]

    # Compute ICC(2,1)
    icc_vals = compute_icc_between_scanner(df_pca, pc_cols)
    summary = icc_summary(icc_vals, f"{model_name}_PCA{n_comp}")

    return {
        "n_pcs": n_comp,
        "var_explained_pct": round(var_explained, 1),
        "median_icc": round(summary["median"], 3),
        "mean_icc": round(summary["mean"], 3),
        "q25": round(summary["q25"], 3),
        "q75": round(summary["q75"], 3),
    }


def main():
    print("=" * 65)
    print("  PCA-reduced ICC robustness check")
    print("=" * 65)

    all_results = {}

    for model_name, csv_file in MODELS.items():
        print(f"\n--- {model_name} ---")
        df, emb_cols = load_model_data(model_name, csv_file)
        orig_dim = len(emb_cols)
        print(f"    Original dims: {orig_dim}, N={len(df)}")

        # Original ICC (full dimensions)
        icc_orig = compute_icc_between_scanner(df, emb_cols)
        orig_summary = icc_summary(icc_orig, f"{model_name}_original")
        print(f"    Original ICC(2,1) median: {orig_summary['median']:.3f}")

        model_results = {
            "original_dim": orig_dim,
            "original_icc": round(orig_summary["median"], 3),
            "pca_results": {},
        }

        for n_pc in PC_COUNTS:
            if n_pc > orig_dim:
                print(f"    PCA-{n_pc}: skipped (> original dim)")
                continue
            res = pca_reduce_and_icc(df, emb_cols, n_pc, model_name)
            model_results["pca_results"][f"pc{n_pc}"] = res
            print(f"    PCA-{n_pc}: ICC={res['median_icc']:.3f} "
                  f"(var explained: {res['var_explained_pct']:.1f}%)")

        all_results[model_name] = model_results

    # Summary table
    print(f"\n{'=' * 65}")
    print(f"  Summary: Median ICC(2,1) at different PC counts")
    print(f"{'=' * 65}")
    header = f"{'Model':<20} {'Orig Dim':>8} {'Orig ICC':>9}"
    for n_pc in PC_COUNTS:
        header += f" {'PC-'+str(n_pc):>8}"
    print(header)
    print("-" * len(header))

    for model_name in MODELS:
        r = all_results[model_name]
        line = f"{model_name:<20} {r['original_dim']:>8} {r['original_icc']:>9.3f}"
        for n_pc in PC_COUNTS:
            key = f"pc{n_pc}"
            if key in r["pca_results"]:
                line += f" {r['pca_results'][key]['median_icc']:>8.3f}"
            else:
                line += f" {'---':>8}"
        print(line)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "pca_icc_robustness.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
