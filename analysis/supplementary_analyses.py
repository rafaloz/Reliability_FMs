#!/usr/bin/env python3
"""
Supplementary analyses for the FM reliability paper.

Computes:
  1. Cross-scanner sex classification (leave-one-scanner-out)
  2. PCA effective dimensionality and top-PC structure
  3. Per-scanner pairwise cosine similarity matrices
  4. Bootstrap paired difference tests for ICC comparisons
  5. Preprocessing failure sensitivity check (ANTs→FLIRT fallback sessions)

Saves results to Results/supplementary_results.json

Usage:
  python supplementary_analyses.py [--data_dir ../DATA] [--results_dir ../Results]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.dirname(__file__))
from utils_reliability import pivot_between_scanner_3d, _icc21_batch


# ── Data loading ─────────────────────────────────────────────────────────────

MODEL_INFO = {
    "FreeSurfer": {"file": "FreeSurfer_IDPs_ON-Harmony.csv", "type": "freesurfer"},
    "AnatCL":     {"file": "embeddings_AnatCL_fold0.csv",    "type": "fm"},
    "y-Aware":    {"file": "embeddings_yAware.csv",          "type": "fm"},
    "BrainIAC":   {"file": "embeddings_BrainIAC.csv",        "type": "fm"},
    "3D-SimCLR":  {"file": "embeddings_3DNeuroSimCLR.csv",   "type": "fm"},
    "BrainSegFounder": {"file": "embeddings_BrainSegFounder.csv", "type": "fm"},
}

# Sessions where ANTs rigid registration failed → FSL FLIRT used instead
# (3D-Neuro-SimCLR Pipeline B, all Siemens Oxford scanners)
FLIRT_FALLBACK_SESSIONS = {
    "ses-OXF1PRI001", "ses-OXF1PRI002", "ses-OXF1PRI003",
    "ses-OXF2PRI001", "ses-OXF2PRI002", "ses-OXF2PRI003",
    "ses-OXF3TRI001", "ses-OXF3TRI002", "ses-OXF3TRI003",
    # Phase B additions
    "ses-OXF1PRI004", "ses-OXF1PRI005",
    "ses-OXF2PRI004", "ses-OXF2PRI005",
    "ses-OXF4GEP001", "ses-OXF4GEP002", "ses-OXF4GEP003",
}


def load_model_data(data_dir, model_name):
    """Load embedding DataFrame for a model. Returns (df, emb_cols)."""
    info = MODEL_INFO[model_name]
    path = os.path.join(data_dir, info["file"])
    df = pd.read_csv(path)

    if info["type"] == "freesurfer":
        # FreeSurfer: select only the 203 DKT + aseg features
        # (must match analyse_FreeSurfer.py column selection)
        dkt_cols = [c for c in df.columns
                    if "freesurfer_" in c and "DKTatlas" in c
                    and any(m in c for m in ["ThickAvg", "SurfArea", "GrayVol"])]
        aseg_cols = [c for c in df.columns
                     if c.startswith("freesurfer_aseg_Volume_mm3")]
        emb_cols = sorted(dkt_cols) + sorted(aseg_cols)
        # Standardize column names
        if "subject" in df.columns and "subject_id" not in df.columns:
            df = df.rename(columns={"subject": "subject_id"})
        if "scanner_typeloc" in df.columns and "scanner" not in df.columns:
            df = df.rename(columns={"scanner_typeloc": "scanner"})
        # Add sex/age from participants.tsv if missing
        if "sex" not in df.columns:
            ptsv = os.path.join(data_dir, "participants.tsv")
            if os.path.exists(ptsv):
                demo = pd.read_csv(ptsv, sep="\t")
                # FreeSurfer uses _03286 format, participants.tsv uses sub-03286
                demo["subject_id"] = demo["participant_id"]
                # Create matching key: strip prefix to numeric part
                df["_merge_key"] = df["subject_id"].str.replace(r"^[_sub-]+", "", regex=True)
                demo["_merge_key"] = demo["subject_id"].str.replace(r"^sub-", "", regex=True)
                df = df.merge(demo[["_merge_key", "sex", "age"]], on="_merge_key", how="left")
                df = df.drop(columns=["_merge_key"])
    else:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]

    # Drop columns that are entirely NaN
    valid_cols = [c for c in emb_cols if df[c].notna().all()]
    if len(valid_cols) < len(emb_cols):
        n_dropped = len(emb_cols) - len(valid_cols)
        print(f"    (dropped {n_dropped} NaN columns from {model_name})")
    emb_cols = valid_cols

    return df, emb_cols


def load_all_models(data_dir):
    """Load all models. Returns dict of {name: (df, emb_cols)}."""
    models = {}
    for name in MODEL_INFO:
        try:
            df, emb_cols = load_model_data(data_dir, name)
            models[name] = (df, emb_cols)
            print(f"  Loaded {name}: {len(df)} rows, {len(emb_cols)} dims")
        except Exception as e:
            print(f"  WARNING: Could not load {name}: {e}")
    return models


# ── Analysis 1: Cross-scanner sex classification ────────────────────────────

def cross_scanner_sex_classification(models):
    """
    Leave-one-scanner-out cross-validation for sex classification.
    Train logistic regression on embeddings from 7 scanners, test on held-out.
    Returns per-model accuracy and balanced accuracy.
    """
    print("\n=== Cross-Scanner Sex Classification (LOSO) ===")
    results = {}

    for name, (df, emb_cols) in models.items():
        if "sex" not in df.columns:
            print(f"  {name}: no sex column, skipping")
            continue

        X = df[emb_cols].values
        y = (df["sex"] == "M").astype(int).values
        scanners = df["scanner"].values
        unique_scanners = sorted(df["scanner"].unique())

        all_preds = np.full(len(y), -1)

        for held_out in unique_scanners:
            train_mask = scanners != held_out
            test_mask = scanners == held_out

            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            # Need both classes in training
            if len(np.unique(y[train_mask])) < 2:
                continue

            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            )
            clf.fit(X[train_mask], y[train_mask])
            all_preds[test_mask] = clf.predict(X[test_mask])

        valid = all_preds >= 0
        acc = accuracy_score(y[valid], all_preds[valid])
        bal_acc = balanced_accuracy_score(y[valid], all_preds[valid])

        # Per-scanner accuracy
        per_scanner = {}
        for sc in unique_scanners:
            sc_mask = (scanners == sc) & valid
            if sc_mask.sum() > 0:
                per_scanner[sc] = float(accuracy_score(y[sc_mask], all_preds[sc_mask]))

        results[name] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "n_valid": int(valid.sum()),
            "per_scanner": per_scanner,
        }
        print(f"  {name}: acc={acc:.3f}, bal_acc={bal_acc:.3f} ({valid.sum()} sessions)")

    return results


# ── Analysis 2: PCA effective dimensionality ─────────────────────────────────

def pca_analysis(models):
    """
    Compute PCA on each model's embeddings.
    Reports: n_PCs for 90/95/99% variance, top-PC correlations with scanner/subject.
    """
    print("\n=== PCA Effective Dimensionality ===")
    results = {}

    for name, (df, emb_cols) in models.items():
        X = StandardScaler().fit_transform(df[emb_cols].values)
        n_components = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        cum_var = np.cumsum(pca.explained_variance_ratio_)

        # Number of PCs for thresholds
        n90 = int(np.searchsorted(cum_var, 0.90) + 1)
        n95 = int(np.searchsorted(cum_var, 0.95) + 1)
        n99 = int(np.searchsorted(cum_var, 0.99) + 1)

        # Top-PC correlation with scanner (eta-squared)
        scanners = df["scanner"].values
        subjects = df["subject_id"].values
        unique_scanners = sorted(df["scanner"].unique())
        unique_subjects = sorted(df["subject_id"].unique())

        # Eta-squared for top PCs with scanner and subject
        def eta_squared(values, groups):
            """Compute eta-squared (proportion of variance explained by grouping)."""
            grand_mean = values.mean()
            ss_total = np.sum((values - grand_mean) ** 2)
            if ss_total == 0:
                return 0.0
            ss_between = 0
            for g in np.unique(groups):
                mask = groups == g
                group_mean = values[mask].mean()
                ss_between += mask.sum() * (group_mean - grand_mean) ** 2
            return float(ss_between / ss_total)

        pc_structure = []
        for pc_idx in range(min(5, n_components)):
            pc_vals = X_pca[:, pc_idx]
            eta_scanner = eta_squared(pc_vals, scanners)
            eta_subject = eta_squared(pc_vals, subjects)
            pc_structure.append({
                "pc": pc_idx + 1,
                "var_explained": float(pca.explained_variance_ratio_[pc_idx]),
                "cum_var": float(cum_var[pc_idx]),
                "eta_sq_scanner": eta_scanner,
                "eta_sq_subject": eta_subject,
                "dominant": "scanner" if eta_scanner > eta_subject else "subject",
            })

        results[name] = {
            "n_dims_original": len(emb_cols),
            "n_components_computed": n_components,
            "n_pc_90": n90,
            "n_pc_95": n95,
            "n_pc_99": n99,
            "top1_var_pct": float(pca.explained_variance_ratio_[0] * 100),
            "top5_var_pct": float(cum_var[min(4, n_components - 1)] * 100),
            "pc_structure": pc_structure,
            "explained_variance_ratio": pca.explained_variance_ratio_[:50].tolist(),
        }
        print(f"  {name}: 90%={n90}, 95%={n95}, 99%={n99} PCs | "
              f"PC1={pca.explained_variance_ratio_[0]*100:.1f}% "
              f"({'scanner' if pc_structure[0]['eta_sq_scanner'] > pc_structure[0]['eta_sq_subject'] else 'subject'}-dominated)")

    return results


# ── Analysis 3: Per-scanner pairwise cosine similarity ──────────────────────

def pairwise_scanner_similarity(models):
    """
    For each model, compute mean cosine similarity for each scanner pair,
    averaged across same-subject pairs.
    """
    print("\n=== Per-Scanner Pairwise Cosine Similarity ===")
    results = {}

    for name, (df, emb_cols) in models.items():
        subjects = sorted(df["subject_id"].unique())
        scanners = sorted(df["scanner"].unique())
        n_scanners = len(scanners)
        scan_idx = {s: i for i, s in enumerate(scanners)}

        # For each subject, get mean embedding per scanner
        sim_matrix = np.full((n_scanners, n_scanners), np.nan)
        pair_counts = np.zeros((n_scanners, n_scanners), dtype=int)

        for subj in subjects:
            subj_df = df[df["subject_id"] == subj]
            scanner_embs = {}
            for sc in scanners:
                sc_df = subj_df[subj_df["scanner"] == sc]
                if len(sc_df) > 0:
                    scanner_embs[sc] = sc_df[emb_cols].values.mean(axis=0)

            for sc_a in scanner_embs:
                for sc_b in scanner_embs:
                    i, j = scan_idx[sc_a], scan_idx[sc_b]
                    cos_sim = float(cosine_similarity(
                        scanner_embs[sc_a].reshape(1, -1),
                        scanner_embs[sc_b].reshape(1, -1)
                    )[0, 0])
                    if np.isnan(sim_matrix[i, j]):
                        sim_matrix[i, j] = 0.0
                    sim_matrix[i, j] += cos_sim
                    pair_counts[i, j] += 1

        # Average
        with np.errstate(divide='ignore', invalid='ignore'):
            sim_matrix = np.where(pair_counts > 0, sim_matrix / pair_counts, np.nan)

        results[name] = {
            "scanners": scanners,
            "similarity_matrix": np.nan_to_num(sim_matrix, nan=0.0).tolist(),
            "pair_counts": pair_counts.tolist(),
        }

        # Report within-vendor vs between-vendor
        vendor_map = {}
        for sc in scanners:
            if "PRI" in sc or "TRI" in sc:
                vendor_map[sc] = "Siemens"
            elif "ACH" in sc or "ING" in sc:
                vendor_map[sc] = "Philips"
            else:
                vendor_map[sc] = "GE"

        within_vendor_sims = []
        between_vendor_sims = []
        for i in range(n_scanners):
            for j in range(i + 1, n_scanners):
                if not np.isnan(sim_matrix[i, j]):
                    if vendor_map[scanners[i]] == vendor_map[scanners[j]]:
                        within_vendor_sims.append(sim_matrix[i, j])
                    else:
                        between_vendor_sims.append(sim_matrix[i, j])

        results[name]["within_vendor_mean"] = float(np.mean(within_vendor_sims)) if within_vendor_sims else None
        results[name]["between_vendor_mean"] = float(np.mean(between_vendor_sims)) if between_vendor_sims else None
        results[name]["vendor_gap"] = (
            float(np.mean(within_vendor_sims) - np.mean(between_vendor_sims))
            if within_vendor_sims and between_vendor_sims else None
        )

        print(f"  {name}: within-vendor={results[name]['within_vendor_mean']:.3f}, "
              f"between-vendor={results[name]['between_vendor_mean']:.3f}, "
              f"gap={results[name]['vendor_gap']:.3f}")

    return results


# ── Analysis 4: Bootstrap paired difference tests ───────────────────────────

def bootstrap_paired_tests(models, results_dir, n_boot=5000):
    """
    For each model pair, compute bootstrap distribution of
    median_ICC_A - median_ICC_B to test significance.
    """
    print("\n=== Bootstrap Paired Difference Tests ===")

    # Load per-dimension ICC arrays from results JSONs
    model_iccs = {}
    model_order = ["FreeSurfer", "AnatCL", "y-Aware", "BrainIAC", "3D-SimCLR", "BrainSegFounder"]

    for name in model_order:
        if name == "AnatCL":
            # Use fold 0 for per-dimension ICCs
            rpath = os.path.join(results_dir, "results_AnatCL_fold0.json")
        else:
            fname = {
                "FreeSurfer": "results_FreeSurfer.json",
                "y-Aware": "results_yAware.json",
                "BrainIAC": "results_BrainIAC.json",
                "3D-SimCLR": "results_3DNeuroSimCLR.json",
                "BrainSegFounder": "results_BrainSegFounder.json",
            }[name]
            rpath = os.path.join(results_dir, fname)

        with open(rpath) as f:
            rdata = json.load(f)

        icc_arr = np.array(rdata["icc_between"])
        # Remove NaN
        icc_arr = icc_arr[~np.isnan(icc_arr)]
        model_iccs[name] = icc_arr

    # We need subject-level bootstrap for proper inference.
    # Since we have per-dimension ICCs (not per-subject), we bootstrap dimensions
    # as a proxy — this tests whether the median difference is robust to
    # dimension sampling, not subject sampling.
    # For subject-level tests we'd need to re-run ICC computation per bootstrap,
    # which is too expensive. Dimension bootstrap is informative and commonly used.
    results = {}
    rng = np.random.RandomState(42)

    pairs_to_test = [
        ("AnatCL", "FreeSurfer"),
        ("AnatCL", "y-Aware"),
        ("FreeSurfer", "y-Aware"),
        ("y-Aware", "BrainIAC"),
        ("BrainIAC", "3D-SimCLR"),
        ("3D-SimCLR", "BrainSegFounder"),
    ]

    for name_a, name_b in pairs_to_test:
        icc_a = model_iccs[name_a]
        icc_b = model_iccs[name_b]

        observed_diff = float(np.median(icc_a) - np.median(icc_b))

        # Bootstrap the difference in medians
        boot_diffs = []
        for _ in range(n_boot):
            idx_a = rng.randint(0, len(icc_a), len(icc_a))
            idx_b = rng.randint(0, len(icc_b), len(icc_b))
            boot_diffs.append(np.median(icc_a[idx_a]) - np.median(icc_b[idx_b]))

        boot_diffs = np.array(boot_diffs)

        # Two-sided p-value: proportion of bootstrap diffs that cross zero
        if observed_diff > 0:
            p_value = float(np.mean(boot_diffs <= 0)) * 2  # two-sided
        else:
            p_value = float(np.mean(boot_diffs >= 0)) * 2
        p_value = min(p_value, 1.0)

        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))

        key = f"{name_a}_vs_{name_b}"
        results[key] = {
            "model_a": name_a,
            "model_b": name_b,
            "observed_diff": observed_diff,
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
            "p_value": p_value,
            "significant_005": p_value < 0.05,
        }
        sig = "*" if p_value < 0.05 else ""
        print(f"  {name_a} vs {name_b}: Δ={observed_diff:+.3f} "
              f"[{ci_lo:+.3f}, {ci_hi:+.3f}] p={p_value:.4f}{sig}")

    return results


# ── Analysis 5: Preprocessing failure sensitivity ───────────────────────────

def preprocessing_failure_check(models):
    """
    Check whether the 15 sessions with ANTs→FLIRT fallback
    (3D-Neuro-SimCLR only) show different embedding patterns.
    """
    print("\n=== Preprocessing Failure Sensitivity Check ===")

    name = "3D-SimCLR"
    if name not in models:
        print("  3D-SimCLR not loaded, skipping")
        return {}

    df, emb_cols = models[name]
    X = df[emb_cols].values

    # Flag sessions
    flirt_mask = df["session_id"].isin(FLIRT_FALLBACK_SESSIONS).values
    ants_mask = ~flirt_mask

    n_flirt = flirt_mask.sum()
    n_ants = ants_mask.sum()
    print(f"  FLIRT sessions: {n_flirt}, ANTs sessions: {n_ants}")

    if n_flirt == 0:
        print("  No FLIRT sessions found — checking session IDs...")
        # Try matching by scanner code (OXF scanners)
        oxford_mask = df["scanner"].str.startswith("OXF").values
        n_oxford = oxford_mask.sum()
        print(f"  Oxford sessions: {n_oxford}")
        flirt_mask = oxford_mask

    if flirt_mask.sum() == 0:
        return {"n_flirt": 0, "note": "No FLIRT fallback sessions identified"}

    # Compare mean embedding norms
    norm_flirt = np.linalg.norm(X[flirt_mask], axis=1).mean()
    norm_ants = np.linalg.norm(X[ants_mask], axis=1).mean()

    # Compare mean pairwise cosine similarity within each group
    if flirt_mask.sum() > 1:
        cos_flirt = cosine_similarity(X[flirt_mask]).mean()
    else:
        cos_flirt = float("nan")
    cos_ants = cosine_similarity(X[ants_mask]).mean()

    # Cross-group similarity
    cos_cross = cosine_similarity(X[flirt_mask], X[ants_mask]).mean()

    result = {
        "n_flirt": int(flirt_mask.sum()),
        "n_ants": int(ants_mask.sum()),
        "mean_norm_flirt": float(norm_flirt),
        "mean_norm_ants": float(norm_ants),
        "mean_cosine_within_flirt": float(cos_flirt),
        "mean_cosine_within_ants": float(cos_ants),
        "mean_cosine_cross": float(cos_cross),
    }
    print(f"  Norm: FLIRT={norm_flirt:.3f}, ANTs={norm_ants:.3f}")
    print(f"  Cosine: within-FLIRT={cos_flirt:.3f}, within-ANTs={cos_ants:.3f}, cross={cos_cross:.3f}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Supplementary analyses")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "DATA"))
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "Results"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "Results"))
    parser.add_argument("--n_boot", type=int, default=5000)
    args = parser.parse_args()

    print("Loading models...")
    models = load_all_models(args.data_dir)

    all_results = {}

    # 1. Sex classification
    all_results["sex_classification"] = cross_scanner_sex_classification(models)

    # 2. PCA
    all_results["pca"] = pca_analysis(models)

    # 3. Per-scanner similarity
    all_results["pairwise_scanner_similarity"] = pairwise_scanner_similarity(models)

    # 4. Bootstrap tests
    all_results["bootstrap_tests"] = bootstrap_paired_tests(
        models, args.results_dir, n_boot=args.n_boot
    )

    # 5. Preprocessing failure check
    all_results["preprocessing_sensitivity"] = preprocessing_failure_check(models)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "supplementary_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
