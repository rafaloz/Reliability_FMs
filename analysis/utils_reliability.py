#!/usr/bin/env python3
"""
Shared utility functions for reliability and reproducibility analyses.

Provides:
  - ICC(2,1) and ICC(3,1) computation (Shrout & Fleiss 1979) — vectorized
  - Variance decomposition (subject, scanner, residual)
  - Coefficient of Variation (within- and between-scanner)
  - Scanner fingerprinting (linear SVM)
  - Subject identification across scanners
  - Bootstrap confidence intervals
  - Summary statistics and result I/O
"""

import sys
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
import json
import os


def _print(msg=""):
    """Print with immediate flush so output appears in real-time."""
    print(msg, flush=True)


# ── Vectorized ICC (all dimensions at once) ─────────────────────────────────

def _icc21_batch(data_3d):
    """
    Vectorized ICC(2,1) over D dimensions simultaneously.

    Parameters
    ----------
    data_3d : np.ndarray, shape (n_subjects, k_scanners, D)
        NaN allowed for missing subject-scanner combinations.

    Returns
    -------
    iccs : np.ndarray, shape (D,)
    """
    # Mask: valid (non-NaN) entries
    mask = ~np.isnan(data_3d)  # (n, k, D)
    n, k, D = data_3d.shape

    # Replace NaN with 0 for summation, use mask for counts
    data = np.where(mask, data_3d, 0.0)

    # Grand mean per dim
    count_total = mask.sum(axis=(0, 1))  # (D,)
    grand_sum = data.sum(axis=(0, 1))    # (D,)
    with np.errstate(divide='ignore', invalid='ignore'):
        grand_mean = np.where(count_total > 0, grand_sum / count_total, 0.0)  # (D,)

    # Row (subject) means
    row_count = mask.sum(axis=1)      # (n, D)
    row_sum = data.sum(axis=1)        # (n, D)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_mean = np.where(row_count > 0, row_sum / row_count, 0.0)  # (n, D)

    # Col (scanner) means
    col_count = mask.sum(axis=0)      # (k, D)
    col_sum = data.sum(axis=0)        # (k, D)
    with np.errstate(divide='ignore', invalid='ignore'):
        col_mean = np.where(col_count > 0, col_sum / col_count, 0.0)  # (k, D)

    # SS_rows = sum_i n_i * (row_mean_i - grand_mean)^2
    SS_rows = (row_count * (row_mean - grand_mean[np.newaxis, :]) ** 2).sum(axis=0)  # (D,)
    # SS_cols = sum_j n_j * (col_mean_j - grand_mean)^2
    SS_cols = (col_count * (col_mean - grand_mean[np.newaxis, :]) ** 2).sum(axis=0)  # (D,)
    # SS_total
    SS_total = (mask * (data_3d - grand_mean[np.newaxis, np.newaxis, :]) ** 2)
    SS_total = np.nansum(SS_total, axis=(0, 1))  # (D,)
    SS_error = SS_total - SS_rows - SS_cols

    # Count valid subjects (those with >=2 non-NaN scanners)
    subj_valid = (mask.sum(axis=1) >= 1).sum(axis=0)  # approx n per dim
    # Use fixed n, k for balanced formula
    df_rows = np.maximum(subj_valid - 1, 1).astype(float)
    df_cols = np.float64(k - 1)
    df_error = np.maximum(df_rows * df_cols, 1.0)

    MS_R = SS_rows / df_rows
    MS_C = SS_cols / df_cols if k > 1 else np.zeros(D)
    MS_E = SS_error / df_error

    n_eff = subj_valid.astype(float)
    denom = MS_R + (k - 1) * MS_E + k * (MS_C - MS_E) / np.maximum(n_eff, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        iccs = np.where(denom != 0, (MS_R - MS_E) / denom, np.nan)
    return iccs


def _icc31_batch(data_3d):
    """
    Vectorized ICC(3,1) over D dimensions simultaneously.

    Parameters
    ----------
    data_3d : np.ndarray, shape (n_subjects, k_sessions, D)

    Returns
    -------
    iccs : np.ndarray, shape (D,)
    """
    mask = ~np.isnan(data_3d)
    n, k, D = data_3d.shape
    data = np.where(mask, data_3d, 0.0)

    count_total = mask.sum(axis=(0, 1))
    grand_sum = data.sum(axis=(0, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        grand_mean = np.where(count_total > 0, grand_sum / count_total, 0.0)

    row_count = mask.sum(axis=1)
    row_sum = data.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_mean = np.where(row_count > 0, row_sum / row_count, 0.0)

    SS_rows = (row_count * (row_mean - grand_mean[np.newaxis, :]) ** 2).sum(axis=0)
    SS_total = np.nansum(mask * (data_3d - grand_mean[np.newaxis, np.newaxis, :]) ** 2,
                         axis=(0, 1))

    col_count = mask.sum(axis=0)
    col_sum = data.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        col_mean = np.where(col_count > 0, col_sum / col_count, 0.0)
    SS_cols = (col_count * (col_mean - grand_mean[np.newaxis, :]) ** 2).sum(axis=0)
    SS_error = SS_total - SS_rows - SS_cols

    subj_valid = (mask.sum(axis=1) >= 1).sum(axis=0)
    df_rows = np.maximum(subj_valid - 1, 1).astype(float)
    df_error = np.maximum(df_rows * (k - 1), 1.0)

    MS_R = SS_rows / df_rows
    MS_E = SS_error / df_error

    denom = MS_R + (k - 1) * MS_E
    with np.errstate(divide='ignore', invalid='ignore'):
        iccs = np.where(denom != 0, (MS_R - MS_E) / denom, np.nan)
    return iccs


# ── Pivot to 3D arrays (all dims at once) ───────────────────────────────────

def pivot_between_scanner_3d(df, emb_cols, subject_col="subject_id",
                             scanner_col="scanner"):
    """
    Build a (n_subjects, k_scanners, D) array for between-scanner ICC.
    Averages repeats of the same subject on the same scanner.
    """
    subjects = sorted(df[subject_col].unique())
    scanners = sorted(df[scanner_col].unique())
    subj_idx = {s: i for i, s in enumerate(subjects)}
    scan_idx = {s: i for i, s in enumerate(scanners)}

    D = len(emb_cols)
    n = len(subjects)
    k = len(scanners)
    arr = np.full((n, k, D), np.nan)

    grouped = df.groupby([subject_col, scanner_col])[emb_cols].mean()
    for (subj, scanner), row in grouped.iterrows():
        arr[subj_idx[subj], scan_idx[scanner], :] = row.values

    return arr


def pivot_within_scanner_3d(df, emb_cols, subject_col="subject_id",
                            scanner_col="scanner"):
    """
    For within-scanner ICC: find scanners with repeat subjects, build
    3D arrays. Returns list of (scanner_name, array(n_subj, k_reps, D)).
    """
    D = len(emb_cols)
    results = []

    for scanner, sdf in df.groupby(scanner_col):
        counts = sdf.groupby(subject_col).size()
        repeat_subjects = sorted(counts[counts >= 2].index.tolist())
        if len(repeat_subjects) < 2:
            continue

        max_reps = int(counts[repeat_subjects].max())
        n_subj = len(repeat_subjects)
        arr = np.full((n_subj, max_reps, D), np.nan)

        for si, subj in enumerate(repeat_subjects):
            vals = sdf[sdf[subject_col] == subj][emb_cols].values
            arr[si, :vals.shape[0], :] = vals

        results.append((scanner, arr))

    return results


# ── High-level ICC functions ─────────────────────────────────────────────────

def compute_icc_between_scanner(df, emb_cols, subject_col="subject_id",
                                scanner_col="scanner"):
    """Compute ICC(2,1) for all embedding dimensions at once."""
    arr = pivot_between_scanner_3d(df, emb_cols, subject_col, scanner_col)
    return _icc21_batch(arr)


def compute_icc_within_scanner(df, emb_cols, subject_col="subject_id",
                               scanner_col="scanner", session_col="session_id"):
    """
    Compute ICC(3,1) for all dims, pooled across scanners with repeats.
    Weighted average by number of repeat subjects per scanner.
    """
    scanner_arrs = pivot_within_scanner_3d(df, emb_cols, subject_col, scanner_col)
    if not scanner_arrs:
        return np.full(len(emb_cols), np.nan)

    weighted_sum = np.zeros(len(emb_cols))
    total_weight = np.zeros(len(emb_cols))

    for scanner, arr in scanner_arrs:
        iccs = _icc31_batch(arr)
        w = arr.shape[0]  # number of repeat subjects
        valid = ~np.isnan(iccs)
        weighted_sum[valid] += iccs[valid] * w
        total_weight[valid] += w

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(total_weight > 0, weighted_sum / total_weight, np.nan)


# ── Variance decomposition (vectorized) ─────────────────────────────────────

def variance_decomposition(df, emb_cols, subject_col="subject_id",
                           scanner_col="scanner"):
    """
    Decompose variance into subject, scanner, and residual components.
    Vectorized over all dimensions.
    """
    subjects = sorted(df[subject_col].unique())
    scanners = sorted(df[scanner_col].unique())
    n_subj = len(subjects)
    n_scan = len(scanners)
    D = len(emb_cols)
    N = len(df)

    vals = df[emb_cols].values  # (N, D)
    grand_mean = np.nanmean(vals, axis=0)  # (D,)

    # Subject means and counts
    subj_map = df[subject_col].map({s: i for i, s in enumerate(subjects)}).values
    subj_sums = np.zeros((n_subj, D))
    subj_counts = np.zeros((n_subj, 1))
    for i in range(n_subj):
        mask = subj_map == i
        subj_sums[i] = vals[mask].sum(axis=0)
        subj_counts[i] = mask.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        subj_means = np.where(subj_counts > 0, subj_sums / subj_counts, 0)

    # Scanner means and counts
    scan_map = df[scanner_col].map({s: i for i, s in enumerate(scanners)}).values
    scan_sums = np.zeros((n_scan, D))
    scan_counts = np.zeros((n_scan, 1))
    for i in range(n_scan):
        mask = scan_map == i
        scan_sums[i] = vals[mask].sum(axis=0)
        scan_counts[i] = mask.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        scan_means = np.where(scan_counts > 0, scan_sums / scan_counts, 0)

    SS_subj = (subj_counts * (subj_means - grand_mean[np.newaxis, :]) ** 2).sum(axis=0)
    SS_scan = (scan_counts * (scan_means - grand_mean[np.newaxis, :]) ** 2).sum(axis=0)
    SS_total = ((vals - grand_mean[np.newaxis, :]) ** 2).sum(axis=0)
    SS_resid = SS_total - SS_subj - SS_scan

    df_subj = max(n_subj - 1, 1)
    df_scan = max(n_scan - 1, 1)
    df_resid = max(N - n_subj - n_scan + 1, 1)

    MS_subj = SS_subj / df_subj
    MS_scan = SS_scan / df_scan
    MS_resid = SS_resid / df_resid

    n_bar = N / n_subj
    k_bar = N / n_scan

    sigma2_resid = np.maximum(MS_resid, 0)
    sigma2_scan = np.maximum((MS_scan - MS_resid) / k_bar, 0)
    sigma2_subj = np.maximum((MS_subj - MS_resid) / n_bar, 0)

    total = sigma2_subj + sigma2_scan + sigma2_resid
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_subj = np.where(total > 0, sigma2_subj / total * 100, np.nan)
        pct_scan = np.where(total > 0, sigma2_scan / total * 100, np.nan)
        pct_resid = np.where(total > 0, sigma2_resid / total * 100, np.nan)

    return {
        "var_subject": sigma2_subj, "var_scanner": sigma2_scan,
        "var_residual": sigma2_resid,
        "pct_subject": pct_subj, "pct_scanner": pct_scan, "pct_residual": pct_resid,
    }


# ── Coefficient of Variation (vectorized) ────────────────────────────────────

def cov_within_scanner(df, emb_cols, subject_col="subject_id",
                       scanner_col="scanner"):
    """
    Within-scanner CoV per dimension: median across subject-scanner groups
    that have >=2 repeats.
    """
    D = len(emb_cols)
    vals = df[emb_cols].values
    groups = df.groupby([subject_col, scanner_col])

    all_covs = []
    for _, idx in groups.groups.items():
        grp = vals[idx]
        if len(grp) < 2:
            continue
        mu = grp.mean(axis=0)
        sd = grp.std(axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cov = np.where(np.abs(mu) > 1e-10, sd / np.abs(mu), np.nan)
        all_covs.append(cov)

    if not all_covs:
        return np.full(D, np.nan)
    return np.nanmedian(np.array(all_covs), axis=0)


def cov_between_scanner(df, emb_cols, subject_col="subject_id",
                        scanner_col="scanner"):
    """
    Between-scanner CoV per dimension: for each subject, std/|mean| across
    scanner means.
    """
    D = len(emb_cols)
    # Get subject-scanner means
    scanner_means = df.groupby([subject_col, scanner_col])[emb_cols].mean()

    all_covs = []
    for subj, grp in scanner_means.groupby(level=0):
        vals = grp.values
        if len(vals) < 2:
            continue
        mu = vals.mean(axis=0)
        sd = vals.std(axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cov = np.where(np.abs(mu) > 1e-10, sd / np.abs(mu), np.nan)
        all_covs.append(cov)

    if not all_covs:
        return np.full(D, np.nan)
    return np.nanmedian(np.array(all_covs), axis=0)


# ── Scanner fingerprinting ───────────────────────────────────────────────────

def _impute_nan(X):
    """Replace NaN with column median (for SVM/cosine which don't handle NaN)."""
    col_median = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X_clean = X.copy()
    X_clean[inds] = np.take(col_median, inds[1])
    return X_clean


def scanner_fingerprinting(df, emb_cols, scanner_col="scanner"):
    """
    Classify scanner identity from embeddings using LOO cross-validation
    with a linear SVM.
    """
    X = _impute_nan(df[emb_cols].values)
    y = df[scanner_col].values

    if len(np.unique(y)) < 2:
        return np.nan

    clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, dual="auto"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(clf, X, y, cv=LeaveOneOut())
    return float(np.mean(y_pred == y))


# ── Subject identification ───────────────────────────────────────────────────

def subject_identification(df, emb_cols, subject_col="subject_id",
                           scanner_col="scanner"):
    """
    For each scan, find the nearest neighbour (cosine similarity) among
    scans from a DIFFERENT scanner. Report proportion of correct matches.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    X = _impute_nan(df[emb_cols].values)
    subjects = df[subject_col].values
    scanners = df[scanner_col].values
    n = len(X)

    sim = cosine_similarity(X)
    # Mask out same-scanner entries
    scanner_match = scanners[:, None] == scanners[None, :]
    sim[scanner_match] = -np.inf
    np.fill_diagonal(sim, -np.inf)

    best_j = sim.argmax(axis=1)
    correct = (subjects[best_j] == subjects).sum()
    return float(correct / n)


# ── Bootstrap CIs ────────────────────────────────────────────────────────────

def bootstrap_icc(df, emb_cols, icc_func, subject_col="subject_id",
                  n_boot=1000, ci=0.95, seed=42, restrict_to_repeat=False,
                  scanner_col="scanner", **kwargs):
    """
    Bootstrap 95% CIs for ICC by resampling subjects.

    Parameters
    ----------
    restrict_to_repeat : bool
        If True, only resample among subjects that have within-scanner repeats
        (i.e., >=2 scans on the same scanner). Use this for within-scanner
        bootstrap to avoid degenerate samples.
    """
    rng = np.random.RandomState(seed)

    if restrict_to_repeat and scanner_col in df.columns:
        # Identify subjects with >=2 scans on at least one scanner
        counts = df.groupby([subject_col, scanner_col]).size().reset_index(name="n")
        repeat_subjs = counts[counts["n"] >= 2][subject_col].unique()
        boot_df_base = df[df[subject_col].isin(repeat_subjs)]
        subjects = boot_df_base[subject_col].unique()
        _print(f"       (bootstrapping among {len(subjects)} repeat subjects)")
    else:
        boot_df_base = df
        subjects = df[subject_col].unique()

    n_subj = len(subjects)
    subj_rows = {s: boot_df_base[boot_df_base[subject_col] == s] for s in subjects}

    boot_iccs = []
    import time as _time
    t0 = _time.time()
    for b in range(n_boot):
        boot_subj = rng.choice(subjects, size=n_subj, replace=True)
        frames = []
        for s_idx, s in enumerate(boot_subj):
            sdf = subj_rows[s].copy()
            sdf[subject_col] = f"{s}__b{s_idx}"
            frames.append(sdf)
        boot_df = pd.concat(frames, ignore_index=True)
        boot_icc = icc_func(boot_df, emb_cols, subject_col=subject_col,
                            scanner_col=scanner_col, **kwargs)
        boot_iccs.append(boot_icc)

        done = b + 1
        if done % 50 == 0 or done == n_boot:
            elapsed = _time.time() - t0
            rate = done / elapsed
            eta = (n_boot - done) / rate if rate > 0 else 0
            pct = done / n_boot * 100
            bar = "=" * int(pct // 5) + ">" + " " * (20 - int(pct // 5))
            _print(f"\r       [{bar}] {done}/{n_boot} ({pct:.0f}%) "
                   f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    boot_iccs = np.array(boot_iccs)
    alpha = (1 - ci) / 2
    lower = np.nanpercentile(boot_iccs, alpha * 100, axis=0)
    upper = np.nanpercentile(boot_iccs, (1 - alpha) * 100, axis=0)
    return lower, upper


# ── Summary statistics ───────────────────────────────────────────────────────

def icc_summary(iccs, label=""):
    """Return a dict with summary stats for an array of per-dimension ICCs."""
    valid = iccs[~np.isnan(iccs)]
    return {
        "label": label,
        "n_dims": len(iccs),
        "n_valid": len(valid),
        "median": float(np.median(valid)) if len(valid) > 0 else np.nan,
        "mean": float(np.mean(valid)) if len(valid) > 0 else np.nan,
        "q25": float(np.percentile(valid, 25)) if len(valid) > 0 else np.nan,
        "q75": float(np.percentile(valid, 75)) if len(valid) > 0 else np.nan,
        "pct_good": float(np.mean(valid >= 0.75) * 100) if len(valid) > 0 else np.nan,
        "pct_excellent": float(np.mean(valid >= 0.90) * 100) if len(valid) > 0 else np.nan,
        "pct_poor": float(np.mean(valid < 0.50) * 100) if len(valid) > 0 else np.nan,
    }


# ── Save results ─────────────────────────────────────────────────────────────

def save_results(results_dict, output_path):
    """Save results dict (may contain numpy arrays) to JSON."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    clean = {}
    for k, v in results_dict.items():
        if isinstance(v, dict):
            clean[k] = {kk: convert(vv) for kk, vv in v.items()}
        else:
            clean[k] = convert(v)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2, default=convert)
    _print(f"Results saved to {output_path}")


# ── Full analysis pipeline ───────────────────────────────────────────────────

def run_full_analysis(df, emb_cols, model_name, output_dir,
                      subject_col="subject_id", scanner_col="scanner",
                      session_col="session_id", n_boot=1000):
    """
    Run the complete reliability analysis for one model and save results.
    """
    import time as _time
    _print(f"\n{'='*60}")
    _print(f"  Reliability analysis: {model_name}")
    _print(f"  {len(df)} observations, {len(emb_cols)} dimensions")
    _print(f"  {df[subject_col].nunique()} subjects, {df[scanner_col].nunique()} scanners")
    _print(f"{'='*60}\n")
    t_start = _time.time()

    results = {"model": model_name, "n_obs": len(df), "n_dims": len(emb_cols),
               "n_subjects": int(df[subject_col].nunique()),
               "n_scanners": int(df[scanner_col].nunique())}

    # 1. Between-scanner ICC(2,1)
    _print("[1/7] Computing between-scanner ICC(2,1)...")
    t0 = _time.time()
    icc_between = compute_icc_between_scanner(df, emb_cols, subject_col, scanner_col)
    results["icc_between"] = icc_between
    results["icc_between_summary"] = icc_summary(icc_between, "ICC(2,1) between-scanner")
    _print(f"       Median ICC(2,1) = {results['icc_between_summary']['median']:.3f}  ({_time.time()-t0:.1f}s)")

    # 2. Within-scanner ICC(3,1)
    _print("[2/7] Computing within-scanner ICC(3,1)...")
    t0 = _time.time()
    icc_within = compute_icc_within_scanner(df, emb_cols, subject_col, scanner_col, session_col)
    results["icc_within"] = icc_within
    results["icc_within_summary"] = icc_summary(icc_within, "ICC(3,1) within-scanner")
    _print(f"       Median ICC(3,1) = {results['icc_within_summary']['median']:.3f}  ({_time.time()-t0:.1f}s)")

    # 3. Bootstrap CIs
    _print(f"[3/7] Bootstrapping between-scanner CIs ({n_boot} resamples)...")
    t0 = _time.time()
    lo_b, hi_b = bootstrap_icc(df, emb_cols, compute_icc_between_scanner,
                               subject_col=subject_col, n_boot=n_boot,
                               scanner_col=scanner_col)
    results["icc_between_ci_lower"] = lo_b
    results["icc_between_ci_upper"] = hi_b
    _print(f"\n       Median ICC(2,1) 95% CI: [{np.nanmedian(lo_b):.3f}, {np.nanmedian(hi_b):.3f}]  ({_time.time()-t0:.1f}s)")

    _print(f"[3b/7] Bootstrapping within-scanner CIs ({n_boot} resamples)...")
    t0 = _time.time()
    lo_w, hi_w = bootstrap_icc(df, emb_cols, compute_icc_within_scanner,
                               subject_col=subject_col, n_boot=n_boot,
                               restrict_to_repeat=True,
                               scanner_col=scanner_col, session_col=session_col)
    results["icc_within_ci_lower"] = lo_w
    results["icc_within_ci_upper"] = hi_w
    _print(f"\n       Median ICC(3,1) 95% CI: [{np.nanmedian(lo_w):.3f}, {np.nanmedian(hi_w):.3f}]  ({_time.time()-t0:.1f}s)")

    # 4. Variance decomposition
    _print("[4/7] Variance decomposition...")
    t0 = _time.time()
    vardecomp = variance_decomposition(df, emb_cols, subject_col, scanner_col)
    results["variance_decomposition"] = vardecomp
    _print(f"       Mean %subject = {np.nanmean(vardecomp['pct_subject']):.1f}%, "
          f"%scanner = {np.nanmean(vardecomp['pct_scanner']):.1f}%, "
          f"%residual = {np.nanmean(vardecomp['pct_residual']):.1f}%  ({_time.time()-t0:.1f}s)")

    # Store aggregated variance proportions
    results["var_prop_subject"] = float(np.nanmean(vardecomp["pct_subject"]))
    results["var_prop_scanner"] = float(np.nanmean(vardecomp["pct_scanner"]))
    results["var_prop_residual"] = float(np.nanmean(vardecomp["pct_residual"]))

    # 5. Coefficient of Variation
    _print("[5/7] Coefficient of Variation...")
    t0 = _time.time()
    cov_w = cov_within_scanner(df, emb_cols, subject_col, scanner_col)
    cov_b = cov_between_scanner(df, emb_cols, subject_col, scanner_col)
    results["cov_within"] = cov_w
    results["cov_between"] = cov_b
    _print(f"       Median CoV within = {np.nanmedian(cov_w):.4f}, "
          f"between = {np.nanmedian(cov_b):.4f}  ({_time.time()-t0:.1f}s)")

    # 6. Scanner fingerprinting
    _print("[6/7] Scanner fingerprinting (LOO-SVM)...")
    t0 = _time.time()
    fp_acc = scanner_fingerprinting(df, emb_cols, scanner_col)
    results["scanner_fingerprint_accuracy"] = fp_acc
    _print(f"       Accuracy = {fp_acc:.1%}  ({_time.time()-t0:.1f}s)")

    # 7. Subject identification
    _print("[7/7] Subject identification (cross-scanner NN)...")
    t0 = _time.time()
    sid_acc = subject_identification(df, emb_cols, subject_col, scanner_col)
    results["subject_identification_accuracy"] = sid_acc
    _print(f"       Accuracy = {sid_acc:.1%}  ({_time.time()-t0:.1f}s)")

    # Save
    out_path = os.path.join(output_dir, f"results_{model_name}.json")
    save_results(results, out_path)
    _print(f"\nDone. Total time: {_time.time()-t_start:.1f}s. Results saved to {out_path}")
    return results
