#!/usr/bin/env python3
"""
Generate supplementary figures for the FM reliability paper.

Produces:
  S1. figS1_icc_histograms.pdf    — Per-dimension ICC histograms for all 6 models
  S2. figS2_pca_structure.pdf     — PCA variance explained + top-2 PC scatter
  S3. figS3_scanner_heatmaps.pdf  — Per-scanner pairwise cosine distance heatmaps
  S4. figS4_sex_classification.pdf — Cross-scanner sex classification accuracy
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")
DATA_DIR = os.path.join(PROJECT_DIR, "DATA")
OUTPUT_DIR = SCRIPT_DIR

# ── Model definitions (same as main figures) ──────────────────────────────────
MODELS = [
    ("FreeSurfer",      "results_FreeSurfer.json"),
    ("AnatCL",          "results_AnatCL_fold0.json"),
    ("y-Aware",         "results_yAware.json"),
    ("BrainIAC",        "results_BrainIAC.json"),
    ("BrainSegFounder", "results_BrainSegFounder.json"),
    ("3D-SimCLR",       "results_3DNeuroSimCLR.json"),
]

COLORS = {
    "FreeSurfer":      "#6c757d",
    "AnatCL":          "#198754",
    "y-Aware":         "#0d6efd",
    "BrainIAC":        "#fd7e14",
    "3D-SimCLR":       "#6f42c1",
    "BrainSegFounder": "#dc3545",
}

SCANNER_VENDOR = {
    "OXF1PRI": "Siemens", "OXF2PRI": "Siemens", "OXF3TRI": "Siemens",
    "NOT1ACH": "Philips", "NOT2ING": "Philips",
    "NOT3GEM": "GE", "NOT4GEP": "GE", "OXF4GEP": "GE",
}

SCANNER_SHORT = {
    "NOT1ACH": "Phi-Ach", "NOT2ING": "Phi-Ing",
    "NOT3GEM": "GE-750", "NOT4GEP": "GE-Pre(N)",
    "OXF1PRI": "Si-Pri32", "OXF2PRI": "Si-Pri64",
    "OXF3TRI": "Si-Trio", "OXF4GEP": "GE-Pre(O)",
}

VENDOR_COLORS = {"Siemens": "#0d6efd", "Philips": "#fd7e14", "GE": "#198754"}

EMB_FILES = {
    "FreeSurfer":      "FreeSurfer_IDPs_ON-Harmony.csv",
    "AnatCL":          "embeddings_AnatCL_fold0.csv",
    "y-Aware":         "embeddings_yAware.csv",
    "BrainIAC":        "embeddings_BrainIAC.csv",
    "3D-SimCLR":       "embeddings_3DNeuroSimCLR.csv",
    "BrainSegFounder": "embeddings_BrainSegFounder.csv",
}

# ── Global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_results():
    results = {}
    for name, fname in MODELS:
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            results[name] = json.load(f)
    return results


def load_supplementary_results():
    path = os.path.join(RESULTS_DIR, "supplementary_results.json")
    with open(path) as f:
        return json.load(f)


def load_embeddings(model_name):
    """Load embedding DataFrame. Returns (df, emb_cols)."""
    path = os.path.join(DATA_DIR, EMB_FILES[model_name])
    df = pd.read_csv(path)
    if model_name == "FreeSurfer":
        emb_cols = [c for c in df.columns if c.startswith("freesurfer_")]
        # Drop NaN columns
        emb_cols = [c for c in emb_cols if df[c].notna().all()]
        if "subject" in df.columns:
            df = df.rename(columns={"subject": "subject_id"})
        if "scanner_typeloc" in df.columns:
            df = df.rename(columns={"scanner_typeloc": "scanner"})
    else:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
    return df, emb_cols


# ── Figure S1: Per-dimension ICC histograms ───────────────────────────────────

def fig_s1_icc_histograms(results):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True)
    axes = axes.ravel()
    model_names = [n for n, _ in MODELS]

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        icc_vals = np.array(results[name]["icc_between"])
        icc_vals = icc_vals[~np.isnan(icc_vals)]

        ax.hist(icc_vals, bins=50, color=COLORS[name], alpha=0.8, edgecolor="white",
                linewidth=0.3)

        # ICC threshold lines
        for thresh, label in [(0.50, ""), (0.75, ""), (0.90, "")]:
            ax.axvline(thresh, color="#999999", linestyle="--", linewidth=0.7, alpha=0.7)

        median = np.nanmedian(icc_vals)
        ax.axvline(median, color="black", linewidth=1.2, linestyle="-")

        ax.set_title(f"{name}", fontsize=10, fontweight="bold", color=COLORS[name])
        ax.text(0.97, 0.92, f"Mdn = {median:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, fontweight="bold")
        ax.text(0.97, 0.80, f"n = {len(icc_vals)} dims", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#666666")

        if idx >= 3:
            ax.set_xlabel("ICC(2,1)")
        if idx % 3 == 0:
            ax.set_ylabel("Count")

    axes[0].set_xlim(-0.5, 1.05)
    fig.suptitle("Per-Dimension Between-Scanner ICC(2,1) Distributions",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── Figure S2: PCA structure ──────────────────────────────────────────────────

def fig_s2_pca_structure(supp_results):
    pca_data = supp_results["pca"]
    model_names = [n for n, _ in MODELS]

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel (a): Cumulative variance explained curves (top row, full width)
    ax_cum = fig.add_subplot(gs[0, :])
    for name in model_names:
        evr = pca_data[name]["explained_variance_ratio"]
        cum = np.cumsum(evr)
        ax_cum.plot(range(1, len(cum) + 1), cum * 100, color=COLORS[name],
                    linewidth=1.5, label=name)

    ax_cum.axhline(90, color="#999999", linestyle="--", linewidth=0.7, alpha=0.7)
    ax_cum.axhline(95, color="#999999", linestyle=":", linewidth=0.7, alpha=0.7)
    ax_cum.text(48, 91, "90%", fontsize=7, color="#666666")
    ax_cum.text(48, 96, "95%", fontsize=7, color="#666666")
    ax_cum.set_xlabel("Number of principal components")
    ax_cum.set_ylabel("Cumulative variance explained (%)")
    ax_cum.set_xlim(1, 50)
    ax_cum.set_ylim(0, 102)
    ax_cum.legend(loc="lower right", ncol=2, fontsize=7)
    ax_cum.set_title("(a) Cumulative variance explained", fontsize=10, fontweight="bold",
                     loc="left")

    # Panels (b): Top-2 PC scatter for each model, colored by scanner
    show_models = model_names
    for i, name in enumerate(show_models):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        df, emb_cols = load_embeddings(name)
        X = StandardScaler().fit_transform(df[emb_cols].values)
        pca_obj = PCA(n_components=min(5, X.shape[1]))
        X_pca = pca_obj.fit_transform(X)

        scanners = df["scanner"].values
        for sc in sorted(set(scanners)):
            vendor = SCANNER_VENDOR.get(sc, "Unknown")
            mask = scanners == sc
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=VENDOR_COLORS.get(vendor, "gray"),
                       s=12, alpha=0.7, edgecolors="white", linewidths=0.3,
                       label=sc if i == 0 else "")

        # Mark dominant signal
        pc1_info = pca_data[name]["pc_structure"][0]
        dom = pc1_info["dominant"]
        dom_color = "#dc3545" if dom == "scanner" else "#198754"
        ax.set_title(f"{name}", fontsize=9, fontweight="bold", color=COLORS[name])
        pct1 = pca_obj.explained_variance_ratio_[0] * 100
        pct2 = pca_obj.explained_variance_ratio_[1] * 100
        ax.set_xlabel(f"PC1 ({pct1:.0f}%, {dom})", fontsize=7)
        ax.set_ylabel(f"PC2 ({pct2:.0f}%)", fontsize=7)
        ax.tick_params(labelsize=6)

    # Legend for vendors
    handles = [mpatches.Patch(color=c, label=v) for v, c in VENDOR_COLORS.items()]
    fig.legend(handles=handles, loc="lower right", ncol=3, fontsize=7,
               bbox_to_anchor=(0.98, 0.01))

    fig.suptitle("", fontsize=1)  # Prevent overlap with title
    return fig


# ── Figure S3: Per-scanner pairwise cosine distance heatmaps ─────────────────

def fig_s3_scanner_heatmaps(supp_results):
    """Per-scanner pairwise similarity using Euclidean distance on z-scored embeddings."""
    model_names = [n for n, _ in MODELS]

    # ── Pass 1: compute all distance matrices and find global vmax ────────
    all_dist = {}
    all_scanners = {}
    global_vmax = 0.0
    for name in model_names:
        df, emb_cols = load_embeddings(name)
        scanners = sorted(df["scanner"].unique())
        subjects = sorted(df["subject_id"].unique())
        n_sc = len(scanners)
        sc_idx = {s: i for i, s in enumerate(scanners)}

        X = StandardScaler().fit_transform(df[emb_cols].values)
        df_z = df[["subject_id", "scanner"]].copy()
        for i, c in enumerate(emb_cols):
            df_z[c] = X[:, i]

        dist_matrix = np.full((n_sc, n_sc), np.nan)
        for subj in subjects:
            subj_df = df_z[df_z["subject_id"] == subj]
            for sc_a in scanners:
                rows_a = subj_df[subj_df["scanner"] == sc_a][emb_cols].values
                if len(rows_a) == 0:
                    continue
                mean_a = rows_a.mean(axis=0)
                for sc_b in scanners:
                    rows_b = subj_df[subj_df["scanner"] == sc_b][emb_cols].values
                    if len(rows_b) == 0:
                        continue
                    mean_b = rows_b.mean(axis=0)
                    d = np.linalg.norm(mean_a - mean_b)
                    i, j = sc_idx[sc_a], sc_idx[sc_b]
                    if np.isnan(dist_matrix[i, j]):
                        dist_matrix[i, j] = 0.0
                    dist_matrix[i, j] += d

        count_matrix = np.zeros((n_sc, n_sc))
        for subj in subjects:
            subj_scanners = set(df[df["subject_id"] == subj]["scanner"].unique())
            for sc_a in subj_scanners:
                for sc_b in subj_scanners:
                    count_matrix[sc_idx[sc_a], sc_idx[sc_b]] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            dist_matrix = np.where(count_matrix > 0, dist_matrix / count_matrix, np.nan)

        off_diag = dist_matrix.copy()
        np.fill_diagonal(off_diag, np.nan)
        local_max = np.nanmax(off_diag) if not np.all(np.isnan(off_diag)) else 0.0
        global_vmax = max(global_vmax, local_max)

        all_dist[name] = dist_matrix
        all_scanners[name] = scanners

    if global_vmax == 0.0:
        global_vmax = 1.0

    # ── Pass 2: plot with shared color scale ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    axes = axes.ravel()

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        dist_matrix = all_dist[name]
        scanners = all_scanners[name]
        n_sc = len(scanners)
        labels = [SCANNER_SHORT.get(s, s) for s in scanners]

        im = ax.imshow(dist_matrix, cmap="YlOrRd", vmin=0, vmax=global_vmax, aspect="equal")

        ax.set_xticks(range(n_sc))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(n_sc))
        ax.set_yticklabels(labels, fontsize=6)

        # Color scanner labels by vendor
        for tick_idx, sc in enumerate(scanners):
            vendor = SCANNER_VENDOR.get(sc, "Unknown")
            color = VENDOR_COLORS.get(vendor, "black")
            ax.get_xticklabels()[tick_idx].set_color(color)
            ax.get_yticklabels()[tick_idx].set_color(color)

        ax.set_title(f"{name}", fontsize=10, fontweight="bold", color=COLORS[name])

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle("Per-Scanner Pairwise Euclidean Distance (z-scored embeddings, same-subject pairs)",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── Figure S4: Sex classification accuracy ────────────────────────────────────

def fig_s4_sex_classification(supp_results):
    sex_data = supp_results["sex_classification"]
    model_names = [n for n, _ in MODELS]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    x = np.arange(len(model_names))
    width = 0.55

    accs = []
    bal_accs = []
    for name in model_names:
        if name in sex_data:
            accs.append(sex_data[name]["accuracy"] * 100)
            bal_accs.append(sex_data[name]["balanced_accuracy"] * 100)
        else:
            accs.append(0)
            bal_accs.append(0)

    bars = ax.bar(x, bal_accs, width, color=[COLORS[n] for n in model_names],
                  edgecolor="white", linewidth=0.5)

    # Add value labels
    for i, (bar, acc, bacc) in enumerate(zip(bars, accs, bal_accs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bacc:.0f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=COLORS[model_names[i]])

    # Chance level (50% for binary)
    ax.axhline(50, color="#999999", linestyle="--", linewidth=0.7)
    ax.text(5.5, 51, "Chance", fontsize=7, color="#999999")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=8)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Cross-Scanner Sex Classification (Leave-One-Scanner-Out)",
                 fontsize=10, fontweight="bold", loc="left")

    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    results = load_results()
    supp_results = load_supplementary_results()
    model_names = [n for n, _ in MODELS]
    print(f"  Loaded {len(results)} models")

    print("\nGenerating supplementary figures...")

    # S1: ICC histograms
    fig = fig_s1_icc_histograms(results)
    path = os.path.join(OUTPUT_DIR, "figS1_icc_histograms.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    # S2: PCA structure
    fig = fig_s2_pca_structure(supp_results)
    path = os.path.join(OUTPUT_DIR, "figS2_pca_structure.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    # S3: Scanner heatmaps
    fig = fig_s3_scanner_heatmaps(supp_results)
    path = os.path.join(OUTPUT_DIR, "figS3_scanner_heatmaps.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    # S4: Sex classification
    fig = fig_s4_sex_classification(supp_results)
    path = os.path.join(OUTPUT_DIR, "figS4_sex_classification.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    print(f"\nDone. All supplementary figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
