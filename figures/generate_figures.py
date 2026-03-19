#!/usr/bin/env python3
"""
Generate publication-quality composite figures for the FM reliability paper.

Produces 4 PDF figures:
  1. fig1_icc_overview.pdf        — ICC violin distributions
  2. fig2_variance_and_within.pdf — (a) within vs between ICC bars + (b) variance decomposition
  3. fig3_fingerprint.pdf         — Scanner fingerprint vs subject identification scatter
  4. fig4_tsne.pdf                — t-SNE projections colored by subject / vendor
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")
DATA_DIR = os.path.join(PROJECT_DIR, "DATA")
OUTPUT_DIR = SCRIPT_DIR

# ── Model definitions ──────────────────────────────────────────────────────────
MODELS = [
    ("FreeSurfer",      "results_FreeSurfer.json"),
    ("AnatCL",          "results_AnatCL_fold0.json"),
    ("y-Aware",         "results_yAware.json"),
    ("BrainIAC",        "results_BrainIAC.json"),
    ("BrainSegFounder", "results_BrainSegFounder.json"),
    ("3D-SimCLR",       "results_3DNeuroSimCLR.json"),
]

# Refined, muted color palette — colorblind-friendly
COLORS = {
    "FreeSurfer":      "#6c757d",  # slate gray
    "AnatCL":          "#198754",  # forest green
    "y-Aware":         "#0d6efd",  # royal blue
    "BrainIAC":        "#fd7e14",  # warm orange
    "3D-SimCLR":       "#6f42c1",  # deep purple
    "BrainSegFounder": "#dc3545",  # crimson
}

REFERENCE = "FreeSurfer"

# Scanner → vendor mapping
SCANNER_VENDOR = {
    "OXF1PRI": "Siemens", "OXF2PRI": "Siemens", "OXF3TRI": "Siemens",
    "NOT1ACH": "Philips", "NOT2ING": "Philips",
    "NOT3GEM": "GE", "NOT4GEP": "GE", "OXF4GEP": "GE",
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

# ── Matplotlib defaults — clean publication style ────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "legend.fontsize":    7.5,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_results():
    data = {}
    for name, fname in MODELS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        with open(path) as f:
            data[name] = json.load(f)
    return data


def sort_models(models, key_fn, reverse=False):
    """Sort models with FreeSurfer always first, rest by key_fn."""
    others = [m for m in models if m != REFERENCE]
    others.sort(key=key_fn, reverse=reverse)
    order = [REFERENCE] if REFERENCE in models else []
    order.extend(others)
    return order


def icc_order(data):
    """FreeSurfer first, then by between-scanner ICC descending."""
    return sort_models(data.keys(),
                       key_fn=lambda m: data[m]["icc_between_summary"]["median"],
                       reverse=True)


def panel_label(ax, label, x=-0.08, y=1.08):
    """Add bold (a)/(b) panel label."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="right")


def add_icc_bands(ax, orientation="horizontal"):
    """Add subtle colored ICC threshold bands."""
    bands = [
        (-0.1, 0.50, "#fee2e2", "Poor"),        # light red
        (0.50, 0.75, "#fef9c3", "Moderate"),    # light yellow
        (0.75, 0.90, "#dcfce7", "Good"),         # light green
        (0.90, 1.05, "#86efac", "Excellent"),    # clearer green
    ]
    for lo, hi, color, _ in bands:
        if orientation == "horizontal":
            ax.axhspan(lo, hi, color=color, alpha=0.25, zorder=0, linewidth=0)
        else:
            ax.axvspan(lo, hi, color=color, alpha=0.25, zorder=0, linewidth=0)


def load_embeddings(model_name):
    """Load embedding CSV and return (X, subject_ids, scanner_codes)."""
    path = os.path.join(DATA_DIR, EMB_FILES[model_name])
    df = pd.read_csv(path)

    if model_name == "FreeSurfer":
        scanner_col, subject_col = "scanner_typeloc", "subject"
        dkt_cols = [c for c in df.columns
                    if "freesurfer_" in c and "DKTatlas" in c
                    and any(m in c for m in ["ThickAvg", "SurfArea", "GrayVol"])]
        aseg_cols = [c for c in df.columns
                     if c.startswith("freesurfer_aseg_Volume_mm3")]
        feat_cols = sorted(dkt_cols) + sorted(aseg_cols)
        X = df[feat_cols].apply(pd.to_numeric, errors="coerce").values
    else:
        scanner_col, subject_col = "scanner", "subject_id"
        feat_cols = [c for c in df.columns if c.startswith("emb_")]
        X = df[feat_cols].values

    return X, df[subject_col].astype(str).values, df[scanner_col].astype(str).values


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: ICC Violin Distributions
# ══════════════════════════════════════════════════════════════════════════════

def fig1_icc_overview(data):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    order = icc_order(data)
    add_icc_bands(ax, "horizontal")

    plot_data, labels = [], []
    for m in order:
        vals = np.array(data[m]["icc_between"], dtype=float)
        vals = vals[np.isfinite(vals)]
        plot_data.append(vals)
        labels.append(m)

    positions = np.arange(len(order))
    parts = ax.violinplot(plot_data, positions=positions,
                          showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[order[i]])
        pc.set_edgecolor(COLORS[order[i]])
        pc.set_linewidth(0.8)
        pc.set_alpha(0.45)

    # Overlay box plots (slim, clean)
    bp = ax.boxplot(plot_data, positions=positions, widths=0.12,
                    showfliers=False, patch_artist=True, zorder=3,
                    medianprops=dict(color="white", linewidth=1.5),
                    boxprops=dict(facecolor="black", edgecolor="black",
                                  linewidth=0.6, alpha=0.7),
                    whiskerprops=dict(color="black", linewidth=0.6),
                    capprops=dict(color="black", linewidth=0.6))

    # Median annotation at the bottom
    for i, m in enumerate(order):
        med = data[m]["icc_between_summary"]["median"]
        ax.text(positions[i], -0.02, f"{med:.3f}",
                ha="center", va="top", fontsize=7.5, fontweight="bold",
                color=COLORS[m])

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("ICC(2,1) between-scanner")
    ax.set_ylim(-0.1, 1.05)

    # Custom legend for ICC bands
    band_patches = [
        mpatches.Patch(facecolor="#fee2e2", alpha=0.4, label="Poor (<0.50)"),
        mpatches.Patch(facecolor="#fef9c3", alpha=0.4, label="Moderate"),
        mpatches.Patch(facecolor="#dcfce7", alpha=0.4, label="Good (0.75–0.90)"),
        mpatches.Patch(facecolor="#86efac", alpha=0.4, label="Excellent (>0.90)"),
    ]
    ax.legend(handles=band_patches, loc="upper right", fontsize=7,
              handlelength=1.2, handletextpad=0.4)

    out = os.path.join(OUTPUT_DIR, "fig1_icc_overview.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Variance & Within/Between — (a) Grouped bars + (b) Stacked bars
# ══════════════════════════════════════════════════════════════════════════════

def fig2_variance_and_within(data):
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.38)

    order = icc_order(data)

    # ── Panel (a): Within vs Between ICC bars ────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    panel_label(ax_a, "(a)")

    x = np.arange(len(order))
    width = 0.32

    between_vals = [data[m]["icc_between_summary"]["median"] for m in order]
    within_vals = [data[m]["icc_within_summary"]["median"] for m in order]

    # Between bars (solid, full color)
    for i, m in enumerate(order):
        ax_a.bar(x[i] - width / 2, between_vals[i], width,
                 color=COLORS[m], edgecolor="white", linewidth=0.5, zorder=2)
    # Within bars (lighter shade, no hatching)
    for i, m in enumerate(order):
        ax_a.bar(x[i] + width / 2, within_vals[i], width,
                 color=COLORS[m], edgecolor="white", linewidth=0.5,
                 alpha=0.35, zorder=2)

    # Value labels (only on top)
    for i in range(len(order)):
        bv, wv = between_vals[i], within_vals[i]
        ax_a.text(x[i] - width / 2, max(bv, 0) + 0.025, f"{bv:.2f}",
                  ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        ax_a.text(x[i] + width / 2, max(wv, 0) + 0.025, f"{wv:.2f}",
                  ha="center", va="bottom", fontsize=6.5, color="#555555")

    # Threshold lines
    for thresh, ls in [(0.75, (0, (5, 5))), (0.50, (0, (3, 3)))]:
        ax_a.axhline(thresh, color="#aaaaaa", linestyle=ls, linewidth=0.6,
                     zorder=1)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(order, fontsize=8.5)
    ax_a.set_ylabel("Median ICC")
    ax_a.set_ylim(-0.08, 1.15)

    # Clean legend
    legend_handles = [
        mpatches.Patch(facecolor="#333333", label="Between-scanner ICC(2,1)"),
        mpatches.Patch(facecolor="#333333", alpha=0.35,
                       label="Within-scanner ICC(3,1)"),
    ]
    ax_a.legend(handles=legend_handles, loc="upper right", fontsize=7.5)

    # ── Panel (b): Variance decomposition stacked bars ───────────────────────
    ax_b = fig.add_subplot(gs[1])
    panel_label(ax_b, "(b)")

    # Build model list: FreeSurfer (harmonized) first, then FreeSurfer,
    # then FMs by subject variance descending — all reversed for bottom-up plot
    harm_path = os.path.join(RESULTS_DIR, "results_FreeSurfer_harmonized.json")
    harm_data = None
    if os.path.exists(harm_path):
        with open(harm_path) as f:
            harm_data = json.load(f)

    fm_models = sort_models(
        data.keys(),
        key_fn=lambda m: np.nanmean(data[m]["variance_decomposition"]["pct_subject"]),
        reverse=True
    )
    # Build ordered list: harmonized first, then the rest (reversed for barh)
    models_sorted = list(reversed(fm_models))
    if harm_data:
        models_sorted.append("FreeSurfer\n(harmonized)")

    y_pos = np.arange(len(models_sorted))
    subj_vals, scan_vals, resid_vals = [], [], []
    for m in models_sorted:
        if m == "FreeSurfer\n(harmonized)":
            vd = harm_data["variance_decomposition"]
        else:
            vd = data[m]["variance_decomposition"]
        subj_vals.append(np.nanmean(vd["pct_subject"]))
        scan_vals.append(np.nanmean(vd["pct_scanner"]))
        resid_vals.append(np.nanmean(vd["pct_residual"]))

    subj_vals = np.array(subj_vals)
    scan_vals = np.array(scan_vals)
    resid_vals = np.array(resid_vals)

    bar_h = 0.55
    ax_b.barh(y_pos, subj_vals, height=bar_h, color="#198754",
              label="Subject", edgecolor="white", linewidth=0.3)
    ax_b.barh(y_pos, scan_vals, height=bar_h, left=subj_vals,
              color="#dc3545", label="Scanner", edgecolor="white",
              linewidth=0.3)
    ax_b.barh(y_pos, resid_vals, height=bar_h, left=subj_vals + scan_vals,
              color="#adb5bd", label="Residual", edgecolor="white",
              linewidth=0.3)

    # Percentage labels inside bars
    for i in range(len(models_sorted)):
        # Subject label
        if subj_vals[i] > 14:
            ax_b.text(subj_vals[i] / 2, y_pos[i], f"{subj_vals[i]:.1f}%",
                      ha="center", va="center", fontsize=7, color="white",
                      fontweight="bold")
        # Scanner label
        if scan_vals[i] > 14:
            ax_b.text(subj_vals[i] + scan_vals[i] / 2, y_pos[i],
                      f"{scan_vals[i]:.1f}%",
                      ha="center", va="center", fontsize=7, color="white",
                      fontweight="bold")
        # Residual label
        if resid_vals[i] > 14:
            ax_b.text(subj_vals[i] + scan_vals[i] + resid_vals[i] / 2,
                      y_pos[i], f"{resid_vals[i]:.1f}%",
                      ha="center", va="center", fontsize=7, color="white",
                      fontweight="bold")

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(models_sorted, fontsize=8.5)
    ax_b.set_xlabel("Variance proportion (%)")
    ax_b.set_xlim(0, 100)
    ax_b.legend(loc="lower right", fontsize=7.5, ncol=3,
                handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
    ax_b.xaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))

    out = os.path.join(OUTPUT_DIR, "fig2_variance_and_within.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Scanner Fingerprint vs Subject Identification (standalone)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_fingerprint(data):
    fig, ax = plt.subplots(figsize=(5.2, 4.5))

    order = icc_order(data)

    # Quadrant shading
    ax.axhspan(50, 110, xmin=0, xmax=0.5, color="#dcfce7", alpha=0.15, zorder=0)
    ax.axhspan(0, 50, xmin=0.5, xmax=1.0, color="#fee2e2", alpha=0.15, zorder=0)

    # Chance level lines
    ax.axhline(100 / 20, color="#cccccc", linestyle="--", linewidth=0.8,
               zorder=1)
    ax.axvline(100 / 8, color="#cccccc", linestyle="--", linewidth=0.8,
               zorder=1)

    # Annotate chance
    ax.text(100 / 8 + 1.5, 104, "Chance\n(scanner)", fontsize=6, color="#999999",
            va="top", ha="left")
    ax.text(95, 100 / 20 + 1, "Chance (subject)", fontsize=6, color="#999999",
            va="bottom", ha="right")

    # Manual offsets to avoid overlap
    offsets = {
        "FreeSurfer":      (8, -6),
        "AnatCL":          (-2, 5),
        "y-Aware":         (8, 3),
        "BrainIAC":        (8, 4),
        "3D-SimCLR":       (-2, -10),
        "BrainSegFounder": (-2, -10),
    }
    align = {
        "AnatCL": "right", "3D-SimCLR": "right", "BrainSegFounder": "right",
    }

    for m in order:
        x = data[m]["scanner_fingerprint_accuracy"] * 100
        y = data[m]["subject_identification_accuracy"] * 100
        ax.scatter(x, y, s=140, color=COLORS[m], edgecolors="white",
                   linewidths=0.8, zorder=3)
        dx, dy = offsets.get(m, (8, 0))
        ha = align.get(m, "left")
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5, fontweight="bold", ha=ha, color=COLORS[m])

    # ── Harmonized FreeSurfer reference point (scanner-variance lower bound) ──
    harm_path = os.path.join(RESULTS_DIR, "results_FreeSurfer_harmonized.json")
    if os.path.exists(harm_path):
        with open(harm_path) as f:
            harm = json.load(f)
        hx = harm["scanner_fingerprint_accuracy"] * 100
        hy = harm["subject_identification_accuracy"] * 100
        ax.scatter(hx, hy, s=120, color=COLORS["FreeSurfer"], marker="D",
                   edgecolors="black", linewidths=0.8, zorder=4)
        ax.annotate("FreeSurfer\n(harmonized)", (hx, hy),
                    textcoords="offset points", xytext=(0, -18),
                    fontsize=6.5, fontstyle="italic", ha="center",
                    color=COLORS["FreeSurfer"])

    ax.set_xlabel("Scanner fingerprint accuracy (%)")
    ax.set_ylabel("Subject identification accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110)

    out = os.path.join(OUTPUT_DIR, "fig3_fingerprint.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: t-SNE Visualization — colored by subject (top) and vendor (bottom)
# ══════════════════════════════════════════════════════════════════════════════

def fig4_tsne(data):
    order = icc_order(data)  # all 6 models, sorted by ICC descending
    # Split into two groups of 3 for a 4-row x 3-col layout
    group1 = order[:3]   # top-performing
    group2 = order[3:]   # bottom-performing

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.5))

    subject_cmap = plt.cm.tab20

    for group_idx, group in enumerate([group1, group2]):
        for col, model in enumerate(group):
            X, subjects, scanners = load_embeddings(model)

            # Handle NaN: drop all-NaN columns, impute remaining with median
            col_mask = ~np.all(np.isnan(X), axis=0)
            X = X[:, col_mask]
            col_medians = np.nanmedian(X, axis=0)
            nan_inds = np.where(np.isnan(X))
            X[nan_inds] = np.take(col_medians, nan_inds[1])

            X_std = StandardScaler().fit_transform(X)
            tsne = TSNE(n_components=2, perplexity=min(30, len(X) - 1),
                         random_state=42, init="pca", learning_rate="auto")
            Z = tsne.fit_transform(X_std)

            unique_subj = sorted(set(subjects))
            subj_colors = {s: subject_cmap(i / len(unique_subj))
                           for i, s in enumerate(unique_subj)}

            # Subject row (rows 0, 2)
            row_s = group_idx * 2
            ax_s = axes[row_s, col]
            for s in unique_subj:
                mask = subjects == s
                ax_s.scatter(Z[mask, 0], Z[mask, 1], c=[subj_colors[s]],
                             s=30, alpha=0.85, edgecolors="white",
                             linewidths=0.2, zorder=2)

            icc_val = data[model]["icc_between_summary"]["median"]
            ax_s.set_title(f"{model}  (ICC = {icc_val:.3f})", fontsize=8, pad=4)
            ax_s.set_xticks([])
            ax_s.set_yticks([])
            for spine in ax_s.spines.values():
                spine.set_visible(True)
                spine.set_color("#dddddd")
                spine.set_linewidth(0.5)
            if col == 0:
                ax_s.set_ylabel("By subject", fontsize=8, fontweight="bold")

            # Vendor row (rows 1, 3)
            row_v = group_idx * 2 + 1
            ax_v = axes[row_v, col]
            vendors = np.array([SCANNER_VENDOR.get(s, "Unknown") for s in scanners])
            for v in ["Siemens", "Philips", "GE"]:
                mask = vendors == v
                if mask.any():
                    ax_v.scatter(Z[mask, 0], Z[mask, 1], c=VENDOR_COLORS[v],
                                 s=30, alpha=0.85, edgecolors="white",
                                 linewidths=0.2, zorder=2, label=v)

            ax_v.set_xticks([])
            ax_v.set_yticks([])
            for spine in ax_v.spines.values():
                spine.set_visible(True)
                spine.set_color("#dddddd")
                spine.set_linewidth(0.5)
            if col == 0:
                ax_v.set_ylabel("By vendor", fontsize=8, fontweight="bold")

    # Vendor legend on bottom-right panel
    vendor_handles = [Line2D([0], [0], marker="o", color="w",
                             markerfacecolor=VENDOR_COLORS[v], markersize=6,
                             markeredgecolor="white", markeredgewidth=0.3,
                             label=v) for v in ["Siemens", "Philips", "GE"]]
    axes[3, 2].legend(handles=vendor_handles, loc="lower right", fontsize=7,
                      handletextpad=0.3, borderpad=0.4)

    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    out = os.path.join(OUTPUT_DIR, "fig4_tsne.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    data = load_results()
    if not data:
        print("ERROR: No results files found.", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(data)} models: {list(data.keys())}")

    print("\nGenerating figures...")
    fig1_icc_overview(data)
    fig2_variance_and_within(data)
    fig3_fingerprint(data)
    fig4_tsne(data)
    print("\nDone. All figures saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
