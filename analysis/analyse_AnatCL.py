#!/usr/bin/env python3
"""
Reliability analysis for AnatCL (ResNet-18, 512-dim, 5-fold evaluation).

AnatCL ships 5 cross-validation fold checkpoints. We run the full reliability
analysis independently for each fold, then report mean ± std across folds.

Input:  ../DATA/embeddings_AnatCL_fold{0..4}.csv
Output: ../Results/results_AnatCL.json          (aggregated mean ± std)
        ../Results/results_AnatCL_fold{0..4}.json (per-fold results)

Usage:
  python analyse_AnatCL.py [--input_dir ../DATA] [--output_dir ../Results] [--n_boot 1000]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils_reliability import run_full_analysis


def aggregate_fold_results(fold_results):
    """Aggregate per-fold results into mean ± std summaries."""
    aggregated = {
        "model": "AnatCL",
        "n_folds": len(fold_results),
        "n_obs": fold_results[0]["n_obs"],
        "n_dims": fold_results[0]["n_dims"],
        "n_subjects": fold_results[0]["n_subjects"],
        "n_scanners": fold_results[0]["n_scanners"],
    }

    # Keys that contain summary dicts with numeric values to aggregate
    summary_keys = [k for k in fold_results[0] if k.endswith("_summary")]
    for sk in summary_keys:
        agg = {}
        sub_keys = fold_results[0][sk].keys()
        for sub_k in sub_keys:
            vals = []
            for fr in fold_results:
                v = fr[sk].get(sub_k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append(v)
            if vals:
                agg[sub_k + "_mean"] = float(np.mean(vals))
                agg[sub_k + "_std"] = float(np.std(vals))
        aggregated[sk] = agg

    # Scalar metrics (fingerprinting accuracy, subject ID accuracy, etc.)
    scalar_keys = [
        "scanner_fingerprint_accuracy",
        "subject_identification_accuracy",
    ]
    for sk in scalar_keys:
        vals = [fr[sk] for fr in fold_results if sk in fr and isinstance(fr[sk], (int, float))]
        if vals:
            aggregated[sk + "_mean"] = float(np.mean(vals))
            aggregated[sk + "_std"] = float(np.std(vals))

    # Variance decomposition proportions
    vd_keys = ["var_prop_subject", "var_prop_scanner", "var_prop_residual"]
    for vk in vd_keys:
        vals = [fr[vk] for fr in fold_results if vk in fr and isinstance(fr[vk], (int, float))]
        if vals:
            aggregated[vk + "_mean"] = float(np.mean(vals))
            aggregated[vk + "_std"] = float(np.std(vals))

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Reliability analysis — AnatCL (5-fold)")
    parser.add_argument("--input_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "DATA"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "Results"))
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    fold_results = []
    for fold_idx in range(5):
        csv_path = os.path.join(args.input_dir, f"embeddings_AnatCL_fold{fold_idx}.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping fold {fold_idx}")
            continue

        df = pd.read_csv(csv_path)
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        print(f"\n*** AnatCL fold {fold_idx}: {len(df)} rows, {len(emb_cols)} dims ***")

        result = run_full_analysis(
            df, emb_cols, model_name=f"AnatCL_fold{fold_idx}",
            output_dir=args.output_dir, n_boot=args.n_boot
        )
        fold_results.append(result)

    if not fold_results:
        print("ERROR: No fold CSVs found.")
        return

    # Aggregate across folds
    print(f"\n{'='*60}")
    print(f"  Aggregating across {len(fold_results)} folds")
    print(f"{'='*60}")

    aggregated = aggregate_fold_results(fold_results)

    # Print summary
    if "icc_between_summary" in aggregated:
        s = aggregated["icc_between_summary"]
        print(f"  ICC(2,1) between-scanner: median = {s.get('median_mean', 0):.3f} ± {s.get('median_std', 0):.3f}")
    if "icc_within_summary" in aggregated:
        s = aggregated["icc_within_summary"]
        print(f"  ICC(3,1) within-scanner:  median = {s.get('median_mean', 0):.3f} ± {s.get('median_std', 0):.3f}")
    if "var_prop_subject_mean" in aggregated:
        print(f"  Variance: subject={aggregated['var_prop_subject_mean']:.1%} ± {aggregated['var_prop_subject_std']:.1%}, "
              f"scanner={aggregated['var_prop_scanner_mean']:.1%} ± {aggregated['var_prop_scanner_std']:.1%}, "
              f"residual={aggregated['var_prop_residual_mean']:.1%} ± {aggregated['var_prop_residual_std']:.1%}")
    if "scanner_fingerprint_accuracy_mean" in aggregated:
        print(f"  Scanner fingerprinting: {aggregated['scanner_fingerprint_accuracy_mean']:.1%} ± {aggregated['scanner_fingerprint_accuracy_std']:.1%}")
    if "subject_identification_accuracy_mean" in aggregated:
        print(f"  Subject identification: {aggregated['subject_identification_accuracy_mean']:.1%} ± {aggregated['subject_identification_accuracy_std']:.1%}")

    # Save aggregated results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "results_AnatCL.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
