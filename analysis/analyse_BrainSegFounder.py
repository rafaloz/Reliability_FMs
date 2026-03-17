#!/usr/bin/env python3
"""
Reliability analysis for BrainSegFounder (Swin-T, 768-dim).

Input:  ../DATA/embeddings_BrainSegFounder.csv
Output: ../Results/results_BrainSegFounder.json

Usage:
  python analyse_BrainSegFounder.py [--input ../DATA/embeddings_BrainSegFounder.csv]
                                    [--output_dir ../Results]
                                    [--n_boot 1000]
"""

import argparse
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils_reliability import run_full_analysis


def main():
    parser = argparse.ArgumentParser(description="Reliability analysis — BrainSegFounder")
    parser.add_argument("--input", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "DATA", "embeddings_BrainSegFounder.csv"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "Results"))
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    print(f"Loaded {len(df)} rows, {len(emb_cols)} embedding dims from {args.input}")

    run_full_analysis(df, emb_cols, model_name="BrainSegFounder",
                      output_dir=args.output_dir, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
