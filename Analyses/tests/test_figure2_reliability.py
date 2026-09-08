from __future__ import annotations

import unittest
from pathlib import Path
import hashlib
import json
import tempfile

import numpy as np
import pandas as pd

from Analyses.figure2_reliability import (
    aggregate_fold_results,
    aggregate_subject_scanner_cells,
    aggregate_trace_results,
    bootstrap_dbicc,
    bootstrap_indices,
    compute_dbicc,
    distance_blocks,
    normalize_metadata,
    numbered_columns,
    paired_bootstrap_contrast,
    paths_for_spec,
    INPUT_SPECS,
    run_reml,
)


def synthetic_frame(*, scanner_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for subject in range(20):
        subject_value = subject * 4.0
        for scanner in range(3):
            for repeat in range(2):
                rows.append(
                    {
                        "subject_id": f"sub-{subject:05d}",
                        "session_id": f"ses-SCN{scanner}{repeat:03d}",
                        "scanner": f"SCN{scanner}",
                        "emb_0": subject_value + scanner_shift * scanner + repeat * 0.01,
                        "emb_1": -subject_value + scanner_shift * scanner - repeat * 0.01,
                    }
                )
    return pd.DataFrame(rows)


class InputContractTests(unittest.TestCase):
    def test_reused_reml_result_must_match_current_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.csv"
            rows = []
            for subject in range(20):
                for scanner in range(8):
                    rows.append(
                        {
                            "subject_id": f"s{subject}",
                            "scanner": f"c{scanner}",
                            "emb_0": float(subject + scanner),
                        }
                    )
            pd.DataFrame(rows + rows[:5]).to_csv(input_path, index=False)
            output_path = root / "result.json"
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "condition": "wrong",
                        "input_csv": str(input_path.resolve()),
                        "n_sessions_input": 165,
                        "n_subjects": 20,
                        "n_scanners": 8,
                        "n_dimensions": 1,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid REML result"):
                run_reml(
                    input_path,
                    output_path,
                    condition="expected",
                    workers=1,
                    rscript="Rscript",
                    reuse_existing=True,
                )

    def test_reused_reml_result_is_bound_to_input_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.csv"
            rows = []
            for subject in range(20):
                for scanner in range(8):
                    rows.append(
                        {
                            "subject_id": f"s{subject}",
                            "scanner": f"c{scanner}",
                            "emb_0": float(subject + scanner),
                        }
                    )
            pd.DataFrame(rows + rows[:5]).to_csv(input_path, index=False)
            output_path = root / "result.json"
            payload = {
                "schema_version": 1,
                "condition": "expected",
                "input_csv": str(input_path.resolve()),
                "input_csv_sha256": "0" * 64,
                "n_sessions_input": 165,
                "n_subject_scanner_cells": 120,
                "n_subjects": 20,
                "n_scanners": 8,
                "n_dimensions": 1,
            }
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input_csv_sha256"):
                run_reml(
                    input_path,
                    output_path,
                    condition="expected",
                    workers=1,
                    rscript="Rscript",
                    reuse_existing=True,
                )

            payload["input_csv_sha256"] = hashlib.sha256(
                input_path.read_bytes()
            ).hexdigest()
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            observed = run_reml(
                input_path,
                output_path,
                condition="expected",
                workers=1,
                rscript="Rscript",
                reuse_existing=True,
            )
            self.assertEqual(observed["input_csv_sha256"], payload["input_csv_sha256"])

    def test_anatcl_uses_dedicated_bfloat16_root(self):
        anatcl = next(spec for spec in INPUT_SPECS if spec.condition == "anatcl")
        other = next(spec for spec in INPUT_SPECS if spec.condition == "yaware")
        roots = (Path("/legacy"), Path("/freesurfer"), Path("/anatcl-bfloat16"))
        anatcl_paths, _ = paths_for_spec(anatcl, *roots)
        other_paths, _ = paths_for_spec(other, *roots)
        self.assertTrue(all(path.parent == roots[2] for path in anatcl_paths))
        self.assertTrue(all(path.parent == roots[0] for path in other_paths))

    def test_numbered_columns_are_numeric_not_lexicographic(self):
        frame = pd.DataFrame(columns=["emb_10", "emb_2", "emb_0", "emb_1"])
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            numbered_columns(frame, "emb_")

        frame = pd.DataFrame(columns=[f"emb_{index}" for index in range(12)])
        self.assertEqual(
            numbered_columns(frame, "emb_"),
            [f"emb_{index}" for index in range(12)],
        )

    def test_curia_style_stems_are_parsed(self):
        frame = pd.DataFrame(
            {
                "stem": ["sub-03286_ses-NOT1ACH001_T1w"],
                "cls_all_0": [1.0],
            }
        )
        normalized = normalize_metadata(frame, condition="curia2")
        self.assertEqual(normalized.loc[0, "subject_id"], "sub-03286")
        self.assertEqual(normalized.loc[0, "session_id"], "ses-NOT1ACH001")
        self.assertEqual(normalized.loc[0, "scanner"], "NOT1ACH")

    def test_freesurfer_metadata_are_normalized(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["_03286"],
                "session_id": ["03286_NOT1ACH001"],
                "scanner": ["NOT1ACH"],
            }
        )
        normalized = normalize_metadata(frame, condition="freesurfer")
        self.assertEqual(normalized.loc[0, "subject_id"], "sub-03286")
        self.assertEqual(normalized.loc[0, "session_id"], "ses-NOT1ACH001")


class DbiccTests(unittest.TestCase):
    def test_acquisition_001_is_selected_within_subject_scanner_cells(self):
        frame = synthetic_frame()
        cells = aggregate_subject_scanner_cells(frame, ["emb_0", "emb_1"])
        self.assertEqual(len(cells), 60)
        self.assertEqual(cells["subject_id"].nunique(), 20)
        observed = cells.loc[
            (cells.subject_id == "sub-00000") & (cells.scanner == "SCN0"),
            "emb_0",
        ].item()
        self.assertEqual(observed, 0.01)

    def test_stable_subject_signal_has_high_dbicc(self):
        frame = synthetic_frame(scanner_shift=0.1)
        samples = bootstrap_indices(20, 50, 42)
        result = compute_dbicc(frame, ["emb_0", "emb_1"], samples)
        self.assertGreater(result["estimate"], 0.99)
        self.assertEqual(len(result["bootstrap_values"]), 50)

    def test_scanner_shift_reduces_dbicc(self):
        samples = bootstrap_indices(20, 50, 42)
        stable = compute_dbicc(
            synthetic_frame(scanner_shift=0.1), ["emb_0", "emb_1"], samples
        )
        shifted = compute_dbicc(
            synthetic_frame(scanner_shift=40.0), ["emb_0", "emb_1"], samples
        )
        self.assertLess(shifted["estimate"], stable["estimate"])

    def test_bootstrap_is_reproducible(self):
        frame = synthetic_frame(scanner_shift=1.0)
        cells = aggregate_subject_scanner_cells(frame, ["emb_0", "emb_1"])
        blocks = distance_blocks(cells, ["emb_0", "emb_1"])
        first = bootstrap_dbicc(blocks, bootstrap_indices(20, 25, 7))
        second = bootstrap_dbicc(blocks, bootstrap_indices(20, 25, 7))
        np.testing.assert_array_equal(first, second)


class AggregationTests(unittest.TestCase):
    def test_fold_aggregation_uses_shared_bootstrap_replicates(self):
        first = {
            "estimate": 0.8,
            "bootstrap_values": [0.7, 0.8, 0.9],
            "n_sessions_input": 165,
            "n_subject_scanner_cells": 160,
            "n_subjects": 20,
            "n_scanners": 8,
            "n_dimensions": 512,
            "distance": "distance",
            "repeat_handling": "cells",
            "formula": "formula",
            "bootstrap": {"n": 3},
            "reference": "reference",
        }
        second = {**first, "estimate": 0.6, "bootstrap_values": [0.5, 0.6, 0.7]}
        aggregate = aggregate_fold_results([first, second])
        self.assertAlmostEqual(aggregate["estimate"], 0.7)
        np.testing.assert_allclose(aggregate["bootstrap_values"], [0.6, 0.7, 0.8])

    def test_trace_pooling_occurs_before_proportions(self):
        first = {
            "trace_subject": 90.0,
            "trace_scanner": 5.0,
            "trace_residual": 5.0,
            "n_dimensions_valid": 1,
            "n_dimensions_failed_or_constant": 0,
            "n_singular": 0,
            "estimator": "REML",
            "formula": "formula",
            "repeat_handling": "cells",
            "response_scaling": "restored",
        }
        second = {
            **first,
            "trace_subject": 0.0,
            "trace_scanner": 1.0,
            "trace_residual": 0.0,
        }
        pooled = aggregate_trace_results([first, second])
        self.assertAlmostEqual(pooled["proportion_subject"], 90.0 / 101.0)
        self.assertNotAlmostEqual(pooled["proportion_subject"], 0.45)

    def test_paired_contrast_uses_matched_bootstrap(self):
        left = {"estimate": 0.8, "bootstrap_values": [0.7, 0.8, 0.9]}
        right = {"estimate": 0.6, "bootstrap_values": [0.5, 0.7, 0.8]}
        result = paired_bootstrap_contrast(left, right, label="left minus right")
        self.assertAlmostEqual(result["estimate"], 0.2)
        np.testing.assert_allclose(result["bootstrap_values"], [0.2, 0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
