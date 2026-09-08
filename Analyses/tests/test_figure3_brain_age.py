from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from Analyses.figure3_brain_age import (
    MODEL_SPECS,
    aggregate_scalar_bootstrap,
    benjamini_hochberg,
    load_internal_test_predictions,
    pairwise_absolute_difference_by_subject,
    participant_scanner_cells,
    scanner_effects,
    sha256_file,
    validate_prediction_audit,
    frozen_prediction_dirs,
)


class AbsoluteDifferenceTests(unittest.TestCase):
    def test_only_anatcl_uses_bfloat16_v3_frozen_roots(self) -> None:
        anatcl = next(spec for spec in MODEL_SPECS if spec.condition == "anatcl")
        yaware = next(spec for spec in MODEL_SPECS if spec.condition == "yaware")
        common = Path("/results")
        internal = Path("/v3/internal")
        onharmony = Path("/v3/onharmony")
        observed = frozen_prediction_dirs(
            common,
            spec=anatcl,
            variant="AnatCL/fold2",
            anatcl_internal_root=internal,
            anatcl_onharmony_root=onharmony,
        )
        self.assertEqual(observed, (internal / "fold2", onharmony / "fold2"))
        observed = frozen_prediction_dirs(
            common,
            spec=yaware,
            variant="yAware/canonical",
            anatcl_internal_root=internal,
            anatcl_onharmony_root=onharmony,
        )
        self.assertEqual(
            observed,
            (
                common / "internal_test/frozen/yAware/canonical",
                common / "onharmony/frozen/yAware/canonical",
            ),
        )

    def test_pairwise_mean_uses_every_scanner_pair(self) -> None:
        cells = pd.DataFrame(
            {
                "subject_id": ["s1", "s1", "s1"],
                "scanner": ["a", "b", "c"],
                "cohort": ["A", "A", "A"],
                "age_pred": [0.0, 2.0, 5.0],
            }
        )
        result = pairwise_absolute_difference_by_subject(cells)
        self.assertAlmostEqual(result.loc["s1"], (2.0 + 5.0 + 3.0) / 3.0)

    def test_within_participant_difference_is_identical_for_bag(self) -> None:
        cells = pd.DataFrame(
            {
                "subject_id": ["s1"] * 3 + ["s2"] * 3,
                "scanner": ["a", "b", "c"] * 2,
                "cohort": ["A"] * 6,
                "age_pred": [30.0, 32.0, 31.0, 50.0, 47.0, 49.0],
                "age": [25.0] * 3 + [40.0] * 3,
            }
        )
        predicted = pairwise_absolute_difference_by_subject(cells)
        bag = cells.assign(age_pred=cells["age_pred"] - cells["age"])
        observed = pairwise_absolute_difference_by_subject(bag)
        np.testing.assert_allclose(predicted.to_numpy(), observed.to_numpy())

    def test_acquisition_001_is_selected_before_scanner_differences(self) -> None:
        rows = []
        for subject_index in range(20):
            subject = f"s{subject_index:02d}"
            for scanner in ("a", "b", "c", "d", "e", "f"):
                rows.append(
                    {
                        "subject_id": subject,
                        "scanner": scanner,
                        "session": f"{scanner}001",
                        "cohort": "A",
                        "age_pred": float(ord(scanner) - ord("a")),
                    }
                )
        rows.append(
            {
                "subject_id": "s00",
                "scanner": "a",
                "session": "a002",
                "cohort": "A",
                "age_pred": 2.0,
            }
        )
        cells = participant_scanner_cells(pd.DataFrame(rows))
        self.assertEqual(len(cells), 120)
        observed = cells.loc[
            (cells.subject_id == "s00") & (cells.scanner == "a"), "age_pred"
        ].item()
        self.assertEqual(observed, 0.0)


class BootstrapTests(unittest.TestCase):
    def test_fold_aggregation_occurs_inside_shared_bootstrap(self) -> None:
        folds = [np.array([1.0, 3.0]), np.array([3.0, 5.0])]
        samples = np.array([[0, 0], [1, 1], [0, 1]], dtype=int)
        result = aggregate_scalar_bootstrap(folds, samples, metric="test")
        self.assertEqual(result["estimate"], 3.0)
        np.testing.assert_allclose(result["bootstrap_values"], [2.0, 4.0, 3.0])

    def test_bh_adjustment_is_monotone_in_rank(self) -> None:
        adjusted = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.04)
        self.assertAlmostEqual(adjusted["c"], 0.04)


class ScannerEffectTests(unittest.TestCase):
    def test_sum_to_zero_scanner_effects_are_recovered(self) -> None:
        scanner_order = ("a", "b", "c")
        true_effects = {"a": -2.0, "b": 0.5, "c": 1.5}
        rows = []
        for index, subject in enumerate(("s1", "s2", "s3", "s4")):
            intercept = 20.0 + 3.0 * index
            for scanner in scanner_order:
                rows.append(
                    {
                        "subject_id": subject,
                        "scanner": scanner,
                        "age_pred": intercept + true_effects[scanner],
                    }
                )
        observed = scanner_effects(pd.DataFrame(rows), scanner_order)
        np.testing.assert_allclose(observed, [-2.0, 0.5, 1.5], atol=1e-10)
        self.assertAlmostEqual(float(observed.sum()), 0.0)

    def test_missing_scanner_is_rejected(self) -> None:
        cells = pd.DataFrame(
            {
                "subject_id": ["s1", "s1"],
                "scanner": ["a", "b"],
                "age_pred": [1.0, 2.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "Scanner design mismatch"):
            scanner_effects(cells, ("a", "b", "c"))


class InputContractTests(unittest.TestCase):
    def test_audit_rejects_prediction_hash_drift(self) -> None:
        spec = MODEL_SPECS[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.tsv"
            predictions.write_text("first\n", encoding="utf-8")
            audit = root / "prediction_audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "status": "ON_HARMONY_PREDICTED",
                        "arm": "frozen",
                        "model": spec.expected_model,
                        "variant": "canonical",
                        "label_blind": True,
                        "outcome_labels_loaded": False,
                        "bias_correction": "none",
                        "n_predictions": 165,
                        "n_subjects": 20,
                        "internal_test_accessed": False,
                        "on_harmony_accessed": True,
                        "predictions_artifact_sha256": sha256_file(predictions),
                    }
                ),
                encoding="utf-8",
            )
            predictions.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_prediction_audit(
                    audit,
                    predictions,
                    spec=spec,
                    variant="yAware/canonical",
                )

    def test_internal_predictions_reject_inconsistent_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.tsv"
            frame = pd.DataFrame(
                {
                    "dataset": ["d"] * 140,
                    "subject": [f"s{i}" for i in range(140)],
                    "split": ["internal_test"] * 140,
                    "age_true": np.arange(140, dtype=float),
                    "age_pred": np.arange(140, dtype=float) + 1.0,
                    "error_years": np.zeros(140),
                    "absolute_error_years": np.ones(140),
                }
            )
            frame.to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "Inconsistent errors"):
                load_internal_test_predictions(path)


if __name__ == "__main__":
    unittest.main()
