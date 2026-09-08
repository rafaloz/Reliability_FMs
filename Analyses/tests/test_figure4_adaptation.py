from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Analyses.figure4_adaptation import (
    MODEL_SPECS,
    aggregate_contrasts,
    aggregate_metric_records,
    aligned_vector,
    all_target_names,
    contrast_metric,
    empirical_two_sided_p,
    metric_record,
    scanner_effects_batched,
    frozen_result_dirs,
)
from Analyses.figure3_brain_age import scanner_effects
from Paper.ImagingNeuroscience.Figures.FiguresScripts.figure4_adaptation import (
    adapted_interval_from_paired_gain,
)


class RunHierarchyTests(unittest.TestCase):
    def test_only_anatcl_uses_bfloat16_v3_frozen_baseline(self) -> None:
        anatcl = next(spec for spec in MODEL_SPECS if spec.condition == "anatcl")
        yaware = next(spec for spec in MODEL_SPECS if spec.condition == "yaware")
        common = Path("/results")
        internal = Path("/v3/internal")
        onharmony = Path("/v3/onharmony")
        self.assertEqual(
            frozen_result_dirs(
                common,
                spec=anatcl,
                variant="fold3",
                anatcl_internal_root=internal,
                anatcl_onharmony_root=onharmony,
            ),
            (internal / "fold3", onharmony / "fold3"),
        )
        self.assertEqual(
            frozen_result_dirs(
                common,
                spec=yaware,
                variant="canonical",
                anatcl_internal_root=internal,
                anatcl_onharmony_root=onharmony,
            ),
            (
                common / "internal_test/frozen/yAware/canonical",
                common / "onharmony/frozen/yAware/canonical",
            ),
        )

    def test_registry_contains_25_unique_runs_and_23_primary_runs(self) -> None:
        targets = all_target_names()
        self.assertEqual(len(targets), 25)
        self.assertEqual(len(set(targets)), 25)
        self.assertEqual(
            sum(len(spec.primary_runs) for spec in MODEL_SPECS),
            23,
        )

    def test_anatcl_keeps_fold_and_seed_sensitivity_separate(self) -> None:
        anatcl = next(spec for spec in MODEL_SPECS if spec.condition == "anatcl")
        self.assertEqual(
            [(run.fold, run.seed) for run in anatcl.primary_runs],
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        )
        secondary = [run for run in anatcl.runs if not run.primary]
        self.assertEqual(
            [(run.fold, run.seed) for run in secondary],
            [(0, 1), (0, 2)],
        )


class PairedGainTests(unittest.TestCase):
    def test_accuracy_and_stability_use_frozen_minus_adapted(self) -> None:
        frozen = metric_record(5.0, np.array([4.0, 5.0, 6.0]), name="frozen")
        adapted = metric_record(3.0, np.array([2.5, 3.0, 3.5]), name="adapted")
        result = contrast_metric(
            frozen,
            adapted,
            adapted_minus_frozen=False,
            name="gain",
        )
        self.assertEqual(result["estimate"], 2.0)
        np.testing.assert_allclose(result["bootstrap_values"], [1.5, 2.0, 2.5])
        self.assertEqual(result["formula"], "frozen minus adapted")

    def test_reliability_uses_adapted_minus_frozen(self) -> None:
        frozen = metric_record(0.7, np.array([0.6, 0.7, 0.8]), name="frozen")
        adapted = metric_record(0.8, np.array([0.75, 0.8, 0.85]), name="adapted")
        result = contrast_metric(
            frozen,
            adapted,
            adapted_minus_frozen=True,
            name="gain",
        )
        self.assertAlmostEqual(result["estimate"], 0.1)
        np.testing.assert_allclose(result["bootstrap_values"], [0.15, 0.1, 0.05])
        self.assertEqual(result["formula"], "adapted minus frozen")

    def test_run_metrics_are_aggregated_inside_the_shared_bootstrap(self) -> None:
        first = metric_record(1.0, np.array([0.0, 1.0, 2.0]), name="first")
        second = metric_record(3.0, np.array([2.0, 3.0, 4.0]), name="second")
        result = aggregate_metric_records([first, second], name="aggregate")
        self.assertEqual(result["estimate"], 2.0)
        np.testing.assert_allclose(result["bootstrap_values"], [1.0, 2.0, 3.0])
        self.assertIn("predictions are not ensembled", result["aggregation"])

    def test_aggregate_contrast_recomputes_empirical_p_value(self) -> None:
        first = {
            **metric_record(1.0, np.array([1.0, 1.0, -1.0]), name="first"),
            "formula": "frozen minus adapted",
        }
        second = {
            **metric_record(3.0, np.array([3.0, 3.0, 1.0]), name="second"),
            "formula": "frozen minus adapted",
        }
        result = aggregate_contrasts([first, second], name="aggregate")
        np.testing.assert_allclose(result["bootstrap_values"], [2.0, 2.0, 0.0])
        self.assertEqual(
            result["p_value_two_sided"],
            empirical_two_sided_p(np.array([2.0, 2.0, 0.0])),
        )

    def test_empirical_p_value_rejects_nonfinite_distributions(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite bootstrap"):
            empirical_two_sided_p(np.array([0.0, np.nan]))

    def test_mae_gain_interval_is_translated_onto_the_adapted_scale(self) -> None:
        gain = metric_record(2.0, np.array([1.0, 2.0, 3.0]), name="gain")
        adapted, lower, upper = adapted_interval_from_paired_gain(
            5.0,
            gain,
            formula="frozen_minus_adapted",
        )
        self.assertEqual(adapted, 3.0)
        self.assertAlmostEqual(lower, 2.05)
        self.assertAlmostEqual(upper, 3.95)

    def test_dbicc_gain_interval_is_translated_onto_the_adapted_scale(self) -> None:
        gain = metric_record(0.10, np.array([0.05, 0.10, 0.15]), name="gain")
        adapted, lower, upper = adapted_interval_from_paired_gain(
            0.70,
            gain,
            formula="adapted_minus_frozen",
        )
        self.assertAlmostEqual(adapted, 0.80)
        self.assertAlmostEqual(lower, 0.7525)
        self.assertAlmostEqual(upper, 0.8475)


class AlignmentTests(unittest.TestCase):
    def test_aligned_vector_pairs_by_identity_not_input_order(self) -> None:
        reference = pd.DataFrame(
            {"dataset": ["b", "a"], "subject": ["s2", "s1"], "value": [2.0, 1.0]}
        )
        candidate = pd.DataFrame(
            {"dataset": ["a", "b"], "subject": ["s1", "s2"], "value": [10.0, 20.0]}
        )
        left, right = aligned_vector(
            reference,
            candidate,
            keys=["dataset", "subject"],
            column="value",
            label="test",
        )
        np.testing.assert_allclose(left, [1.0, 2.0])
        np.testing.assert_allclose(right, [10.0, 20.0])

    def test_aligned_vector_rejects_different_subjects(self) -> None:
        reference = pd.DataFrame(
            {"dataset": ["a"], "subject": ["s1"], "value": [1.0]}
        )
        candidate = pd.DataFrame(
            {"dataset": ["a"], "subject": ["s2"], "value": [2.0]}
        )
        with self.assertRaisesRegex(ValueError, "row identity mismatch"):
            aligned_vector(
                reference,
                candidate,
                keys=["dataset", "subject"],
                column="value",
                label="test",
            )


class ScannerEffectBatchTests(unittest.TestCase):
    def test_batched_solver_matches_independent_reference_fits(self) -> None:
        scanner_order = ("s0", "s1", "s2")
        rows = []
        for subject_index, subject in enumerate(("p0", "p1", "p2", "p3")):
            for scanner_index, scanner in enumerate(scanner_order):
                rows.append(
                    {
                        "subject_id": subject,
                        "scanner": scanner,
                        "cohort": "A",
                        "first": 40.0 + subject_index + 2.0 * scanner_index,
                        "second": 70.0 - subject_index - 0.5 * scanner_index,
                    }
                )
        cells = pd.DataFrame(rows)
        predictions = cells[["first", "second"]].to_numpy(dtype=np.float64)
        observed = scanner_effects_batched(cells, predictions, scanner_order)
        expected = np.vstack(
            [
                scanner_effects(
                    cells.assign(age_pred=cells[column]), scanner_order
                )
                for column in ("first", "second")
            ]
        )
        np.testing.assert_allclose(observed, expected, atol=1e-12)
        np.testing.assert_allclose(observed.sum(axis=1), 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
