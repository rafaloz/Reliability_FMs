import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Paper.ImagingNeuroscience.Figures.FiguresScripts import (
    figureS4_frozen_ridge_onharmony_scatter as frozen,
    figureS6_adapted_onharmony_scatter as adapted,
)
from Paper.ImagingNeuroscience.Figures.FiguresScripts.scatter_accuracy import accuracy_summary


class OnHarmonyScatterMetricsTests(unittest.TestCase):
    def setUp(self):
        from Analyses.figure2_reliability import bootstrap_indices

        samples = bootstrap_indices(20, 1000, 42)
        contract = json.dumps({"cohorts": {"onharmony": {
            "dbicc_bootstrap_n": 1000,
            "dbicc_bootstrap_seed": 42,
            "dbicc_bootstrap_samples_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        }}})
        original_text, original_bytes = Path.read_text, Path.read_bytes
        # Keep the unit tests independent of private study result artifacts.
        target = "Results/Figure3/figure3_brain_age_results.json"
        patches = [
            patch.object(Path, "read_text", lambda path, *a, **kw:
                contract if path.as_posix().endswith(target) else original_text(path, *a, **kw)),
            patch.object(Path, "read_bytes", lambda path:
                contract.encode() if path.as_posix().endswith(target) else original_bytes(path)),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_single_run_reports_accuracy_and_retains_negative_r2(self):
        summary = {"metrics": {
            "mae_years": {"estimate": 10, "ci_lower": 8, "ci_upper": 12},
            "r2": {"estimate": -0.5, "ci_lower": -0.9, "ci_upper": -0.1},
        }}
        label = frozen.metric_text({"accuracy_summary": summary})
        self.assertEqual(label, "MAE 10.00 y\n[8.00, 12.00]\n$R^2$ -0.50\n[-0.90, -0.10]")
        self.assertNotIn("dbICC", label)
        self.assertNotIn("Scanner", label)

    def test_run_metrics_are_averaged_not_predictions(self):
        truth = np.repeat(np.linspace(20, 50, 20), 6)
        frames = [pd.DataFrame({
            "subject_id": np.repeat([f"s{i:02}" for i in range(20)], 6),
            "scanner": np.tile(np.arange(6), 20),
            "age_true": truth, "age_pred": truth + shift,
        }) for shift in (-10, 10)]
        for module in (frozen, adapted):
            with self.subTest(module=module.__name__):
                condition = {"frames": frames}
                module.metric_text(condition)
                summary = condition["accuracy_summary"]["metrics"]
                self.assertAlmostEqual(summary["mae_years"]["estimate"], 10)
                self.assertAlmostEqual(summary["mae_years"]["ci_lower"], 10)
                self.assertLess(summary["r2"]["estimate"], 1)

    def test_bootstrap_retains_whole_participants(self):
        from Analyses.figure2_reliability import bootstrap_indices
        errors = np.arange(20, dtype=float)
        truth = np.repeat(np.linspace(20, 50, 20), 6)
        frame = pd.DataFrame({
            "subject_id": np.repeat([f"s{i:02}" for i in range(20)], 6),
            "scanner": np.tile(np.arange(6), 20),
            "age_true": truth, "age_pred": truth + np.repeat(errors, 6),
        })
        result = accuracy_summary([frame], external=True)["metrics"]["mae_years"]
        expected = np.percentile(errors[bootstrap_indices(20, 1000, 42)].mean(axis=1), [2.5, 97.5])
        np.testing.assert_allclose([result["ci_lower"], result["ci_upper"]], expected)
        with self.assertRaisesRegex(ValueError, "different participants"):
            changed = frame.copy()
            changed.loc[0, "age_true"] += 1
            accuracy_summary([frame, changed], external=True)

    def test_age_is_joined_by_identity_not_row_position(self):
        participants = pd.DataFrame({"participant_id": ["a", "b"], "age": [20, 40]})
        predictions = pd.DataFrame({"subject_id": ["b", "a", "b"], "age_pred": [41, 22, 43]})
        for module in (frozen, adapted):
            with self.subTest(module=module.__name__):
                joined = module.attach_age(predictions, participants)
                self.assertEqual(joined["age_true"].tolist(), [40, 20, 40])
                metrics = module.regression_metrics(joined["age_true"], joined["age_pred"])
                self.assertEqual(metrics["n"], 3)
                self.assertEqual(metrics["mae_years"], 2)

    def test_outcomes_and_nonfinite_predictions_are_rejected(self):
        participants = pd.DataFrame({"participant_id": ["a"], "age": [20]})
        for module in (frozen, adapted):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(ValueError, "outcomes"):
                    module.attach_age(pd.DataFrame({"subject_id": ["a"], "age_pred": [20], "age": [20]}), participants)
                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    module.attach_age(pd.DataFrame({"subject_id": ["a"], "age_pred": [np.nan]}), participants)


if __name__ == "__main__":
    unittest.main()
