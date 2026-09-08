import unittest

import numpy as np
import pandas as pd

from Paper.ImagingNeuroscience.Figures.FiguresScripts import (
    figureS4_frozen_ridge_onharmony_scatter as frozen,
    figureS6_adapted_onharmony_scatter as adapted,
)


class OnHarmonyScatterMetricsTests(unittest.TestCase):
    def test_single_run_reports_accuracy_and_retains_negative_r2(self):
        metrics = frozen.regression_metrics([20, 30, 40], [30, 40, 50])
        label = frozen.metric_text({"accuracy": [metrics]})
        self.assertEqual(label, "MAE 10.00 y\n$R^2$ -0.500")
        self.assertNotIn("dbICC", label)
        self.assertNotIn("Scanner", label)

    def test_run_metrics_are_averaged_not_predictions(self):
        truth = np.array([20, 30, 40])
        metrics = [frozen.regression_metrics(truth, truth + shift) for shift in (-10, 10)]
        expected = "MAE 10.00 $\\pm$ 0.00 y\n$R^2$ -0.500 $\\pm$ 0.000"
        for module in (frozen, adapted):
            with self.subTest(module=module.__name__):
                self.assertEqual(module.metric_text({"accuracy": metrics}), expected)

    def test_spread_is_sample_standard_deviation(self):
        metrics = [{"mae_years": 2, "r2": 0.2}, {"mae_years": 4, "r2": 0.4}]
        expected = "MAE 3.00 $\\pm$ 1.41 y\n$R^2$ 0.300 $\\pm$ 0.141"
        for module in (frozen, adapted):
            with self.subTest(module=module.__name__):
                self.assertEqual(module.metric_text({"accuracy": metrics}), expected)

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
