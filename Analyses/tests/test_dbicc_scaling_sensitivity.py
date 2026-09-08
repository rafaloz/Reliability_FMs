from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from Paper.ImagingNeuroscience.Tables.build_supp_dbicc_scaling import (
    load_bound_input,
    render_table,
    scaling_scores,
    sha256_file,
    summarise,
)


def cohort() -> pd.DataFrame:
    rows = []
    for subject in range(20):
        scanners = range(6) if subject < 10 else range(2, 8)
        for scanner in scanners:
            rows.append({
                "subject_id": f"sub-{subject:05d}",
                "scanner": f"SC{scanner}",
                "session_id": f"ses-SC{scanner}001",
                "emb_0": subject * 1000.0 + scanner,
                "emb_1": subject + scanner * 8.0,
                "emb_2": 3.0,
            })
    for row in rows[:9]:
        for repeat in range(2, 7):
            rows.append({**row, "session_id": f"ses-{row['scanner']}{repeat:03d}", "emb_0": 1e6})
    return pd.DataFrame(rows)


class ScalingSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.frame = cohort()
        self.features = ["emb_0", "emb_1", "emb_2"]

    def test_scaling_is_fitted_only_on_primary_acquisitions(self):
        selected = self.frame[self.frame.session_id.str.endswith("001")]
        self.assertEqual(scaling_scores(self.frame, self.features), scaling_scores(selected, self.features))

    def test_coordinate_rescaling_changes_native_but_not_standardised_dbicc(self):
        original = scaling_scores(self.frame, self.features)
        modified = self.frame.copy()
        modified["emb_0"] *= 0.001
        modified["emb_1"] = modified["emb_1"] * 17 + 50
        result = scaling_scores(modified, self.features)
        self.assertGreater(abs(original["native_dbicc"] - result["native_dbicc"]), 0.1)
        self.assertAlmostEqual(original["standardised_dbicc"], result["standardised_dbicc"], places=12)
        self.assertEqual(result["constant_dimensions"], 1)

    def test_global_scale_does_not_change_either_dbicc(self):
        modified = self.frame.copy()
        modified[self.features] *= 10
        for key in ("native_dbicc", "standardised_dbicc"):
            self.assertAlmostEqual(scaling_scores(self.frame, self.features)[key], scaling_scores(modified, self.features)[key], places=12)

    def test_nonfinite_features_fail(self):
        self.frame.loc[0, "emb_0"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            scaling_scores(self.frame, self.features)

    def test_input_binding_and_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            self.frame.to_csv(path, index=False)
            digest = sha256_file(path)
            self.assertEqual(len(load_bound_input(path, digest, 3)), 165)
            with self.assertRaisesRegex(ValueError, "size"):
                load_bound_input(path, digest, 4)
            self.frame.loc[0, "emb_0"] = 5.0
            self.frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "SHA"):
                load_bound_input(path, digest, 3)

    def test_missing_first_acquisition_fails(self):
        self.frame.loc[0, "session_id"] = "ses-SC0007"
        with self.assertRaisesRegex(ValueError, "acquisition 001"):
            scaling_scores(self.frame, self.features)

    def test_fold_metrics_are_averaged_before_ranking(self):
        folds = pd.DataFrame([
            {"condition": "a", "model": "A", "dimensions": 3, "native_dbicc": 0.8, "standardised_dbicc": 0.9},
            {"condition": "a", "model": "A", "dimensions": 3, "native_dbicc": 0.4, "standardised_dbicc": 0.7},
            {"condition": "b", "model": "B", "dimensions": 4, "native_dbicc": 0.7, "standardised_dbicc": 0.5},
            {"condition": "c", "model": "C", "dimensions": 5, "native_dbicc": 0.7, "standardised_dbicc": 0.4},
        ])
        result = summarise(folds).set_index("condition")
        self.assertAlmostEqual(result.loc["a", "native_dbicc"], 0.6)
        self.assertEqual(result.loc["a", "native_rank"], 3)
        self.assertEqual(result.loc["a", "standardised_rank"], 1)
        self.assertEqual(result.loc["b", "native_rank"], 1)
        self.assertEqual(result.loc["c", "native_rank"], 1)
        self.assertIn("0.600", render_table(result.reset_index()))


if __name__ == "__main__":
    unittest.main()
