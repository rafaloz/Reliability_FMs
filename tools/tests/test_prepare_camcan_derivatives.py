import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from tools.prepare_camcan_derivatives import select_frame


class CamcanExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "source.tsv"
        self.frame = pd.DataFrame({
            "dataset": ["CamCAN", "Other"], "subject": ["sub-CCtest", "other"],
            "split": ["validation", "validation"], "age_pred": [23.25, 99],
            "age_true": [20, 100], "sex": ["F", "M"],
            "error_years": [3.25, -1], "input_path": ["/private/a", "/private/b"],
        })

    def read(self):
        self.frame.to_csv(self.path, sep="\t", index=False)
        return select_frame(self.path, {"sub-CCtest"}, "validation")[0]

    def test_allowlist_excludes_other_cohorts_demographics_and_paths(self):
        result = self.read()
        self.assertEqual(result.columns.tolist(), ["participant_id", "split", "age_pred"])
        self.assertEqual(result.participant_id.tolist(), ["sub-CCtest"])
        self.assertEqual(result.age_pred.tolist(), [23.25])

    def test_duplicate_subject_rejected(self):
        self.frame = pd.concat([self.frame, self.frame.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "subject mismatch"):
            self.read()

    def test_wrong_identity_rejected(self):
        self.frame.loc[0, "subject"] = "sub-CCwrong"
        with self.assertRaisesRegex(ValueError, "subject mismatch"):
            self.read()

    def test_wrong_split_rejected(self):
        self.frame.loc[0, "split"] = "train"
        with self.assertRaisesRegex(ValueError, "Wrong split"):
            self.read()

    def test_nonfinite_rejected(self):
        self.frame.loc[0, "age_pred"] = np.inf
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.read()


if __name__ == "__main__":
    unittest.main()
