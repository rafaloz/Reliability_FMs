from __future__ import annotations

import io
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "preprocessings" / "FineTunningDataset" / "preprocessing_BrainIAC.py"
SPEC = importlib.util.spec_from_file_location("preprocessing_brainiac", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
brainiac = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(brainiac)


class BrainIACPreprocessingTests(unittest.TestCase):
    def test_definitive_manifest_contract_and_selection(self) -> None:
        rows = brainiac.load_manifest(brainiac.DEFAULT_MANIFEST)
        self.assertEqual(len(rows), 1411)
        selected = brainiac.select_rows(
            rows,
            datasets=["CamCAN"],
            splits=["train"],
            subjects=["sub-CC110033"],
            limit=None,
        )
        self.assertEqual(
            [(row["dataset"], row["subject"]) for row in selected],
            [("CamCAN", "sub-CC110033")],
        )

    def test_output_paths_are_dataset_scoped(self) -> None:
        paths = brainiac.output_paths(Path("/outputs"), "CamCAN", "sub-CC110033")
        self.assertEqual(
            paths["ready"],
            Path("/outputs/CamCAN/sub-CC110033/sub-CC110033_brainiac_ready.nii.gz"),
        )
        self.assertEqual(paths["mask"].name, "sub-CC110033_brain_mask.nii.gz")
        self.assertEqual(paths["qc"].name, "preprocessing_qc.json")

    def test_official_template_hash_and_geometry(self) -> None:
        provenance = brainiac.validate_template(brainiac.DEFAULT_TEMPLATE)
        self.assertEqual(provenance["sha256"], brainiac.EXPECTED_TEMPLATE_SHA256)
        self.assertEqual(tuple(provenance["size"]), brainiac.EXPECTED_TEMPLATE_SIZE)
        self.assertEqual(
            tuple(provenance["spacing_mm"]), brainiac.EXPECTED_TEMPLATE_SPACING
        )

    def test_ready_output_validation_accepts_nonzero_zscore(self) -> None:
        rng = np.random.default_rng(42)
        data = np.zeros(brainiac.EXPECTED_READY_SHAPE, dtype=np.float32)
        foreground = rng.normal(size=(40, 44, 48)).astype(np.float32)
        foreground = (foreground - foreground.mean()) / foreground.std()
        data[28:68, 26:70, 24:72] = foreground
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "ready.nii.gz"
            nib.save(nib.Nifti1Image(data, np.eye(4)), output)
            statistics = brainiac.validate_ready_output(output)

        self.assertEqual(statistics["shape"], [96, 96, 96])
        self.assertLess(abs(statistics["mean_nonzero"]), 2e-4)
        self.assertLess(abs(statistics["standard_deviation_nonzero"] - 1.0), 2e-4)

    def test_contract_separates_offline_preprocessing_from_augmentation(self) -> None:
        self.assertNotIn("augmentation", brainiac.CONTRACT_PAYLOAD)
        self.assertFalse(brainiac.CONTRACT_PAYLOAD["n4"]["explicit_mask"])
        self.assertTrue(
            brainiac.CONTRACT_PAYLOAD["skull_stripping"]["retain_mask"]
        )
        self.assertEqual(
            brainiac.CONTRACT_PAYLOAD["model_view"]["shape"], (96, 96, 96)
        )

    def test_hdbet_command_retains_brain_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "subject_brain.nii.gz"
            generated_mask = Path(temporary) / "subject_brain_bet.nii.gz"

            def fake_run(command, **kwargs):
                self.assertIn("--disable_tta", command)
                self.assertIn("--save_bet_mask", command)
                output.touch()
                generated_mask.touch()
                return mock.Mock(returncode=0)

            with mock.patch.object(brainiac.subprocess, "run", side_effect=fake_run):
                command = brainiac.skull_strip_hdbet(
                    Path(temporary) / "registered.nii.gz",
                    output,
                    "hd-bet",
                    "cpu",
                    None,
                    io.StringIO(),
                )

        self.assertIn("--save_bet_mask", command)
        self.assertEqual(
            brainiac.hdbet_generated_mask_path(output),
            generated_mask,
        )


if __name__ == "__main__":
    unittest.main()
