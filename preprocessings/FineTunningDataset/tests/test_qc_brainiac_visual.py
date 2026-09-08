from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from preprocessings.QC_BrainIAC_visual import (
    Case,
    FT_CONTRACT_SHA256,
    FT_PIPELINE_VERSION,
    ONH_HISTORICAL_GENERATOR_SHA256,
    add_priorities,
    atomic_json,
    atomic_tsv,
    command_finalize,
    load_onharmony_cases,
    process_case,
    report_html,
    sha256_file,
)


class BrainIACVisualQCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.report_dir = self.root / "report"
        self.report_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _png(path: Path) -> None:
        Image.new("RGB", (705, 474), "black").save(path)

    def _fine_tuning_case(self) -> Case:
        root = self.root / "ft" / "CamCAN" / "sub-001"
        root.mkdir(parents=True)
        ready = root / "sub-001_brainiac_ready.nii.gz"
        brain = root / "sub-001_brain.nii.gz"
        png = root / "preprocessing_qc.png"
        ready.write_bytes(b"ready")
        brain.write_bytes(b"brain")
        self._png(png)
        qc = {
            "status": "PASS",
            "pipeline_version": FT_PIPELINE_VERSION,
            "pipeline_contract_sha256": FT_CONTRACT_SHA256,
            "script_sha256": "a" * 64,
            "dataset": "CamCAN",
            "subject": "sub-001",
            "split": "train",
            "registration": {"metric_value": -0.42},
            "spatial_outputs": {"brain_nonzero_fraction": 0.22},
            "model_ready_output": {
                "shape": [96, 96, 96],
                "dtype_on_disk": "float32",
                "nonzero_fraction": 0.23,
                "mean_nonzero": 0.0,
                "standard_deviation_nonzero": 1.0,
            },
            "output_sha256": {
                "ready": sha256_file(ready),
                "brain": sha256_file(brain),
                "qc_png": sha256_file(png),
            },
        }
        (root / "preprocessing_qc.json").write_text(json.dumps(qc), encoding="utf-8")
        return Case(
            case_id="brain_age::CamCAN::sub-001",
            scope="brain_age",
            group="brain_age::CamCAN",
            dataset="CamCAN",
            subject="sub-001",
            session="",
            acquisition_id="",
            split="train",
            ready_path=ready,
            brain_path=brain,
            qc_path=root / "preprocessing_qc.json",
            qc_png_path=png,
        )

    def _onharmony_case(self) -> Case:
        acquisition = "sub-12345_ses-OXF2PRI001_T1w"
        root = self.root / "onh" / acquisition
        root.mkdir(parents=True)
        ready = root / f"{acquisition}_brainiac_ready.nii.gz"
        brain = root / f"{acquisition}_brain.nii.gz"
        png = root / f"{acquisition}_qc.png"
        qc_path = root / f"{acquisition}_qc.json"
        log_path = self.root / f"brainiac_{acquisition}.log"
        ready.write_bytes(b"ready")
        brain.write_bytes(b"brain")
        self._png(png)
        qc = {
            "raw_input": f"/raw/{acquisition}.nii.gz",
            "brain_shape": [170, 206, 162],
            "brainiac_shape": [96, 96, 96],
            "brain_nonzero_voxels": 1_250_000,
            "brainiac_mean_nonzero": 0.0,
            "brainiac_std_nonzero": 1.0,
            "brainiac_nonzero_fraction": 0.23,
            "shape_ok": True,
            "norm_ok": True,
            "brain_ok": True,
            "status": "PASS",
        }
        qc_path.write_text(json.dumps(qc), encoding="utf-8")
        log_path.write_text(
            "\n".join(
                [
                    f"script   : {ONH_HISTORICAL_GENERATOR_SHA256}",
                    "[Preprocessing 4-5/5] Resample 96",
                    "status=PASS",
                    "All done.",
                    "exit_code: 0",
                ]
            ),
            encoding="utf-8",
        )
        return Case(
            case_id=f"onharmony::{acquisition}",
            scope="onharmony",
            group="onharmony::OXF2PRI",
            dataset="ON-Harmony",
            subject="sub-12345",
            session="OXF2PRI001",
            acquisition_id=acquisition,
            split="external",
            ready_path=ready,
            brain_path=brain,
            qc_path=qc_path,
            qc_png_path=png,
            log_path=log_path,
        )

    def test_current_fine_tuning_qc_is_bound_to_outputs(self) -> None:
        row = process_case(self._fine_tuning_case(), self.report_dir)
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(row["provenance_schema"], "current")
        self.assertEqual(row["ready_sha256"], sha256_file(Path(row["ready_path"])))
        self.assertTrue(row["qc_png_file"].endswith("preprocessing_qc.png"))

    def test_historical_onharmony_qc_requires_a_successful_pinned_log(self) -> None:
        case = self._onharmony_case()
        row = process_case(case, self.report_dir)
        self.assertEqual(row["provenance_schema"], "historical")
        case.log_path.write_text("exit_code: 0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unrecognized|Incomplete"):
            process_case(case, self.report_dir)

    def test_onharmony_filelist_is_relocated_to_the_configured_root(self) -> None:
        filelist = self.root / "filelist.txt"
        filelist.write_text(
            "/cluster/old/sub-12345_ses-OXF2PRI001_T1w_brainiac_ready.nii.gz\n",
            encoding="utf-8",
        )
        cases = load_onharmony_cases(filelist, self.root / "new", self.root / "logs", 1)
        self.assertEqual(cases[0].ready_path.parent, self.root / "new/sub-12345_ses-OXF2PRI001_T1w")
        self.assertEqual(cases[0].group, "onharmony::OXF2PRI")

    def test_html_is_label_blind_and_supports_keyboard_review(self) -> None:
        rows = [
            process_case(self._fine_tuning_case(), self.report_dir),
            process_case(self._onharmony_case(), self.report_dir),
        ]
        frame = add_priorities(pd.DataFrame(rows))
        rendered = report_html(frame, "2026-08-30T00:00:00+00:00")
        self.assertIn("arrowdown", rendered)
        self.assertIn("decide('PASS')", rendered)
        self.assertIn("No age, diagnosis, prediction, BAG or error was loaded", rendered)
        self.assertIn("qc_review.tsv", rendered)
        self.assertIn(r"rows.join('\n')", rendered)
        self.assertNotIn("rows.join('\n')", rendered)

    def test_finalize_requires_and_freezes_one_decision_per_case(self) -> None:
        metrics = pd.DataFrame({"case_id": ["case-a", "case-b"], "status": ["PASS", "PASS"]})
        metrics_path = self.report_dir / "qc_metrics.tsv"
        atomic_tsv(metrics_path, metrics)
        atomic_json(
            self.report_dir / "qc_summary.json",
            {"status": "COMPLETED", "n_cases": 2, "metrics_sha256": sha256_file(metrics_path)},
        )
        review_path = self.root / "qc_review.tsv"
        atomic_tsv(
            review_path,
            pd.DataFrame(
                {
                    "case_id": ["case-a"],
                    "decision": ["PASS"],
                    "reviewed_at_utc": ["2026-08-30T00:00:00Z"],
                }
            ),
        )
        args = argparse.Namespace(
            report_dir=self.report_dir,
            review_tsv=review_path,
            allow_incomplete=False,
        )
        with self.assertRaisesRegex(ValueError, "Incomplete review"):
            command_finalize(args)
        atomic_tsv(
            review_path,
            pd.DataFrame(
                {
                    "case_id": ["case-a", "case-b"],
                    "decision": ["PASS", "REVIEW"],
                    "reviewed_at_utc": [
                        "2026-08-30T00:00:00Z",
                        "2026-08-30T00:01:00Z",
                    ],
                }
            ),
        )
        command_finalize(args)
        final = json.loads((self.report_dir / "qc_final_summary.json").read_text())
        self.assertEqual(final["status"], "HUMAN_REVIEW_COMPLETED")
        self.assertEqual(final["decision_counts"], {"PASS": 1, "REVIEW": 1})


if __name__ == "__main__":
    unittest.main()
