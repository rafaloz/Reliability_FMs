#!/usr/bin/env python3
"""Dependency-light tests for the BrainSegFounder visual-QC contract."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from FineTuning.BrainSegFounder.model import (  # noqa: E402
    PREPROCESSING_CONTRACT_SHA256,
    PREPROCESSING_PIPELINE_VERSION,
)
from FineTuning.BrainSegFounder.predict_onharmony import (  # noqa: E402
    ALLOWED_GENERATOR_HASHES,
    INPUT_CONTRACT_SHA256,
    INPUT_CONTRACT_VERSION,
)
from FineTuning.shared.cohort import sha256_file  # noqa: E402
from preprocessings.QC_BrainSegFounder_visual import (  # noqa: E402
    Case,
    METRIC_COLUMNS,
    assign_priority,
    command_finalize,
    inspect_case,
    report_html,
    validate_qc,
)


class BrainSegFounderVisualQCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def write_nifti(path: Path, shape: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.zeros(shape, dtype=np.float32)
        slices = tuple(slice(max(0, size // 4), max(1, 3 * size // 4)) for size in shape)
        data[slices] = np.linspace(0.1, 1.0, int(np.prod(data[slices].shape))).reshape(
            data[slices].shape
        )
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))

    def fine_tuning_case(self) -> Case:
        raw = self.root / "raw.nii.gz"
        self.write_nifti(raw, (20, 20, 20))
        directory = self.root / "inputs" / "D" / "sub-001"
        case = Case(
            case_id="finetuning::D::sub-001",
            scope="finetuning",
            group="finetuning::D",
            dataset="D",
            subject="sub-001",
            split="train",
            acquisition_id="",
            raw_path=raw.resolve(),
            directory=directory,
            stem="sub-001",
        )
        paths = case.paths()
        for key in ("reoriented_t1", "fov_brain", "reduced_t1", "fnirt_brain"):
            self.write_nifti(paths[key], (20, 20, 20))
        self.write_nifti(paths["ready"], (96, 96, 96))
        reference = self.root / "MNI152_T1_1mm_brain.nii.gz"
        self.write_nifti(reference, (20, 20, 20))
        paths["qc_png"].write_bytes(b"synthetic-png")
        qc = {
            "status": "PASS",
            "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
            "pipeline_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
            "dataset": case.dataset,
            "subject": case.subject,
            "split": case.split,
            "raw_input": {"path": str(raw.resolve())},
            "registration_recovery": {"used": False},
            "mni_assets": {
                "brain": {
                    "path": str(reference.resolve()),
                    "sha256": sha256_file(reference),
                }
            },
            "output_sha256": {
                key: sha256_file(paths[key])
                for key in (
                    "reoriented_t1",
                    "fov_brain",
                    "reduced_t1",
                    "fnirt_brain",
                    "ready",
                    "qc_png",
                )
            },
        }
        paths["qc"].write_text(json.dumps(qc), encoding="utf-8")
        return case

    def onharmony_case(self) -> Case:
        raw = self.root / "sub-001_ses-SCAN001_T1w.nii.gz"
        self.write_nifti(raw, (20, 20, 20))
        acquisition = "sub-001_ses-SCAN001_T1w"
        directory = self.root / "onharmony" / acquisition
        case = Case(
            case_id=f"onharmony::{acquisition}",
            scope="onharmony",
            group="onharmony::SCAN",
            dataset="ON-Harmony",
            subject="sub-001",
            split="external",
            acquisition_id=acquisition,
            raw_path=raw.resolve(),
            directory=directory,
            stem=acquisition,
        )
        paths = case.paths()
        for key in ("reoriented_t1", "fov_brain", "reduced_t1", "fnirt_brain"):
            self.write_nifti(paths[key], (20, 20, 20))
        self.write_nifti(paths["ready"], (96, 96, 96))
        reference = self.root / "MNI152_T1_1mm_brain.nii.gz"
        self.write_nifti(reference, (20, 20, 20))
        paths["log"].write_text(
            "\n".join(
                [
                    f"pipeline_version={INPUT_CONTRACT_VERSION}",
                    f"contract={INPUT_CONTRACT_SHA256}",
                    "/usr/local/fsl/bin/fnirt --in=x ",
                    "/usr/local/fsl/bin/convertwarp --ref=x ",
                    "/usr/local/fsl/bin/applywarp --in=x ",
                ]
            ),
            encoding="utf-8",
        )
        qc = {
            "status": "PASS",
            "pipeline_version": INPUT_CONTRACT_VERSION,
            "pipeline_contract_sha256": INPUT_CONTRACT_SHA256,
            "script_sha256": sorted(ALLOWED_GENERATOR_HASHES)[0],
            "raw_input": {"path": str(raw.resolve())},
            "registration_recovery": {"used": False},
            "assets": {
                "brain": {
                    "path": str(reference.resolve()),
                    "sha256": sha256_file(reference),
                }
            },
            "output_sha256": {
                key: sha256_file(paths[key])
                for key in (
                    "reoriented_t1",
                    "fov_brain",
                    "reduced_t1",
                    "fnirt_brain",
                    "ready",
                )
            },
        }
        paths["qc"].write_text(json.dumps(qc), encoding="utf-8")
        return case

    def test_fine_tuning_inspection_renders_relative_report_image(self) -> None:
        report = self.root / "report"
        row = inspect_case(self.fine_tuning_case(), report)
        image = report / row["image_relpath"]
        self.assertTrue(image.is_file())
        self.assertEqual(row["processing_status"], "PASS")
        self.assertEqual(row["image_sha256"], sha256_file(image))
        self.assertEqual(row["montage_version"], "brainsegfounder_visual_qc_v2")
        self.assertEqual(row["reference_brain_sha256"], sha256_file(self.root / "MNI152_T1_1mm_brain.nii.gz"))
        self.assertEqual(row["ready_nonzero_std"] > 0, True)

    def test_onharmony_contract_requires_successful_log_and_hashes(self) -> None:
        case = self.onharmony_case()
        qc = validate_qc(case, case.paths())
        self.assertEqual(qc["status"], "PASS")
        case.paths()["log"].write_text("incomplete", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "preprocessing log"):
            validate_qc(case, case.paths())

    def test_priority_is_ordering_and_html_uses_relative_images_and_keys(self) -> None:
        rows = []
        for index in range(8):
            row = {metric: float(index + 1) for metric in METRIC_COLUMNS}
            row.update(
                case_id=f"case-{index}",
                scope="finetuning",
                group="finetuning::D",
                dataset="D",
                subject=f"sub-{index:03d}",
                split="train",
                acquisition_id="",
                image_relpath=f"images/{index}.png",
                registration_recovery_used=False,
            )
            rows.append(row)
        frame = assign_priority(pd.DataFrame(rows))
        self.assertTrue((frame["decision"] == "UNREVIEWED").all())
        rendered = report_html(frame, "now")
        self.assertIn("images/0.png", rendered)
        self.assertIn("ArrowRight", rendered)
        self.assertIn("qc_review.tsv", rendered)
        self.assertIn("not a quality score", rendered)
        self.assertIn("registered brain with edges from the exact MNI reference", rendered)
        self.assertIn("brainsegfounder-qc-v2", rendered)
        self.assertIn(r"replace(/[\t\r\n]+/g", rendered)
        self.assertIn(r"columns.join('\t')", rendered)
        self.assertIn(r"lines.join('\n')+'\n'", rendered)
        self.assertIn("try{saved=JSON.parse(localStorage.getItem(key)", rendered)
        self.assertIn("catch(error){storageAvailable=false", rendered)

    def test_finalize_requires_and_publishes_one_decision_per_case(self) -> None:
        report = self.root / "final"
        report.mkdir()
        row = {metric: 0.1 for metric in METRIC_COLUMNS}
        row.update(
            case_id="case-1",
            scope="finetuning",
            group="finetuning::D",
            dataset="D",
            subject="sub-001",
            split="train",
            acquisition_id="",
            image_relpath="images/one.png",
            image_sha256="not-used-here",
            registration_recovery_used=False,
            automatic_priority="ROUTINE",
            automatic_max_robust_z=0.0,
            automatic_outlier_count=0,
            montage_version="brainsegfounder_visual_qc_v2",
            decision="UNREVIEWED",
            reviewer_notes="",
        )
        metrics = pd.DataFrame([row])
        metrics.to_csv(report / "qc_metrics.tsv", sep="\t", index=False)
        (report / "qc_source_manifest.json").write_text(
            json.dumps(
                {
                    "metrics_sha256": sha256_file(report / "qc_metrics.tsv"),
                    "montage_version": "brainsegfounder_visual_qc_v2",
                }
            ),
            encoding="utf-8",
        )
        review = self.root / "qc_review.tsv"
        pd.DataFrame(
            [{"case_id": "case-1", "decision": "PASS", "reviewer_notes": "ok"}]
        ).to_csv(review, sep="\t", index=False)
        command_finalize(
            Namespace(report_dir=report, review_tsv=review, allow_incomplete=False)
        )
        summary = json.loads((report / "qc_final_summary.json").read_text())
        self.assertEqual(summary["status"], "HUMAN_REVIEW_COMPLETED")
        self.assertEqual(summary["decision_counts"], {"PASS": 1})
        self.assertTrue((report / "qc_review_completed.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
