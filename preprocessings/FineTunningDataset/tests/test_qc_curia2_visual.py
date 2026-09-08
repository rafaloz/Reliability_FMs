from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from FineTuning.shared.cohort import sha256_file
from preprocessings.QC_Curia2_visual import (
    MONTAGE_VERSION,
    add_automatic_assessment,
    command_finalize,
    command_reassess,
    process_pl_slices,
    reorient_array,
    report_html,
    selected_slice_indices,
)


def complete_row(case_id: str = "finetuning::Demo::sub-001") -> dict[str, object]:
    return {
        "case_id": case_id,
        "scope": "finetuning",
        "metric_group": "finetuning::Demo",
        "dataset": "Demo",
        "subject": "sub-001",
        "session": "",
        "acquisition_id": "",
        "split": "train",
        "raw_path": "/raw/sub-001.nii.gz",
        "raw_sha256": "a" * 64,
        "raw_sha256_verified": True,
        "raw_shape": "8x9x10",
        "ras_shape": "8x9x10",
        "raw_axcodes": "RAS",
        "ras_voxel_spacing_mm": "1x1x1",
        "voxel_volume_mm3": 1.0,
        "n_slices": 10,
        "selected_slice_indices": "1,3,4,6,8",
        "raw_dynamic_range": 100.0,
        "active_slice_fraction": 0.8,
        "core_active_fraction": 1.0,
        "active_span_fraction": 0.8,
        "active_center_offset_fraction": 0.01,
        "inferior_active_fraction": 0.5,
        "superior_active_fraction": 0.5,
        "selected_constant_fraction": 0.0,
        "processed_abs_mean_max": 1e-7,
        "processed_std_deviation_max": 1e-7,
        "image_relpath": "images/example.jpg",
        "image_sha256": "",
        "montage_version": MONTAGE_VERSION,
        "technical_status": "PASS",
        "technical_error": "",
        "automatic_priority": "ROUTINE",
        "automatic_typicality_score": 10.0,
        "automatic_max_robust_z": 0.0,
        "automatic_outlier_count": 0,
        "decision": "UNREVIEWED",
        "reviewer_notes": "",
    }


class Curia2VisualQCTests(unittest.TestCase):
    def test_reorientation_to_las_flips_ras_first_axis(self) -> None:
        volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        affine = np.eye(4)
        observed = reorient_array(volume, affine, ("L", "A", "S"))
        np.testing.assert_array_equal(observed, volume[::-1])

    def test_selected_indices_span_the_axial_stack(self) -> None:
        self.assertEqual(selected_slice_indices(101), [10, 30, 50, 70, 90])
        with self.assertRaises(ValueError):
            selected_slice_indices(4)

    def test_processor_reconstruction_zscores_and_centers_constant_slice(self) -> None:
        gradient = np.arange(35, dtype=np.float32).reshape(5, 7)
        constant = np.full((5, 7), 3.0, dtype=np.float32)
        observed = process_pl_slices([gradient, constant])
        self.assertEqual(observed.shape, (2, 1, 512, 512))
        self.assertTrue(np.isfinite(observed).all())
        self.assertAlmostEqual(float(observed[0].mean()), 0.0, places=5)
        self.assertAlmostEqual(float(observed[0].std(ddof=1)), 1.0, places=5)
        self.assertLessEqual(float(np.abs(observed[1]).max()), 1e-6)

    def test_html_uses_relative_images_keyboard_and_complete_export(self) -> None:
        frame = pd.DataFrame([complete_row()])
        generated = report_html(frame, "2026-09-02T00:00:00+00:00")
        self.assertIn('src="${esc(row.image_relpath)}"', generated)
        self.assertIn("arrowright", generated)
        self.assertIn("key==='p'", generated)
        self.assertIn("a.download='qc_review.tsv'", generated)
        self.assertIn("rows.forEach", generated)
        self.assertNotIn("/raw/sub-001.nii.gz", generated)

    def test_automatic_score_only_orders_within_group(self) -> None:
        rows = []
        for index in range(20):
            row = complete_row(f"case::{index}")
            row["subject"] = f"sub-{index:03d}"
            row["n_slices"] = 100 if index < 19 else 500
            rows.append(row)
        observed = add_automatic_assessment(pd.DataFrame(rows))
        extreme = observed.loc[observed["case_id"].eq("case::19")].iloc[0]
        self.assertIn(extreme["automatic_priority"], {"REVIEW", "HIGH"})
        self.assertLess(float(extreme["automatic_typicality_score"]), 10.0)
        self.assertTrue((observed["decision"] == "UNREVIEWED").all())

    def test_finalize_requires_and_freezes_every_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "example.jpg"
            image_path.parent.mkdir()
            Image.new("RGB", (20, 20), "black").save(image_path)
            row = complete_row()
            row["image_sha256"] = sha256_file(image_path)
            metrics = pd.DataFrame([row])
            metrics_path = root / "qc_metrics.tsv"
            metrics.to_csv(metrics_path, sep="\t", index=False)
            source = {
                "metrics_sha256": sha256_file(metrics_path),
                "montage_version": MONTAGE_VERSION,
                "created_utc": "2026-09-02T09:00:00Z",
            }
            (root / "qc_source_manifest.json").write_text(
                json.dumps(source), encoding="utf-8"
            )
            review_path = root / "qc_review.tsv"
            pd.DataFrame(
                [
                    {
                        "case_id": row["case_id"],
                        "decision": "PASS",
                        "reviewer_notes": "",
                        "reviewed_at_utc": "2026-09-02T10:00:00Z",
                    }
                ]
            ).to_csv(review_path, sep="\t", index=False)
            command_reassess(argparse.Namespace(report_dir=root))
            command_finalize(
                argparse.Namespace(
                    report_dir=root,
                    review_tsv=review_path,
                    allow_incomplete=False,
                )
            )
            summary = json.loads((root / "qc_final_summary.json").read_text())
            self.assertEqual(summary["status"], "HUMAN_REVIEW_COMPLETED")
            self.assertEqual(summary["decision_counts"], {"PASS": 1})
            self.assertTrue((root / "qc_review_completed.tsv").is_file())
            self.assertTrue((root / "qc_final_report.html").is_file())


if __name__ == "__main__":
    unittest.main()
