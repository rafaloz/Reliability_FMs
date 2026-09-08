from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "preprocessings" / "QC_CAT12_visual.py"
SPEC = importlib.util.spec_from_file_location("qc_cat12_visual", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qc
SPEC.loader.exec_module(qc)


def gaussian(shape: tuple[int, int, int]) -> np.ndarray:
    axes = np.ogrid[tuple(slice(-1.0, 1.0, complex(size)) for size in shape)]
    radius = sum((axis / scale) ** 2 for axis, scale in zip(axes, (0.72, 0.82, 0.70)))
    return np.exp(-4.0 * radius).astype(np.float32)


class Cat12VisualQCTests(unittest.TestCase):
    def test_acquisition_and_scanner_parsing(self) -> None:
        value = "/x/sub-03286_ses-NOT1ACH001_T1w.nii.gz"
        self.assertEqual(qc.acquisition_id(value), "sub-03286_ses-NOT1ACH001_T1w")
        self.assertEqual(qc.scanner_from_session("NOT1ACH001"), "NOT1ACH")

    def test_metrics_are_finite_and_geometry_based(self) -> None:
        data = gaussian((21, 23, 19))
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        metrics = qc.gm_metrics(data, affine)
        self.assertGreater(metrics["gm_volume_ml"], 0.0)
        self.assertGreater(metrics["gm_support_fraction"], 0.0)
        self.assertLess(metrics["gm_edge_mass_fraction"], 0.10)
        self.assertAlmostEqual(metrics["gm_lr_asymmetry"], 0.0, places=5)
        raw = qc.raw_metrics(data * 100.0)
        self.assertEqual(raw["raw_finite_fraction"], 1.0)
        self.assertGreater(raw["raw_robust_dynamic_range"], 0.0)

    def test_robust_z_uses_mad_and_flags_extreme_value(self) -> None:
        values = np.array([1.0, 1.1, 0.9, 1.05, 10.0])
        scores = qc.robust_z_scores(values)
        self.assertGreater(scores[-1], qc.ROBUST_SEVERE_Z)
        self.assertLess(np.max(np.abs(scores[:-1])), qc.ROBUST_OUTLIER_Z)

    def test_quality_score_inverts_siqr_and_keeps_relative_priority_separate(self) -> None:
        rows = []
        for case_id, adverse, siqr, status in (
            ("good", 0.0, 1.0, "PASS"),
            ("outlier", 5.0, 1.0, "PASS"),
            ("lower_quality", 0.0, 5.0, "PASS"),
            ("failure", 0.0, 1.0, "FAIL"),
        ):
            row = {
                "case_id": case_id,
                "source": "finetuning",
                "technical_status": status,
                "gm_negative_fraction": 0.0,
                "cat12_siqr": siqr,
            }
            for metric, direction in qc.ROBUST_METRIC_DIRECTIONS.items():
                if metric == "within_subject_ncc":
                    value = np.nan
                elif direction == "low":
                    value = -adverse
                else:
                    value = adverse
                row[f"{metric}_robust_z"] = value
            rows.append(row)
        scored = qc.add_automatic_assessment(pd.DataFrame(rows)).set_index("case_id")
        self.assertEqual(scored.loc["good", "auto_quality_score"], 9.5)
        self.assertEqual(scored.loc["outlier", "auto_quality_score"], 9.5)
        self.assertEqual(scored.loc["lower_quality", "auto_quality_score"], 5.5)
        self.assertEqual(scored.loc["failure", "auto_quality_score"], 0.0)
        self.assertEqual(scored.loc["outlier", "auto_priority"], "HIGH")
        self.assertLess(
            scored.loc["outlier", "auto_typicality_score"],
            scored.loc["good", "auto_typicality_score"],
        )
        self.assertTrue(scored["auto_quality_score"].between(0.0, 10.0).all())
        self.assertFalse(qc.quality_score_contract()["exclusion_decision"])

    def test_finetuning_loader_reads_no_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.tsv"
            pd.DataFrame(
                [
                    {
                        "dataset": "Demo",
                        "subject": "sub-001",
                        "raw_t1w_path": "/raw/sub-001.nii.gz",
                        "age": "SECRET_AGE",
                        "sex": "SECRET_SEX",
                        "split": "train",
                        "inclusion_status": "INCLUDED",
                        "required_offline_groups": "cat12_ixi555_gm",
                    }
                ]
            ).to_csv(manifest, sep="\t", index=False)
            cases = qc.load_ft_cases(manifest, root / "preproc", expected_cases=None)
        self.assertEqual(len(cases), 1)
        self.assertNotIn("age", qc.asdict(cases[0]))
        self.assertNotIn("sex", qc.asdict(cases[0]))

    def test_onharmony_loader_pairs_paths_by_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.txt"
            gm = root / "gm.txt"
            provenance = root / "provenance.tsv"
            ids = [
                "sub-001_ses-NOT1ACH001_T1w",
                "sub-002_ses-OXF1PRI001_T1w",
            ]
            raw.write_text("\n".join(f"/raw/{stem}.nii.gz" for stem in ids) + "\n")
            gm.write_text("\n".join(f"/gm/{stem}_anatcl_ready.nii.gz" for stem in reversed(ids)) + "\n")
            pd.DataFrame(
                {
                    "acquisition_id": ids,
                    "raw_t1w_sha256": ["a" * 64, "b" * 64],
                    "output_sha256": ["c" * 64, "d" * 64],
                    "status": ["PASS", "PASS"],
                }
            ).to_csv(provenance, sep="\t", index=False)
            cases = qc.load_onh_cases(raw, gm, provenance, expected_cases=None)
        self.assertEqual([case.acquisition_id for case in cases], ids)
        self.assertTrue(cases[0].gm_path.name.startswith(ids[0]))
        self.assertEqual(cases[0].group, "NOT1ACH")

    def test_case_inspection_accepts_historical_raw_without_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw.nii.gz"
            gm_dir = root / "Demo" / "sub-001"
            gm_dir.mkdir(parents=True)
            gm_path = gm_dir / "sub-001_anatcl_ready.nii.gz"
            qc_path = gm_dir / "preprocessing_qc.json"
            report_dir = gm_dir / "cat12_work" / "report"
            report_dir.mkdir(parents=True)
            raw_data = gaussian((35, 37, 33)) * 100.0
            gm_data = gaussian(qc.EXPECTED_GM_SHAPE)
            affine = np.diag([-1.5, 1.5, 1.5, 1.0])
            nib.save(nib.Nifti1Image(raw_data, np.eye(4)), raw_path)
            nib.save(nib.Nifti1Image(gm_data, affine), gm_path)
            qc_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "dataset": "Demo",
                        "subject": "sub-001",
                        "split": "train",
                        "raw_input": {"path": str(raw_path)},
                        "anatcl_output_sha256": "e" * 64,
                    }
                )
            )
            (report_dir / "cat_sub-001.xml").write_text(
                "<S><qualityratings><SIQR>1.75</SIQR><NCR>1.6</NCR>"
                "<ICR>1.8</ICR><res_RMS>2.0</res_RMS></qualityratings></S>"
            )
            case = qc.Case(
                case_id="finetuning::Demo::sub-001",
                source="finetuning",
                group="Demo",
                dataset="Demo",
                subject="sub-001",
                session="",
                acquisition_id="sub-001",
                split="train",
                raw_path=raw_path,
                gm_path=gm_path,
                qc_path=qc_path,
            )
            record, downsample = qc.inspect_case(
                case,
                gaussian(qc.EXPECTED_GM_SHAPE),
                root / "montages",
                verify_hashes=False,
            )
        self.assertEqual(record["technical_status"], "PASS")
        self.assertEqual(record["raw_sha256"], "")
        self.assertTrue(record["gm_loaded"])
        self.assertEqual(record["cat12_xml_status"], "PASS")
        self.assertEqual(record["cat12_siqr"], 1.75)
        self.assertEqual(downsample.shape, (61, 73, 61))

    def test_report_payload_is_label_blind(self) -> None:
        row = {
            "case_id": "finetuning::Demo::sub-001",
            "source": "finetuning",
            "group": "Demo",
            "dataset": "Demo",
            "subject": "sub-001",
            "session": "",
            "acquisition_id": "sub-001",
            "split": "train",
            "auto_priority": "LOW",
            "auto_score": 0,
            "auto_quality_score": 9.4,
            "auto_quality_coverage": 1.0,
            "auto_review_registration": 9.2,
            "auto_review_morphology": 9.5,
            "auto_review_cat12": 9.6,
            "auto_review_repeatability": np.nan,
            "auto_flags": "",
            "technical_status": "PASS",
            "technical_errors": "",
            "montage": "montages/a.jpg",
            "raw_sha256": "",
            "gm_sha256": "a" * 64,
            "montage_sha256": "b" * 64,
            "gm_volume_ml": 600,
            "gm_template_ncc": 0.9,
            "gm_quartic_mean_z": 1.0,
            "gm_com_distance_mm": 1.0,
            "gm_edge_mass_fraction": 0.01,
            "gm_lr_asymmetry": 0.1,
            "within_subject_ncc": np.nan,
        }
        report = qc.build_report_html(pd.DataFrame([row]), created_utc="2026-01-01T00:00:00Z")
        self.assertIn("sub-001", report)
        self.assertNotIn("SECRET_AGE", report)
        self.assertNotIn("SECRET_SEX", report)
        self.assertIn("No age, prediction, BAG, diagnosis or outcome", report)
        self.assertIn("CAT12 QUALITY $", report)
        self.assertIn("Automatic CAT12 quality score (0-10)", report)

    def test_rescore_updates_completed_build_without_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_dir = Path(temporary)
            row = {
                "case_id": "finetuning::Demo::sub-001",
                "source": "finetuning",
                "group": "Demo",
                "dataset": "Demo",
                "subject": "sub-001",
                "session": "",
                "acquisition_id": "sub-001",
                "split": "train",
                "auto_priority": "LOW",
                "auto_score": 0,
                "auto_flags": "",
                "technical_status": "PASS",
                "technical_errors": "",
                "gm_negative_fraction": 0.0,
                "cat12_siqr": 2.0,
                "montage": "montages/a.jpg",
                "raw_sha256": "",
                "gm_sha256": "a" * 64,
                "montage_sha256": "b" * 64,
                "gm_volume_ml": 600.0,
                "gm_template_ncc": 0.9,
                "gm_quartic_mean_z": 1.0,
                "gm_com_distance_mm": 1.0,
                "gm_edge_mass_fraction": 0.01,
                "gm_lr_asymmetry": 0.1,
                "within_subject_ncc": np.nan,
            }
            for metric in qc.ROBUST_METRIC_DIRECTIONS:
                row[f"{metric}_robust_z"] = np.nan if metric == "within_subject_ncc" else 0.0
            metrics = pd.DataFrame([row])
            metrics_path = report_dir / "qc_metrics.tsv"
            review_path = report_dir / "qc_review.tsv"
            html_path = report_dir / "index.html"
            summary_path = report_dir / "qc_summary.json"
            metrics.to_csv(metrics_path, sep="\t", index=False)
            qc.initial_review_frame(metrics).to_csv(review_path, sep="\t", index=False)
            html_path.write_text("old")
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "outputs": {
                            "metrics_sha256": qc.sha256_file(metrics_path),
                            "blank_review_sha256": qc.sha256_file(review_path),
                            "html_sha256": qc.sha256_file(html_path),
                        },
                    }
                )
            )
            qc.command_rescore(SimpleNamespace(report_dir=report_dir))
            rescored = pd.read_csv(metrics_path, sep="\t")
            summary = json.loads(summary_path.read_text())
            html_text = html_path.read_text()
        self.assertEqual(rescored.loc[0, "auto_quality_score"], 8.5)
        self.assertEqual(summary["quality_score_contract"]["version"], qc.QUALITY_SCORE_VERSION)
        self.assertIn("QUALITY", html_text)

    def test_final_review_requires_exact_identity_and_complete_decisions(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "case_id": "a",
                    "source": "finetuning",
                    "dataset": "Demo",
                    "subject": "sub-001",
                    "session": "",
                    "acquisition_id": "sub-001",
                    "raw_sha256": "",
                    "gm_sha256": "g",
                    "montage_sha256": "m",
                }
            ]
        )
        review = pd.DataFrame(
            [
                {
                    "case_id": "a",
                    "source": "finetuning",
                    "dataset": "Demo",
                    "subject": "sub-001",
                    "session": "",
                    "acquisition_id": "sub-001",
                    "decision": "PASS",
                    "reasons": "",
                    "notes": "",
                    "rater": "Rater",
                    "reviewed_at_utc": "2026-01-01T00:00:00Z",
                    "raw_sha256": "",
                    "gm_sha256": "g",
                    "montage_sha256": "m",
                }
            ]
        )
        joined = qc._validate_review(metrics, review, allow_incomplete=False)
        self.assertEqual(joined.loc[0, "decision"], "PASS")
        review.loc[0, "decision"] = "UNREVIEWED"
        with self.assertRaises(ValueError):
            qc._validate_review(metrics, review, allow_incomplete=False)


if __name__ == "__main__":
    unittest.main()
