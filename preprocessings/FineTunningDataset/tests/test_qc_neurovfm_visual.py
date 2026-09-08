#!/usr/bin/env python3
"""Dependency-light tests for the NeuroVFM visual-QC contract."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from FineTuning.NeuroVFM.model import (  # noqa: E402
    CACHE_BATCH_REQUIRED_KEYS,
    CACHE_PREPROCESSOR_CODE_FILES,
    NEUROVFM_CODE_SHA256,
    NEUROVFM_GITHUB_COMMIT,
    NEUROVFM_MODEL_REPO,
    NEUROVFM_MODEL_REVISION,
    PREPROCESSING_CACHE_VERSION,
)
from FineTuning.shared.cohort import sha256_file  # noqa: E402
from preprocessings.QC_NeuroVFM_visual import (  # noqa: E402
    MODEL_CONFIG_SHA256,
    MONTAGE_VERSION,
    PATCH_SHAPE,
    ROBUST_METRICS,
    Case,
    add_automatic_assessment,
    atomic_json,
    atomic_tsv,
    canonical_json_hash,
    command_finalize,
    inspect_case,
    infer_slice_axis_xyz,
    load_validated_cache,
    model_dhw_to_ras,
    reconstruct_cache,
    render_panel,
    report_html,
)


class NeuroVFMVisualQCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_case(self, *, duplicate_coordinate: bool = False) -> Case:
        raw_path = self.root / "raw" / "sub-001_T1w.nii.gz"
        raw_path.parent.mkdir(parents=True)
        axes = np.ogrid[
            tuple(slice(-1.0, 1.0, 24j) for _ in range(3))
        ]
        raw = np.exp(
            -4.0
            * sum(
                (axis / scale) ** 2
                for axis, scale in zip(axes, (0.7, 0.8, 0.7))
            )
        ).astype(np.float32)
        nib.save(nib.Nifti1Image(raw, np.eye(4)), str(raw_path))
        raw_path = raw_path.resolve()
        raw_stat = raw_path.stat()
        raw_source = {
            "path": str(raw_path),
            "size_bytes": int(raw_stat.st_size),
            "mtime_ns": int(raw_stat.st_mtime_ns),
            "sha256": sha256_file(raw_path),
        }

        directory = self.root / "cache" / "Demo" / "sub-001"
        directory.mkdir(parents=True)
        cache_path = directory / "study_preprocessor.pt"
        sidecar_path = directory / "study_preprocessor.json"
        grid = np.stack(
            np.meshgrid(
                np.arange(2),
                np.arange(2),
                np.arange(2),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        if duplicate_coordinate:
            grid[-1] = grid[0]
        n_tokens = len(grid)
        img = torch.linspace(
            0.0,
            1.0,
            n_tokens * int(np.prod(PATCH_SHAPE)),
            dtype=torch.float32,
        ).reshape(n_tokens, -1)
        batch = {
            "img": img,
            "coords": torch.from_numpy(grid).long(),
            "series_masks_indices": torch.empty(0),
            "series_cu_seqlens": torch.tensor(
                [0, n_tokens], dtype=torch.int32
            ),
            "series_max_len": n_tokens,
            "mode": ["mri"],
            "path": [str(raw_path)],
            "size": [(8, 32, 32)],
        }
        self.assertEqual(set(batch), CACHE_BATCH_REQUIRED_KEYS)
        contract = {
            "cache_version": PREPROCESSING_CACHE_VERSION,
            "model_repository": NEUROVFM_MODEL_REPO,
            "model_revision": NEUROVFM_MODEL_REVISION,
            "config_sha256": MODEL_CONFIG_SHA256,
            "github_commit": NEUROVFM_GITHUB_COMMIT,
            "preprocessor_files_sha256": {
                relative: NEUROVFM_CODE_SHA256[relative]
                for relative in CACHE_PREPROCESSOR_CODE_FILES
            },
            "study_preprocessor": {
                "patch_size": list(PATCH_SHAPE),
                "target_spacing": [1.0, 1.0, 4.0],
                "remove_background": True,
            },
            "preprocessing_software": {
                "neurovfm": "test",
                "torch": str(torch.__version__),
                "numpy": str(np.__version__),
                "nibabel": str(nib.__version__),
                "SimpleITK": "test",
            },
            "modality": "mri",
            "batch_keys": sorted(CACHE_BATCH_REQUIRED_KEYS),
        }
        contract["contract_sha256"] = canonical_json_hash(contract)
        payload = {
            "cache_version": PREPROCESSING_CACHE_VERSION,
            "preprocessing_contract_sha256": contract["contract_sha256"],
            "dataset": "Demo",
            "subject": "sub-001",
            "raw_source": raw_source,
            "batch": batch,
        }
        torch.save(payload, cache_path)
        metadata = {
            "status": "PASS",
            "cache_version": PREPROCESSING_CACHE_VERSION,
            "preprocessing_contract": contract,
            "dataset": "Demo",
            "subject": "sub-001",
            "raw_source": raw_source,
            "artifact": str(cache_path),
            "artifact_sha256": sha256_file(cache_path),
            "artifact_size_bytes": cache_path.stat().st_size,
            "batch": {
                "keys": sorted(batch),
                "img": {
                    "shape": list(img.shape),
                    "dtype": str(img.dtype),
                },
                "coords": {
                    "shape": list(batch["coords"].shape),
                    "dtype": str(batch["coords"].dtype),
                },
                "series_masks_indices": {
                    "shape": [0],
                    "dtype": str(
                        batch["series_masks_indices"].dtype
                    ),
                },
                "series_cu_seqlens": {
                    "shape": [2],
                    "dtype": str(batch["series_cu_seqlens"].dtype),
                },
                "series_max_len": n_tokens,
                "mode": ["mri"],
                "path": [str(raw_path)],
                "size": [[8, 32, 32]],
                "n_tokens": n_tokens,
            },
        }
        sidecar_path.write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return Case(
            case_id="finetuning::Demo::sub-001",
            scope="finetuning",
            metric_group="finetuning::Demo",
            dataset="Demo",
            subject="sub-001",
            session="",
            acquisition_id="",
            split="train",
            raw_path=raw_path,
            cache_path=cache_path,
            sidecar_path=sidecar_path,
        )

    def test_cache_reconstruction_is_bit_exact_and_renders_relative_image(
        self,
    ) -> None:
        case = self.make_case()
        batch, provenance = load_validated_cache(
            case, verify_raw_sha256=True
        )
        volume, support, occupancy = reconstruct_cache(batch)
        self.assertEqual(volume.shape, (8, 32, 32))
        self.assertTrue(support.all())
        self.assertEqual(int(occupancy.sum()), 8)
        self.assertEqual(provenance["raw_sha256_verified"], True)
        report_dir = self.root / "report"
        row = inspect_case(
            case, report_dir, verify_raw_sha256=True
        )
        image_path = report_dir / row["image_relpath"]
        self.assertTrue(image_path.is_file())
        self.assertEqual(
            row["image_sha256"], sha256_file(image_path)
        )
        self.assertEqual(row["montage_version"], MONTAGE_VERSION)
        self.assertEqual(row["n_tokens"], 8)

    def test_duplicate_coordinates_are_rejected(self) -> None:
        case = self.make_case(duplicate_coordinate=True)
        with self.assertRaisesRegex(
            ValueError, "duplicate patch coordinates"
        ):
            load_validated_cache(case, verify_raw_sha256=True)

    def test_model_layout_is_restored_from_rpi_to_ras(self) -> None:
        model = np.zeros((2, 3, 4), dtype=np.float32)
        model[0, 0, 3] = 1.0
        restored = model_dhw_to_ras(model, slice_axis_xyz=2)
        self.assertEqual(restored.shape, (4, 3, 2))
        self.assertEqual(float(restored[3, 2, 1]), 1.0)
        self.assertEqual(float(restored.sum()), 1.0)

    def test_slice_axis_and_small_token_panels_are_visible(self) -> None:
        self.assertEqual(
            infer_slice_axis_xyz(
                (32, 32, 32), (1.0, 1.0, 1.0), (8, 32, 32)
            ),
            2,
        )
        panel = render_panel(
            np.asarray([[0.2, 0.8], [0.4, 1.0]], dtype=np.float32),
            low=0.0,
            high=1.0,
            support=np.ones((2, 2), dtype=bool),
            resample=Image.Resampling.NEAREST,
        )
        pixels = np.asarray(panel)
        visible = np.any(pixels != 0, axis=2)
        rows, columns = np.where(visible)
        self.assertGreater(rows.max() - rows.min(), 200)
        self.assertGreater(columns.max() - columns.min(), 200)

    def test_typicality_score_prioritises_outliers_without_excluding(
        self,
    ) -> None:
        rows = []
        for index, value in enumerate([1.0, 1.1, 0.9, 1.05, 8.0]):
            row = {
                "case_id": f"case-{index}",
                "scope": "finetuning",
                "metric_group": "finetuning::Demo",
                "dataset": "Demo",
                "subject": f"sub-{index}",
                "session": "",
            }
            for metric in ROBUST_METRICS:
                row[metric] = (
                    value if metric == "n_tokens" else 1.0
                )
            rows.append(row)
        scored = add_automatic_assessment(
            pd.DataFrame(rows)
        ).set_index("case_id")
        self.assertEqual(
            scored.loc["case-4", "automatic_priority"], "HIGH"
        )
        self.assertLess(
            scored.loc["case-4", "automatic_typicality_score"],
            scored.loc["case-0", "automatic_typicality_score"],
        )
        self.assertTrue(
            scored["automatic_typicality_score"]
            .between(0.0, 10.0)
            .all()
        )
        self.assertTrue((scored["decision"] == "UNREVIEWED").all())

    def test_html_uses_relative_images_keyboard_and_label_blind_copy(
        self,
    ) -> None:
        row = inspect_case(
            self.make_case(),
            self.root / "html",
            verify_raw_sha256=False,
        )
        frame = add_automatic_assessment(pd.DataFrame([row]))
        rendered = report_html(frame, "now")
        self.assertIn(row["image_relpath"], rendered)
        self.assertIn("ArrowRight", rendered)
        self.assertIn("decide('PASS')", rendered)
        self.assertIn("qc_review.tsv", rendered)
        self.assertIn(
            "No age, sex, diagnosis, prediction, BAG or error was loaded",
            rendered,
        )
        self.assertIn("visible.length", rendered)
        self.assertIn(r"replace(/[\t\r\n]+/g", rendered)
        self.assertIn(r"columns.join('\t')", rendered)
        self.assertIn(r"lines.join('\n')+'\n'", rendered)

    def test_finalize_freezes_one_decision_per_case(self) -> None:
        report_dir = self.root / "final"
        row = inspect_case(
            self.make_case(),
            report_dir,
            verify_raw_sha256=False,
        )
        frame = add_automatic_assessment(pd.DataFrame([row]))
        metrics_path = report_dir / "qc_metrics.tsv"
        atomic_tsv(metrics_path, frame)
        atomic_json(
            report_dir / "qc_source_manifest.json",
            {
                "metrics_sha256": sha256_file(metrics_path),
                "montage_version": MONTAGE_VERSION,
            },
        )
        review_path = self.root / "qc_review.tsv"
        atomic_tsv(
            review_path,
            pd.DataFrame(
                [
                    {
                        "case_id": row["case_id"],
                        "decision": "PASS",
                        "reviewer_notes": "visually acceptable",
                    }
                ]
            ),
        )
        command_finalize(
            Namespace(
                report_dir=report_dir,
                review_tsv=review_path,
                allow_incomplete=False,
            )
        )
        summary = json.loads(
            (report_dir / "qc_final_summary.json").read_text()
        )
        self.assertEqual(
            summary["status"], "HUMAN_REVIEW_COMPLETED"
        )
        self.assertEqual(summary["decision_counts"], {"PASS": 1})
        self.assertTrue(
            (report_dir / "qc_review_completed.tsv").is_file()
        )
        self.assertTrue(
            (report_dir / "qc_final_report.html").is_file()
        )


if __name__ == "__main__":
    unittest.main()
