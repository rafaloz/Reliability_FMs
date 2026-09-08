from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "preprocessing_BrainSegFounder.py"
)
SPEC = importlib.util.spec_from_file_location("preprocessing_brainsegfounder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    nib.save(
        nib.Nifti1Image(
            np.asarray(data),
            np.eye(4, dtype=np.float64) if affine is None else affine,
        ),
        str(path),
    )


class BrainSegFounderPreprocessingTests(unittest.TestCase):
    def test_only_pinned_v3_and_v4_contract_pairs_are_reusable(self) -> None:
        for version, contract in MODULE.REUSABLE_PIPELINE_CONTRACTS.items():
            self.assertTrue(
                MODULE.reusable_pipeline_qc(
                    {
                        "pipeline_version": version,
                        "pipeline_contract_sha256": contract,
                    }
                )
            )
        self.assertFalse(
            MODULE.reusable_pipeline_qc(
                {
                    "pipeline_version": "brainsegfounder-ukb-gdcoff-finetuning-v3",
                    "pipeline_contract_sha256": MODULE.PIPELINE_CONTRACT_SHA256,
                }
            )
        )
        self.assertFalse(MODULE.reusable_pipeline_qc({}))
        self.assertFalse(
            MODULE.reusable_pipeline_qc(
                {
                    "pipeline_version": "unknown",
                    "pipeline_contract_sha256": None,
                }
            )
        )

    def test_manifest_requires_brainsegfounder_and_preserves_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.tsv"
            manifest.write_text(
                "dataset\tsubject\traw_t1w_path\tage\tsex\tsplit\t"
                "inclusion_status\trequired_offline_groups\n"
                "CamCAN\tsub-01\t/raw/sub-01.nii.gz\t20\tF\ttrain\t"
                "INCLUDED\tpreproc_brainsegfounder;other\n",
                encoding="utf-8",
            )
            rows = MODULE.load_manifest(manifest)
            self.assertEqual(len(rows), 1)
            selected = MODULE.select_rows(rows, ["CamCAN"], ["train"], ["sub-01"], None)
            self.assertEqual(selected[0]["subject"], "sub-01")

            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    "preproc_brainsegfounder;other", "other"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not request"):
                MODULE.load_manifest(manifest)

    def test_output_paths_are_dataset_scoped(self) -> None:
        root = Path("/out")
        paths = MODULE.output_paths(root, "WAND", "sub-42")
        self.assertEqual(paths["subject_dir"], root / "WAND" / "sub-42")
        self.assertEqual(
            paths["ready"].name,
            "sub-42_brainsegfounder_ready.nii.gz",
        )
        self.assertEqual(
            paths["reoriented_t1"].name,
            "sub-42_ukb_reoriented.nii.gz",
        )
        self.assertEqual(paths["linear_mat"].suffix, ".mat")

    def test_raw_input_rejects_non_3d(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "four_dimensional.nii.gz"
            save_nifti(path, np.ones((8, 9, 10, 1), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "requires a 3D"):
                MODULE.validate_raw_input(path)

    def test_mni_assets_require_matching_binary_grid_and_pinned_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            MODULE, "EXPECTED_MNI_SHAPE", (8, 9, 10)
        ):
            root = Path(directory)
            standard = root / "standard"
            ukb = root / "ukb"
            standard.mkdir()
            ukb.mkdir()
            head = np.ones((8, 9, 10), dtype=np.float32)
            mask = np.zeros_like(head, dtype=np.uint8)
            mask[1:-1, 1:-1, 1:-1] = 1
            save_nifti(standard / "MNI152_T1_1mm.nii.gz", head)
            save_nifti(standard / "MNI152_T1_1mm_brain.nii.gz", head * mask)
            save_nifti(ukb / "MNI152_T1_1mm_brain_mask.nii.gz", mask)
            save_nifti(ukb / "MNI152_T1_1mm_brain_mask_dil_GD7.nii.gz", mask)
            (ukb / "bb_fnirt.cnf").write_text("--subsamp=4,2,1\n", encoding="utf-8")
            schedule = root / "xyztrans.sch"
            schedule.write_text("setscale 1\n", encoding="utf-8")
            expected_hashes = {
                "fnirt_config": MODULE.sha256_file(ukb / "bb_fnirt.cnf"),
                "mask": MODULE.sha256_file(ukb / "MNI152_T1_1mm_brain_mask.nii.gz"),
                "dilated_mask": MODULE.sha256_file(
                    ukb / "MNI152_T1_1mm_brain_mask_dil_GD7.nii.gz"
                ),
            }
            with mock.patch.object(MODULE, "EXPECTED_UKB_HASHES", expected_hashes):
                result = MODULE.validate_mni_assets(standard, ukb, schedule)
            self.assertEqual(result["head"]["shape"], [8, 9, 10])
            self.assertEqual(result["mask"]["foreground_voxels"], int(mask.sum()))

            bad = mask.astype(np.float32)
            bad[2, 2, 2] = 0.5
            save_nifti(ukb / "MNI152_T1_1mm_brain_mask.nii.gz", bad)
            with self.assertRaisesRegex(ValueError, "not binary"):
                MODULE.validate_mni_assets(standard, ukb, schedule)

    def test_robustfov_parser_matches_released_shell_pipeline(self) -> None:
        output = "Final FOV is:\n0.0 182.0 0.0 218.0 7.0 170.0\n"
        self.assertEqual(MODULE.parse_robustfov_head_top(output), 7)
        with self.assertRaisesRegex(RuntimeError, "Could not parse"):
            MODULE.parse_robustfov_head_top("Final FOV is:\n")

    def test_reorientation_requires_superior_on_voxel_z_and_preserves_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            MODULE, "run_external", side_effect=lambda cmd, *_: cmd
        ):
            root = Path(directory)
            source = root / "source.nii.gz"
            target = root / "target.nii.gz"
            source_affine = np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            save_nifti(source, np.ones((4, 5, 6), dtype=np.float32), source_affine)
            save_nifti(
                target,
                np.ones((6, 4, 5), dtype=np.float32),
                np.diag([-1.0, 1.0, 1.0, 1.0]),
            )

            command, record = MODULE.reorient_to_fsl_standard(
                source, target, "fslreorient2std", mock.Mock(), None
            )
            self.assertEqual(command, ["fslreorient2std", str(source), str(target)])
            self.assertEqual(record["input_axcodes"], ["P", "S", "R"])
            self.assertEqual(record["output_axcodes"], ["L", "A", "S"])

            save_nifti(target, np.ones((4, 5, 6), dtype=np.float32), source_affine)
            with self.assertRaisesRegex(ValueError, "voxel z"):
                MODULE.reorient_to_fsl_standard(
                    source, target, "fslreorient2std", mock.Mock(), None
                )

    def test_fsl_commands_match_ukb_gdc_off_contract(self) -> None:
        log = mock.Mock()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            MODULE, "run_external", side_effect=lambda cmd, *_: cmd
        ):
            root = Path(directory)
            bet = MODULE.skull_strip_bet(
                Path("fov.nii.gz"), Path("brain.nii.gz"), "bet", log, None
            )
            matrices = MODULE.build_linear_transforms(
                Path("T1.nii.gz"),
                Path("raw.nii.gz"),
                Path("roi_to_mni.mat"),
                Path("T1_to_raw.mat"),
                Path("raw_to_T1.mat"),
                Path("T1_to_MNI.mat"),
                "flirt",
                "convert_xfm",
                Path("xyztrans.sch"),
                log,
                None,
            )
            recovery = MODULE.build_header_geometry_transforms(
                Path("T1.nii.gz"),
                Path("raw.nii.gz"),
                Path("roi_to_mni.mat"),
                Path("T1_to_raw.mat"),
                Path("raw_to_T1.mat"),
                Path("T1_to_MNI.mat"),
                root / "header_geometry_preview.nii.gz",
                "flirt",
                "convert_xfm",
                log,
                None,
            )
            fnirt = MODULE.nonlinear_register_fnirt(
                Path("T1.nii.gz"),
                Path("MNI.nii.gz"),
                Path("T1_to_MNI.mat"),
                Path("bb_fnirt.cnf"),
                Path("mask_GD7.nii.gz"),
                Path("coef.nii.gz"),
                Path("warp.nii.gz"),
                Path("jac.nii.gz"),
                Path("preview.nii.gz"),
                Path("fnirt.log"),
                "fnirt",
                log,
                None,
            )
            warped = MODULE.warp_original_to_mni(
                Path("raw.nii.gz"),
                Path("MNI.nii.gz"),
                Path("raw_to_T1.mat"),
                Path("warp.nii.gz"),
                Path("combined.nii.gz"),
                Path("warped.nii.gz"),
                "convertwarp",
                "applywarp",
                log,
                None,
            )
            masked = MODULE.apply_mni_brain_mask(
                Path("warped.nii.gz"),
                Path("mask.nii.gz"),
                Path("fnirt_brain.nii.gz"),
                "fslmaths",
                log,
                None,
            )

        self.assertEqual(bet[-1], "-R")
        self.assertEqual(matrices["flirt"][-2:], ["-schedule", "xyztrans.sch"])
        self.assertIn("-applyxfm", recovery["flirt_usesqform"])
        self.assertIn("-usesqform", recovery["flirt_usesqform"])
        self.assertNotIn("-schedule", recovery["flirt_usesqform"])
        self.assertFalse((root / "header_geometry_preview.nii.gz").exists())
        self.assertIn("--config=bb_fnirt.cnf", fnirt)
        self.assertIn("--refmask=mask_GD7.nii.gz", fnirt)
        self.assertIn("--interp=spline", fnirt)
        self.assertIn("--premat=raw_to_T1.mat", warped["convertwarp"])
        self.assertIn("--rel", warped["applywarp"])
        self.assertEqual(masked[2], "-mul")

    def test_spatial_validation_accepts_fnirt_coefficient_lattice(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            MODULE, "EXPECTED_MNI_SHAPE", (8, 9, 10)
        ):
            root = Path(directory)
            paths = MODULE.output_paths(root, "dataset", "subject")
            paths["subject_dir"].mkdir(parents=True)
            reference_data = np.ones((8, 9, 10), dtype=np.float32)
            brain_mask = np.zeros_like(reference_data)
            brain_mask[1:-1, 1:-1, 1:-1] = 1
            head = root / "MNI152_T1_1mm.nii.gz"
            mask = root / "MNI152_T1_1mm_brain_mask.nii.gz"
            save_nifti(head, reference_data)
            save_nifti(mask, brain_mask)
            for key in (
                "reoriented_t1", "fov_roi", "fov_brain", "reduced_t1", "fnirt_preview",
                "warp_jac", "warped_unmasked",
            ):
                save_nifti(paths[key], reference_data)
            save_nifti(paths["fnirt_brain"], reference_data * brain_mask)
            for key in ("warp_field", "combined_warp"):
                save_nifti(paths[key], np.ones((8, 9, 10, 3), dtype=np.float32))
            save_nifti(
                paths["warp_coef"],
                np.ones((4, 5, 6, 3), dtype=np.float32),
                np.diag([2.0, 2.0, 2.0, 1.0]),
            )
            for key in (
                "roi_to_mni_mat", "t1_to_raw_mat", "raw_to_t1_mat", "linear_mat",
            ):
                np.savetxt(paths[key], np.eye(4))
            paths["fnirt_log"].write_text("FNIRT PASS\n", encoding="utf-8")

            result = MODULE.validate_spatial_outputs(
                paths,
                {
                    "head": {"path": str(head)},
                    "mask": {"path": str(mask)},
                },
            )

            self.assertEqual(result["warp_coef"]["shape"], [4, 5, 6, 3])
            self.assertEqual(
                result["warp_coef"]["geometry"],
                "FNIRT B-spline coefficient lattice",
            )

    def test_ready_validation_enforces_shape_and_zscore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ready.nii.gz"
            rng = np.random.default_rng(42)
            data = np.zeros(MODULE.EXPECTED_READY_SHAPE, dtype=np.float32)
            foreground = np.zeros_like(data, dtype=bool)
            foreground[20:76, 20:76, 20:76] = True
            values = rng.normal(size=int(foreground.sum())).astype(np.float32)
            values = (values - values.mean()) / values.std()
            data[foreground] = values
            save_nifti(path, data)
            result = MODULE.validate_ready_output(path)
            self.assertEqual(result["shape"], [96, 96, 96])
            self.assertAlmostEqual(result["mean_nonzero"], 0.0, places=4)
            self.assertAlmostEqual(result["standard_deviation_nonzero"], 1.0, places=4)

            data[foreground] *= 0.93
            save_nifti(path, data)
            interpolated = MODULE.validate_ready_output(path)
            self.assertAlmostEqual(
                interpolated["standard_deviation_nonzero"], 0.93, places=4
            )

    def test_qc_match_requires_current_hashes_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.nii.gz"
            save_nifti(raw, np.ones((4, 4, 4), dtype=np.float32))
            paths = MODULE.output_paths(root, "dataset", "subject")
            paths["subject_dir"].mkdir(parents=True)
            nifti_keys = (
                "reoriented_t1", "fov_roi", "fov_brain", "reduced_t1", "warp_coef", "warp_field",
                "warp_jac", "fnirt_preview", "combined_warp", "warped_unmasked",
                "fnirt_brain", "ready",
            )
            for key in nifti_keys:
                save_nifti(paths[key], np.ones((4, 4, 4), dtype=np.float32))
            matrix_keys = ("roi_to_mni_mat", "t1_to_raw_mat", "raw_to_t1_mat", "linear_mat")
            for key in matrix_keys:
                paths[key].write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
            paths["fnirt_log"].write_text("FNIRT PASS\n", encoding="utf-8")
            paths["qc_png"].write_bytes(b"png")
            mni_assets = {"head": {"sha256": "head"}}
            artifact_keys = (*nifti_keys, *matrix_keys, "fnirt_log", "qc_png")
            qc = {
                "status": "PASS",
                "pipeline_version": MODULE.PIPELINE_VERSION,
                "pipeline_contract_sha256": MODULE.PIPELINE_CONTRACT_SHA256,
                "mni_assets": mni_assets,
                "raw_input": MODULE.raw_fingerprint(raw),
                "output_sha256": {
                    key: MODULE.sha256_file(paths[key]) for key in artifact_keys
                },
            }
            paths["qc"].write_text(json.dumps(qc), encoding="utf-8")
            self.assertTrue(MODULE.qc_matches(paths, raw, mni_assets))
            paths["ready"].write_bytes(paths["ready"].read_bytes() + b"changed")
            self.assertFalse(MODULE.qc_matches(paths, raw, mni_assets))


if __name__ == "__main__":
    unittest.main()
