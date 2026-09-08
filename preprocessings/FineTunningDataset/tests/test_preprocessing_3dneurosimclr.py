from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "preprocessings"
    / "FineTunningDataset"
    / "preprocessing_3DNeuroSimCLR.py"
)
SPEC = importlib.util.spec_from_file_location("preprocessing_3dneurosimclr", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
simclr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(simclr)


class NeuroSimCLRPreprocessingTests(unittest.TestCase):
    def test_definitive_manifest_contract_and_selection(self) -> None:
        rows = simclr.load_manifest(simclr.DEFAULT_MANIFEST)
        self.assertEqual(len(rows), 1411)
        selected = simclr.select_rows(
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

    def test_output_paths_are_dataset_scoped_and_extractor_compatible(self) -> None:
        paths = simclr.output_paths(Path("/outputs"), "CamCAN", "sub-CC110033")
        self.assertEqual(
            paths["ready"],
            Path(
                "/outputs/CamCAN/sub-CC110033/"
                "sub-CC110033_simclr_ready.nii.gz"
            ),
        )
        self.assertEqual(paths["mask_mni"].name, "sub-CC110033_mask_mni.nii.gz")
        self.assertEqual(paths["qc"].name, "preprocessing_qc.json")

    def test_template_hash_and_geometry_match_the_release_template(self) -> None:
        provenance = simclr.validate_template(simclr.DEFAULT_TEMPLATE)
        self.assertEqual(
            simclr.DEFAULT_TEMPLATE.name,
            "mni_icbm152_t1_tal_nlin_sym_09c.nii",
        )
        self.assertEqual(provenance["sha256"], simclr.EXPECTED_TEMPLATE_SHA256)
        self.assertEqual(tuple(provenance["shape"]), simclr.EXPECTED_TEMPLATE_SHAPE)
        self.assertEqual(
            tuple(provenance["spacing_mm"]),
            simclr.EXPECTED_TEMPLATE_SPACING,
        )

    def test_contract_separates_offline_pipeline_from_model_loader(self) -> None:
        self.assertEqual(simclr.CONTRACT_PAYLOAD["registration"]["transform"], "rigid (-t r)")
        self.assertEqual(
            simclr.CONTRACT_PAYLOAD["skull_stripping"]["negative_voxels"],
            "universally clamp post-SynthStrip intensities below zero to zero",
        )
        self.assertEqual(simclr.CONTRACT_PAYLOAD["normalization"]["modality"], "t1")
        self.assertEqual(simclr.CONTRACT_PAYLOAD["normalization"]["width"], 0.05)
        self.assertNotIn("fallback", simclr.CONTRACT_PAYLOAD["normalization"])
        self.assertEqual(
            simclr.CONTRACT_PAYLOAD["deferred_model_input"]["shape"],
            (1, 150, 192, 192),
        )

    def test_registered_mask_is_numpy_labels_greater_than_or_equal_to_two(self) -> None:
        labels = np.array(
            [[[0, 1], [2, 3]], [[4, 0], [1, 2]]],
            dtype=np.int16,
        )
        expected = (labels >= 2).astype(np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segmentation = root / "segmentation.nii.gz"
            mask = root / "mask.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), segmentation)

            operation = simclr.derive_registered_mask(
                segmentation,
                mask,
                io.StringIO(),
            )
            result = nib.load(str(mask)).get_fdata().astype(np.uint8)

        np.testing.assert_array_equal(result, expected)
        self.assertEqual(operation["expression"], "segmentation >= 2")
        self.assertEqual(operation["foreground_voxels"], int(expected.sum()))

    def test_whitestripe_calls_the_python_class_with_image_and_boolean_mask(self) -> None:
        calls: list[tuple[np.ndarray, np.ndarray]] = []

        class FakeWhiteStripeNormalize:
            def __call__(self, image, mask):
                calls.append((image.copy(), mask.copy()))
                return image + mask.astype(np.float32)

        package = types.ModuleType("intensity_normalization")
        normalize = types.ModuleType("intensity_normalization.normalize")
        whitestripe = types.ModuleType("intensity_normalization.normalize.whitestripe")
        whitestripe.WhiteStripeNormalize = FakeWhiteStripeNormalize
        modules = {
            "intensity_normalization": package,
            "intensity_normalization.normalize": normalize,
            "intensity_normalization.normalize.whitestripe": whitestripe,
        }

        image_data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        mask_data = np.array(
            [[[0, 1], [1, 0]], [[1, 1], [0, 0]]],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.nii.gz"
            mask = root / "mask.nii.gz"
            output = root / "normalized.nii.gz"
            nib.save(nib.Nifti1Image(image_data, np.eye(4)), image)
            nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask)

            with mock.patch.dict(sys.modules, modules):
                operation = simclr.whitestripe_normalize(
                    image,
                    mask,
                    output,
                    io.StringIO(),
                )
            result = nib.load(str(output)).get_fdata(dtype=np.float32)

        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0], image_data)
        np.testing.assert_array_equal(calls[0][1], mask_data.astype(bool))
        np.testing.assert_array_equal(result, image_data + mask_data)
        self.assertEqual(operation["modality"], "t1")
        self.assertEqual(operation["width"], 0.05)

    def test_post_synthstrip_clamp_is_value_based_and_preserves_geometry(self) -> None:
        data = np.array(
            [[[-2.0, 0.0], [1.0, -0.5]], [[3.0, 4.0], [0.0, 2.0]]],
            dtype=np.float32,
        )
        affine = np.array(
            [[1.5, 0.0, 0.0, -4.0], [0.0, 1.5, 0.0, -5.0],
             [0.0, 0.0, 1.5, -6.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "brain.nii.gz"
            nib.save(nib.Nifti1Image(data, affine), path)
            operation = simclr.clamp_negative_voxels(path, io.StringIO())
            result = nib.load(str(path))
            result_data = result.get_fdata(dtype=np.float32)

        np.testing.assert_array_equal(result_data, np.maximum(data, 0.0))
        np.testing.assert_allclose(result.affine, affine, rtol=0.0, atol=0.0)
        self.assertEqual(operation["negative_voxels"], 2)
        self.assertEqual(operation["scope"], "applied uniformly to every session")

    def test_rigid_registration_command_uses_release_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "subject_mni.nii.gz"
            transform = root / "subject_rigid.mat"

            def fake_run(command, log_handle, timeout_seconds):
                prefix = Path(command[command.index("-o") + 1])
                Path(str(prefix) + "Warped.nii.gz").touch()
                Path(str(prefix) + "0GenericAffine.mat").touch()

            with mock.patch.object(simclr, "run_external", side_effect=fake_run):
                command = simclr.rigid_register_ants(
                    root / "brain.nii.gz",
                    root / "template.nii.gz",
                    output,
                    transform,
                    "antsRegistrationSyNQuick.sh",
                    4,
                    io.StringIO(),
                    None,
                )
            self.assertTrue(output.is_file())
            self.assertTrue(transform.is_file())

        self.assertEqual(command[command.index("-t") + 1], "r")
        self.assertEqual(command[command.index("-n") + 1], "4")

    def test_synthstrip_cpu_omits_gpu_and_cuda_requests_it(self) -> None:
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_run(command, log_handle, timeout_seconds):
                commands.append(command)
                Path(command[command.index("-o") + 1]).touch()
                Path(command[command.index("-m") + 1]).touch()

            with mock.patch.object(simclr, "run_external", side_effect=fake_run):
                simclr.skull_strip_synthstrip(
                    root / "n4.nii.gz",
                    root / "brain_cpu.nii.gz",
                    root / "mask_cpu.nii.gz",
                    "mri_synthstrip",
                    "cpu",
                    io.StringIO(),
                    None,
                )
                simclr.skull_strip_synthstrip(
                    root / "n4.nii.gz",
                    root / "brain_gpu.nii.gz",
                    root / "mask_gpu.nii.gz",
                    "mri_synthstrip",
                    "cuda",
                    io.StringIO(),
                    None,
                )

        self.assertNotIn("--gpu", commands[0])
        self.assertIn("--gpu", commands[1])

    def test_synthseg_is_explicitly_cpu_and_receives_thread_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "synthseg.nii.gz"

            def fake_run(command, log_handle, timeout_seconds):
                Path(command[command.index("--o") + 1]).touch()

            with mock.patch.object(simclr, "run_external", side_effect=fake_run):
                command = simclr.run_synthseg(
                    root / "registered.nii.gz",
                    output,
                    "mri_synthseg",
                    6,
                    io.StringIO(),
                    None,
                )

        self.assertIn("--fast", command)
        self.assertIn("--cpu", command)
        self.assertEqual(command[command.index("--threads") + 1], "6")


if __name__ == "__main__":
    unittest.main()
