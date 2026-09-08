#!/usr/bin/env python3
"""Dependency-light tests for the 3D-Neuro-SimCLR brain-age workflow."""

from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


TEST_PATH = Path(__file__).resolve()
MODEL_DIR = TEST_PATH.parents[1]
PROJECT_ROOT = TEST_PATH.parents[3]
for import_root in (PROJECT_ROOT, MODEL_DIR):
    import_path = str(import_root)
    if import_path in sys.path:
        sys.path.remove(import_path)
    sys.path.insert(0, import_path)

for module_alias in (
    "model",
    "extract_embeddings",
    "finetune",
    "fit_ridge",
    "predict_onharmony",
):
    sys.modules.pop(module_alias, None)

import model as neuro_simclr_model  # noqa: E402
import fit_ridge as fit_ridge_module  # noqa: E402
import Emb_extractors.extract_3DNeuroSimCLR as onharmony_extractor  # noqa: E402
from model import (  # noqa: E402
    MODEL_INPUT_SHAPE,
    PREPROCESSING_CONTRACT_SHA256,
    PREPROCESSING_PIPELINE_VERSION,
    REGISTERED_SHAPE,
    NeuroSimCLRModel,
    NeuroSimCLROptions,
    clean_encoder_state_dict,
    load_and_validate_manifest,
    masked_standardize,
    transpose_crop,
)
from extract_embeddings import (  # noqa: E402
    PIPELINE_VERSION as EXTRACTION_PIPELINE_VERSION,
    command_audit as audit_embeddings,
    command_extract as extract_embeddings,
)
from finetune import (  # noqa: E402
    PIPELINE_VERSION as ADAPTED_PIPELINE_VERSION,
    validate_evaluation_options,
    validate_selection_candidates,
)
from fit_ridge import (  # noqa: E402
    PIPELINE_VERSION as RIDGE_PIPELINE_VERSION,
    fit_code_hashes,
    validate_evaluation_lock,
)
from predict_onharmony import (  # noqa: E402
    EXPECTED_TEMPLATE_SHA256,
    OnHarmonyNeuroSimCLRModel,
    artifact_paths,
    validate_artifact_set,
)
from FineTuning.shared.cohort import frame_for_splits, sha256_file  # noqa: E402
from FineTuning.shared.evaluation import validate_selected_evaluation_run  # noqa: E402
from FineTuning.shared.metrics import (  # noqa: E402
    embedding_columns,
    regression_metrics,
    validate_embedding_frame,
)
from FineTuning.shared.protocol import (  # noqa: E402
    ADAPTED_FIXED_HYPERPARAMETERS,
    ADAPTED_LR_CONFIGS,
    FROZEN_RIDGE_ALPHAS,
    adapted_learning_rates,
    select_adapted_learning_rate,
)
from FineTuning.shared.ridge import (  # noqa: E402
    atomic_npz,
    fit_ridge_grid,
    load_npz,
    parse_ridge_alphas,
    predict_ridge,
    validate_frozen_embedding_contract,
)
from FineTuning.shared.onharmony import (  # noqa: E402
    ArtifactRegistry,
    FORBIDDEN_LABEL_COLUMNS,
    effective_precision,
    load_onharmony_manifest,
    validate_lock_header,
    validate_prediction_frame,
)
from FineTuning.shared.runtime import atomic_json, atomic_tsv  # noqa: E402


QC_SCRIPT_PATH = PROJECT_ROOT / "preprocessings/QC_3DNeuroSimCLR_visual.py"
QC_MODULE_SPEC = importlib.util.spec_from_file_location(
    "qc_3dneurosimclr_visual", QC_SCRIPT_PATH
)
if QC_MODULE_SPEC is None or QC_MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot import visual QC script: {QC_SCRIPT_PATH}")
qc_visual = importlib.util.module_from_spec(QC_MODULE_SPEC)
sys.modules[QC_MODULE_SPEC.name] = qc_visual
QC_MODULE_SPEC.loader.exec_module(qc_visual)


class VisualQCReportTests(unittest.TestCase):
    def test_report_javascript_escapes_tsv_control_characters(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "case_id": "brain_age::Dataset::sub-001",
                    "scope": "brain_age",
                    "group": "brain_age::Dataset",
                    "dataset": "Dataset",
                    "subject": "sub-001",
                    "session": "",
                    "acquisition_id": "",
                    "split": "train",
                    "montage_file": "montages/example.jpg",
                    "auto_priority": "ROUTINE",
                    "auto_reasons": "",
                    "mask_volume_ml": 1200.0,
                    "mask_largest_component_fraction": 0.999,
                    "mask_boundary_fraction": 0.001,
                    "cropped_mask_retention_fraction": 0.995,
                }
            ]
        )

        generated = qc_visual.report_html(frame, "2026-08-30T00:00:00+00:00")

        self.assertIn("rows.join('\\n')+'\\n'", generated)
        self.assertIn(".join('\\t')", generated)
        self.assertNotIn("rows.join('\n')+'\n'", generated)
        self.assertNotIn(".join('\t')", generated)
        self.assertIn('src="${esc(c.montage_file)}"', generated)
        self.assertIn("key==='arrowdown'||key==='j'", generated)
        self.assertIn("key==='arrowup'||key==='k'", generated)
        self.assertIn("decide('PASS')", generated)
        self.assertIn("decide('REVIEW')", generated)
        self.assertIn("decide('FAIL')", generated)
        self.assertIn('aria-keyshortcuts="P 1"', generated)


class FrozenOnHarmonyExtractorTests(unittest.TestCase):
    def test_extractor_locks_bfloat16_seed_and_exact_adapter(self) -> None:
        args = onharmony_extractor.parse_args([])
        self.assertEqual(args.amp, "bfloat16")
        self.assertEqual(args.seed, 42)
        self.assertIs(
            onharmony_extractor.OnHarmonyNeuroSimCLRModel,
            OnHarmonyNeuroSimCLRModel,
        )

        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        cuda = torch.device("cuda")
        with patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            onharmony_extractor.validate_runtime_contract(
                cuda, "bfloat16", 42
            )
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            onharmony_extractor.validate_runtime_contract(
                torch.device("cpu"), "bfloat16", 42
            )
        with self.assertRaisesRegex(ValueError, "requires amp"):
            onharmony_extractor.validate_runtime_contract(cuda, "off", 42)
        with self.assertRaisesRegex(ValueError, "requires seed"):
            onharmony_extractor.validate_runtime_contract(cuda, "bfloat16", 0)

    def test_build_model_loads_and_freezes_common_adapter(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        registry = object()
        contract = {"contract": "value"}
        device = torch.device("cpu")
        with patch.object(
            onharmony_extractor, "OnHarmonyNeuroSimCLRModel"
        ) as adapter_class:
            adapter = adapter_class.return_value
            adapter.set_frozen.return_value = {
                "encoder_trainable_parameters": 0
            }
            observed, summary = onharmony_extractor.build_model(
                Path("checkpoint.tar"), registry, contract, device
            )
        options, observed_registry, observed_contract = adapter_class.call_args.args
        self.assertEqual(options.checkpoint, Path("checkpoint.tar"))
        self.assertIs(observed_registry, registry)
        self.assertEqual(observed_contract, contract)
        adapter.load.assert_called_once_with(device)
        adapter.set_frozen.assert_called_once_with()
        self.assertIs(observed, adapter)
        self.assertEqual(summary["encoder_trainable_parameters"], 0)

    def test_embedding_is_serialized_as_float32_after_forward(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummyAdapter:
            def __init__(self) -> None:
                self.mode_calls = 0

            def controlled_train_mode(self) -> None:
                self.mode_calls += 1

            @staticmethod
            def encode(record, *, training, epoch, seed):
                del record, training, epoch, seed
                return torch.arange(
                    neuro_simclr_model.EMBEDDING_DIM,
                    dtype=torch.bfloat16,
                )

        adapter = DummyAdapter()
        embedding = onharmony_extractor.extract_embedding(
            adapter,
            {"acquisition_id": "sub-1_ses-A001_T1w"},
            device=torch.device("cpu"),
            amp="off",
            seed=42,
        )
        self.assertEqual(adapter.mode_calls, 1)
        self.assertEqual(embedding.shape, (neuro_simclr_model.EMBEDDING_DIM,))
        self.assertEqual(embedding.dtype, np.float32)

    def test_output_metadata_is_label_blind(self) -> None:
        observed = onharmony_extractor.output_metadata(
            {"subject": "sub-03286", "session": "NOT1ACH001"},
            Path("/tmp/input.nii.gz"),
        )
        self.assertEqual(
            observed,
            {
                "file_path": "/tmp/input.nii.gz",
                "subject_id": "sub-03286",
                "session_id": "ses-NOT1ACH001",
                "scanner": "NOT1ACH",
            },
        )
        self.assertNotIn("age", observed)
        self.assertNotIn("sex", observed)


class CohortAndPathTests(unittest.TestCase):
    def test_manifest_permits_offline_simclr_inputs(self) -> None:
        frame, summary = load_and_validate_manifest()
        self.assertEqual(len(frame), 1411)
        self.assertEqual(len(frame), summary["n_subjects"])
        self.assertEqual(
            summary["split_counts"],
            {"train": 1130, "validation": 141, "internal_test": 140},
        )
        self.assertEqual(
            summary["input_contract"],
            {"required_offline_group": "preproc_3d_neuro_simclr"},
        )
        self.assertEqual(summary["on_harmony_firewall"], "PASS")

    def test_dataset_scoped_paths_are_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = NeuroSimCLRModel(
                NeuroSimCLROptions(
                    checkpoint=Path(directory) / "checkpoint.tar",
                    preproc_root=Path(directory) / "preprocessed",
                )
            )
            record = {"dataset": "CamCAN", "subject": "sub-001", "split": "train"}
            self.assertEqual(
                model.input_path(record),
                Path(directory)
                / "preprocessed"
                / "CamCAN"
                / "sub-001"
                / "sub-001_simclr_ready.nii.gz",
            )
            self.assertEqual(
                model.mask_path(record).name,
                "sub-001_mask_mni.nii.gz",
            )
            self.assertEqual(model.qc_path(record).name, "preprocessing_qc.json")


class OnHarmonyPredictionContractTests(unittest.TestCase):
    def test_effective_precision_matches_autocast_device_semantics(self) -> None:
        class Device:
            def __init__(self, device_type: str) -> None:
                self.type = device_type

        self.assertEqual(effective_precision(Device("cpu"), "bfloat16"), "float32")
        self.assertEqual(effective_precision(Device("cuda"), "off"), "float32")
        self.assertEqual(effective_precision(Device("cuda"), "float16"), "float16")
        self.assertEqual(effective_precision(Device("cuda"), "bfloat16"), "bfloat16")

    @staticmethod
    def _write_external_manifest(root: Path) -> tuple[Path, Path]:
        paths = []
        metadata_rows = []
        for participant, session, cohort in (
            ("00001", "NOT1ACH001", "A"),
            ("00002", "OXF1PRI001", "B"),
        ):
            acquisition = f"sub-{participant}_ses-{session}_T1w"
            path = root / "original" / f"{acquisition}.nii.gz"
            paths.append(str(path.resolve()))
            metadata_rows.append(
                {
                    "session": f"{participant}_{session}",
                    "cohort": cohort,
                    "scanner_code": "Ach" if cohort == "A" else "Pri32",
                    "scanner_name": "scanner",
                    "vendor": "vendor",
                    "scanner_type": "type",
                    "scanner_typeloc": "location",
                    "scan_session_num": "_001",
                    "scan_session": session,
                    "subject": f"_{participant}",
                    "age": "DO_NOT_LOAD",
                }
            )
        filelist = root / "filelist.txt"
        filelist.write_text("\n".join(paths) + "\n", encoding="utf-8")
        metadata = root / "metadata.csv"
        pd.DataFrame(metadata_rows).to_csv(metadata, index=False)
        return filelist, metadata

    def test_external_manifest_is_ordered_and_label_blind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            filelist, metadata = self._write_external_manifest(Path(directory))
            with patch.object(pd, "read_csv", wraps=pd.read_csv) as read_csv:
                frame, summary = load_onharmony_manifest(
                    filelist,
                    metadata,
                    expected_acquisitions=2,
                    expected_subjects=2,
                )
            self.assertTrue(read_csv.called)
            self.assertEqual(frame["subject"].tolist(), ["sub-00001", "sub-00002"])
            self.assertEqual(frame["session"].tolist(), ["NOT1ACH001", "OXF1PRI001"])
            self.assertFalse(FORBIDDEN_LABEL_COLUMNS & set(frame.columns))
            self.assertTrue(summary["label_blind"])

    def test_external_manifest_rejects_a_preprocessed_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            filelist, metadata = self._write_external_manifest(Path(directory))
            lines = filelist.read_text(encoding="utf-8").splitlines()
            filelist.write_text(
                "\n".join(value.replace(".nii.gz", "_simclr_ready.nii.gz") for value in lines)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-raw T1"):
                load_onharmony_manifest(
                    filelist,
                    metadata,
                    expected_acquisitions=2,
                    expected_subjects=2,
                )

    def test_prediction_frame_rejects_labels_and_order_drift(self) -> None:
        expected = pd.DataFrame(
            {
                "subject": ["sub-1", "sub-2"],
                "session": ["A", "B"],
                "acquisition_id": ["sub-1_ses-A_T1w", "sub-2_ses-B_T1w"],
            }
        )
        predictions = expected.assign(age_pred=[40.0, 50.0])
        validate_prediction_frame(predictions, expected)
        with self.assertRaisesRegex(ValueError, "labels"):
            validate_prediction_frame(predictions.assign(age_true=[41.0, 49.0]), expected)
        with self.assertRaisesRegex(ValueError, "order differs"):
            validate_prediction_frame(predictions.iloc[::-1].reset_index(drop=True), expected)

    def test_external_provenance_binds_image_mask_and_success_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition = "sub-00001_ses-NOT1ACH001_T1w"
            subject_dir = root / acquisition
            subject_dir.mkdir()
            image = subject_dir / f"{acquisition}_simclr_ready.nii.gz"
            mask = subject_dir / f"{acquisition}_mask_mni.nii.gz"
            image.write_bytes(b"image")
            mask.write_bytes(b"mask")
            log_root = root / "logs"
            log_root.mkdir()
            log = log_root / f"simclr_{acquisition}.log"
            raw = root / f"{acquisition}.nii.gz"
            raw.write_bytes(b"raw")
            log.write_text(
                f"input    : {raw}\n"
                "3D-Neuro-SimCLR preprocessing (TurboPrep-equivalent)\n"
                f"Template SHA-256: {EXPECTED_TEMPLATE_SHA256}\n"
                "[Preprocessing 5/5] WhiteStripe normalization\n"
                f"Done: {image}\n"
                "All done.\n"
                "exit_code: 0\n",
                encoding="utf-8",
            )
            record = {
                "acquisition_id": acquisition,
                "raw_t1w_path": str(raw),
            }
            registry = ArtifactRegistry(
                contract_version="test",
                resolver=lambda value: artifact_paths(value, root, log_root),
                validator=validate_artifact_set,
            )
            model = OnHarmonyNeuroSimCLRModel(
                NeuroSimCLROptions(
                    checkpoint=root / "checkpoint.tar",
                    preproc_root=root / "training_inputs",
                ),
                registry,
                {
                    "template_affine": np.eye(4).tolist(),
                    "template_spacing": [1.0, 1.0, 1.0],
                },
            )
            provenance = model.validate_input_provenance(record)
            self.assertEqual(provenance["input_sha256"], sha256_file(image))
            self.assertEqual(provenance["mask_sha256"], sha256_file(mask))
            log.write_text(
                log.read_text(encoding="utf-8").replace("exit_code: 0", "exit_code: 01"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not a successful corrected run"):
                model.validate_input_provenance(record)
            log.write_text(
                log.read_text(encoding="utf-8").replace("exit_code: 01", "exit_code: 0")
                + "WhiteStripe failed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not a successful corrected run"):
                model.validate_input_provenance(record)

    def test_external_lock_requires_validation_only_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "selected_model.pt"
            checkpoint.write_bytes(b"adapted")
            lock = {
                "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                "arm": "adapted",
                "model": "3D-Neuro-SimCLR",
                "variant": "canonical",
                "checkpoint_sha256": sha256_file(checkpoint),
                "internal_test_accessed": False,
                "on_harmony_accessed": False,
            }
            atomic_json(lock, run_dir / "LOCKED.json")
            observed, observed_checkpoint, _ = validate_lock_header(
                run_dir,
                expected_model="3D-Neuro-SimCLR",
                expected_variant="canonical",
            )
            self.assertEqual(observed["checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(observed_checkpoint, checkpoint)
            atomic_json({**lock, "on_harmony_accessed": True}, run_dir / "LOCKED.json")
            with self.assertRaisesRegex(ValueError, "Invalid validation-only"):
                validate_lock_header(
                    run_dir,
                    expected_model="3D-Neuro-SimCLR",
                    expected_variant="canonical",
                )


class ReleasedInputContractTests(unittest.TestCase):
    def test_transpose_crop_matches_released_indices(self) -> None:
        array = np.zeros((1, 193, 229, 193), dtype=np.uint8)
        array[0, 1, 19, 10] = 7
        array[0, 192, 210, 159] = 9
        observed = transpose_crop(array)
        self.assertEqual(observed.shape, MODEL_INPUT_SHAPE)
        self.assertEqual(int(observed[0, 0, 0, 0]), 7)
        self.assertEqual(int(observed[0, -1, -1, -1]), 9)

    def test_transpose_crop_rejects_grid_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected pre-crop shape"):
            transpose_crop(np.zeros((1, 192, 229, 193), dtype=np.float32))

    def test_masked_standardization_uses_only_foreground(self) -> None:
        images = np.asarray([[[[[1.0, 2.0, 100.0, 4.0]]]]], dtype=np.float32)
        masks = np.asarray([[[[[1.0, 1.0, 0.0, 1.0]]]]], dtype=np.float32)
        observed = masked_standardize(images, masks)
        foreground = observed[masks.astype(bool)]
        self.assertAlmostEqual(float(foreground.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(foreground.std(ddof=0)), 1.0, places=6)
        self.assertEqual(float(observed[0, 0, 0, 0, 2]), 0.0)

    def test_masked_standardization_rejects_empty_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "mask is empty"):
            masked_standardize(
                np.ones((1, 1, 2, 2, 2), dtype=np.float32),
                np.zeros((1, 1, 2, 2, 2), dtype=np.float32),
            )

    def test_load_input_wires_grid_affine_value_and_mask_validation(self) -> None:
        class FakeImage:
            def __init__(self, shape, data, affine=None) -> None:
                self.shape = shape
                self._data = np.asarray(data, dtype=np.float32)
                self.affine = np.eye(4) if affine is None else affine

            def get_fdata(self, dtype=None):
                return self._data.astype(dtype or self._data.dtype, copy=False)

        model = NeuroSimCLRModel(
            NeuroSimCLROptions(checkpoint=Path("."), preproc_root=Path("."))
        )
        record = {"dataset": "CamCAN", "subject": "sub-001", "split": "train"}
        valid = FakeImage(REGISTERED_SHAPE, np.ones((1, 1, 1), dtype=np.float32))

        cases = (
            (
                FakeImage((1, 2, 3), np.ones((1, 1, 1))),
                valid,
                "Unexpected image/mask grid",
            ),
            (
                valid,
                FakeImage(
                    REGISTERED_SHAPE,
                    np.ones((1, 1, 1)),
                    affine=np.diag([1.0, 1.0, 1.0, 2.0]),
                ),
                "affines differ",
            ),
            (
                FakeImage(REGISTERED_SHAPE, np.asarray([[[np.nan]]])),
                valid,
                "non-finite",
            ),
            (
                valid,
                FakeImage(REGISTERED_SHAPE, np.full((1, 1, 1), 0.5)),
                "not binary",
            ),
        )
        for image, mask, message in cases:
            with self.subTest(message=message), patch.object(
                model, "validate_input_provenance", return_value={}
            ), patch("nibabel.load", side_effect=[image, mask]):
                with self.assertRaisesRegex(ValueError, message):
                    model.load_input(record)

    @staticmethod
    def _write_provenance_fixture(
        root: Path, *, dataset: str = "CamCAN", subject: str = "sub-001"
    ) -> tuple[NeuroSimCLRModel, dict[str, str], Path, Path, Path]:
        subject_dir = root / "preprocessed" / dataset / subject
        subject_dir.mkdir(parents=True)
        image_path = subject_dir / f"{subject}_simclr_ready.nii.gz"
        mask_path = subject_dir / f"{subject}_mask_mni.nii.gz"
        qc_path = subject_dir / "preprocessing_qc.json"
        image_path.write_bytes(b"image-v1")
        mask_path.write_bytes(b"mask-v1")
        record = {"dataset": dataset, "subject": subject, "split": "train"}
        atomic_json(
            {
                "status": "PASS",
                "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
                "pipeline_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
                **record,
                "output_sha256": {
                    "ready": sha256_file(image_path),
                    "mask_mni": sha256_file(mask_path),
                },
            },
            qc_path,
        )
        model = NeuroSimCLRModel(
            NeuroSimCLROptions(
                checkpoint=root / "checkpoint.tar",
                preproc_root=root / "preprocessed",
            )
        )
        return model, record, image_path, mask_path, qc_path

    def test_provenance_hashes_are_cached_while_files_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, record, _, _, _ = self._write_provenance_fixture(Path(directory))
            with patch.object(
                neuro_simclr_model, "sha256_file", wraps=sha256_file
            ) as mocked_hash:
                first = model.validate_input_provenance(record)
                second = model.validate_input_provenance(record)
            self.assertEqual(first, second)
            self.assertEqual(mocked_hash.call_count, 3)

    def test_provenance_cache_is_invalidated_when_an_input_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, record, image_path, _, _ = self._write_provenance_fixture(Path(directory))
            model.validate_input_provenance(record)
            image_path.write_bytes(b"image-v2-with-a-different-size")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                model.validate_input_provenance(record)

    def test_provenance_cache_is_scoped_by_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, first_record, _, _, _ = self._write_provenance_fixture(root)
            _, second_record, _, _, _ = self._write_provenance_fixture(
                root, dataset="WAND"
            )
            with patch.object(
                neuro_simclr_model, "sha256_file", wraps=sha256_file
            ) as mocked_hash:
                model.validate_input_provenance(first_record)
                model.validate_input_provenance(second_record)
            self.assertEqual(mocked_hash.call_count, 6)
            self.assertEqual(len(model._input_provenance_cache), 2)


class CheckpointContractTests(unittest.TestCase):
    def test_only_encoder_weights_survive_prefix_cleaning(self) -> None:
        state = {
            "module.encoder.conv1.weight": "encoder",
            "module.projector.0.weight": "projector",
        }
        self.assertEqual(clean_encoder_state_dict(state), {"conv1.weight": "encoder"})

    def test_unknown_checkpoint_keys_are_rejected(self) -> None:
        state = {
            "module.encoder.conv1.weight": "encoder",
            "module.projector.0.weight": "projector",
            "module.classifier.weight": "unexpected",
        }
        with self.assertRaisesRegex(ValueError, "Unexpected checkpoint keys"):
            clean_encoder_state_dict(state)

    def test_only_layer4_is_trainable(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummyResNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Linear(2, 2)
                self.layer3 = torch.nn.Linear(2, 2)
                self.layer4 = torch.nn.Sequential(
                    torch.nn.Linear(2, 2),
                    torch.nn.Linear(2, 2),
                )

        adapter = NeuroSimCLRModel(
            NeuroSimCLROptions(checkpoint=Path("."), preproc_root=Path("."))
        )
        adapter.model = DummyResNet()
        adapter.device = torch.device("cpu")
        adapter._projector_parameters = 10
        expected_trainable = {
            name
            for name, _ in adapter.model.named_parameters()
            if name.startswith("layer4.")
        }
        summary = adapter.set_controlled_adaptation()
        observed_trainable = {
            name for name, parameter in adapter.model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(expected_trainable)
        self.assertSetEqual(observed_trainable, expected_trainable)
        self.assertSetEqual(
            set(summary["trainable_parameter_names"]), expected_trainable
        )
        self.assertEqual(summary["trainable_components"], ["layer4"])
        self.assertEqual(summary["contrastive_projection_parameters"], 10)


class MetricsAndProtocolTests(unittest.TestCase):
    def test_entry_points_share_one_pipeline_contract(self) -> None:
        self.assertEqual(EXTRACTION_PIPELINE_VERSION, "3d_neuro_simclr_brain_age_v1")
        self.assertEqual(RIDGE_PIPELINE_VERSION, EXTRACTION_PIPELINE_VERSION)
        self.assertEqual(ADAPTED_PIPELINE_VERSION, EXTRACTION_PIPELINE_VERSION)

    def test_regression_and_embedding_contracts(self) -> None:
        metrics = regression_metrics([20, 30, 40], [22, 29, 43])
        self.assertAlmostEqual(metrics["mae_years"], 2.0)
        self.assertEqual(
            embedding_columns(pd.DataFrame({"emb_0": [1], "emb_1": [2]})),
            ["emb_0", "emb_1"],
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            embedding_columns(pd.DataFrame({"emb_0": [1], "emb_2": [2]}))

    def test_embedding_rows_use_canonical_variant(self) -> None:
        manifest, _ = load_and_validate_manifest()
        expected = frame_for_splits(manifest, ["validation"]).iloc[:2].reset_index(drop=True)
        table = pd.DataFrame(
            {
                "dataset": expected["dataset"],
                "subject": expected["subject"],
                "split": expected["split"],
                "age": expected["age"],
                "sex": expected["sex"],
                "input_path": ["a", "b"],
                "model": ["3D-Neuro-SimCLR", "3D-Neuro-SimCLR"],
                "variant": ["canonical", "canonical"],
                "emb_0": [1.0, 2.0],
            }
        )
        self.assertEqual(
            validate_embedding_frame(
                table,
                expected_manifest=expected,
                expected_model="3D-Neuro-SimCLR",
                expected_variant="canonical",
            ),
            ["emb_0"],
        )

        validate_frozen_embedding_contract(
            table,
            expected,
            ["emb_0"],
            sidecar_dimension=1,
            expected_dimension=1,
            model_name="3D-Neuro-SimCLR",
        )
        wrong_sex = table.copy()
        wrong_sex.loc[0, "sex"] = "__DRIFT__"
        with self.assertRaisesRegex(ValueError, "sex differs"):
            validate_frozen_embedding_contract(
                wrong_sex,
                expected,
                ["emb_0"],
                sidecar_dimension=1,
                expected_dimension=1,
                model_name="3D-Neuro-SimCLR",
            )
        with self.assertRaisesRegex(ValueError, "dimensionality drift"):
            validate_frozen_embedding_contract(
                table,
                expected,
                ["emb_0"],
                sidecar_dimension=512,
                expected_dimension=512,
                model_name="3D-Neuro-SimCLR",
            )

    def test_ridge_code_manifest_includes_the_model_contract(self) -> None:
        key = "FineTuning/3DNeuroSimCLR/model.py"
        self.assertEqual(fit_code_hashes()[key], sha256_file(PROJECT_ROOT / key))

    def test_fixed_protocol_and_ridge_roundtrip(self) -> None:
        self.assertEqual(
            FROZEN_RIDGE_ALPHAS,
            (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
        )
        self.assertEqual(adapted_learning_rates("conservative"), (1e-6, 1e-4))
        self.assertEqual(adapted_learning_rates("intermediate"), (3e-6, 3e-4))
        self.assertEqual(adapted_learning_rates("high"), (1e-5, 1e-3))
        self.assertEqual(parse_ridge_alphas("10,0.1,1"), [0.1, 1.0, 10.0])

        rng = np.random.default_rng(7)
        train_x = rng.normal(size=(80, 6))
        validation_x = rng.normal(size=(20, 6))
        weights = np.asarray([4.0, -2.0, 0.5, 0.0, 1.0, -0.2])
        train_age = 50 + train_x @ weights + rng.normal(scale=0.2, size=80)
        validation_age = 50 + validation_x @ weights + rng.normal(scale=0.2, size=20)
        artifact, _, expected = fit_ridge_grid(
            train_x, train_age, validation_x, validation_age, [0.01, 0.1, 1.0]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            atomic_npz(artifact, path)
            observed = predict_ridge(load_npz(path), validation_x)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)

    def test_lr_selection_is_validation_only(self) -> None:
        maes = {"conservative": 4.2, "intermediate": 3.8, "high": 4.0}
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            runs = []
            for config, mae in maes.items():
                run = root / config
                run.mkdir()
                checkpoint = run / "selected_model.pt"
                checkpoint.write_bytes(config.encode("ascii"))
                lr_encoder, lr_head = ADAPTED_LR_CONFIGS[config].values()
                arguments = {
                    "checkpoint": "/fixed/checkpoint.tar",
                    "preproc_root": "/fixed/inputs",
                    "seed": 0,
                    "amp": "bfloat16",
                    "lr_config": config,
                    "lr_encoder": lr_encoder,
                    "lr_head": lr_head,
                    **ADAPTED_FIXED_HYPERPARAMETERS,
                }
                atomic_json(
                    {
                        "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                        "arm": "adapted",
                        "model": "3D-Neuro-SimCLR",
                        "variant": "canonical",
                        "checkpoint": checkpoint.name,
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "selected_epoch": 3,
                        "selection_value": mae,
                        "locked_validation_metrics": {
                            "mae_years": mae,
                            "rmse_years": mae + 0.5,
                            "r2": 0.5,
                        },
                        "manifest": {"manifest_sha256": "manifest-test"},
                        "release": {"checkpoint": "fixed"},
                        "code_sha256": {"code.py": "fixed"},
                        "run_config": {"arguments": arguments},
                        "training_elapsed_seconds": 10.0,
                        "internal_test_accessed": False,
                        "on_harmony_accessed": False,
                    },
                    run / "LOCKED.json",
                )
                atomic_tsv(pd.DataFrame({"epoch": [1, 2, 3]}), run / "train_history.tsv")
                runs.append(run)
            output = root / "selection"
            output.mkdir()
            result = select_adapted_learning_rate(
                runs,
                output,
                expected_model="3D-Neuro-SimCLR",
                expected_variant="canonical",
                manifest_sha256="manifest-test",
            )
            self.assertEqual(result["selected_lr_config"], "intermediate")
            self.assertFalse(result["internal_test_accessed"])
            selected_run = root / "intermediate"
            selected_lock = json.loads(
                (selected_run / "LOCKED.json").read_text(encoding="utf-8")
            )
            provenance = validate_selected_evaluation_run(
                selection_dir=output,
                run_dir=selected_run,
                lock=selected_lock,
                manifest_sha256="manifest-test",
                expected_model="3D-Neuro-SimCLR",
                evaluation_variant="canonical",
            )
            self.assertEqual(provenance["selected_lr_config"], "intermediate")
            self.assertEqual(provenance["evaluation_seed"], 0)

            wrong_run = root / "conservative"
            wrong_lock = json.loads(
                (wrong_run / "LOCKED.json").read_text(encoding="utf-8")
            )
            with self.assertRaisesRegex(ValueError, "validation-selected LR"):
                validate_selected_evaluation_run(
                    selection_dir=output,
                    run_dir=wrong_run,
                    lock=wrong_lock,
                    manifest_sha256="manifest-test",
                    expected_model="3D-Neuro-SimCLR",
                    evaluation_variant="canonical",
                )

    def test_lr_selection_rejects_nonzero_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "seed1"
            run.mkdir()
            atomic_json(
                {"run_config": {"arguments": {"seed": 1}}},
                run / "LOCKED.json",
            )
            with self.assertRaisesRegex(ValueError, "requires seed 0"):
                validate_selection_candidates([run])


class LockedEvaluationTests(unittest.TestCase):
    @staticmethod
    def _frozen_evaluation_fixture(root: Path) -> Namespace:
        run_dir = root / "frozen_run"
        run_dir.mkdir()
        model_path = run_dir / fit_ridge_module.MODEL_FILENAME
        model_path.write_bytes(b"frozen-ridge-artifact")
        atomic_json(
            {
                "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                "arm": "frozen",
                "model": "3D-Neuro-SimCLR",
                "variant": "canonical",
                "model_artifact_sha256": sha256_file(model_path),
                "manifest": {"manifest_sha256": "manifest"},
                "fit_code_sha256": {"fit.py": "locked"},
                "internal_test_accessed": False,
                "on_harmony_accessed": False,
            },
            run_dir / fit_ridge_module.LOCK_NAME,
        )
        return Namespace(
            acknowledge_locked_test=True,
            run_dir=run_dir,
            manifest=root / "manifest.tsv",
            test_embeddings=root / "test_embeddings.tsv",
            output_dir=root / "evaluation",
        )

    def test_smoke_forward_rejects_internal_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "locked internal-test"):
            audit_embeddings(Namespace(split="internal_test", smoke_forward=True))

    def test_embedding_extraction_requires_internal_test_acknowledgement(self) -> None:
        with self.assertRaisesRegex(ValueError, "acknowledge-locked-test"):
            extract_embeddings(
                Namespace(split="internal_test", acknowledge_locked_test=False)
            )

    def test_amp_lock_is_enforced(self) -> None:
        args = Namespace(amp="bfloat16")
        checkpoint = {"run_config": {"arguments": vars(args).copy()}}
        validate_evaluation_options(args, checkpoint)
        with self.assertRaisesRegex(ValueError, "--amp differs"):
            validate_evaluation_options(
                Namespace(amp="float16"),
                checkpoint,
            )

    def test_frozen_evaluation_requires_a_validation_only_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "frozen_linear_model.npz"
            model_path.write_bytes(b"locked-ridge")
            lock = {
                "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                "arm": "frozen",
                "model": "3D-Neuro-SimCLR",
                "variant": "canonical",
                "model_artifact_sha256": sha256_file(model_path),
                "internal_test_accessed": False,
                "on_harmony_accessed": False,
            }
            validate_evaluation_lock(lock, model_path, variant="canonical")
            for update in (
                {"status": "INCOMPLETE"},
                {"internal_test_accessed": True},
                {"on_harmony_accessed": True},
            ):
                with self.assertRaisesRegex(ValueError, "validation-only locked"):
                    validate_evaluation_lock(
                        {**lock, **update}, model_path, variant="canonical"
                    )

            for update in (
                {"arm": "adapted"},
                {"model": "Other"},
                {"variant": "other"},
            ):
                with self.assertRaisesRegex(ValueError, "not the expected frozen"):
                    validate_evaluation_lock(
                        {**lock, **update}, model_path, variant="canonical"
                    )

            model_path.write_bytes(b"tampered-ridge")
            with self.assertRaisesRegex(ValueError, "Frozen model hash differs"):
                validate_evaluation_lock(lock, model_path, variant="canonical")

    def test_frozen_evaluation_rejects_manifest_and_fit_code_drift(self) -> None:
        for drift_kind in ("manifest", "code"):
            with self.subTest(drift=drift_kind), tempfile.TemporaryDirectory() as directory:
                args = self._frozen_evaluation_fixture(Path(directory))
                manifest_sha256 = "different" if drift_kind == "manifest" else "manifest"
                fit_code = (
                    {"fit.py": "different"}
                    if drift_kind == "code"
                    else {"fit.py": "locked"}
                )
                with (
                    patch.object(
                        fit_ridge_module,
                        "load_and_validate_manifest",
                        return_value=(pd.DataFrame(), {"manifest_sha256": manifest_sha256}),
                    ),
                    patch.object(fit_ridge_module, "fit_code_hashes", return_value=fit_code),
                ):
                    message = (
                        "different subject manifest"
                        if drift_kind == "manifest"
                        else "Runtime Ridge code differs"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        fit_ridge_module.command_evaluate(args)

if __name__ == "__main__":
    unittest.main()
