#!/usr/bin/env python3
"""Dependency-light tests for the BrainIAC brain-age workflow."""

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

import model as brainiac_model  # noqa: E402
import predict_onharmony as brainiac_onharmony  # noqa: E402
import Emb_extractors.extract_BrainIAC as onharmony_extractor  # noqa: E402

PRECISION_AUDIT_PATH = (
    PROJECT_ROOT
    / "ExcelsChecks/paper_actual/audits/BrainIAC/audit_fp32_bfloat16.py"
)
PRECISION_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "brainiac_precision_audit", PRECISION_AUDIT_PATH
)
if PRECISION_AUDIT_SPEC is None or PRECISION_AUDIT_SPEC.loader is None:
    raise RuntimeError(f"Cannot import BrainIAC precision audit: {PRECISION_AUDIT_PATH}")
precision_audit = importlib.util.module_from_spec(PRECISION_AUDIT_SPEC)
sys.modules[PRECISION_AUDIT_SPEC.name] = precision_audit
PRECISION_AUDIT_SPEC.loader.exec_module(precision_audit)
from model import (  # noqa: E402
    EMBEDDING_DIM,
    MODEL_INPUT_SHAPE,
    PREPROCESSING_CONTRACT_SHA256,
    PREPROCESSING_PIPELINE_VERSION,
    BrainIACModel,
    BrainIACOptions,
    clean_backbone_state_dict,
    load_and_validate_manifest,
)
from extract_embeddings import (  # noqa: E402
    PIPELINE_VERSION as EXTRACTION_PIPELINE_VERSION,
    command_audit as audit_embeddings,
    command_extract as extract_embeddings,
)
from finetune import (  # noqa: E402
    BRAINIAC_SENSITIVITY_LR_CONFIGS,
    BRAINIAC_TRAINING_LR_CONFIGS,
    PIPELINE_VERSION as ADAPTED_PIPELINE_VERSION,
    INTERNAL_TEST_SELECTION_RUNTIME_MIGRATION,
    brainiac_learning_rates,
    validate_evaluation_options,
    validate_selection_candidates,
)
import fit_ridge as fit_ridge_module  # noqa: E402
from fit_ridge import PIPELINE_VERSION as RIDGE_PIPELINE_VERSION  # noqa: E402
from fit_ridge import fit_code_hashes, validate_evaluation_lock  # noqa: E402
from FineTuning.shared.cohort import frame_for_splits, sha256_file  # noqa: E402
from FineTuning.shared.evaluation import validate_selected_evaluation_run  # noqa: E402
from FineTuning.shared.metrics import regression_metrics, validate_embedding_frame  # noqa: E402
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
    predict_ridge,
)
from FineTuning.shared.internal_test import internal_test_runtime_migration  # noqa: E402
from FineTuning.shared.runtime import atomic_json  # noqa: E402
from FineTuning.shared.runtime import atomic_tsv  # noqa: E402
from FineTuning.shared.onharmony import (  # noqa: E402
    adapted_runtime_migration,
    adapted_selection_runtime_migration,
    validate_prediction_runtime_code,
)


class FakeTensor:
    def __init__(self, size: int = 1) -> None:
        self.size = size

    def numel(self) -> int:
        return self.size


class FrozenOnHarmonyExtractorTests(unittest.TestCase):
    def test_extractor_locks_bfloat16_seed_and_exact_adapter(self) -> None:
        args = onharmony_extractor.parse_args([])
        self.assertEqual(args.amp, "bfloat16")
        self.assertEqual(args.seed, 42)
        self.assertIs(
            onharmony_extractor.OnHarmonyBrainIACModel,
            brainiac_onharmony.OnHarmonyBrainIACModel,
        )
        self.assertFalse(hasattr(args, "participants"))

        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        cuda = torch.device("cuda")
        with patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            onharmony_extractor.validate_runtime_contract(cuda, "bfloat16", 42)
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
        device = torch.device("cpu")
        with patch.object(onharmony_extractor, "OnHarmonyBrainIACModel") as adapter_class:
            adapter = adapter_class.return_value
            adapter.set_frozen.return_value = {"encoder_trainable_parameters": 0}
            observed, summary = onharmony_extractor.build_model(
                Path("BrainIAC.ckpt"), registry, device
            )
        options, observed_registry = adapter_class.call_args.args
        self.assertEqual(options.checkpoint, Path("BrainIAC.ckpt"))
        self.assertIs(observed_registry, registry)
        adapter.load.assert_called_once_with(device)
        adapter.set_frozen.assert_called_once_with()
        self.assertIs(observed, adapter)
        self.assertEqual(summary["encoder_trainable_parameters"], 0)

    def test_all_readouts_use_the_same_complete_token_sequence(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        tokens = torch.arange(
            216 * EMBEDDING_DIM, dtype=torch.float32
        ).reshape(1, 216, EMBEDDING_DIM)
        torch.testing.assert_close(
            onharmony_extractor.apply_readout(tokens, "first_patch"),
            tokens[:, 0, :],
        )
        torch.testing.assert_close(
            onharmony_extractor.apply_readout(tokens, "mean_pool"),
            tokens.mean(dim=1),
        )
        torch.testing.assert_close(
            onharmony_extractor.apply_readout(tokens, "last_patch"),
            tokens[:, -1, :],
        )
        torch.testing.assert_close(
            onharmony_extractor.apply_readout(tokens, "max_pool"),
            tokens.max(dim=1).values,
        )
        with self.assertRaisesRegex(ValueError, "Unknown BrainIAC readout"):
            onharmony_extractor.apply_readout(tokens, "unknown")

    def test_readouts_are_serialized_as_float32_after_one_forward(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummyViT(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.forward_calls = 0

            def forward(self, volume):
                del volume
                self.forward_calls += 1
                tokens = torch.arange(
                    216 * EMBEDDING_DIM, dtype=torch.float32
                ).reshape(1, 216, EMBEDDING_DIM)
                return tokens, []

        class DummyAdapter:
            def __init__(self) -> None:
                self.model = DummyViT()
                self.mode_calls = 0

            def controlled_train_mode(self) -> None:
                self.mode_calls += 1

            @staticmethod
            def load_input(record):
                del record
                return torch.zeros((1, 1, *MODEL_INPUT_SHAPE), dtype=torch.float32)

            def _require_loaded(self):
                return self.model

        adapter = DummyAdapter()
        observed = onharmony_extractor.extract_readouts(
            adapter,
            {"acquisition_id": "sub-1_ses-A001_T1w"},
            readouts=onharmony_extractor.READOUTS,
            device=torch.device("cpu"),
            amp="off",
        )
        self.assertEqual(adapter.mode_calls, 1)
        self.assertEqual(adapter.model.forward_calls, 1)
        self.assertEqual(set(observed), set(onharmony_extractor.READOUTS))
        for embedding in observed.values():
            self.assertEqual(embedding.shape, (EMBEDDING_DIM,))
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

    def test_precision_audit_npz_is_pickle_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.npz"
            precision_audit._atomic_npz(
                path,
                {
                    "subject": np.asarray(["sub-001", "sub-002"], dtype=str),
                    "embedding": np.ones((2, EMBEDDING_DIM), dtype=np.float32),
                },
            )
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(payload["subject"].tolist(), ["sub-001", "sub-002"])
                self.assertEqual(payload["embedding"].dtype, np.float32)
            with self.assertRaisesRegex(TypeError, "object array"):
                precision_audit._atomic_npz(
                    path,
                    {"unsafe": np.asarray([object()], dtype=object)},
                )


class CohortAndPathTests(unittest.TestCase):
    def test_manifest_permits_offline_brainiac_inputs(self) -> None:
        frame, summary = load_and_validate_manifest()
        self.assertEqual(len(frame), 1411)
        self.assertEqual(
            summary["split_counts"],
            {"train": 1130, "validation": 141, "internal_test": 140},
        )
        self.assertEqual(
            summary["input_contract"], {"required_offline_group": "preproc_brainiac"}
        )
        self.assertEqual(summary["on_harmony_firewall"], "PASS")

    def test_dataset_scoped_paths_are_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = BrainIACModel(
                BrainIACOptions(
                    checkpoint=Path(directory) / "BrainIAC.ckpt",
                    preproc_root=Path(directory) / "preprocessed",
                )
            )
            record = {"dataset": "CamCAN", "subject": "sub-001", "split": "train"}
            self.assertEqual(
                adapter.input_path(record),
                Path(directory)
                / "preprocessed"
                / "CamCAN"
                / "sub-001"
                / "sub-001_brainiac_ready.nii.gz",
            )
            self.assertEqual(adapter.qc_path(record).name, "preprocessing_qc.json")


class InputAndCheckpointTests(unittest.TestCase):
    @staticmethod
    def _provenance_fixture(root: Path) -> tuple[BrainIACModel, dict[str, str], Path]:
        subject_dir = root / "preprocessed" / "CamCAN" / "sub-001"
        subject_dir.mkdir(parents=True)
        image = subject_dir / "sub-001_brainiac_ready.nii.gz"
        qc = subject_dir / "preprocessing_qc.json"
        image.write_bytes(b"brainiac-image-v1")
        record = {"dataset": "CamCAN", "subject": "sub-001", "split": "train"}
        atomic_json(
            {
                "status": "PASS",
                "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
                "pipeline_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
                **record,
                "output_sha256": {"ready": sha256_file(image)},
            },
            qc,
        )
        adapter = BrainIACModel(
            BrainIACOptions(
                checkpoint=root / "BrainIAC.ckpt",
                preproc_root=root / "preprocessed",
            )
        )
        return adapter, record, image

    def test_provenance_cache_and_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter, record, image = self._provenance_fixture(Path(directory))
            with patch.object(
                brainiac_model, "sha256_file", wraps=sha256_file
            ) as mocked_hash:
                first = adapter.validate_input_provenance(record)
                second = adapter.validate_input_provenance(record)
            self.assertEqual(first, second)
            self.assertEqual(mocked_hash.call_count, 2)
            image.write_bytes(b"brainiac-image-v2-with-different-size")
            with self.assertRaisesRegex(ValueError, "differs"):
                adapter.validate_input_provenance(record)

    def test_checkpoint_cleaning_accounts_for_projection_head(self) -> None:
        cleaned, tensors, parameters = clean_backbone_state_dict(
            {
                "backbone.patch_embedding.weight": FakeTensor(10),
                "projection_head.0.weight": FakeTensor(6),
                "projection_head.0.bias": FakeTensor(2),
            }
        )
        self.assertEqual(list(cleaned), ["patch_embedding.weight"])
        self.assertEqual(tensors, 2)
        self.assertEqual(parameters, 8)

    def test_checkpoint_cleaning_rejects_unknown_branches(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected BrainIAC checkpoint keys"):
            clean_backbone_state_dict(
                {
                    "backbone.x": FakeTensor(),
                    "projection_head.x": FakeTensor(),
                    "classifier.x": FakeTensor(),
                }
            )

    def test_only_final_pretrained_block_and_norm_are_trainable(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummyViT(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_embedding = torch.nn.Linear(2, 2)
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(12)])
                self.norm = torch.nn.LayerNorm(2)

        adapter = BrainIACModel(
            BrainIACOptions(checkpoint=Path("."), preproc_root=Path("."))
        )
        adapter.model = DummyViT()
        adapter.device = torch.device("cpu")
        adapter._loaded_parameter_names = {
            name for name, _ in adapter.model.named_parameters()
        }
        adapter._projector_parameters = 10
        expected_trainable = {
            name
            for name, _ in adapter.model.named_parameters()
            if name.startswith("blocks.11.") or name.startswith("norm.")
        }
        summary = adapter.set_controlled_adaptation()
        observed_trainable = {
            name for name, _ in adapter.trainable_named_parameters()
        }
        self.assertTrue(expected_trainable)
        self.assertSetEqual(observed_trainable, expected_trainable)
        self.assertSetEqual(
            set(summary["trainable_parameter_names"]), expected_trainable
        )
        self.assertEqual(summary["trainable_components"], ["blocks.11", "norm"])

    def test_encode_selects_first_patch_without_collapsing_tokens(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummyViT(torch.nn.Module):
            def forward(self, volume):
                del volume
                tokens = torch.arange(216 * EMBEDDING_DIM, dtype=torch.float32).reshape(
                    1, 216, EMBEDDING_DIM
                )
                return tokens, []

        adapter = BrainIACModel(
            BrainIACOptions(checkpoint=Path("."), preproc_root=Path("."))
        )
        adapter.model = DummyViT()
        adapter.device = torch.device("cpu")
        adapter.load_input = lambda record: torch.zeros((1, 1, *MODEL_INPUT_SHAPE))
        observed = adapter.encode({}, training=False, epoch=0, seed=0)
        torch.testing.assert_close(observed, torch.arange(EMBEDDING_DIM, dtype=torch.float32))

    def test_load_input_rejects_shape_nonfinite_and_empty_volumes(self) -> None:
        try:
            import nibabel as nib
            import torch
        except ImportError:
            self.skipTest("nibabel and torch are required")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = BrainIACModel(
                BrainIACOptions(
                    checkpoint=root / "BrainIAC.ckpt",
                    preproc_root=root / "preprocessed",
                )
            )
            record = {"dataset": "CamCAN", "subject": "sub-001", "split": "train"}
            input_path = adapter.input_path(record)
            input_path.parent.mkdir(parents=True)
            adapter.validate_input_provenance = lambda _: {"status": "PASS"}

            def save(array: np.ndarray) -> None:
                nib.save(nib.Nifti1Image(array, np.eye(4)), input_path)

            save(np.ones((95, 96, 96), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "exact 3D"):
                adapter.load_input(record)

            nonfinite = np.ones(MODEL_INPUT_SHAPE, dtype=np.float32)
            nonfinite[0, 0, 0] = np.nan
            save(nonfinite)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                adapter.load_input(record)

            save(np.zeros(MODEL_INPUT_SHAPE, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "empty"):
                adapter.load_input(record)

            save(np.ones(MODEL_INPUT_SHAPE, dtype=np.float32))
            observed = adapter.load_input(record)
            self.assertEqual(tuple(observed.shape), (1, 1, *MODEL_INPUT_SHAPE))
            self.assertEqual(observed.dtype, torch.float32)


class ProtocolTests(unittest.TestCase):
    def test_all_entry_points_share_one_contract(self) -> None:
        self.assertEqual(EXTRACTION_PIPELINE_VERSION, "brainiac_brain_age_v1")
        self.assertEqual(RIDGE_PIPELINE_VERSION, EXTRACTION_PIPELINE_VERSION)
        self.assertEqual(ADAPTED_PIPELINE_VERSION, EXTRACTION_PIPELINE_VERSION)

    def test_ridge_code_manifest_includes_the_model_contract(self) -> None:
        key = "FineTuning/BrainIAC/model.py"
        self.assertEqual(fit_code_hashes()[key], sha256_file(PROJECT_ROOT / key))

    def test_metrics_rows_and_learning_rates_are_common(self) -> None:
        metrics = regression_metrics([20, 30, 40], [22, 29, 43])
        self.assertAlmostEqual(metrics["mae_years"], 2.0)
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
                "model": ["BrainIAC", "BrainIAC"],
                "variant": ["canonical", "canonical"],
                "emb_0": [1.0, 2.0],
            }
        )
        self.assertEqual(
            validate_embedding_frame(
                table,
                expected_manifest=expected,
                expected_model="BrainIAC",
                expected_variant="canonical",
            ),
            ["emb_0"],
        )
        self.assertEqual(adapted_learning_rates("conservative"), (1e-6, 1e-4))
        self.assertEqual(adapted_learning_rates("intermediate"), (3e-6, 3e-4))
        self.assertEqual(adapted_learning_rates("high"), (1e-5, 1e-3))

    def test_brainiac_sensitivity_is_separate_from_the_primary_lr_grid(self) -> None:
        self.assertEqual(
            tuple(ADAPTED_LR_CONFIGS),
            ("conservative", "intermediate", "high"),
        )
        self.assertEqual(
            BRAINIAC_SENSITIVITY_LR_CONFIGS,
            {
                "encoder_1e4_sensitivity": {
                    "lr_encoder": 1e-4,
                    "lr_head": 1e-3,
                },
                "encoder_3e4_sensitivity": {
                    "lr_encoder": 3e-4,
                    "lr_head": 1e-3,
                },
            },
        )
        self.assertEqual(
            tuple(BRAINIAC_TRAINING_LR_CONFIGS),
            (
                "conservative",
                "intermediate",
                "high",
                "encoder_1e4_sensitivity",
                "encoder_3e4_sensitivity",
            ),
        )
        self.assertEqual(
            brainiac_learning_rates("encoder_1e4_sensitivity"),
            (1e-4, 1e-3),
        )
        self.assertEqual(
            brainiac_learning_rates("encoder_3e4_sensitivity"),
            (3e-4, 1e-3),
        )

    def test_fixed_protocol_and_ridge_roundtrip(self) -> None:
        self.assertEqual(
            FROZEN_RIDGE_ALPHAS,
            (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
        )
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
            adapter = BrainIACModel(
                BrainIACOptions(
                    checkpoint=root / "BrainIAC.ckpt",
                    preproc_root=root / "preprocessed",
                )
            ).audit_description()
            runs: list[Path] = []
            for config, mae in maes.items():
                run = root / config
                run.mkdir()
                checkpoint = run / "selected_model.pt"
                checkpoint.write_bytes(config.encode("ascii"))
                lr_encoder = ADAPTED_LR_CONFIGS[config]["lr_encoder"]
                lr_head = ADAPTED_LR_CONFIGS[config]["lr_head"]
                arguments = {
                    "checkpoint": str(adapter["checkpoint"]),
                    "preproc_root": str(adapter["input_root"]),
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
                        "model": "BrainIAC",
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
                        "adapter": adapter,
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

            validate_selection_candidates(runs)
            output = root / "selection"
            output.mkdir()
            result = select_adapted_learning_rate(
                runs,
                output,
                expected_model="BrainIAC",
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
                expected_model="BrainIAC",
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
                    expected_model="BrainIAC",
                    evaluation_variant="canonical",
                )

    @staticmethod
    def _selection_candidate(root: Path, name: str, adapter: dict) -> Path:
        run_dir = root / name
        run_dir.mkdir()
        atomic_json(
            {
                "run_config": {"arguments": {"seed": 0}},
                "adapter": adapter,
            },
            run_dir / "LOCKED.json",
        )
        return run_dir

    def test_selection_candidates_accept_the_resolved_brainiac_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = BrainIACModel(
                BrainIACOptions(
                    checkpoint=root / "BrainIAC.ckpt",
                    preproc_root=root / "preprocessed",
                )
            ).audit_description()
            self.assertNotIn("registered_shape", adapter)
            run_dirs = [
                self._selection_candidate(root, f"run-{index}", adapter)
                for index in range(3)
            ]
            validate_selection_candidates(run_dirs)

    def test_selection_candidates_reject_different_patch_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = BrainIACModel(
                BrainIACOptions(
                    checkpoint=root / "BrainIAC.ckpt",
                    preproc_root=root / "preprocessed",
                )
            ).audit_description()
            changed = dict(adapter)
            changed["n_patch_tokens"] = int(adapter["n_patch_tokens"]) + 1
            run_dirs = [
                self._selection_candidate(root, "run-0", adapter),
                self._selection_candidate(root, "run-1", adapter),
                self._selection_candidate(root, "run-2", changed),
            ]
            with self.assertRaisesRegex(ValueError, "different resolved input contracts"):
                validate_selection_candidates(run_dirs)

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
        lock = {
            "status": "LOCKED_AFTER_TRAIN_VALIDATION",
            "arm": "frozen",
            "model": "BrainIAC",
            "variant": "canonical",
            "model_artifact_sha256": sha256_file(model_path),
            "manifest": {"manifest_sha256": "manifest"},
            "fit_code_sha256": {"fit.py": "locked"},
            "embedding_dim": EMBEDDING_DIM,
            "extraction_release": {"revision": "fixed"},
            "extraction_code_sha256": {"extract.py": "fixed"},
            "extraction_adapter": {"input_contract": "fixed"},
            "extraction_amp": "bfloat16",
            "internal_test_accessed": False,
            "on_harmony_accessed": False,
        }
        atomic_json(lock, run_dir / fit_ridge_module.LOCK_NAME)
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

    def test_evaluation_runtime_lock_is_enforced(self) -> None:
        args = Namespace(amp="bfloat16", seed=1)
        checkpoint = {"run_config": {"arguments": vars(args).copy()}}
        validate_evaluation_options(args, checkpoint)
        with self.assertRaisesRegex(ValueError, "--amp differs"):
            validate_evaluation_options(Namespace(amp="float16", seed=1), checkpoint)
        with self.assertRaisesRegex(ValueError, "--seed differs"):
            validate_evaluation_options(Namespace(amp="bfloat16", seed=2), checkpoint)

    def test_internal_test_runtime_migration_is_exactly_hash_pinned(self) -> None:
        key = "FineTuning/BrainIAC/finetune.py"
        metrics_key = "FineTuning/shared/metrics.py"
        locked = {
            key: "bed265cdefaaf8465bbd294a1e16e8ce177bebdb0e445a3c33d63da909b7d328",
            metrics_key: "d7ec28a8f8d4d921efc397da4662af7bec972bb567f7061aee7d1544cd0ebd3e",
            "unchanged.py": "same",
        }
        migration = internal_test_runtime_migration("BrainIAC", locked)
        self.assertIsNotNone(migration)
        old_hash, new_hash = migration.changed[key]
        self.assertEqual(new_hash, sha256_file(PROJECT_ROOT / key))
        current = {
            key: new_hash,
            metrics_key: migration.changed[metrics_key][1],
            "unchanged.py": "same",
        }
        provenance = validate_prediction_runtime_code(
            locked, current, migration
        )
        self.assertEqual(provenance["mode"], "hash_pinned_evaluation_only_migration")
        self.assertEqual(provenance["migration_id"], migration.migration_id)
        with self.assertRaisesRegex(ValueError, "Unapproved runtime hash transition"):
            validate_prediction_runtime_code(
                locked, {**current, key: "different"},
                migration,
            )
        with self.assertRaisesRegex(ValueError, "lacks a code-hash mapping"):
            internal_test_runtime_migration("BrainIAC", {})
        with self.assertRaises(TypeError):
            internal_test_runtime_migration("BrainIAC")

    def test_selection_and_current_runtime_migrations_are_independent(self) -> None:
        finetune_key = "FineTuning/BrainIAC/finetune.py"
        metrics_key = "FineTuning/shared/metrics.py"
        selection_migration = adapted_selection_runtime_migration("BrainIAC")
        self.assertIsNotNone(selection_migration)
        selection_old, selection_new = selection_migration.changed[finetune_key]
        self.assertEqual(
            INTERNAL_TEST_SELECTION_RUNTIME_MIGRATION,
            selection_migration,
        )
        self.assertEqual(
            selection_old,
            "bed265cdefaaf8465bbd294a1e16e8ce177bebdb0e445a3c33d63da909b7d328",
        )
        self.assertEqual(
            selection_new,
            "48f4f243140a1c40b8858d0dfaf35563445270edca07df8c4a4f7b5382ce480f",
        )
        current_finetune = sha256_file(PROJECT_ROOT / finetune_key)
        old_hashes = (selection_old, selection_new)
        for old_hash in old_hashes:
            with self.subTest(old_hash=old_hash):
                locked_code = {
                    finetune_key: old_hash,
                    metrics_key: (
                        "d7ec28a8f8d4d921efc397da4662af7bec972bb567f7061aee7d1544cd0ebd3e"
                    ),
                }
                expected = adapted_runtime_migration("BrainIAC", locked_code)
                self.assertEqual(
                    internal_test_runtime_migration("BrainIAC", locked_code),
                    expected,
                )
                current_code = dict(locked_code)
                for changed_key, (_, new_hash) in expected.changed.items():
                    current_code[changed_key] = new_hash
                provenance = validate_prediction_runtime_code(
                    locked_code,
                    current_code,
                    expected,
                )
                self.assertEqual(
                    provenance["mode"], "hash_pinned_evaluation_only_migration"
                )

        current_code = {
            finetune_key: current_finetune,
            metrics_key: sha256_file(PROJECT_ROOT / metrics_key),
        }
        self.assertIsNone(
            internal_test_runtime_migration("BrainIAC", current_code)
        )
        with self.assertRaisesRegex(ValueError, "No unique approved BrainIAC"):
            internal_test_runtime_migration(
                "BrainIAC", {**current_code, finetune_key: "unknown"}
            )

    def test_frozen_evaluation_requires_a_validation_only_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "frozen_linear_model.npz"
            model_path.write_bytes(b"locked-ridge")
            lock = {
                "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                "arm": "frozen",
                "model": "BrainIAC",
                "variant": "canonical",
                "model_artifact_sha256": sha256_file(model_path),
                "internal_test_accessed": False,
                "on_harmony_accessed": False,
            }
            validate_evaluation_lock(lock, model_path, variant="canonical")

            invalid_cases = {
                "wrong status": {"status": "INCOMPLETE"},
                "test already accessed": {"internal_test_accessed": True},
                "ON-Harmony accessed": {"on_harmony_accessed": True},
            }
            for label, update in invalid_cases.items():
                with self.subTest(label=label):
                    invalid = {**lock, **update}
                    with self.assertRaisesRegex(ValueError, "validation-only locked"):
                        validate_evaluation_lock(invalid, model_path, variant="canonical")

            identity_cases = {
                "wrong arm": {"arm": "adapted"},
                "wrong model": {"model": "Other"},
                "wrong variant": {"variant": "other"},
            }
            for label, update in identity_cases.items():
                with self.subTest(label=label):
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
                root = Path(directory)
                args = self._frozen_evaluation_fixture(root)
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
