#!/usr/bin/env python3
"""Dependency-light tests for the y-Aware brain-age workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

import FineTuning.yAware.fit_ridge as fit_ridge_module
import Emb_extractors.extract_yAware as onharmony_extractor
from FineTuning.shared.cohort import frame_for_splits, sha256_file
from FineTuning.shared.evaluation import validate_selected_evaluation_run
from FineTuning.shared.metrics import embedding_columns, regression_metrics, validate_embedding_frame
from FineTuning.shared.protocol import (
    ADAPTED_FIXED_HYPERPARAMETERS,
    ADAPTED_LR_CONFIGS,
    FROZEN_RIDGE_ALPHAS,
    adapted_learning_rates,
    select_adapted_learning_rate,
)
from FineTuning.shared.ridge import atomic_npz, fit_ridge_grid, load_npz, parse_ridge_alphas, predict_ridge
from FineTuning.shared.runtime import atomic_json, atomic_tsv
from FineTuning.yAware.extract_embeddings import (
    command_audit as audit_embeddings,
    command_extract as extract_embeddings,
)
from FineTuning.yAware.finetune import (
    validate_evaluation_options,
    validate_selection_candidates,
)
from FineTuning.yAware.fit_ridge import fit_code_hashes, validate_evaluation_lock
from FineTuning.yAware.model import (
    EMBEDDING_DIM,
    MODEL_INPUT_SHAPE,
    PREPROCESSING_CONTRACT_SHA256,
    PREPROCESSING_PIPELINE_VERSION,
    VARIANT,
    YAwareModel,
    YAwareOptions,
    _build_pretraining_model,
    _checkpoint_state_dict,
    _strip_data_parallel_prefix,
    load_and_validate_manifest,
    make_yaware_input,
)


class CohortTests(unittest.TestCase):
    def test_authoritative_manifest_and_offline_contract(self) -> None:
        frame, summary = load_and_validate_manifest()
        self.assertEqual(len(frame), 1411)
        self.assertEqual(
            summary["split_counts"],
            {"train": 1130, "validation": 141, "internal_test": 140},
        )
        self.assertEqual(summary["input_contract"], {"required_offline_group": "cat12_ixi555_gm"})
        self.assertEqual(summary["on_harmony_firewall"], "PASS")

    def test_firewall_rejects_on_harmony_alias(self) -> None:
        frame, _ = load_and_validate_manifest()
        altered = frame.copy()
        altered.loc[0, "dataset"] = "ON-Harmony"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.tsv"
            altered.to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "firewall"):
                load_and_validate_manifest(path, expected_sha256=None, expected_counts=None)

    def test_split_selection_preserves_manifest_order(self) -> None:
        frame, _ = load_and_validate_manifest()
        validation = frame_for_splits(frame, ["validation"])
        self.assertEqual(len(validation), 141)
        self.assertEqual(validation.iloc[0]["subject"], frame.loc[frame["split"] == "validation"].iloc[0]["subject"])

    def test_cat12_failure_is_absent_from_common_manifest(self) -> None:
        frame, _ = load_and_validate_manifest()
        self.assertEqual(len(frame_for_splits(frame, ["internal_test"])), 140)
        self.assertFalse(
            ((frame["dataset"] == "CamCAN") & (frame["subject"] == "sub-CC620821")).any()
        )


class ModelContractTests(unittest.TestCase):
    def test_input_is_validated_but_not_transformed(self) -> None:
        rng = np.random.default_rng(4)
        data = rng.normal(size=MODEL_INPUT_SHAPE).astype(np.float32)
        data = ((data - data.mean()) / data.std()).astype(np.float32)
        observed = make_yaware_input(data)
        self.assertEqual(observed.shape, MODEL_INPUT_SHAPE)
        np.testing.assert_array_equal(observed, data)

    def test_input_rejects_shape_nonfinite_and_nonstandardized_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be"):
            make_yaware_input(np.zeros((121, 145, 121), dtype=np.float32))
        invalid = np.zeros(MODEL_INPUT_SHAPE, dtype=np.float32)
        invalid.flat[0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            make_yaware_input(invalid)
        with self.assertRaisesRegex(ValueError, "z-score"):
            make_yaware_input(np.arange(np.prod(MODEL_INPUT_SHAPE), dtype=np.float32).reshape(MODEL_INPUT_SHAPE))

    def test_checkpoint_layouts_and_prefix_removal_are_explicit(self) -> None:
        state = {"module.features.weight": np.ones(1)}
        self.assertIs(_checkpoint_state_dict({"model": state}), state)
        self.assertEqual(set(_strip_data_parallel_prefix(state)), {"features.weight"})
        with self.assertRaisesRegex(ValueError, "Unexpected y-Aware checkpoint layout"):
            _checkpoint_state_dict({"weights": "invalid"})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _strip_data_parallel_prefix({"module.a": object(), "a": object()})

    def test_qc_provenance_binds_yaware_hash_and_shared_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            input_dir = root / "preproc_yAware" / "CamCAN" / "sub-test"
            qc_dir = root / "preproc_AnatCL" / "CamCAN" / "sub-test"
            input_dir.mkdir(parents=True)
            qc_dir.mkdir(parents=True)
            array_path = input_dir / "sub-test_yaware_ready.npy"
            with array_path.open("wb") as handle:
                np.save(handle, np.zeros(MODEL_INPUT_SHAPE, dtype=np.float32))
            qc = {
                "status": "PASS",
                "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
                "pipeline_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
                "dataset": "CamCAN",
                "subject": "sub-test",
                "split": "train",
                "yaware_output_sha256": sha256_file(array_path),
                "runtime_views": {
                    "yaware": {
                        "shape": list(MODEL_INPUT_SHAPE),
                        "normalization": "whole-volume z-score",
                    }
                },
            }
            qc_path = qc_dir / "preprocessing_qc.json"
            qc_path.write_text(json.dumps(qc), encoding="utf-8")
            model = YAwareModel(
                YAwareOptions(
                    checkpoint=root / "checkpoint.pth",
                    preproc_root=root / "preproc_yAware",
                    qc_root=root / "preproc_AnatCL",
                )
            )
            record = {"dataset": "CamCAN", "subject": "sub-test", "split": "train"}
            provenance = model.validate_input_provenance(record)
            self.assertEqual(provenance["input_sha256"], sha256_file(array_path))
            array_path.write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                model.validate_input_provenance(record)

    def test_release_architecture_has_no_norm5_and_has_expected_projection(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        model = _build_pretraining_model(torch)
        keys = set(model.state_dict())
        self.assertIn("features.denseblock4.denselayer16.conv2.weight", keys)
        self.assertIn("hidden_representation.weight", keys)
        self.assertIn("head_projection.weight", keys)
        self.assertFalse(any("norm5" in key for key in keys))

    def test_onharmony_extractor_uses_canonical_adapter_and_bfloat16_default(self) -> None:
        self.assertIs(onharmony_extractor.YAwareModel, YAwareModel)
        args = onharmony_extractor.parse_args([])
        self.assertEqual(args.amp, "bfloat16")
        self.assertEqual(args.seed, 42)

        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        device = torch.device("cpu")
        with patch.object(onharmony_extractor, "YAwareModel") as adapter_class:
            adapter = adapter_class.return_value
            adapter.set_frozen.return_value = {"encoder_trainable_parameters": 0}
            observed, summary = onharmony_extractor.build_model(
                Path("checkpoint.pth"), device
            )
        options = adapter_class.call_args.args[0]
        self.assertEqual(options.checkpoint, Path("checkpoint.pth"))
        adapter.load.assert_called_once_with(device)
        adapter.set_frozen.assert_called_once_with()
        self.assertIs(observed, adapter)
        self.assertEqual(summary["encoder_trainable_parameters"], 0)

    def test_onharmony_input_reuses_canonical_validation(self) -> None:
        rng = np.random.default_rng(41)
        data = rng.normal(size=MODEL_INPUT_SHAPE).astype(np.float32)
        data = ((data - data.mean()) / data.std()).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npy"
            np.save(path, np.asfortranarray(data))
            tensor = onharmony_extractor.load_volume(path)
        self.assertEqual(tuple(tensor.shape), (1, *MODEL_INPUT_SHAPE))
        self.assertTrue(tensor.is_contiguous())
        np.testing.assert_array_equal(tensor.numpy()[0], data)

    def test_onharmony_embedding_uses_amp_and_serializes_float32(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummyModel:
            @staticmethod
            def preprojection(volume):
                del volume
                return torch.arange(EMBEDDING_DIM, dtype=torch.bfloat16).reshape(1, -1)

        adapter = SimpleNamespace(model=DummyModel())
        device = torch.device("cpu")
        volume = torch.zeros((1, *MODEL_INPUT_SHAPE), dtype=torch.float32)
        with patch.object(
            onharmony_extractor,
            "autocast_context",
            return_value=torch.autocast(device_type="cpu", enabled=False),
        ) as autocast:
            embedding = onharmony_extractor.extract_embedding(
                adapter, volume, device, "bfloat16"
            )
        autocast.assert_called_once_with(torch, device, "bfloat16")
        self.assertEqual(embedding.shape, (EMBEDDING_DIM,))
        self.assertEqual(embedding.dtype, np.float32)

    def test_onharmony_runtime_is_locked_to_cuda_bfloat16_seed42(self) -> None:
        cuda = SimpleNamespace(type="cuda")
        onharmony_extractor.validate_runtime_contract(cuda, "bfloat16", 42)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            onharmony_extractor.validate_runtime_contract(
                SimpleNamespace(type="cpu"), "bfloat16", 42
            )
        with self.assertRaisesRegex(ValueError, "amp=bfloat16"):
            onharmony_extractor.validate_runtime_contract(cuda, "off", 42)
        with self.assertRaisesRegex(ValueError, "seed=42"):
            onharmony_extractor.validate_runtime_contract(cuda, "bfloat16", 0)

    def test_only_denseblock4_is_trainable(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = torch.nn.Module()
                self.features.denseblock1 = torch.nn.Linear(2, 2)
                self.features.denseblock4 = torch.nn.Sequential(
                    *[torch.nn.Linear(2, 2) for _ in range(16)]
                )
                self.hidden_representation = torch.nn.Linear(2, 2)
                self.head_projection = torch.nn.Linear(2, 1)

        adapter = YAwareModel(
            YAwareOptions(checkpoint=Path("."), preproc_root=Path("."), qc_root=Path("."))
        )
        adapter.model = DummyModel()
        adapter.device = torch.device("cpu")
        expected_trainable = {
            name
            for name, _ in adapter.model.named_parameters()
            if name.startswith("features.denseblock4.")
        }
        summary = adapter.set_controlled_adaptation()
        observed_trainable = {
            name for name, value in adapter.model.named_parameters() if value.requires_grad
        }
        self.assertTrue(expected_trainable)
        self.assertSetEqual(observed_trainable, expected_trainable)
        self.assertSetEqual(
            set(summary["trainable_parameter_names"]), expected_trainable
        )
        self.assertEqual(summary["trainable_components"], ["features.denseblock4"])


class MetricsAndEmbeddingTests(unittest.TestCase):
    def test_regression_metrics_are_uncorrected(self) -> None:
        metrics = regression_metrics([20, 30, 40], [22, 29, 43])
        self.assertAlmostEqual(metrics["mae_years"], 2.0)
        self.assertEqual(metrics["n"], 3)

    def test_embedding_columns_are_contiguous(self) -> None:
        self.assertEqual(embedding_columns(pd.DataFrame({"emb_0": [1], "emb_1": [2]})), ["emb_0", "emb_1"])
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
                "model": ["y-Aware", "y-Aware"],
                "variant": [VARIANT, VARIANT],
                "emb_0": [1.0, 2.0],
            }
        )
        self.assertEqual(
            validate_embedding_frame(
                table,
                expected_manifest=expected,
                expected_model="y-Aware",
                expected_variant=VARIANT,
            ),
            ["emb_0"],
        )


class RidgeAndProtocolTests(unittest.TestCase):
    def test_ridge_code_manifest_includes_the_model_contract(self) -> None:
        key = "FineTuning/yAware/model.py"
        self.assertEqual(fit_code_hashes()[key], sha256_file(PROJECT_ROOT / key))

    def test_fixed_protocol_values(self) -> None:
        self.assertEqual(FROZEN_RIDGE_ALPHAS, (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))
        self.assertEqual(adapted_learning_rates("conservative"), (1e-6, 1e-4))
        self.assertEqual(adapted_learning_rates("intermediate"), (3e-6, 3e-4))
        self.assertEqual(adapted_learning_rates("high"), (1e-5, 1e-3))

    def test_alpha_grid_validation_and_roundtrip(self) -> None:
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

    def test_lr_selection_is_canonical_and_validation_only(self) -> None:
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
                    "checkpoint": "/fixed/checkpoint.pth",
                    "preproc_root": "/fixed/inputs",
                    "qc_root": "/fixed/qc",
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
                        "model": "y-Aware",
                        "variant": VARIANT,
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
                expected_model="y-Aware",
                expected_variant=VARIANT,
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
                expected_model="y-Aware",
                evaluation_variant=VARIANT,
            )
            self.assertEqual(provenance["selected_lr_config"], "intermediate")

            wrong_run = root / "high"
            wrong_lock = json.loads(
                (wrong_run / "LOCKED.json").read_text(encoding="utf-8")
            )
            with self.assertRaisesRegex(ValueError, "validation-selected LR"):
                validate_selected_evaluation_run(
                    selection_dir=output,
                    run_dir=wrong_run,
                    lock=wrong_lock,
                    manifest_sha256="manifest-test",
                    expected_model="y-Aware",
                    evaluation_variant=VARIANT,
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
                "model": "y-Aware",
                "variant": VARIANT,
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

    def test_identical_runtime_settings_are_accepted(self) -> None:
        args = Namespace(amp="bfloat16")
        checkpoint = {"run_config": {"arguments": vars(args).copy()}}
        validate_evaluation_options(args, checkpoint)

    def test_amp_drift_is_rejected(self) -> None:
        checkpoint = {"run_config": {"arguments": {"amp": "bfloat16"}}}
        with self.assertRaisesRegex(ValueError, "--amp differs"):
            validate_evaluation_options(Namespace(amp="float16"), checkpoint)

    def test_frozen_evaluation_requires_a_validation_only_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "frozen_linear_model.npz"
            model_path.write_bytes(b"locked-ridge")
            lock = {
                "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                "arm": "frozen",
                "model": "y-Aware",
                "variant": VARIANT,
                "model_artifact_sha256": sha256_file(model_path),
                "internal_test_accessed": False,
                "on_harmony_accessed": False,
            }
            validate_evaluation_lock(lock, model_path, variant=VARIANT)
            for update in (
                {"status": "INCOMPLETE"},
                {"internal_test_accessed": True},
                {"on_harmony_accessed": True},
            ):
                with self.assertRaisesRegex(ValueError, "validation-only locked"):
                    validate_evaluation_lock(
                        {**lock, **update}, model_path, variant=VARIANT
                    )

            for update in (
                {"arm": "adapted"},
                {"model": "Other"},
                {"variant": "other"},
            ):
                with self.assertRaisesRegex(ValueError, "not the expected frozen"):
                    validate_evaluation_lock(
                        {**lock, **update}, model_path, variant=VARIANT
                    )

            model_path.write_bytes(b"tampered-ridge")
            with self.assertRaisesRegex(ValueError, "Frozen model hash differs"):
                validate_evaluation_lock(lock, model_path, variant=VARIANT)

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
