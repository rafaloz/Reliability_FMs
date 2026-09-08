#!/usr/bin/env python3
"""Dependency-light tests for the BrainSegFounder brain-age workflow."""

from __future__ import annotations

import json
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

import model as brainsegfounder_model  # noqa: E402
import fit_ridge as fit_ridge_module  # noqa: E402
from model import (  # noqa: E402
    BOTTLENECK_SHAPE,
    COMPATIBLE_PREPROCESSING_CONTRACTS,
    EMBEDDING_DIM,
    MODEL_INPUT_SHAPE,
    PREPROCESSING_CONTRACT_SHA256,
    PREPROCESSING_PIPELINE_VERSION,
    BrainSegFounderModel,
    BrainSegFounderOptions,
    clean_swin_state_dict,
    load_and_validate_manifest,
)
from extract_embeddings import (  # noqa: E402
    PIPELINE_VERSION as EXTRACTION_PIPELINE_VERSION,
    command_audit as audit_embeddings,
    command_extract as extract_embeddings,
)
from finetune import (  # noqa: E402
    LOCK_NAME,
    PIPELINE_VERSION as ADAPTED_PIPELINE_VERSION,
    validate_evaluation_options,
    validate_selection_candidates,
)
from fit_ridge import (  # noqa: E402
    PIPELINE_VERSION as RIDGE_PIPELINE_VERSION,
    fit_code_hashes,
    validate_evaluation_lock,
)
from Supplementary.Analyses.select_lr_migration_BrainSegFounder import (  # noqa: E402
    CURRENT_RUNTIME_SHA256,
    LEGACY_RUNTIME_SHA256,
    SELECTOR_RUNTIME_SHA256,
    verify_source_migration,
)
from FineTuning.shared.cohort import frame_for_splits, sha256_file  # noqa: E402
from FineTuning.shared.metrics import regression_metrics, validate_embedding_frame  # noqa: E402
from FineTuning.shared.protocol import adapted_learning_rates  # noqa: E402
from FineTuning.shared.runtime import atomic_json  # noqa: E402


class FakeTensor:
    def __init__(self, size: int = 1) -> None:
        self.size = size

    def numel(self) -> int:
        return self.size


class CohortAndPathTests(unittest.TestCase):
    def test_manifest_permits_offline_bsf_inputs(self) -> None:
        frame, summary = load_and_validate_manifest()
        self.assertEqual(len(frame), 1411)
        self.assertEqual(
            summary["split_counts"],
            {"train": 1130, "validation": 141, "internal_test": 140},
        )
        self.assertEqual(
            summary["input_contract"],
            {"required_offline_group": "preproc_brainsegfounder"},
        )
        self.assertEqual(summary["on_harmony_firewall"], "PASS")

    def test_dataset_scoped_paths_are_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = BrainSegFounderModel(
                BrainSegFounderOptions(
                    checkpoint=Path(directory) / "checkpoint.pt",
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
                / "sub-001_brainsegfounder_ready.nii.gz",
            )
            self.assertEqual(adapter.qc_path(record).name, "preprocessing_qc.json")


class InputAndCheckpointTests(unittest.TestCase):
    @staticmethod
    def _provenance_fixture(
        root: Path,
    ) -> tuple[BrainSegFounderModel, dict[str, str], Path]:
        subject_dir = root / "preprocessed" / "CamCAN" / "sub-001"
        subject_dir.mkdir(parents=True)
        image = subject_dir / "sub-001_brainsegfounder_ready.nii.gz"
        qc = subject_dir / "preprocessing_qc.json"
        image.write_bytes(b"brainsegfounder-image-v1")
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
        adapter = BrainSegFounderModel(
            BrainSegFounderOptions(
                checkpoint=root / "checkpoint.pt",
                preproc_root=root / "preprocessed",
            )
        )
        return adapter, record, image

    def test_provenance_cache_and_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter, record, image = self._provenance_fixture(Path(directory))
            with patch.object(
                brainsegfounder_model, "sha256_file", wraps=sha256_file
            ) as mocked_hash:
                first = adapter.validate_input_provenance(record)
                second = adapter.validate_input_provenance(record)
            self.assertEqual(first, second)
            self.assertEqual(mocked_hash.call_count, 2)
            image.write_bytes(b"brainsegfounder-image-v2-with-different-size")
            with self.assertRaisesRegex(ValueError, "differs"):
                adapter.validate_input_provenance(record)

    def test_load_input_rejects_shape_nonfinite_and_empty_volumes(self) -> None:
        import nibabel as nib

        with tempfile.TemporaryDirectory() as directory:
            adapter, record, image = self._provenance_fixture(Path(directory))
            qc_path = adapter.qc_path(record)

            def save(data: np.ndarray) -> None:
                nib.save(
                    nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)),
                    image,
                )
                qc = json.loads(qc_path.read_text(encoding="utf-8"))
                qc["output_sha256"]["ready"] = sha256_file(image)
                atomic_json(qc, qc_path)

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

    def test_v3_primary_outputs_remain_provenance_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter, record, _ = self._provenance_fixture(Path(directory))
            qc_path = adapter.qc_path(record)
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            version = "brainsegfounder-ukb-gdcoff-finetuning-v3"
            qc["pipeline_version"] = version
            qc["pipeline_contract_sha256"] = COMPATIBLE_PREPROCESSING_CONTRACTS[
                version
            ]
            atomic_json(qc, qc_path)
            provenance = adapter.validate_input_provenance(record)
            self.assertEqual(provenance["pipeline_version"], version)

    def test_crossed_v3_v4_contract_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter, record, _ = self._provenance_fixture(Path(directory))
            qc_path = adapter.qc_path(record)
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            qc["pipeline_version"] = "brainsegfounder-ukb-gdcoff-finetuning-v3"
            qc["pipeline_contract_sha256"] = PREPROCESSING_CONTRACT_SHA256
            atomic_json(qc, qc_path)
            with self.assertRaisesRegex(ValueError, "preprocessing_contract"):
                adapter.validate_input_provenance(record)

    def test_missing_preprocessing_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter, record, _ = self._provenance_fixture(Path(directory))
            qc_path = adapter.qc_path(record)
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            qc.pop("pipeline_version")
            qc.pop("pipeline_contract_sha256")
            atomic_json(qc, qc_path)
            with self.assertRaisesRegex(ValueError, "preprocessing_contract"):
                adapter.validate_input_provenance(record)

    def test_checkpoint_cleaning_isolates_swin_and_accounts_for_ssl_heads(self) -> None:
        cleaned, dropped, parameters = clean_swin_state_dict(
            {
                "module.swinViT.patch_embed.proj.weight": FakeTensor(10),
                "module.rotation_head.weight": FakeTensor(6),
                "module.contrastive_head.weight": FakeTensor(7),
                "module.conv.weight": FakeTensor(8),
            }
        )
        self.assertEqual(list(cleaned), ["patch_embed.proj.weight"])
        self.assertEqual(dropped, {"rotation_head": 1, "contrastive_head": 1, "conv": 1})
        self.assertEqual(parameters, 21)

    def test_checkpoint_cleaning_rejects_unknown_branches(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected BrainSegFounder checkpoint keys"):
            clean_swin_state_dict(
                {
                    "module.swinViT.x": FakeTensor(),
                    "module.rotation_head.x": FakeTensor(),
                    "module.contrastive_head.x": FakeTensor(),
                    "module.conv.x": FakeTensor(),
                    "module.classifier.x": FakeTensor(),
                }
            )

    def test_only_final_swin_stage_is_trainable(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummySwin(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers3 = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
                self.layers4 = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

        class DummyWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.swinViT = DummySwin()
                self.decoder = torch.nn.Linear(2, 2)

        adapter = BrainSegFounderModel(
            BrainSegFounderOptions(checkpoint=Path("."), preproc_root=Path("."))
        )
        adapter.model = DummyWrapper()
        adapter.device = torch.device("cpu")
        adapter._ssl_parameters = 10
        expected_trainable = {
            name
            for name, _ in adapter.model.named_parameters()
            if name.startswith("swinViT.layers4.")
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
        self.assertEqual(summary["trainable_components"], ["swinViT.layers4"])

    def test_frozen_summary_declares_no_trainable_parameters(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummySwin(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

        class DummyWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.swinViT = DummySwin()

        adapter = BrainSegFounderModel(
            BrainSegFounderOptions(checkpoint=Path("."), preproc_root=Path("."))
        )
        adapter.model = DummyWrapper()
        adapter.device = torch.device("cpu")
        summary = adapter.set_frozen()
        self.assertEqual(summary["encoder_trainable_parameters"], 0)
        self.assertEqual(
            summary["mode_during_adaptation"],
            "full wrapper eval; no trainable parameters",
        )

    def test_encode_spatially_averages_the_final_bottleneck(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummySwin(torch.nn.Module):
            def forward(self, volume, normalize):
                del volume, normalize
                final = torch.arange(
                    EMBEDDING_DIM * 27, dtype=torch.float32
                ).reshape(BOTTLENECK_SHAPE)
                return [torch.zeros(1)] * 4 + [final]

        class DummyWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.swinViT = DummySwin()
                self.normalize = True

        adapter = BrainSegFounderModel(
            BrainSegFounderOptions(checkpoint=Path("."), preproc_root=Path("."))
        )
        adapter.model = DummyWrapper()
        adapter.device = torch.device("cpu")
        adapter.load_input = lambda record: torch.zeros((1, 1, *MODEL_INPUT_SHAPE))
        observed = adapter.encode({}, training=False, epoch=0, seed=0)
        expected = torch.arange(EMBEDDING_DIM * 27, dtype=torch.float32).reshape(
            BOTTLENECK_SHAPE
        ).mean(dim=(2, 3, 4))[0]
        torch.testing.assert_close(observed, expected)


class ProtocolTests(unittest.TestCase):
    def test_all_entry_points_share_one_contract(self) -> None:
        self.assertEqual(EXTRACTION_PIPELINE_VERSION, "brainsegfounder_brain_age_v1")
        self.assertEqual(RIDGE_PIPELINE_VERSION, EXTRACTION_PIPELINE_VERSION)
        self.assertEqual(ADAPTED_PIPELINE_VERSION, EXTRACTION_PIPELINE_VERSION)

    def test_ridge_code_manifest_includes_the_model_contract(self) -> None:
        key = "FineTuning/BrainSegFounder/model.py"
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
                "model": ["BrainSegFounder", "BrainSegFounder"],
                "variant": ["bottleneck_mean", "bottleneck_mean"],
                "emb_0": [1.0, 2.0],
            }
        )
        self.assertEqual(
            validate_embedding_frame(
                table,
                expected_manifest=expected,
                expected_model="BrainSegFounder",
                expected_variant="bottleneck_mean",
            ),
            ["emb_0"],
        )
        self.assertEqual(adapted_learning_rates("conservative"), (1e-6, 1e-4))
        self.assertEqual(adapted_learning_rates("intermediate"), (3e-6, 3e-4))
        self.assertEqual(adapted_learning_rates("high"), (1e-5, 1e-3))

    def test_selection_candidates_use_the_resolved_brainsegfounder_contract(self) -> None:
        adapter = BrainSegFounderModel(
            BrainSegFounderOptions(checkpoint=Path("checkpoint.pt"), preproc_root=Path("inputs"))
        ).audit_description()
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            runs = []
            for config in ("conservative", "intermediate", "high"):
                run = root / config
                run.mkdir()
                atomic_json(
                    {
                        "adapter": adapter,
                        "run_config": {"arguments": {"seed": 0}},
                    },
                    run / LOCK_NAME,
                )
                runs.append(run)
            validate_selection_candidates(runs)

            lock_path = runs[-1] / LOCK_NAME
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["adapter"]["bottleneck_shape"] = [1, EMBEDDING_DIM, 1, 1, 1]
            atomic_json(lock, lock_path)
            with self.assertRaisesRegex(ValueError, "different resolved input contracts"):
                validate_selection_candidates(runs)

    def test_lr_selection_rejects_nonzero_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "seed1"
            run.mkdir()
            atomic_json(
                {"run_config": {"arguments": {"seed": 1}}},
                run / LOCK_NAME,
            )
            with self.assertRaisesRegex(ValueError, "requires seed 0"):
                validate_selection_candidates([run])

    def test_selector_migration_reconstructs_the_allowlisted_legacy_source(self) -> None:
        migration = verify_source_migration()
        self.assertEqual(migration["legacy_sha256"], LEGACY_RUNTIME_SHA256)
        self.assertEqual(
            migration["selector_corrected_sha256"], SELECTOR_RUNTIME_SHA256
        )
        self.assertEqual(migration["current_sha256"], CURRENT_RUNTIME_SHA256)


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
                "model": "BrainSegFounder",
                "variant": "bottleneck_mean",
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

    def test_amp_and_seed_locks_are_enforced(self) -> None:
        args = Namespace(amp="bfloat16", seed=0)
        checkpoint = {"run_config": {"arguments": vars(args).copy()}}
        validate_evaluation_options(args, checkpoint)
        with self.assertRaisesRegex(ValueError, "--amp differs"):
            validate_evaluation_options(Namespace(amp="float16", seed=0), checkpoint)
        with self.assertRaisesRegex(ValueError, "--seed differs"):
            validate_evaluation_options(Namespace(amp="bfloat16", seed=1), checkpoint)

    def test_frozen_evaluation_requires_a_validation_only_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "frozen_linear_model.npz"
            model_path.write_bytes(b"locked-ridge")
            lock = {
                "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                "arm": "frozen",
                "model": "BrainSegFounder",
                "variant": "bottleneck_mean",
                "model_artifact_sha256": sha256_file(model_path),
                "internal_test_accessed": False,
                "on_harmony_accessed": False,
            }
            validate_evaluation_lock(lock, model_path, variant="bottleneck_mean")
            for update in (
                {"status": "INCOMPLETE"},
                {"internal_test_accessed": True},
                {"on_harmony_accessed": True},
            ):
                with self.assertRaisesRegex(ValueError, "validation-only locked"):
                    validate_evaluation_lock(
                        {**lock, **update}, model_path, variant="bottleneck_mean"
                    )

            for update in (
                {"arm": "adapted"},
                {"model": "Other"},
                {"variant": "other"},
            ):
                with self.assertRaisesRegex(ValueError, "not the expected frozen"):
                    validate_evaluation_lock(
                        {**lock, **update}, model_path, variant="bottleneck_mean"
                    )

            model_path.write_bytes(b"tampered-ridge")
            with self.assertRaisesRegex(ValueError, "Frozen model hash differs"):
                validate_evaluation_lock(lock, model_path, variant="bottleneck_mean")

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
