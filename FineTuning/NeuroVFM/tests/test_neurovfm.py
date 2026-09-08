#!/usr/bin/env python3
"""Dependency-light tests for the NeuroVFM brain-age workflow."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

import FineTuning.NeuroVFM.fit_ridge as fit_ridge_module
import FineTuning.NeuroVFM.extract_embeddings as extract_embeddings_module
import FineTuning.NeuroVFM.model as model_module
from FineTuning.NeuroVFM.extract_embeddings import (
    command_audit as audit_embeddings,
    command_cache as cache_embeddings,
    command_extract as extract_embeddings,
    parse_args as parse_embedding_args,
)
from FineTuning.NeuroVFM.finetune import (
    validate_evaluation_options,
    validate_selection_candidates,
)
from FineTuning.NeuroVFM.model import (
    DEFAULT_MANIFEST,
    NEUROVFM_MODEL_REPO,
    NEUROVFM_MODEL_REVISION,
    NeuroVFMModel,
    NeuroVFMOptions,
    load_and_validate_manifest,
    preprocessing_cache_paths,
    resolve_neurovfm_snapshot,
    validate_neurovfm_code,
    validate_neurovfm_device,
)
from FineTuning.shared.cohort import frame_for_splits, sha256_file
from FineTuning.shared.evaluation import validate_selected_evaluation_run
from FineTuning.shared.metrics import (
    embedding_columns,
    regression_metrics,
    validate_embedding_frame,
)
from FineTuning.shared.protocol import (
    ADAPTED_FIXED_HYPERPARAMETERS,
    ADAPTED_LR_CONFIGS,
    FROZEN_RIDGE_ALPHAS,
    adapted_learning_rates,
    select_adapted_learning_rate,
)
from FineTuning.shared.ridge import (
    atomic_npz,
    fit_ridge_grid,
    load_npz,
    parse_ridge_alphas,
    predict_ridge,
)
from FineTuning.shared.runtime import atomic_json, atomic_tsv


class CohortTests(unittest.TestCase):
    def test_authoritative_manifest(self) -> None:
        frame, summary = load_and_validate_manifest(DEFAULT_MANIFEST)
        self.assertEqual(len(frame), 1411)
        self.assertEqual(
            summary["split_counts"],
            {"train": 1130, "validation": 141, "internal_test": 140},
        )
        self.assertEqual(summary["input_contract"], {"required_runtime_condition": "NeuroVFM"})
        self.assertEqual(summary["on_harmony_firewall"], "PASS")

    def test_firewall_rejects_on_harmony_alias(self) -> None:
        frame = pd.read_csv(DEFAULT_MANIFEST, sep="\t").copy()
        frame.loc[0, "raw_t1w_path"] = "/tmp/ON_Harmony/sub-001_T1w.nii.gz"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.tsv"
            frame.to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "ON-Harmony firewall"):
                load_and_validate_manifest(path, expected_sha256=None, expected_counts=None)

    def test_split_selection_preserves_manifest_order(self) -> None:
        frame, _ = load_and_validate_manifest()
        selected = frame_for_splits(frame, ["train", "validation"])
        self.assertEqual(len(selected), 1271)
        self.assertFalse((selected["split"] == "internal_test").any())


class ReleaseContractTests(unittest.TestCase):
    def test_default_source_is_revision_pinned(self) -> None:
        options = NeuroVFMOptions()
        self.assertEqual(options.model_source, NEUROVFM_MODEL_REPO)
        self.assertEqual(options.model_revision, NEUROVFM_MODEL_REVISION)
        options.validate()

    def test_remote_source_requires_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable revision"):
            NeuroVFMOptions(model_source="org/moving-model", model_revision=None).validate()

    def test_local_snapshot_is_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "pytorch_model.bin").write_bytes(b"weights")
            snapshot, details = resolve_neurovfm_snapshot(str(root), None)
            self.assertEqual(snapshot, root.resolve())
            self.assertEqual(details["source_kind"], "local_directory")
            self.assertEqual(
                details["artifacts_sha256"]["pytorch_model.bin"],
                sha256_file(root / "pytorch_model.bin"),
            )

    def test_huggingface_symlinks_preserve_snapshot_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models--mlinslab--neurovfm-encoder"
            blobs = cache / "blobs"
            snapshot_root = cache / "snapshots" / NEUROVFM_MODEL_REVISION
            blobs.mkdir(parents=True)
            snapshot_root.mkdir(parents=True)

            config_blob = blobs / "config-hash"
            weights_blob = blobs / "weights-hash"
            config_blob.write_text("{}\n", encoding="utf-8")
            weights_blob.write_bytes(b"weights")
            (snapshot_root / "config.json").symlink_to(config_blob)
            (snapshot_root / "pytorch_model.bin").symlink_to(weights_blob)

            pointers = {
                "config.json": snapshot_root / "config.json",
                "pytorch_model.bin": snapshot_root / "pytorch_model.bin",
            }

            def fake_download(*, filename: str, **_options: object) -> str:
                return str(pointers[filename])

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.hf_hub_download = fake_download
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                snapshot, details = resolve_neurovfm_snapshot(
                    NEUROVFM_MODEL_REPO,
                    NEUROVFM_MODEL_REVISION,
                )

            self.assertEqual(snapshot, snapshot_root)
            self.assertEqual(details["source_kind"], "huggingface_snapshot")
            self.assertEqual(
                details["artifacts_sha256"]["pytorch_model.bin"],
                sha256_file(weights_blob),
            )

    def test_code_validation_accepts_exact_hash_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "models" / "vit.py"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"audited")
            expected = hashlib.sha256(b"audited").hexdigest()
            with patch.dict(model_module.NEUROVFM_CODE_SHA256, {"models/vit.py": expected}, clear=True):
                result = validate_neurovfm_code(root)
                self.assertEqual(result["files_sha256"]["models/vit.py"], expected)
                path.write_bytes(b"drift")
                with self.assertRaisesRegex(RuntimeError, "differs from the audited release"):
                    validate_neurovfm_code(root)


class _FakeCuda:
    def __init__(self, capability: tuple[int, int], bf16: bool = True) -> None:
        self.capability = capability
        self.bf16 = bf16

    def get_device_capability(self, _device):
        return self.capability

    def is_bf16_supported(self):
        return self.bf16

    def get_device_name(self, _device):
        return "Fake GPU"


class _FakeTorch:
    def __init__(self, capability: tuple[int, int], bf16: bool = True) -> None:
        self.cuda = _FakeCuda(capability, bf16)


class _FakeDevice:
    type = "cuda"

    def __str__(self) -> str:
        return "cuda"


class RuntimeAndBoundaryTests(unittest.TestCase):
    def test_sm86_bfloat16_is_accepted(self) -> None:
        result = validate_neurovfm_device(_FakeTorch((8, 6)), _FakeDevice(), "bfloat16")
        self.assertEqual(result["compute_capability"], "sm_86")

    def test_sm75_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "sm_80 or newer"):
            validate_neurovfm_device(_FakeTorch((7, 5), bf16=False), _FakeDevice(), "bfloat16")

    def test_non_bfloat16_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --amp bfloat16"):
            validate_neurovfm_device(_FakeTorch((8, 6)), _FakeDevice(), "float16")

    def test_controlled_boundary_is_final_block_and_norm(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class FakeEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_embed = torch.nn.Linear(2, 2)
                self.blocks = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2) for _ in range(12)]
                )
                self.norm = torch.nn.LayerNorm(2)

        workflow = NeuroVFMModel(NeuroVFMOptions())
        workflow.model = FakeEncoder()
        workflow.pipeline = types.SimpleNamespace(
            norm_module=torch.nn.Linear(2, 2)
        )
        workflow.preprocessor = object()
        components = workflow.controlled_modules()
        self.assertEqual([name for name, _ in components], ["blocks.11", "norm"])
        self.assertIs(components[0][1], workflow.model.blocks[-1])
        self.assertIs(components[1][1], workflow.model.norm)
        expected_trainable = {
            name
            for name, _ in workflow.model.named_parameters()
            if name.startswith("blocks.11.") or name.startswith("norm.")
        }
        summary = workflow.set_controlled_adaptation()
        observed_trainable = {
            name for name, _ in workflow.trainable_named_parameters()
        }
        self.assertTrue(expected_trainable)
        self.assertSetEqual(observed_trainable, expected_trainable)
        self.assertSetEqual(
            set(summary["trainable_parameter_names"]), expected_trainable
        )
        self.assertEqual(summary["trainable_components"], ["blocks.11", "norm"])
        self.assertFalse(
            any(parameter.requires_grad for parameter in workflow.pipeline.norm_module.parameters())
        )


class PreprocessingCacheTests(unittest.TestCase):
    @staticmethod
    def _record(raw_path: Path) -> dict[str, object]:
        return {
            "dataset": "DatasetA",
            "subject": "sub-001",
            "split": "train",
            "raw_t1w_path": str(raw_path),
        }

    @staticmethod
    def _batch(torch, raw_path: Path) -> dict[str, object]:
        n_tokens = 3
        return {
            "img": torch.arange(n_tokens * 1024, dtype=torch.float32).reshape(
                n_tokens, 1024
            ),
            "coords": torch.arange(n_tokens * 3, dtype=torch.int64).reshape(n_tokens, 3),
            "series_masks_indices": torch.tensor([]),
            "series_cu_seqlens": torch.tensor([0, n_tokens], dtype=torch.int32),
            "series_max_len": np.int64(n_tokens),
            "mode": ["mri"],
            "path": [str(raw_path)],
            "size": [(np.int64(4), np.int64(16), np.int64(16))],
        }

    @staticmethod
    def _contract() -> dict[str, str]:
        return {"cache_version": "test", "contract_sha256": "fixed-contract"}

    def _build_cache(self, root: Path, raw_path: Path):
        import torch

        class FakePreprocessor:
            calls = 0

            def load_study(inner_self, path: str, modality: str):
                self.assertEqual(path, str(raw_path))
                self.assertEqual(modality, "mri")
                inner_self.calls += 1
                return self._batch(torch, raw_path)

        workflow = NeuroVFMModel(
            NeuroVFMOptions(preprocessing_cache_root=root)
        )
        preprocessor = FakePreprocessor()
        workflow.preprocessor = preprocessor
        record = self._record(raw_path)
        with patch.object(
            workflow,
            "preprocessing_cache_contract",
            return_value=self._contract(),
        ):
            result = workflow.build_preprocessing_cache(record)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(preprocessor.calls, 1)
        self.assertEqual(workflow.preprocessing_cache_statistics()["raw_preprocessing_calls"], 1)
        return workflow, record

    def test_batch_validation_rejects_shape_empty_and_nonfinite_tokens(self) -> None:
        import torch

        raw_path = Path("raw.nii.gz")
        valid = self._batch(torch, raw_path)

        wrong_shape = dict(valid)
        wrong_shape["img"] = valid["img"][:, :-1]
        with self.assertRaisesRegex(ValueError, "patch shape"):
            NeuroVFMModel._validate_preprocessed_batch(torch, wrong_shape)

        empty = dict(valid)
        empty["img"] = torch.empty((0, 1024), dtype=torch.float32)
        empty["coords"] = torch.empty((0, 3), dtype=torch.int64)
        empty["series_cu_seqlens"] = torch.tensor([0, 0], dtype=torch.int32)
        empty["series_max_len"] = np.int64(0)
        with self.assertRaisesRegex(ValueError, "no foreground tokens"):
            NeuroVFMModel._validate_preprocessed_batch(torch, empty)

        nonfinite = dict(valid)
        nonfinite["img"] = valid["img"].clone()
        nonfinite["img"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            NeuroVFMModel._validate_preprocessed_batch(torch, nonfinite)

    def test_cache_roundtrip_reads_disk_once_then_reuses_ram(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.nii.gz"
            raw_path.write_bytes(b"raw-volume")
            _, record = self._build_cache(root / "cache", raw_path)

            class PreprocessorMustNotRun:
                def load_study(self, *_args, **_kwargs):
                    raise AssertionError("raw preprocessing was repeated")

            reader = NeuroVFMModel(
                NeuroVFMOptions(
                    preprocessing_cache_root=root / "cache",
                    require_preprocessing_cache=True,
                    memory_cache=True,
                )
            )
            reader.model = object()
            reader.pipeline = object()
            reader.preprocessor = PreprocessorMustNotRun()
            with patch.object(
                reader,
                "preprocessing_cache_contract",
                return_value=self._contract(),
            ):
                first = reader.load_cached_preprocessing(record)
                second = reader.load_cached_preprocessing(record)

            torch.testing.assert_close(first["img"], second["img"])
            statistics = reader.preprocessing_cache_statistics()
            self.assertEqual(statistics["disk_loads"], 1)
            self.assertEqual(statistics["memory_hits"], 1)
            self.assertEqual(statistics["raw_preprocessing_calls"], 0)
            self.assertEqual(statistics["memory_items"], 1)

    def test_cache_rejects_corrupt_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.nii.gz"
            raw_path.write_bytes(b"raw-volume")
            workflow, record = self._build_cache(root / "cache", raw_path)
            artifact, _ = workflow.preprocessing_cache_paths(record)
            artifact.write_bytes(artifact.read_bytes() + b"corrupt")
            with patch.object(
                workflow,
                "preprocessing_cache_contract",
                return_value=self._contract(),
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    workflow._load_preprocessing_cache(record, retain_in_memory=False)

    def test_cache_rejects_changed_raw_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.nii.gz"
            raw_path.write_bytes(b"raw-volume")
            workflow, record = self._build_cache(root / "cache", raw_path)
            raw_path.write_bytes(b"raw-volume-changed")
            with patch.object(
                workflow,
                "preprocessing_cache_contract",
                return_value=self._contract(),
            ):
                with self.assertRaisesRegex(ValueError, "raw source changed"):
                    workflow._load_preprocessing_cache(record, retain_in_memory=False)

    def test_incomplete_cache_pair_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.nii.gz"
            raw_path.write_bytes(b"raw-volume")
            workflow, record = self._build_cache(root / "cache", raw_path)
            _, sidecar = workflow.preprocessing_cache_paths(record)
            sidecar.unlink()
            with patch.object(
                workflow,
                "preprocessing_cache_contract",
                return_value=self._contract(),
            ):
                result = workflow.build_preprocessing_cache(record)
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(sidecar.is_file())
            self.assertEqual(
                workflow.preprocessing_cache_statistics()["raw_preprocessing_calls"],
                2,
            )

    def test_required_inputs_are_cache_artifact_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root / "raw.nii.gz")
            workflow = NeuroVFMModel(
                NeuroVFMOptions(
                    preprocessing_cache_root=root / "cache",
                    require_preprocessing_cache=True,
                )
            )
            expected = list(preprocessing_cache_paths(root / "cache", record))
            self.assertEqual(workflow.required_input_paths(record), expected)

    def test_cache_paths_are_scoped_by_dataset_and_subject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            raw_path = Path(directory) / "raw.nii.gz"
            first = self._record(raw_path)
            second = {**first, "dataset": "DatasetB"}

            first_artifact, first_sidecar = preprocessing_cache_paths(root, first)
            second_artifact, second_sidecar = preprocessing_cache_paths(root, second)

            self.assertNotEqual(first_artifact, second_artifact)
            self.assertNotEqual(first_sidecar, second_sidecar)
            self.assertEqual(first_artifact.parent, root.resolve() / "DatasetA" / "sub-001")
            self.assertEqual(second_artifact.parent, root.resolve() / "DatasetB" / "sub-001")

    def test_cache_paths_reject_unsafe_manifest_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            raw_path = Path(directory) / "raw.nii.gz"
            valid = self._record(raw_path)
            unsafe_records = (
                {**valid, "dataset": "../escape"},
                {**valid, "subject": "nested/sub-001"},
                {**valid, "subject": ".."},
            )

            for record in unsafe_records:
                with self.subTest(record=record):
                    with self.assertRaisesRegex(ValueError, "Unsafe .* value"):
                        preprocessing_cache_paths(root, record)

    def test_cache_parser_allows_label_free_internal_test_materialization(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["extract_embeddings.py", "cache", "--splits", "internal_test"],
        ):
            args = parse_embedding_args()
        self.assertEqual(args.splits, ["internal_test"])
        self.assertFalse(args.acknowledge_locked_test)

    def test_cache_requires_internal_test_acknowledgement(self) -> None:
        with self.assertRaisesRegex(ValueError, "acknowledge-locked-test"):
            cache_embeddings(
                Namespace(splits=["internal_test"], acknowledge_locked_test=False)
            )

    def test_cache_command_publishes_ledger_and_summary(self) -> None:
        class FakeModel:
            @staticmethod
            def input_path(record) -> Path:
                return Path(str(record["raw_t1w_path"]))

            @staticmethod
            def required_input_paths(record):
                return [Path(str(record["raw_t1w_path"]))]

            @staticmethod
            def load_preprocessor() -> None:
                return None

            @staticmethod
            def preprocessing_cache_contract() -> dict[str, str]:
                return {"contract_sha256": "fixed-contract"}

            @staticmethod
            def release_provenance() -> dict[str, str]:
                return {"revision": "fixed-release"}

            @staticmethod
            def build_preprocessing_cache(_record, *, overwrite=False):
                self.assertFalse(overwrite)
                return {
                    "status": "SKIPPED",
                    "artifact": "cached.pt",
                    "artifact_sha256": "a" * 64,
                    "artifact_size_bytes": 128,
                    "n_tokens": 3,
                }

            @staticmethod
            def preprocessing_cache_statistics() -> dict[str, int]:
                return {"disk_loads": 0, "memory_hits": 0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.nii.gz"
            raw_path.write_bytes(b"raw")
            frame = pd.DataFrame([self._record(raw_path)])
            args = Namespace(
                splits=["train"],
                acknowledge_locked_test=False,
                dataset=None,
                subject=None,
                manifest=root / "manifest.tsv",
                cache_root=root / "cache",
                overwrite=False,
            )
            with (
                patch.object(
                    extract_embeddings_module,
                    "model_from_args",
                    return_value=FakeModel(),
                ),
                patch.object(
                    extract_embeddings_module,
                    "load_and_validate_manifest",
                    return_value=(frame, {"manifest_sha256": "fixed-manifest"}),
                ),
                patch.object(
                    extract_embeddings_module,
                    "runtime_code_hashes",
                    return_value={"extract.py": "fixed-code"},
                ),
                patch.object(
                    extract_embeddings_module,
                    "package_versions",
                    return_value={"python": "test"},
                ),
            ):
                cache_embeddings(args)

            manifests = list((root / "cache" / "_runs").glob("cache_run_*.tsv"))
            summaries = list((root / "cache" / "_runs").glob("cache_run_*.json"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(summaries), 1)
            ledger = pd.read_csv(manifests[0], sep="\t")
            self.assertEqual(ledger.loc[0, "status"], "SKIPPED")
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["status_counts"], {"SKIPPED": 1})


class MetricsAndRidgeTests(unittest.TestCase):
    @staticmethod
    def _embedding_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dataset": ["A", "A", "B"],
                "subject": ["s1", "s2", "s3"],
                "split": ["train", "train", "train"],
                "age": [20.0, 40.0, 60.0],
                "sex": ["F", "M", "F"],
                "input_path": ["a", "b", "c"],
                "model": ["NeuroVFM"] * 3,
                "variant": ["canonical"] * 3,
                "n_tokens": [10, 12, 11],
                "emb_0": [0.0, 1.0, 2.0],
                "emb_1": [2.0, 1.0, 0.0],
            }
        )

    def test_embedding_validation_allows_token_count_metadata(self) -> None:
        frame = self._embedding_frame()
        self.assertEqual(
            validate_embedding_frame(
                frame, expected_model="NeuroVFM", expected_variant="canonical"
            ),
            ["emb_0", "emb_1"],
        )
        self.assertEqual(embedding_columns(frame), ["emb_0", "emb_1"])

    def test_metrics_are_uncorrected(self) -> None:
        metrics = regression_metrics([20, 40], [22, 37])
        self.assertEqual(metrics["mae_years"], 2.5)
        self.assertEqual(metrics["mean_error_years"], -0.5)

    def test_ridge_roundtrip(self) -> None:
        train_x = np.arange(24, dtype=float).reshape(8, 3)
        train_age = np.linspace(20, 70, 8)
        validation_x = np.arange(12, dtype=float).reshape(4, 3) + 0.5
        validation_age = np.linspace(25, 65, 4)
        artifact, table, prediction = fit_ridge_grid(
            train_x, train_age, validation_x, validation_age, [0.1, 1.0]
        )
        self.assertEqual(set(table["alpha"]), {0.1, 1.0})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            atomic_npz(artifact, path)
            observed = predict_ridge(load_npz(path), validation_x)
        np.testing.assert_allclose(observed, prediction, rtol=0, atol=1e-12)

    def test_alpha_grid_validation(self) -> None:
        self.assertEqual(parse_ridge_alphas("0.1,1,10"), [0.1, 1.0, 10.0])
        with self.assertRaises(ValueError):
            parse_ridge_alphas("0,1")


class ProtocolTests(unittest.TestCase):
    @staticmethod
    def _frozen_evaluation_fixture(root: Path) -> tuple[Namespace, dict[str, object]]:
        run_dir = root / "frozen_run"
        run_dir.mkdir()
        manifest_path = root / "manifest.tsv"
        manifest_path.write_text(
            "dataset\tsubject\tsplit\nCamCAN\tsub-test\tinternal_test\n",
            encoding="utf-8",
        )
        model_path = run_dir / fit_ridge_module.MODEL_FILENAME
        model_path.write_bytes(b"frozen-ridge-artifact")
        lock: dict[str, object] = {
            "status": "LOCKED_AFTER_TRAIN_VALIDATION",
            "arm": "frozen",
            "model": "NeuroVFM",
            "variant": "canonical",
            "model_artifact_sha256": sha256_file(model_path),
            "manifest": {"manifest_sha256": sha256_file(manifest_path)},
            "fit_code_sha256": {"fit.py": "locked"},
            "embedding_dim": 768,
            "extraction_release": {"revision": "fixed"},
            "extraction_code_sha256": {"extract.py": "fixed"},
            "extraction_adapter": {"readout": "fixed"},
            "internal_test_accessed": False,
            "on_harmony_accessed": False,
        }
        atomic_json(lock, run_dir / fit_ridge_module.LOCK_NAME)
        args = Namespace(
            acknowledge_locked_test=True,
            run_dir=run_dir,
            manifest=manifest_path,
            test_embeddings=root / "test_embeddings.tsv",
            output_dir=root / "evaluation",
        )
        return args, lock

    @staticmethod
    def _manifest_summary(args: Namespace) -> dict[str, str]:
        return {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        }

    def test_ridge_code_manifest_includes_the_model_contract(self) -> None:
        key = "FineTuning/NeuroVFM/model.py"
        self.assertEqual(
            fit_ridge_module.fit_code_hashes()[key],
            sha256_file(PROJECT_ROOT / key),
        )

    def test_smoke_forward_rejects_internal_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "locked internal-test"):
            audit_embeddings(Namespace(split="internal_test", smoke_forward=True))

    def test_embedding_extraction_requires_internal_test_acknowledgement(self) -> None:
        with self.assertRaisesRegex(ValueError, "acknowledge-locked-test"):
            extract_embeddings(
                Namespace(split="internal_test", acknowledge_locked_test=False)
            )

    def test_frozen_evaluation_requires_a_validation_only_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "frozen_linear_model.npz"
            model_path.write_bytes(b"locked-ridge")
            lock = {
                "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                "arm": "frozen",
                "model": "NeuroVFM",
                "variant": "canonical",
                "model_artifact_sha256": sha256_file(model_path),
                "internal_test_accessed": False,
                "on_harmony_accessed": False,
            }
            fit_ridge_module.validate_evaluation_lock(
                lock, model_path, variant="canonical"
            )
            for update in (
                {"status": "INCOMPLETE"},
                {"internal_test_accessed": True},
                {"on_harmony_accessed": True},
            ):
                with self.assertRaisesRegex(ValueError, "validation-only locked"):
                    fit_ridge_module.validate_evaluation_lock(
                        {**lock, **update}, model_path, variant="canonical"
                    )

            for update in (
                {"arm": "adapted"},
                {"model": "Other"},
                {"variant": "other"},
            ):
                with self.assertRaisesRegex(ValueError, "not the expected frozen"):
                    fit_ridge_module.validate_evaluation_lock(
                        {**lock, **update}, model_path, variant="canonical"
                    )

            model_path.write_bytes(b"tampered-ridge")
            with self.assertRaisesRegex(ValueError, "Frozen model hash differs"):
                fit_ridge_module.validate_evaluation_lock(
                    lock, model_path, variant="canonical"
                )

    def test_fixed_protocol_values(self) -> None:
        self.assertEqual(fit_ridge_module.PIPELINE_VERSION, "neurovfm_brain_age_v2")
        self.assertEqual(len(FROZEN_RIDGE_ALPHAS), 7)
        self.assertEqual(tuple(ADAPTED_LR_CONFIGS), ("conservative", "intermediate", "high"))
        self.assertEqual(ADAPTED_FIXED_HYPERPARAMETERS["epochs"], 30)
        self.assertEqual(adapted_learning_rates("conservative"), (1e-6, 1e-4))

    def test_evaluation_rejects_runtime_drift(self) -> None:
        locked = {
            "amp": "bfloat16",
            "seed": 0,
            "model_source": NEUROVFM_MODEL_REPO,
            "model_revision": NEUROVFM_MODEL_REVISION,
            "cache_root": "/tmp/cache",
        }
        checkpoint = {
            "run_config": {"arguments": locked.copy()}
        }
        matching = Namespace(**{**locked, "cache_root": Path(locked["cache_root"])})
        validate_evaluation_options(matching, checkpoint)

        for key, drifted_value in (
            ("amp", "float16"),
            ("seed", 1),
            ("model_source", "different/model"),
            ("model_revision", "moving"),
        ):
            with self.subTest(key=key):
                requested = {**locked, "cache_root": Path(locked["cache_root"])}
                requested[key] = drifted_value
                option = key.replace("_", "-")
                with self.assertRaisesRegex(ValueError, f"--{option} differs"):
                    validate_evaluation_options(Namespace(**requested), checkpoint)

        cache_drift = Namespace(
            **{**locked, "cache_root": Path("/tmp/other-cache")}
        )
        with self.assertRaisesRegex(ValueError, "cache-root"):
            validate_evaluation_options(cache_drift, checkpoint)

    def test_lr_selection_rejects_nonzero_seed_before_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for index, seed in enumerate((0, 0, 1)):
                run = root / f"run-{index}"
                run.mkdir()
                atomic_json(
                    {"run_config": {"arguments": {"seed": seed}}},
                    run / "LOCKED.json",
                )
                runs.append(run)
            with self.assertRaisesRegex(ValueError, "requires seed 0"):
                validate_selection_candidates(runs)

    def test_frozen_evaluation_rejects_manifest_and_fit_code_drift(self) -> None:
        for drift_kind in ("manifest", "code"):
            with self.subTest(drift=drift_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, _ = self._frozen_evaluation_fixture(root)
                summary = self._manifest_summary(args)
                if drift_kind == "manifest":
                    summary["manifest_sha256"] = "different"
                fit_code = {"fit.py": "different"} if drift_kind == "code" else {"fit.py": "locked"}
                with (
                    patch.object(
                        fit_ridge_module,
                        "load_and_validate_manifest",
                        return_value=(pd.DataFrame(), summary),
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

    def test_frozen_evaluation_reserves_output_before_reading_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._frozen_evaluation_fixture(root)
            with (
                patch.object(
                    fit_ridge_module,
                    "load_and_validate_manifest",
                    return_value=(pd.DataFrame(), self._manifest_summary(args)),
                ),
                patch.object(
                    fit_ridge_module,
                    "fit_code_hashes",
                    return_value={"fit.py": "locked"},
                ),
                patch.object(
                    fit_ridge_module,
                    "prepare_internal_test_output_dir",
                    side_effect=RuntimeError("output reservation reached"),
                ),
                patch.object(fit_ridge_module, "read_embedding_table") as read_embeddings,
            ):
                with self.assertRaisesRegex(RuntimeError, "output reservation reached"):
                    fit_ridge_module.command_evaluate(args)
                read_embeddings.assert_not_called()

    def test_frozen_evaluation_rejects_extraction_adapter_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._frozen_evaluation_fixture(root)
            args.test_embeddings.write_text("placeholder\n", encoding="utf-8")
            fit_ridge_module.embedding_sidecar_path(args.test_embeddings).write_text(
                "{}\n", encoding="utf-8"
            )
            test_manifest = pd.DataFrame({"split": pd.Series(dtype=str)})
            sidecar = {
                "release": {"revision": "fixed"},
                "code_sha256": {"extract.py": "fixed"},
                "adapter": {"readout": "different"},
            }
            with (
                patch.object(
                    fit_ridge_module,
                    "load_and_validate_manifest",
                    return_value=(test_manifest, self._manifest_summary(args)),
                ),
                patch.object(
                    fit_ridge_module,
                    "fit_code_hashes",
                    return_value={"fit.py": "locked"},
                ),
                patch.object(
                    fit_ridge_module,
                    "read_embedding_table",
                    return_value=(pd.DataFrame(), ["emb_0000"], sidecar),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "extraction/input contract"):
                    fit_ridge_module.command_evaluate(args)

    def test_lr_selection_uses_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for name, value in (("conservative", 6.0), ("intermediate", 5.0), ("high", 7.0)):
                run = root / name
                run.mkdir()
                lr_encoder, lr_head = adapted_learning_rates(name)
                lock = {
                    "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                    "arm": "adapted",
                    "model": "NeuroVFM",
                    "variant": "canonical",
                    "checkpoint": "selected_model.pt",
                    "checkpoint_sha256": "fake",
                    "selection_value": value,
                    "manifest": {"manifest_sha256": "manifest"},
                    "release": {"revision": "fixed"},
                    "code_sha256": {"code.py": "hash"},
                    "run_config": {
                        "arguments": {
                            "lr_config": name,
                            "lr_encoder": lr_encoder,
                            "lr_head": lr_head,
                            "seed": 0,
                            "amp": "bfloat16",
                            "model_source": NEUROVFM_MODEL_REPO,
                            "model_revision": NEUROVFM_MODEL_REVISION,
                            **ADAPTED_FIXED_HYPERPARAMETERS,
                        },
                        "internal_test_used_for_training_or_selection": False,
                        "on_harmony_accessed": False,
                    },
                    "selected_epoch": 2,
                    "locked_validation_metrics": {
                        "mae_years": value,
                        "rmse_years": value + 0.5,
                        "r2": 0.5,
                    },
                    "training_elapsed_seconds": 10.0,
                    "internal_test_accessed": False,
                    "on_harmony_accessed": False,
                }
                atomic_json(lock, run / "LOCKED.json")
                (run / "selected_model.pt").write_bytes(b"checkpoint")
                lock["checkpoint_sha256"] = sha256_file(run / "selected_model.pt")
                atomic_json(lock, run / "LOCKED.json")
                atomic_tsv(
                    pd.DataFrame(
                        {
                            "epoch": [1, 2],
                            "train_mae_years_online": [10.0, 8.0],
                            "validation_mae_years": [value + 1, value],
                            "selected": [False, True],
                        }
                    ),
                    run / "train_history.tsv",
                )
                runs.append(run)
            result = select_adapted_learning_rate(
                runs,
                root / "selection",
                expected_model="NeuroVFM",
                expected_variant="canonical",
                manifest_sha256="manifest",
            )
            self.assertEqual(result["selected_lr_config"], "intermediate")
            selected_run = root / "intermediate"
            selected_lock = json.loads(
                (selected_run / "LOCKED.json").read_text(encoding="utf-8")
            )
            provenance = validate_selected_evaluation_run(
                selection_dir=root / "selection",
                run_dir=selected_run,
                lock=selected_lock,
                manifest_sha256="manifest",
                expected_model="NeuroVFM",
                evaluation_variant="canonical",
            )
            self.assertEqual(provenance["selected_lr_config"], "intermediate")

            wrong_run = root / "high"
            wrong_lock = json.loads(
                (wrong_run / "LOCKED.json").read_text(encoding="utf-8")
            )
            with self.assertRaisesRegex(ValueError, "validation-selected LR"):
                validate_selected_evaluation_run(
                    selection_dir=root / "selection",
                    run_dir=wrong_run,
                    lock=wrong_lock,
                    manifest_sha256="manifest",
                    expected_model="NeuroVFM",
                    evaluation_variant="canonical",
                )


if __name__ == "__main__":
    unittest.main()
