#!/usr/bin/env python3
"""Dependency-light tests for the AnatCL brain-age workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from Emb_extractors.extract_AnatCL import (
    _validate_descriptor_checkpoint_dir,
    validate_preprocessing_provenance,
)
from FineTuning.AnatCL.finetune import (
    validate_evaluation_options,
    validate_selection_candidates,
)
from FineTuning.AnatCL import extract_embeddings as extract_embeddings_module
from FineTuning.AnatCL import fit_ridge as fit_ridge_module
from FineTuning.AnatCL.extract_embeddings import (
    INTERNAL_TEST_REVISION_ID,
    command_audit as audit_embeddings,
    command_extract as extract_embeddings,
    resolve_internal_test_receipt,
    verify_internal_test_revision,
)
from FineTuning.AnatCL.fit_ridge import (
    extraction_contract,
    validate_evaluation_lock,
    validate_onharmony_precision_contract,
)
from FineTuning.AnatCL.predict_onharmony import (
    artifact_paths as onharmony_artifact_paths,
    validate_artifact_set as validate_onharmony_artifact_set,
)
from FineTuning.AnatCL.model import (
    AnatCLModel,
    AnatCLOptions,
    CROP_Y,
    DEFAULT_MANIFEST,
    MODEL_INPUT_SHAPE,
    PREPROCESSING_CONTRACT_SHA256,
    PREPROCESSING_PIPELINE_VERSION,
    SHARED_INPUT_SHAPE,
    _checkpoint_state_dict,
    load_and_validate_manifest,
    make_anatcl_input,
    variant_for_fold,
)
from FineTuning.shared.cohort import frame_for_splits, sha256_file
from FineTuning.shared.evaluation import (
    validate_evaluation_runtime_code,
    validate_selected_evaluation_run,
)
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
from preprocessings.OnHarmonyDataset.audit_AnatCL_provenance import (
    inspect_batch,
    inspect_catlog,
)
from preprocessings.OnHarmonyDataset.preprocessing_AnatCL import DEFAULT_TEMPLATE


class CohortTests(unittest.TestCase):
    def test_authoritative_manifest_and_offline_contract(self) -> None:
        frame, summary = load_and_validate_manifest(DEFAULT_MANIFEST)
        self.assertEqual(len(frame), 1411)
        self.assertEqual(
            summary["split_counts"],
            {"train": 1130, "validation": 141, "internal_test": 140},
        )
        self.assertEqual(summary["input_contract"], {"required_offline_group": "cat12_ixi555_gm"})
        self.assertEqual(summary["on_harmony_firewall"], "PASS")

    def test_firewall_rejects_on_harmony_alias(self) -> None:
        frame = pd.read_csv(DEFAULT_MANIFEST, sep="\t").copy()
        frame.loc[0, "dataset"] = "ON-Harmony"
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
        self.assertEqual(selected.iloc[0]["subject"], frame.iloc[0]["subject"])

    def test_release_descriptor_guard_accepts_pathlike_default(self) -> None:
        self.assertIsNone(
            _validate_descriptor_checkpoint_dir(
                Path("Models/AnatCL/NEW_anatcl-g3"), "global"
            )
        )
        with self.assertRaisesRegex(AssertionError, "mismatched descriptor"):
            _validate_descriptor_checkpoint_dir(Path("checkpoints/local"), "global")

    def test_cat12_failure_is_absent_from_common_manifest(self) -> None:
        frame, _ = load_and_validate_manifest()
        self.assertEqual(len(frame_for_splits(frame, ["internal_test"])), 140)
        self.assertFalse(
            ((frame["dataset"] == "CamCAN") & (frame["subject"] == "sub-CC620821")).any()
        )


class OnHarmonyPreprocessingProvenanceTests(unittest.TestCase):
    def test_default_template_is_the_repository_ixi555_template(self) -> None:
        self.assertEqual(DEFAULT_TEMPLATE.name, "Template_1_IXI555_MNI152.nii")
        self.assertEqual(DEFAULT_TEMPLATE.parent.name, "templates")
        self.assertTrue(DEFAULT_TEMPLATE.is_file())

    def test_historical_batch_and_catlog_require_dartel_evidence(self) -> None:
        batch = "\n".join(
            (
                "spm.tools.cat.estwrite.opts.affreg = 'mni';",
                "spm.tools.cat.estwrite.extopts.registration.dartel.darteltpm = {'/old/Template_1_IXI555_MNI152.nii'};",
                "spm.tools.cat.estwrite.extopts.vox = 1.5;",
                "spm.tools.cat.estwrite.output.GM.mod = 1;",
            )
        )
        template, failures = inspect_batch(batch)
        self.assertEqual(template, "/old/Template_1_IXI555_MNI152.nii")
        self.assertEqual(failures, [])
        dartel, version, failures = inspect_catlog(
            "CAT12.7 r1742\nDartel registration\n"
        )
        self.assertEqual(dartel, "Dartel registration")
        self.assertEqual(version, "CAT12.7 r1742")
        self.assertEqual(failures, [])

    def test_adapted_prediction_consumes_batch_and_runtime_log(self) -> None:
        acquisition = "sub-001_ses-A_T1w"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = onharmony_artifact_paths(
                {"acquisition_id": acquisition}, root / "preproc", root / "logs"
            )
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            paths["input"].write_bytes(b"nifti")
            paths["cat12_batch"].write_text(
                "\n".join(
                    (
                        "spm.tools.cat.estwrite.opts.affreg = 'mni';",
                        "spm.tools.cat.estwrite.extopts.registration.dartel.darteltpm = {'template'};",
                        "spm.tools.cat.estwrite.extopts.vox = 1.5;",
                        "spm.tools.cat.estwrite.output.GM.mod = 1;",
                    )
                ),
                encoding="utf-8",
            )
            paths["cat12_runtime_log"].write_text(
                "CAT12.7 r1742\nDartel registration\n", encoding="utf-8"
            )
            paths["preprocessing_log"].write_text(
                "\n".join(
                    (
                        "AnatCL preprocessing (IXI Dartel, vox=1.5)",
                        "Saving CAT12 output (native shape, no resample)",
                        "Shape: (121, 145, 121)",
                        "Zooms: (1.5, 1.5, 1.5)",
                        f"Done: /derived/{acquisition}/{paths['input'].name}",
                        "All done.",
                    )
                ),
                encoding="utf-8",
            )
            details = validate_onharmony_artifact_set(
                {"acquisition_id": acquisition}, paths
            )
            self.assertEqual(details["preprocessing_status"], "PASS")
            paths["cat12_runtime_log"].write_text(
                "CAT12.7 r1742\nShooting registration\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Dartel registration"):
                validate_onharmony_artifact_set(
                    {"acquisition_id": acquisition}, paths
                )

    def test_frozen_extractor_binds_each_input_to_case_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            rows = []
            for index in range(2):
                acquisition = f"sub-00{index + 1}_ses-A_T1w"
                path = root / f"{acquisition}_anatcl_ready.nii.gz"
                path.write_bytes(f"volume-{index}".encode("ascii"))
                inputs.append(str(path))
                rows.append(
                    {
                        "acquisition_id": acquisition,
                        "output_path": f"preproc_AnatCL/{acquisition}/{path.name}",
                        "output_sha256": sha256_file(path),
                        "status": "PASS",
                    }
                )
            table = root / "cases.tsv"
            pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
            manifest = root / "provenance.json"
            auditor = (
                Path(__file__).resolve().parents[3]
                / "preprocessings"
                / "OnHarmonyDataset"
                / "audit_AnatCL_provenance.py"
            )
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "anatcl_onharmony_retrospective_provenance_v1",
                        "status": "PASS",
                        "auditor": {
                            "path": str(auditor),
                            "sha256": sha256_file(auditor),
                        },
                        "template_reference": {"identity_matches_expected": True},
                        "evidence": {
                            "case_table": table.name,
                            "case_table_sha256": sha256_file(table),
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = validate_preprocessing_provenance(manifest, inputs)
            self.assertEqual(result["n_acquisitions"], 2)
            self.assertEqual(result["status"], "VALIDATED")

            Path(inputs[0]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs from historical"):
                validate_preprocessing_provenance(manifest, inputs)


class ModelContractTests(unittest.TestCase):
    def test_fold_labels_and_validation(self) -> None:
        self.assertEqual([variant_for_fold(value) for value in range(5)], [f"fold{x}" for x in range(5)])
        with self.assertRaisesRegex(ValueError, "0..4"):
            AnatCLOptions(fold=5).validate()

    def test_crop_is_exact_and_does_not_normalize(self) -> None:
        data = np.arange(np.prod(SHARED_INPUT_SHAPE), dtype=np.float32).reshape(SHARED_INPUT_SHAPE)
        observed = make_anatcl_input(data)
        expected = data[:, CROP_Y[0] : CROP_Y[1], :][np.newaxis, ...]
        self.assertEqual(observed.shape, MODEL_INPUT_SHAPE)
        np.testing.assert_array_equal(observed, expected)
        self.assertNotAlmostEqual(float(observed.mean()), 0.0)
        self.assertNotAlmostEqual(float(observed.std()), 1.0)

    def test_crop_rejects_wrong_shape_or_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be"):
            make_anatcl_input(np.zeros((121, 128, 121), dtype=np.float32))
        nonfinite = np.ones(SHARED_INPUT_SHAPE, dtype=np.float32)
        nonfinite.flat[0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            make_anatcl_input(nonfinite)
        invalid = np.ones(SHARED_INPUT_SHAPE, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "zero variance"):
            make_anatcl_input(invalid)

    def test_checkpoint_layouts_are_explicit(self) -> None:
        state = {"encoder.weight": object()}
        self.assertIs(_checkpoint_state_dict({"model": state}), state)

        class Bundle:
            def state_dict(self):
                return state

        self.assertIs(_checkpoint_state_dict(Bundle()), state)
        with self.assertRaisesRegex(ValueError, "Unexpected AnatCL checkpoint layout"):
            _checkpoint_state_dict({"weights": state})

    def test_qc_provenance_binds_subject_contract_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            subject_dir = root / "CamCAN" / "sub-test"
            subject_dir.mkdir(parents=True)
            image = subject_dir / "sub-test_anatcl_ready.nii.gz"
            image.write_bytes(b"fixed-input")
            qc = {
                "status": "PASS",
                "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
                "pipeline_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
                "dataset": "CamCAN",
                "subject": "sub-test",
                "split": "train",
                "anatcl_output_sha256": sha256_file(image),
                "runtime_views": {
                    "anatcl": {
                        "shape": list(MODEL_INPUT_SHAPE),
                        "crop_y": list(CROP_Y),
                        "normalization": "none",
                    }
                },
            }
            (subject_dir / "preprocessing_qc.json").write_text(json.dumps(qc), encoding="utf-8")
            model = AnatCLModel(AnatCLOptions(fold=0, preproc_root=root, checkpoint_dir=root))
            record = {"dataset": "CamCAN", "subject": "sub-test", "split": "train"}
            provenance = model.validate_input_provenance(record)
            self.assertEqual(provenance["input_sha256"], sha256_file(image))
            image.write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                model.validate_input_provenance(record)

    def test_only_layer4_is_trainable(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                encoder = torch.nn.Module()
                encoder.layer1 = torch.nn.Sequential(torch.nn.Linear(2, 2))
                encoder.layer4 = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
                backbone = torch.nn.Module()
                backbone.encoder = encoder
                backbone.head = torch.nn.Linear(2, 1)
                self.backbone = backbone

        adapter = AnatCLModel(AnatCLOptions(fold=0, checkpoint_dir=Path("."), preproc_root=Path(".")))
        adapter.model = DummyModel()
        adapter.device = torch.device("cpu")
        expected_trainable = {
            name
            for name, _ in adapter.model.named_parameters()
            if name.startswith("backbone.encoder.layer4.")
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
        self.assertEqual(summary["trainable_components"], ["backbone.encoder.layer4"])


class MetricsAndEmbeddingTests(unittest.TestCase):
    def test_regression_metrics_are_uncorrected(self) -> None:
        metrics = regression_metrics([20, 30, 40], [22, 29, 43])
        self.assertAlmostEqual(metrics["mae_years"], 2.0)
        self.assertAlmostEqual(metrics["rmse_years"], np.sqrt(14 / 3))
        self.assertEqual(metrics["n"], 3)

    def test_embedding_columns_are_contiguous(self) -> None:
        self.assertEqual(embedding_columns(pd.DataFrame({"emb_0": [1], "emb_1": [2]})), ["emb_0", "emb_1"])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            embedding_columns(pd.DataFrame({"emb_0": [1], "emb_2": [2]}))

    def test_embedding_rows_use_fold_variant(self) -> None:
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
                "model": ["AnatCL", "AnatCL"],
                "variant": ["fold3", "fold3"],
                "emb_0": [1.0, 2.0],
            }
        )
        self.assertEqual(
            validate_embedding_frame(
                table,
                expected_manifest=expected,
                expected_model="AnatCL",
                expected_variant="fold3",
            ),
            ["emb_0"],
        )


class RidgeAndProtocolTests(unittest.TestCase):
    def test_fixed_protocol_values(self) -> None:
        self.assertEqual(FROZEN_RIDGE_ALPHAS, (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))
        self.assertEqual(adapted_learning_rates("conservative"), (1e-6, 1e-4))
        self.assertEqual(adapted_learning_rates("intermediate"), (3e-6, 3e-4))
        self.assertEqual(adapted_learning_rates("high"), (1e-5, 1e-3))

    def test_extraction_contract_is_relocatable_but_amp_locked(self) -> None:
        sidecar = {
            "release": {"checkpoint": "/old/fold0.pth", "checkpoint_sha256": "weights"},
            "adapter": {
                "checkpoint_dir": "/old/checkpoints",
                "input_root": "/old/inputs",
                "model": "AnatCL",
            },
            "code_sha256": {"model.py": "code"},
            "runtime": {"device": "cuda:0", "amp": "bfloat16", "seed": 42, "fold": 0},
            "checkpoint_load": {
                "strict": True,
                "backbone_keys": 122,
                "probe_key": "encoder.conv1.weight",
                "probe_changed_l2": 1.0,
                "probe_bit_exact": True,
                "projection_head_used": False,
                "checkpoint_sha256": "weights",
                "installed_source_sha256": {"resnet3d.py": "source"},
            },
        }
        expected = extraction_contract(sidecar)
        relocated = json.loads(json.dumps(sidecar))
        relocated["release"]["checkpoint"] = "/new/fold0.pth"
        relocated["adapter"]["checkpoint_dir"] = "/new/checkpoints"
        relocated["adapter"]["input_root"] = "/new/inputs"
        relocated["runtime"]["device"] = "cuda:3"
        relocated["runtime"]["seed"] = 999
        relocated["checkpoint_load"]["probe_changed_l2"] = 2.0
        self.assertEqual(extraction_contract(relocated), expected)
        relocated["runtime"]["amp"] = "off"
        with self.assertRaisesRegex(ValueError, "must use.*bfloat16"):
            extraction_contract(relocated)

    def test_onharmony_precision_is_bound_to_manifest_and_ridge_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            embeddings.write_text("file_path,emb_0\n", encoding="utf-8")
            manifest = root / "manifest.json"
            lock_dir = root / "ridge"
            lock_dir.mkdir()
            lock = lock_dir / "LOCKED.json"
            args = Namespace(
                embeddings=embeddings,
                embedding_manifest=manifest,
                run_dir=lock_dir,
            )

            manifest.write_text(json.dumps({"amp": "off"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must declare.*bfloat16"):
                validate_onharmony_precision_contract(
                    args, require_locked_ridge=False
                )

            manifest.write_text(
                json.dumps({"amp": "bfloat16"}), encoding="utf-8"
            )
            lock.write_text(
                json.dumps({"extraction_contract": {"amp": "off"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "differs from the locked"):
                validate_onharmony_precision_contract(
                    args, require_locked_ridge=True
                )

            lock.write_text(
                json.dumps({"extraction_contract": {"amp": "bfloat16"}}),
                encoding="utf-8",
            )
            validate_onharmony_precision_contract(args, require_locked_ridge=True)

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

    def test_lr_selection_is_fold_specific_and_validation_only(self) -> None:
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
                    "fold": 0,
                    "checkpoint_dir": "/fixed/checkpoints",
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
                        "model": "AnatCL",
                        "variant": "fold0",
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
                expected_model="AnatCL",
                expected_variant="fold0",
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
                expected_model="AnatCL",
                evaluation_variant="fold0",
                selection_variant="fold0",
            )
            self.assertFalse(provenance["cross_variant_lr_transfer"])

            fold1_run = root / "fold1_intermediate"
            fold1_run.mkdir()
            fold1_checkpoint = fold1_run / "selected_model.pt"
            fold1_checkpoint.write_bytes(b"fold1-intermediate")
            fold1_lock = json.loads(json.dumps(selected_lock))
            fold1_lock["variant"] = "fold1"
            fold1_lock["checkpoint_sha256"] = sha256_file(fold1_checkpoint)
            fold1_lock["run_config"]["arguments"]["fold"] = 1
            atomic_json(fold1_lock, fold1_run / "LOCKED.json")
            transferred = validate_selected_evaluation_run(
                selection_dir=output,
                run_dir=fold1_run,
                lock=fold1_lock,
                manifest_sha256="manifest-test",
                expected_model="AnatCL",
                evaluation_variant="fold1",
                selection_variant="fold0",
            )
            self.assertTrue(transferred["cross_variant_lr_transfer"])

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
                    expected_model="AnatCL",
                    evaluation_variant="fold0",
                    selection_variant="fold0",
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
    def _frozen_evaluation_fixture(root: Path) -> tuple[Namespace, dict[str, object]]:
        run_dir = root / "frozen_run"
        run_dir.mkdir()
        model_path = run_dir / fit_ridge_module.MODEL_FILENAME
        model_path.write_bytes(b"frozen-ridge-artifact")
        lock: dict[str, object] = {
            "status": "LOCKED_AFTER_TRAIN_VALIDATION",
            "arm": "frozen",
            "model": "AnatCL",
            "variant": "fold2",
            "model_artifact_sha256": sha256_file(model_path),
            "manifest": {"manifest_sha256": "manifest"},
            "fit_code_sha256": {"fit.py": "locked"},
            "embedding_dim": 512,
            "extraction_contract": {"release": "fixed"},
            "internal_test_accessed": False,
            "on_harmony_accessed": False,
        }
        atomic_json(lock, run_dir / fit_ridge_module.LOCK_NAME)
        args = Namespace(
            fold=2,
            acknowledge_locked_test=True,
            run_dir=run_dir,
            manifest=root / "manifest.tsv",
            test_embeddings=root / "test_embeddings.tsv",
            output_dir=root / "evaluation",
        )
        return args, lock

    def test_smoke_forward_rejects_internal_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "locked internal-test"):
            audit_embeddings(Namespace(split="internal_test", smoke_forward=True))

    def test_frozen_extraction_requires_internal_test_acknowledgement(self) -> None:
        with self.assertRaisesRegex(ValueError, "acknowledge-locked-test"):
            extract_embeddings(
                Namespace(split="internal_test", acknowledge_locked_test=False)
            )

    def test_internal_test_revision_arguments_are_split_scoped(self) -> None:
        args = Namespace(
            split="train",
            internal_test_revision=INTERNAL_TEST_REVISION_ID,
            internal_test_revision_lock=Path("LOCKED.json"),
        )
        model = Namespace(model_name="AnatCL", variant="fold2", fold=2)
        with self.assertRaisesRegex(ValueError, "forbidden outside internal_test"):
            resolve_internal_test_receipt(
                args,
                model,
                Path("train.tsv").resolve(),
                {"manifest_sha256": "a" * 64},
                {"extract.py": "b" * 64},
                Path("AnatCL__fold2.json").resolve(),
            )

    def test_bfloat16_revision_preserves_and_binds_the_original_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            variant = "fold2"
            model = Namespace(
                model_name="AnatCL",
                variant=variant,
                fold=2,
                readout="global-average-pooled encoder output",
                embedding_dim=512,
            )
            manifest_sha256 = "a" * 64
            code_sha256 = {"extract.py": "b" * 64}

            original_receipt = root / "receipts" / "AnatCL__fold2.json"
            original_receipt.parent.mkdir(parents=True)
            original_dir = root / "legacy" / variant
            original_dir.mkdir(parents=True)
            original_artifact = original_dir / "internal_test.tsv"
            original_artifact.write_bytes(b"original-fp32-embeddings")
            original_sidecar = original_dir / "internal_test.tsv.audit.json"
            attempt_id = "1" * 32
            atomic_json(
                {
                    "pipeline_version": extract_embeddings_module.PIPELINE_VERSION,
                    "model": "AnatCL",
                    "variant": variant,
                    "split": "internal_test",
                    "artifact_sha256": sha256_file(original_artifact),
                    "runtime": {"amp": "off"},
                    "manifest": {"manifest_sha256": manifest_sha256},
                    "publication": {"status": "COMMITTED"},
                    "internal_test_receipt": {
                        "path": str(original_receipt),
                        "attempt_id": attempt_id,
                    },
                },
                original_sidecar,
            )
            atomic_json(
                {
                    "status": "INTERNAL_TEST_EMBEDDINGS_EXTRACTED",
                    "operation": "embedding extraction",
                    "attempt_id": attempt_id,
                    "pipeline_version": extract_embeddings_module.PIPELINE_VERSION,
                    "model": "AnatCL",
                    "variant": variant,
                    "split": "internal_test",
                    "manifest_sha256": manifest_sha256,
                    "artifact": str(original_artifact),
                    "artifact_sha256": sha256_file(original_artifact),
                    "sidecar": str(original_sidecar),
                    "sidecar_sha256": sha256_file(original_sidecar),
                },
                original_receipt,
            )

            fold_dir = (
                root
                / "Results"
                / "brain_age"
                / "embeddings_FT_bfloat16_v3"
                / "AnatCL"
                / variant
            )
            fold_dir.mkdir(parents=True)
            split_hashes = {}
            for split in ("train", "validation"):
                artifact = fold_dir / f"{split}.tsv"
                artifact.write_bytes(f"{split}-bf16".encode("ascii"))
                split_hashes[split] = sha256_file(artifact)
                atomic_json(
                    {
                        "pipeline_version": extract_embeddings_module.PIPELINE_VERSION,
                        "model": "AnatCL",
                        "variant": variant,
                        "split": split,
                        "artifact_sha256": split_hashes[split],
                        "code_sha256": code_sha256,
                        "runtime": {"amp": "bfloat16"},
                        "manifest": {"manifest_sha256": manifest_sha256},
                        "publication": {"status": "COMMITTED"},
                    },
                    Path(str(artifact) + ".audit.json"),
                )

            run_dir = (
                root
                / "Results"
                / "brain_age"
                / "frozen_bfloat16_v3"
                / "AnatCL"
                / variant
            )
            run_dir.mkdir(parents=True)
            ridge_model = run_dir / "frozen_linear_model.npz"
            ridge_model.write_bytes(b"ridge-selected-with-validation")
            lock_path = run_dir / "LOCKED.json"
            atomic_json(
                {
                    "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                    "pipeline_version": extract_embeddings_module.PIPELINE_VERSION,
                    "arm": "frozen",
                    "model": "AnatCL",
                    "variant": variant,
                    "fold": 2,
                    "readout": model.readout,
                    "embedding_dim": model.embedding_dim,
                    "manifest": {"manifest_sha256": manifest_sha256},
                    "extraction_contract": {
                        "code_sha256": code_sha256,
                        "amp": "bfloat16",
                        "fold": 2,
                    },
                    "train_embeddings_sha256": split_hashes["train"],
                    "validation_embeddings_sha256": split_hashes["validation"],
                    "model_artifact_sha256": sha256_file(ridge_model),
                    "selected_alpha": 100.0,
                    "internal_test_accessed": False,
                    "on_harmony_accessed": False,
                },
                lock_path,
            )
            output = fold_dir / "internal_test.tsv"
            args = Namespace(
                split="internal_test",
                amp="bfloat16",
                internal_test_revision=INTERNAL_TEST_REVISION_ID,
                internal_test_revision_lock=lock_path,
            )
            with patch.object(extract_embeddings_module, "PROJECT_ROOT", root):
                receipt, context = resolve_internal_test_receipt(
                    args,
                    model,
                    output,
                    {"manifest_sha256": manifest_sha256},
                    code_sha256,
                    original_receipt,
                )
            self.assertIsNotNone(context)
            assert context is not None
            self.assertEqual(context["revision_id"], INTERNAL_TEST_REVISION_ID)
            self.assertEqual(context["previous_amp"], "off")
            self.assertEqual(context["revised_amp"], "bfloat16")
            self.assertEqual(context["selection_lock"], str(lock_path))
            self.assertEqual(
                receipt,
                original_receipt.parent
                / "revisions"
                / INTERNAL_TEST_REVISION_ID
                / original_receipt.name,
            )
            verify_internal_test_revision(context)
            original_artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                verify_internal_test_revision(context)

    def test_identical_runtime_settings_are_accepted(self) -> None:
        args = Namespace(amp="bfloat16", fold=2)
        checkpoint = {"run_config": {"arguments": vars(args).copy()}}
        validate_evaluation_options(args, checkpoint)

    def test_fold_or_amp_drift_is_rejected(self) -> None:
        checkpoint = {"run_config": {"arguments": {"amp": "bfloat16", "fold": 2}}}
        with self.assertRaisesRegex(ValueError, "--amp differs"):
            validate_evaluation_options(Namespace(amp="float16", fold=2), checkpoint)
        with self.assertRaisesRegex(ValueError, "--fold differs"):
            validate_evaluation_options(Namespace(amp="bfloat16", fold=3), checkpoint)

    def test_frozen_evaluation_requires_a_validation_only_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "frozen_linear_model.npz"
            model_path.write_bytes(b"locked-ridge")
            lock = {
                "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                "arm": "frozen",
                "model": "AnatCL",
                "variant": "fold2",
                "model_artifact_sha256": sha256_file(model_path),
                "internal_test_accessed": False,
                "on_harmony_accessed": False,
            }
            validate_evaluation_lock(lock, model_path, variant="fold2")
            for update in (
                {"status": "INCOMPLETE"},
                {"internal_test_accessed": True},
                {"on_harmony_accessed": True},
            ):
                with self.assertRaisesRegex(ValueError, "validation-only locked"):
                    validate_evaluation_lock(
                        {**lock, **update}, model_path, variant="fold2"
                    )

            for update in (
                {"arm": "adapted"},
                {"model": "Other"},
                {"variant": "other"},
            ):
                with self.assertRaisesRegex(ValueError, "not the expected frozen"):
                    validate_evaluation_lock(
                        {**lock, **update}, model_path, variant="fold2"
                    )

            model_path.write_bytes(b"tampered-ridge")
            with self.assertRaisesRegex(ValueError, "Frozen model hash differs"):
                validate_evaluation_lock(lock, model_path, variant="fold2")

    def test_frozen_evaluation_rejects_manifest_and_fit_code_drift(self) -> None:
        for drift_kind in ("manifest", "code"):
            with self.subTest(drift=drift_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, _ = self._frozen_evaluation_fixture(root)
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

    def test_frozen_evaluation_reserves_output_before_reading_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._frozen_evaluation_fixture(root)
            with (
                patch.object(
                    fit_ridge_module,
                    "load_and_validate_manifest",
                    return_value=(pd.DataFrame(), {"manifest_sha256": "manifest"}),
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

    def test_legacy_training_lock_accepts_only_scoped_evaluation_migration(self) -> None:
        script = "FineTuning/AnatCL/finetune.py"
        guard = "FineTuning/shared/evaluation.py"
        locked = {script: "legacy", "FineTuning/shared/cohort.py": "same"}
        current = {script: "current", "FineTuning/shared/cohort.py": "same", guard: "guard"}
        provenance = validate_evaluation_runtime_code(
            locked_code=locked,
            current_code=current,
            script_key=script,
            evaluation_guard_key=guard,
            approved_legacy_script_hashes=("legacy",),
        )
        self.assertEqual(provenance["mode"], "approved_evaluation_guard_migration")
        current["FineTuning/shared/cohort.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "outside the approved"):
            validate_evaluation_runtime_code(
                locked_code=locked,
                current_code=current,
                script_key=script,
                evaluation_guard_key=guard,
                approved_legacy_script_hashes=("legacy",),
            )


if __name__ == "__main__":
    unittest.main()
