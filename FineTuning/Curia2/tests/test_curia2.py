#!/usr/bin/env python3
"""Dependency-light tests for the Curia-2 brain-age workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

import FineTuning.Curia2.fit_ridge as fit_ridge_module
from FineTuning.Curia2.extract_embeddings import (
    command_audit as audit_embeddings,
    command_extract as extract_embeddings,
    parse_args as parse_embedding_args,
)
from FineTuning.Curia2.finetune import (
    FINETUNE_CODE_KEY,
    LEGACY_FINETUNE_SHA256,
    LEGACY_MODEL_SHA256,
    LOADER_MIGRATION_ID,
    MODEL_CODE_KEY,
    STRICT_3D_MODEL_SHA256,
    validate_evaluation_options,
    validate_loader_migration_receipt,
    validate_selection_candidates,
)
from FineTuning.Curia2.fit_ridge import (
    extraction_contract,
    fit_code_hashes,
    matching_extraction_contract,
    validate_evaluation_lock,
)
from FineTuning.Curia2.model import (
    Curia2Model,
    Curia2Options,
    DEFAULT_MANIFEST,
    _load_las_volume,
    las_slice_to_pl,
    load_and_validate_manifest,
)
from FineTuning.shared.cohort import frame_for_splits, sha256_file
from FineTuning.shared.internal_test import internal_test_runtime_migration
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
from FineTuning.shared.onharmony import validate_prediction_runtime_code


def canonical_curia_extraction_sidecar() -> dict:
    """Build the minimal complete sidecar contract used by Ridge tests."""

    return {
        "release": {"checkpoint": "fixed"},
        "code_sha256": {"extract.py": "fixed"},
        "adapter": {"model": "Curia-2", "variant": "canonical"},
        "software": {"python": "3.10.6", "torch": "2.5.0"},
        "runtime": {
            "amp": "off",
            "slice_batch_size": 8,
            "input_hashes_verified": True,
        },
    }


class CohortTests(unittest.TestCase):
    def test_authoritative_manifest(self) -> None:
        frame, summary = load_and_validate_manifest(DEFAULT_MANIFEST)
        self.assertEqual(len(frame), 1411)
        self.assertEqual(
            summary["split_counts"],
            {"train": 1130, "validation": 141, "internal_test": 140},
        )
        self.assertEqual(summary["input_contract"], {"required_runtime_condition": "Curia-2"})
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


class SliceContractTests(unittest.TestCase):
    def test_options_reject_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "slice_batch_size"):
            Curia2Options(slice_batch_size=0).validate()

    def test_las_axial_slice_rotates_to_pl(self) -> None:
        volume = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
        observed = las_slice_to_pl(volume, 0)
        np.testing.assert_array_equal(observed, np.rot90(volume[..., 0], k=1))
        self.assertTrue(observed.flags.c_contiguous)

    def test_loader_rejects_4d_nifti_instead_of_selecting_a_plane(self) -> None:
        import nibabel as nib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "singleton_4d.nii.gz"
            image = nib.Nifti1Image(
                np.zeros((2, 3, 4, 1), dtype=np.float32),
                np.eye(4),
            )
            nib.save(image, path)
            with self.assertRaisesRegex(ValueError, "exactly 3D"):
                _load_las_volume(path)

    def test_only_final_block_and_norm_are_trainable(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class DummyEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2) for _ in range(12)]
                )

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings = torch.nn.Linear(2, 2)
                self.encoder = DummyEncoder()
                self.layernorm = torch.nn.LayerNorm(2)

        adapter = Curia2Model(Curia2Options(model_dir=Path(".")))
        adapter.model = DummyModel()
        adapter.device = torch.device("cpu")
        expected_trainable = {
            name
            for name, _ in adapter.model.named_parameters()
            if name.startswith("encoder.layer.11.") or name.startswith("layernorm.")
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
        self.assertEqual(
            summary["trainable_components"], ["encoder.layer.11", "layernorm"]
        )


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

    def test_embedding_rows_match_manifest(self) -> None:
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
                "model": ["Curia-2", "Curia-2"],
                "variant": ["canonical", "canonical"],
                "emb_0": [1.0, 2.0],
            }
        )
        self.assertEqual(
            validate_embedding_frame(
                table,
                expected_manifest=expected,
                expected_model="Curia-2",
                expected_variant="canonical",
            ),
            ["emb_0"],
        )

        wrong_sex = table.copy()
        wrong_sex.loc[0, "sex"] = "INVALID"
        with self.assertRaisesRegex(ValueError, "sex labels"):
            validate_embedding_frame(
                wrong_sex,
                expected_manifest=expected,
                expected_model="Curia-2",
                expected_variant="canonical",
            )


class RidgeAndProtocolTests(unittest.TestCase):
    def test_ridge_code_manifest_includes_the_model_contract(self) -> None:
        key = "FineTuning/Curia2/model.py"
        self.assertEqual(fit_code_hashes()[key], sha256_file(PROJECT_ROOT / key))

    def test_extraction_contract_locks_numerical_runtime(self) -> None:
        sidecar = canonical_curia_extraction_sidecar()
        observed = extraction_contract(sidecar)
        self.assertEqual(observed["amp"], "off")
        self.assertEqual(observed["slice_batch_size"], 8)
        self.assertEqual(observed["software"], sidecar["software"])
        self.assertTrue(observed["input_hashes_verified"])

        invalid_cases = (
            ({"amp": "bfloat16"}, "require amp=off"),
            ({"slice_batch_size": 0}, "invalid slice_batch_size"),
            ({"input_hashes_verified": False}, "did not verify its input hashes"),
        )
        for runtime_update, message in invalid_cases:
            with self.subTest(runtime_update=runtime_update):
                invalid = canonical_curia_extraction_sidecar()
                invalid["runtime"].update(runtime_update)
                with self.assertRaisesRegex(ValueError, message):
                    extraction_contract(invalid)

        missing_software = canonical_curia_extraction_sidecar()
        missing_software["software"] = {}
        with self.assertRaisesRegex(ValueError, "lacks contract fields"):
            extraction_contract(missing_software)

    def test_train_validation_extraction_contract_must_match(self) -> None:
        train = canonical_curia_extraction_sidecar()
        validation = canonical_curia_extraction_sidecar()
        self.assertEqual(
            matching_extraction_contract(train, validation),
            extraction_contract(train),
        )

        for update in ("runtime", "software"):
            with self.subTest(update=update):
                drifted = canonical_curia_extraction_sidecar()
                if update == "runtime":
                    drifted["runtime"]["slice_batch_size"] = 16
                else:
                    drifted["software"]["torch"] = "different"
                with self.assertRaisesRegex(ValueError, "extraction contracts differ"):
                    matching_extraction_contract(train, drifted)

    def test_fixed_protocol_values(self) -> None:
        self.assertEqual(FROZEN_RIDGE_ALPHAS, (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))
        self.assertEqual(adapted_learning_rates("conservative"), (1e-6, 1e-4))
        self.assertEqual(adapted_learning_rates("intermediate"), (3e-6, 3e-4))
        self.assertEqual(adapted_learning_rates("high"), (1e-5, 1e-3))

    def test_alpha_grid_validation(self) -> None:
        self.assertEqual(parse_ridge_alphas("10,0.1,1"), [0.1, 1.0, 10.0])
        with self.assertRaises(ValueError):
            parse_ridge_alphas("0,1")
        with self.assertRaises(ValueError):
            parse_ridge_alphas("1,1")

    def test_saved_ridge_reproduces_predictions(self) -> None:
        rng = np.random.default_rng(7)
        train_x = rng.normal(size=(80, 6))
        validation_x = rng.normal(size=(20, 6))
        weights = np.asarray([4.0, -2.0, 0.5, 0.0, 1.0, -0.2])
        train_age = 50 + train_x @ weights + rng.normal(scale=0.2, size=80)
        validation_age = 50 + validation_x @ weights + rng.normal(scale=0.2, size=20)
        artifact, candidates, expected = fit_ridge_grid(
            train_x, train_age, validation_x, validation_age, [0.01, 0.1, 1.0]
        )
        self.assertEqual(candidates.iloc[0]["mae_years"], candidates["mae_years"].min())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            atomic_npz(artifact, path)
            observed = predict_ridge(load_npz(path), validation_x)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)

    def test_exact_tie_uses_larger_alpha(self) -> None:
        artifact, _, _ = fit_ridge_grid(
            np.zeros((20, 2)),
            np.linspace(20, 60, 20),
            np.zeros((5, 2)),
            np.linspace(25, 55, 5),
            [0.1, 1.0, 10.0],
        )
        self.assertEqual(float(artifact["alpha"][0]), 10.0)

    def test_lr_selection_uses_validation_only(self) -> None:
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
                    "model_dir": "/fixed/model",
                    "seed": 0,
                    "amp": "bfloat16",
                    "lr_config": config,
                    "lr_encoder": lr_encoder,
                    "lr_head": lr_head,
                    "slice_batch_size": 8,
                    **ADAPTED_FIXED_HYPERPARAMETERS,
                }
                atomic_json(
                    {
                        "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                        "arm": "adapted",
                        "model": "Curia-2",
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
                expected_model="Curia-2",
                expected_variant="canonical",
                manifest_sha256="manifest-test",
            )
            self.assertEqual(result["selected_lr_config"], "intermediate")
            self.assertFalse(result["internal_test_accessed"])

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
                "model": "Curia-2",
                "variant": "canonical",
                "model_artifact_sha256": sha256_file(model_path),
                "manifest": {"manifest_sha256": "manifest"},
                "fit_code_sha256": {"fit.py": "locked"},
                "extraction_contract": extraction_contract(
                    canonical_curia_extraction_sidecar()
                ),
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

    def test_frozen_extraction_is_locked_to_float32(self) -> None:
        for amp in ("float16", "bfloat16"):
            with self.subTest(amp=amp), self.assertRaisesRegex(
                ValueError, "requires --amp off"
            ):
                extract_embeddings(
                    Namespace(
                        split="train",
                        acknowledge_locked_test=False,
                        amp=amp,
                    )
                )

    def test_extract_parser_defaults_to_float32(self) -> None:
        with patch(
            "sys.argv",
            [
                "extract_embeddings.py",
                "extract",
                "--split",
                "train",
                "--output",
                "embeddings.tsv",
            ],
        ):
            args = parse_embedding_args()
        self.assertEqual(args.amp, "off")

    def test_identical_runtime_settings_are_accepted(self) -> None:
        args = Namespace(amp="bfloat16", seed=0, slice_batch_size=8)
        checkpoint = {"run_config": {"arguments": vars(args).copy()}}
        validate_evaluation_options(args, checkpoint)

    def test_runtime_drift_is_rejected(self) -> None:
        locked = {"amp": "bfloat16", "seed": 0, "slice_batch_size": 8}
        checkpoint = {"run_config": {"arguments": locked.copy()}}
        for key, drifted_value in (
            ("amp", "float16"),
            ("seed", 1),
            ("slice_batch_size", 16),
        ):
            with self.subTest(key=key):
                requested = locked.copy()
                requested[key] = drifted_value
                option = key.replace("_", "-")
                with self.assertRaisesRegex(ValueError, f"--{option} differs"):
                    validate_evaluation_options(Namespace(**requested), checkpoint)

    def test_internal_test_runtime_migration_is_exactly_hash_pinned(self) -> None:
        key = "FineTuning/Curia2/finetune.py"
        metrics_key = "FineTuning/shared/metrics.py"
        locked = {
            key: "df8ee990e880a16029b86362eb9f08100638b4033dae0de28e5f290f7baadabc",
            metrics_key: "d7ec28a8f8d4d921efc397da4662af7bec972bb567f7061aee7d1544cd0ebd3e",
            "unchanged.py": "same",
        }
        migration = internal_test_runtime_migration("Curia-2", locked)
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
        with self.assertRaisesRegex(ValueError, "Unapproved runtime hash transition"):
            validate_prediction_runtime_code(
                locked, {**current, key: "different"},
                migration,
            )
        self.assertEqual(
            internal_test_runtime_migration("Curia-2", locked),
            migration,
        )
        self.assertIsNone(
            internal_test_runtime_migration("Curia-2", current)
        )
        with self.assertRaisesRegex(ValueError, "No unique approved Curia-2"):
            internal_test_runtime_migration(
                "Curia-2", {key: "unregistered", "unchanged.py": "same"}
            )
        with self.assertRaisesRegex(ValueError, "lacks a code-hash mapping"):
            internal_test_runtime_migration("Curia-2", {})

    def test_frozen_evaluation_requires_a_validation_only_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "frozen_linear_model.npz"
            model_path.write_bytes(b"locked-ridge")
            lock = {
                "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                "arm": "frozen",
                "model": "Curia-2",
                "variant": "canonical",
                "model_artifact_sha256": sha256_file(model_path),
                "extraction_contract": extraction_contract(
                    canonical_curia_extraction_sidecar()
                ),
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

    def test_audited_3d_loader_migration_preserves_both_code_manifests(self) -> None:
        old_code = {
            FINETUNE_CODE_KEY: LEGACY_FINETUNE_SHA256,
            MODEL_CODE_KEY: LEGACY_MODEL_SHA256,
            "FineTuning/shared/runtime.py": "unchanged",
        }
        new_code = {
            FINETUNE_CODE_KEY: "new-finetune-hash",
            MODEL_CODE_KEY: STRICT_3D_MODEL_SHA256,
            "FineTuning/shared/runtime.py": "unchanged",
        }
        audit = {
            "n_manifest_rows": 1411,
            "n_headers_read": 1411,
            "dimensionality_counts": {"3": 1411},
            "n_non_3d": 0,
            "n_header_errors": 0,
        }
        receipt = {
            "status": "AUDITED_RUNTIME_CODE_MIGRATION",
            "migration_id": LOADER_MIGRATION_ID,
            "source_lock_sha256": "lock-hash",
            "checkpoint_sha256": "checkpoint-hash",
            "manifest_sha256": "manifest-hash",
            "old_code_sha256": old_code,
            "new_code_sha256": new_code,
            "differing_code_keys": [FINETUNE_CODE_KEY, MODEL_CODE_KEY],
            "nifti_dimensionality_audit": audit,
        }
        observed = validate_loader_migration_receipt(
            receipt,
            lock={"code_sha256": old_code},
            lock_sha256="lock-hash",
            current_code=new_code,
            manifest_sha256="manifest-hash",
            checkpoint_sha256="checkpoint-hash",
            dimension_audit=audit,
        )
        self.assertEqual(observed["mode"], "audited_runtime_code_migration")

        tampered = dict(receipt)
        tampered["checkpoint_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "receipt is inconsistent"):
            validate_loader_migration_receipt(
                tampered,
                lock={"code_sha256": old_code},
                lock_sha256="lock-hash",
                current_code=new_code,
                manifest_sha256="manifest-hash",
                checkpoint_sha256="checkpoint-hash",
                dimension_audit=audit,
            )

        with self.assertRaisesRegex(ValueError, "no runtime code manifest"):
            validate_loader_migration_receipt(
                receipt,
                lock={},
                lock_sha256="lock-hash",
                current_code=new_code,
                manifest_sha256="manifest-hash",
                checkpoint_sha256="checkpoint-hash",
                dimension_audit=audit,
            )

        def receipt_for(
            candidate_old: dict[str, str],
            candidate_new: dict[str, str],
            candidate_audit: dict[str, object],
        ) -> dict[str, object]:
            differing = sorted(
                key
                for key in set(candidate_old) | set(candidate_new)
                if candidate_old.get(key) != candidate_new.get(key)
            )
            return {
                **receipt,
                "old_code_sha256": candidate_old,
                "new_code_sha256": candidate_new,
                "differing_code_keys": differing,
                "nifti_dimensionality_audit": candidate_audit,
            }

        semantic_failures = (
            (
                "extra runtime change",
                old_code,
                {**new_code, "FineTuning/shared/runtime.py": "changed"},
                audit,
                "Unaudited Curia-2 runtime changes",
            ),
            (
                "wrong legacy finetune",
                {**old_code, FINETUNE_CODE_KEY: "wrong"},
                new_code,
                audit,
                "legacy Curia-2 finetune",
            ),
            (
                "wrong legacy loader",
                {**old_code, MODEL_CODE_KEY: "wrong"},
                new_code,
                audit,
                "legacy Curia-2 loader",
            ),
            (
                "wrong strict loader",
                old_code,
                {**new_code, MODEL_CODE_KEY: "wrong"},
                audit,
                "strict-3D Curia-2 loader",
            ),
            (
                "incomplete dimensionality audit",
                old_code,
                new_code,
                {**audit, "n_non_3d": 1},
                "Unexpected Curia-2 cohort dimensionality audit",
            ),
        )
        for name, candidate_old, candidate_new, candidate_audit, message in semantic_failures:
            with self.subTest(failure=name), self.assertRaisesRegex(ValueError, message):
                validate_loader_migration_receipt(
                    receipt_for(candidate_old, candidate_new, candidate_audit),
                    lock={"code_sha256": candidate_old},
                    lock_sha256="lock-hash",
                    current_code=candidate_new,
                    manifest_sha256="manifest-hash",
                    checkpoint_sha256="checkpoint-hash",
                    dimension_audit=candidate_audit,
                )


if __name__ == "__main__":
    unittest.main()
