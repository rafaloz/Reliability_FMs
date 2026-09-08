"""Focused tests for label-blind frozen and adapted inference on ON-Harmony."""

from __future__ import annotations

import ast
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import FineTuning.shared.ridge as ridge_module
import FineTuning.shared.onharmony as onharmony_module
from FineTuning.shared.cohort import PROJECT_ROOT, sha256_file
from FineTuning.shared.internal_test import validate_internal_test_runtime_code
from FineTuning.shared.onharmony import (
    ArtifactRegistry,
    RuntimeMigration,
    adapted_runtime_migration,
    adapted_selection_runtime_migration,
    capture_prediction_artifact_integrity,
    effective_precision,
    load_onharmony_manifest,
    merge_prediction_provenance,
    publish_predictions,
    selection_compatible_lock,
    validate_prediction_frame,
    validate_prediction_runtime_code,
)
from FineTuning.shared.ridge import (
    MODEL_FILENAME,
    ONHARMONY_EMBEDDING_MANIFEST_VERSION,
    audit_frozen_onharmony_source,
    atomic_npz,
    load_locked_ridge_artifact,
    read_label_blind_onharmony_embeddings,
    run_frozen_onharmony_prediction,
    validate_ridge_artifact,
    validate_frozen_onharmony_lock,
)


def isolated_model_import(name: str):
    """Import a model entry point without depending on prior legacy aliases."""

    project_root = Path(__file__).resolve().parents[3]
    family = name.split(".", 2)[1]
    model_dir = str((project_root / "FineTuning" / family).resolve())
    aliases = ("model", "finetune")
    missing = object()
    previous_aliases = {
        alias: sys.modules.pop(alias, missing) for alias in aliases
    }
    previous_target = sys.modules.pop(name, missing)
    previous_path = list(sys.path)
    sys.path[:] = [model_dir, *[entry for entry in sys.path if entry != model_dir]]
    try:
        imported = importlib.import_module(name)
    finally:
        sys.path[:] = previous_path
        for alias, module in previous_aliases.items():
            sys.modules.pop(alias, None)
            if module is not missing:
                sys.modules[alias] = module
        if previous_target is not missing:
            sys.modules[name] = previous_target
        else:
            sys.modules.pop(name, None)
    return imported


anatcl = isolated_model_import("FineTuning.AnatCL.predict_onharmony")
brainiac = isolated_model_import("FineTuning.BrainIAC.predict_onharmony")
brainsegfounder = isolated_model_import(
    "FineTuning.BrainSegFounder.predict_onharmony"
)
curia2 = isolated_model_import("FineTuning.Curia2.predict_onharmony")
neurovfm = isolated_model_import("FineTuning.NeuroVFM.predict_onharmony")
neurosimclr = isolated_model_import(
    "FineTuning.3DNeuroSimCLR.predict_onharmony"
)
yaware = isolated_model_import("FineTuning.yAware.predict_onharmony")


def metadata_row(participant: str, session: str) -> dict[str, str]:
    return {
        "session": f"{participant}_{session}",
        "cohort": "phase1",
        "scanner_code": "scanner1",
        "scanner_name": "Scanner One",
        "vendor": "Vendor",
        "scanner_type": "3T",
        "scanner_typeloc": "Vendor 3T Site",
        "scan_session_num": "1",
        "scan_session": session,
        "subject": f"_{participant}",
        "age": "SHOULD_NOT_BE_LOADED",
    }


class ManifestTests(unittest.TestCase):
    def test_manifest_is_ordered_and_drops_outcome_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [
                "sub-001_ses-B_T1w.nii.gz",
                "sub-001_ses-A_T1w.nii.gz",
            ]
            paths = [root / name for name in names]
            for path in paths:
                path.write_bytes(b"nifti")
            filelist = root / "filelist.txt"
            filelist.write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")
            metadata = root / "metadata.csv"
            pd.DataFrame(
                [metadata_row("001", "A"), metadata_row("001", "B")]
            ).to_csv(metadata, index=False)

            frame, summary = load_onharmony_manifest(
                filelist,
                metadata,
                expected_acquisitions=2,
                expected_subjects=1,
            )

            self.assertEqual(frame["session"].tolist(), ["B", "A"])
            self.assertNotIn("age", frame.columns)
            self.assertTrue(summary["label_blind"])
            self.assertEqual(summary["n_acquisitions"], 2)

    def test_duplicate_filelist_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sub-001_ses-A_T1w.nii.gz"
            path.write_bytes(b"nifti")
            filelist = root / "filelist.txt"
            filelist.write_text(f"{path}\n{path}\n", encoding="utf-8")
            metadata = root / "metadata.csv"
            pd.DataFrame(
                [metadata_row("001", "A"), metadata_row("001", "B")]
            ).to_csv(metadata, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate paths"):
                load_onharmony_manifest(
                    filelist,
                    metadata,
                    expected_acquisitions=2,
                    expected_subjects=1,
                )


class CommonAdaptedInferenceTests(unittest.TestCase):
    def test_all_seven_models_delegate_to_the_same_inference_engine(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        model_directories = (
            "3DNeuroSimCLR",
            "AnatCL",
            "BrainIAC",
            "BrainSegFounder",
            "Curia2",
            "NeuroVFM",
            "yAware",
        )
        required_names = {
            "validate_run_binding",
            "load_adapted_model",
            "run_standard_audit",
            "run_standard_prediction",
        }
        forbidden_local_definitions = {
            "load_onharmony_manifest",
            "validate_lock_header",
            "predict_frame",
            "publish_predictions",
        }
        for directory in model_directories:
            path = project_root / "FineTuning" / directory / "predict_onharmony.py"
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            shared_imports = {
                alias.asname or alias.name
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                and node.module == "FineTuning.shared.onharmony"
                for alias in node.names
            }
            called_names = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            local_definitions = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            with self.subTest(model=directory):
                self.assertTrue(
                    required_names <= shared_imports,
                    f"Missing shared imports: {sorted(required_names - shared_imports)}",
                )
                self.assertTrue(
                    required_names <= called_names,
                    f"Missing real shared calls: {sorted(required_names - called_names)}",
                )
                self.assertTrue(
                    forbidden_local_definitions.isdisjoint(local_definitions),
                    "Forbidden local definitions: "
                    f"{sorted(forbidden_local_definitions & local_definitions)}",
                )


class RuntimeAndArtifactTests(unittest.TestCase):
    def test_effective_precision_matches_autocast_device_semantics(self) -> None:
        class Device:
            def __init__(self, device_type: str) -> None:
                self.type = device_type

        self.assertEqual(effective_precision(Device("cpu"), "bfloat16"), "float32")
        self.assertEqual(effective_precision(Device("cuda"), "off"), "float32")
        self.assertEqual(effective_precision(Device("cuda"), "float16"), "float16")
        self.assertEqual(effective_precision(Device("cuda"), "bfloat16"), "bfloat16")

    @staticmethod
    def publication_inputs(root: Path) -> dict[str, object]:
        class Model:
            model_name = "TestModel"
            variant = "canonical"

        class Device:
            type = "cpu"

            def __str__(self) -> str:
                return "cpu"

        class Torch:
            class cuda:
                @staticmethod
                def get_device_name(device: object) -> str:
                    raise AssertionError("CPU publication must not query CUDA")

        lock_path = root / "run" / "LOCKED.json"
        lock_path.parent.mkdir()
        checkpoint_path = lock_path.parent / "selected_model.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        lock = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "code_sha256": {"finetune.py": "training-hash"},
        }
        lock_path.write_text(json.dumps(lock) + "\n", encoding="utf-8")
        selection_path = root / "selection" / "SELECTED_LR.json"
        selection_path.parent.mkdir()
        selection_payload = {
            "model": "TestModel",
            "variant": "canonical",
            "selection_seed": 0,
            "selected_lr_config": "high",
            "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        }
        selection_path.write_text(
            json.dumps(selection_payload) + "\n", encoding="utf-8"
        )
        selection = {
            "selection_artifact": str(selection_path),
            "selection_artifact_sha256": sha256_file(selection_path),
            "selected_lr_config": "high",
            "selection_model": "TestModel",
            "selection_variant": "canonical",
            "selection_seed": 0,
            "selection_checkpoint_sha256": sha256_file(checkpoint_path),
        }
        integrity = capture_prediction_artifact_integrity(
            lock=lock,
            lock_path=lock_path,
            checkpoint_path=checkpoint_path,
            selection=selection,
        )
        predictions = pd.DataFrame({"subject": ["sub-1"], "age_pred": [42.0]})
        predictions.attrs["elapsed_seconds"] = 1.25
        return {
            "pipeline_version": "test-v1",
            "model": Model(),
            "predictions": predictions,
            "manifest": {"n_acquisitions": 1, "n_subjects": 1},
            "input_contract": {"contract": "test"},
            "binding": {
                "lock": lock,
                "lock_path": lock_path,
                "checkpoint_path": checkpoint_path,
                "selection": selection,
                "runtime_code_provenance": {"mode": "exact"},
                "seed": 0,
                "amp": "off",
                "artifact_integrity": integrity,
            },
            "runtime": {
                "torch": Torch,
                "device": Device(),
                "boundary": {"trainable_components": ["final_block"]},
            },
            "script_path": Path(__file__).resolve(),
        }

    def test_prediction_bundle_is_published_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions"
            inputs = self.publication_inputs(root)
            audit = publish_predictions(output_dir=output, **inputs)

            self.assertTrue((output / "predictions.tsv").is_file())
            self.assertTrue((output / "prediction_audit.json").is_file())
            self.assertEqual(
                audit["predictions_artifact"], str(output / "predictions.tsv")
            )
            self.assertEqual(audit["runtime"]["effective_precision"], "float32")
            self.assertEqual(list(root.glob(".predictions.tmp.*")), [])

    def test_prediction_bundle_failure_leaves_no_final_or_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions"
            inputs = self.publication_inputs(root)
            with patch(
                "FineTuning.shared.onharmony.atomic_json",
                side_effect=RuntimeError("interrupted"),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    publish_predictions(output_dir=output, **inputs)

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".predictions.tmp.*")), [])

    def test_extra_audit_cannot_override_protected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions"
            inputs = self.publication_inputs(root)
            with self.assertRaisesRegex(ValueError, "protected fields"):
                publish_predictions(
                    output_dir=output,
                    extra_audit={"status": "FAKE_PASS"},
                    **inputs,
                )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".predictions.tmp.*")), [])

    def test_checkpoint_drift_during_publication_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions"
            inputs = self.publication_inputs(root)
            checkpoint = Path(inputs["binding"]["checkpoint_path"])
            original_atomic_json = onharmony_module.atomic_json

            def write_then_mutate(payload: object, path: Path) -> None:
                original_atomic_json(payload, path)
                checkpoint.write_bytes(b"checkpoint-changed-during-publication")

            with patch(
                "FineTuning.shared.onharmony.atomic_json",
                side_effect=write_then_mutate,
            ):
                with self.assertRaisesRegex(RuntimeError, "adapted_checkpoint changed"):
                    publish_predictions(output_dir=output, **inputs)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".predictions.tmp.*")), [])

    def test_training_lock_drift_during_publication_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions"
            inputs = self.publication_inputs(root)
            lock_path = Path(inputs["binding"]["lock_path"])
            original_atomic_json = onharmony_module.atomic_json

            def write_then_mutate(payload: object, path: Path) -> None:
                original_atomic_json(payload, path)
                lock_path.write_text('{"changed": true}\n', encoding="utf-8")

            with patch(
                "FineTuning.shared.onharmony.atomic_json",
                side_effect=write_then_mutate,
            ):
                with self.assertRaisesRegex(RuntimeError, "training_lock changed"):
                    publish_predictions(output_dir=output, **inputs)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".predictions.tmp.*")), [])

    def test_existing_prediction_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions"
            output.mkdir()
            sentinel = output / "KEEP.txt"
            sentinel.write_text("original\n", encoding="utf-8")
            inputs = self.publication_inputs(root)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                publish_predictions(output_dir=output, **inputs)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original\n")
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["KEEP.txt"])
            self.assertEqual(list(root.glob(".predictions.tmp.*")), [])

    def test_selection_drift_before_publication_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions"
            inputs = self.publication_inputs(root)
            selection = Path(
                inputs["binding"]["artifact_integrity"]["lr_selection"]["path"]
            )
            selection.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "lr_selection changed"):
                publish_predictions(output_dir=output, **inputs)
            self.assertFalse(output.exists())

    def test_anatcl_release_normalizer_accepts_only_verified_installed_sources(
        self,
    ) -> None:
        expected = {"anatcl/anatcl.py": "one", "anatcl/models/resnet3d.py": "two"}
        current = {
            "checkpoint": "/new/location/fold0.pth",
            "expected_source_sha256": expected,
        }
        locked = {
            "checkpoint": "/old/location/fold0.pth",
            "expected_source_sha256": expected,
            "installed_source_sha256": dict(expected),
        }
        self.assertEqual(
            anatcl.prediction_release_contract(current),
            anatcl.prediction_release_contract(locked),
        )
        locked["installed_source_sha256"] = dict(expected, **{"unknown.py": "bad"})
        with self.assertRaisesRegex(ValueError, "installed-source hashes"):
            anatcl.prediction_release_contract(locked)

    def test_runtime_migration_requires_exact_old_and_new_hashes(self) -> None:
        migration = RuntimeMigration(
            changed={"model.py": ("old", "new")},
            added={"guard.py": "guard"},
        )
        result = validate_prediction_runtime_code(
            {"model.py": "old"},
            {"model.py": "new", "guard.py": "guard"},
            migration,
        )
        self.assertEqual(result["mode"], "hash_pinned_evaluation_only_migration")
        with self.assertRaisesRegex(ValueError, "Unapproved runtime hash transition"):
            validate_prediction_runtime_code(
                {"model.py": "different"},
                {"model.py": "new", "guard.py": "guard"},
                migration,
            )
        with self.assertRaisesRegex(ValueError, "Unapproved runtime hash transition"):
            validate_prediction_runtime_code(
                {"model.py": "old"},
                {"model.py": "different", "guard.py": "guard"},
                migration,
            )
        with self.assertRaisesRegex(ValueError, "Unapproved added runtime dependency"):
            validate_prediction_runtime_code(
                {"model.py": "old"},
                {"model.py": "new", "guard.py": "different"},
                migration,
            )
        with self.assertRaisesRegex(ValueError, "drift exceeds"):
            validate_prediction_runtime_code(
                {"model.py": "old", "removed.py": "same"},
                {"model.py": "new", "guard.py": "guard"},
                migration,
            )
        with self.assertRaisesRegex(ValueError, "differs from the adapted checkpoint"):
            validate_prediction_runtime_code(
                {"model.py": "old"},
                {"model.py": "new"},
            )

    def test_all_adapted_model_migrations_are_exact_and_complete(self) -> None:
        old_metrics = (
            "d7ec28a8f8d4d921efc397da4662af7bec972bb567f7061aee7d1544cd0ebd3e"
        )
        cases = (
            (
                "3D-Neuro-SimCLR",
                {
                    "FineTuning/3DNeuroSimCLR/finetune.py": (
                        "045ff93d27b61b491c8b745b1af4e3845bebe399b2c7020b1056b594553dbd3a"
                    ),
                    "FineTuning/shared/metrics.py": old_metrics,
                },
            ),
            (
                "AnatCL",
                {
                    "FineTuning/AnatCL/finetune.py": (
                        "90eb9c3ee459b75f820fd2c70826ab8601905d25a77f03b417f46bbe5e9aec08"
                    ),
                    "FineTuning/shared/metrics.py": old_metrics,
                },
            ),
            (
                "BrainIAC",
                {
                    "FineTuning/BrainIAC/finetune.py": (
                        "bed265cdefaaf8465bbd294a1e16e8ce177bebdb0e445a3c33d63da909b7d328"
                    ),
                    "FineTuning/shared/metrics.py": old_metrics,
                },
            ),
            (
                "BrainIAC",
                {
                    "FineTuning/BrainIAC/finetune.py": (
                        "48f4f243140a1c40b8858d0dfaf35563445270edca07df8c4a4f7b5382ce480f"
                    ),
                    "FineTuning/shared/metrics.py": old_metrics,
                },
            ),
            (
                "BrainSegFounder",
                {
                    "FineTuning/BrainSegFounder/finetune.py": (
                        "c162123d7273effda333bc173df2ca336cc1e186f4d08441e8825b53f298f69f"
                    ),
                    "FineTuning/shared/metrics.py": old_metrics,
                },
            ),
            (
                "Curia-2",
                {
                    "FineTuning/Curia2/finetune.py": (
                        "df8ee990e880a16029b86362eb9f08100638b4033dae0de28e5f290f7baadabc"
                    ),
                    "FineTuning/shared/metrics.py": old_metrics,
                },
            ),
            (
                "NeuroVFM",
                {
                    "FineTuning/NeuroVFM/finetune.py": (
                        "f2a10474132aa5e8863de1555ea150511d4247ef9bde6085b7c9b7315d92be0d"
                    ),
                    "FineTuning/shared/metrics.py": old_metrics,
                    "Emb_extractors/extract_NeuroVFM.py": (
                        "6fc5fd23f08d00fcebca597cd7090a9238df7de5b3541e63177d41066d207a32"
                    ),
                },
            ),
            (
                "y-Aware",
                {
                    "FineTuning/yAware/finetune.py": (
                        "ac4b7073ac85ca32a85b4694f3fde713e9c5fa7811b6c3a09b36e32e2c1df209"
                    ),
                    "FineTuning/shared/metrics.py": old_metrics,
                },
            ),
        )
        for model_name, locked in cases:
            with self.subTest(model=model_name, finetune=next(iter(locked.values()))):
                migration = adapted_runtime_migration(model_name, locked)
                self.assertIsNotNone(migration)
                self.assertTrue(migration.migration_id)
                self.assertTrue(migration.reason)
                current = dict(locked)
                for key, (_, new_hash) in migration.changed.items():
                    current[key] = new_hash
                current.update(migration.added)
                provenance = validate_prediction_runtime_code(
                    locked, current, migration
                )
                self.assertEqual(provenance["migration_id"], migration.migration_id)
                self.assertEqual(provenance["reason"], migration.reason)
                self.assertEqual(
                    provenance["migration_registry_code"],
                    {
                        "FineTuning/shared/onharmony.py": sha256_file(
                            PROJECT_ROOT / "FineTuning/shared/onharmony.py"
                        )
                    },
                )
                internal_provenance = validate_internal_test_runtime_code(
                    model_name=model_name,
                    locked_code=locked,
                    current_code=current,
                )
                self.assertEqual(
                    internal_provenance["internal_test_guard_code"],
                    {
                        "FineTuning/shared/internal_test.py": sha256_file(
                            PROJECT_ROOT / "FineTuning/shared/internal_test.py"
                        )
                    },
                )
                self.assertIsNone(adapted_runtime_migration(model_name, current))

        unknown = dict(cases[0][1])
        unknown["FineTuning/shared/metrics.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "No unique approved"):
            adapted_runtime_migration("3D-Neuro-SimCLR", unknown)
        with self.assertRaisesRegex(ValueError, "No adapted runtime contract"):
            adapted_runtime_migration("unknown-model", {"code.py": "0" * 64})

    def test_brainiac_selection_migration_is_separate_from_current_runtime(self) -> None:
        migration = adapted_selection_runtime_migration("BrainIAC")
        self.assertIsNotNone(migration)
        self.assertEqual(
            set(migration.changed), {"FineTuning/BrainIAC/finetune.py"}
        )
        self.assertIsNone(adapted_selection_runtime_migration("Curia-2"))

    def test_selection_anchor_accepts_only_the_pinned_code_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_run = root / "selected"
            selected_run.mkdir()
            (selected_run / "LOCKED.json").write_text(
                json.dumps({"code_sha256": {"finetune.py": "old"}}),
                encoding="utf-8",
            )
            selection = root / "selection"
            selection.mkdir()
            (selection / "SELECTED_LR.json").write_text(
                json.dumps({"selected_run_dir": str(selected_run)}),
                encoding="utf-8",
            )
            normalized, provenance = selection_compatible_lock(
                {"code_sha256": {"finetune.py": "new"}},
                selection_dir=selection,
                migration=RuntimeMigration(
                    changed={"finetune.py": ("old", "new")}, added={}
                ),
            )
            self.assertEqual(normalized["code_sha256"], {"finetune.py": "old"})
            self.assertEqual(
                provenance["mode"], "hash_pinned_evaluation_only_migration"
            )
            with self.assertRaisesRegex(ValueError, "without an approved migration"):
                selection_compatible_lock(
                    {"code_sha256": {"finetune.py": "new"}},
                    selection_dir=selection,
                    migration=None,
                )
            with self.assertRaisesRegex(
                ValueError, "Unapproved runtime hash transition"
            ):
                selection_compatible_lock(
                    {"code_sha256": {"finetune.py": "new"}},
                    selection_dir=selection,
                    migration=RuntimeMigration(
                        changed={"finetune.py": ("different", "new")}, added={}
                    ),
                )

    def test_artifact_registry_revalidates_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"first")
            calls: list[str] = []
            registry = ArtifactRegistry(
                contract_version="test-v1",
                resolver=lambda record: {"input": path},
                validator=lambda record, paths: calls.append("validated") or {},
            )
            record = {"acquisition_id": "acq"}
            first = registry.validate(record)
            registry.validate(record)
            self.assertEqual(calls, ["validated"])
            path.write_bytes(b"second-longer")
            second = registry.validate(record)
            self.assertEqual(calls, ["validated", "validated"])
            self.assertNotEqual(first["input_sha256"], second["input_sha256"])

    def test_artifact_registry_rejects_drift_after_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"first")
            registry = ArtifactRegistry(
                contract_version="test-v1",
                resolver=lambda record: {"input": path},
            )
            record = {"acquisition_id": "acq"}
            registry.freeze(record)
            path.write_bytes(b"second-longer")
            with self.assertRaisesRegex(RuntimeError, "changed after being frozen"):
                registry.validate(record)

    def test_artifact_registry_rejects_changes_during_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"first")

            def mutate_input(
                record: dict[str, str], paths: dict[str, Path]
            ) -> dict[str, str]:
                del record
                paths["input"].write_bytes(b"second-longer")
                return {"validator": "mutated input"}

            registry = ArtifactRegistry(
                contract_version="test-v1",
                resolver=lambda record: {"input": path},
                validator=mutate_input,
            )
            with self.assertRaisesRegex(RuntimeError, "changed during validation"):
                registry.validate({"acquisition_id": "acq"})

    def test_prediction_provenance_cannot_override_protected_fields(self) -> None:
        prediction = {"age_pred": 42.0, "checkpoint_sha256": "checkpoint"}
        self.assertEqual(
            merge_prediction_provenance(prediction, {"input_sha256": "input"}),
            {
                "age_pred": 42.0,
                "checkpoint_sha256": "checkpoint",
                "input_sha256": "input",
            },
        )
        with self.assertRaisesRegex(ValueError, "cannot override prediction fields"):
            merge_prediction_provenance(prediction, {"age_pred": -1.0})

    def test_prediction_frame_rejects_labels_and_wrong_order(self) -> None:
        expected = pd.DataFrame(
            {
                "subject": ["sub-1", "sub-2"],
                "session": ["A", "B"],
                "acquisition_id": ["one", "two"],
            }
        )
        valid = expected.copy()
        valid["age_pred"] = [50.0, 60.0]
        validate_prediction_frame(valid, expected)
        labelled = valid.copy()
        labelled["age_true"] = [51.0, 61.0]
        with self.assertRaisesRegex(ValueError, "labels"):
            validate_prediction_frame(labelled, expected)
        with self.assertRaisesRegex(ValueError, "count differs"):
            validate_prediction_frame(valid.iloc[:1].copy(), expected)
        nonfinite = valid.copy()
        nonfinite.loc[1, "age_pred"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite ages"):
            validate_prediction_frame(nonfinite, expected)
        duplicated = valid.copy()
        duplicated.loc[1, "acquisition_id"] = duplicated.loc[0, "acquisition_id"]
        with self.assertRaisesRegex(ValueError, "duplicate acquisitions"):
            validate_prediction_frame(duplicated, expected)
        with self.assertRaisesRegex(ValueError, "order"):
            validate_prediction_frame(valid.iloc[::-1].reset_index(drop=True), expected)


class FrozenRidgeOnHarmonyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            [
                {
                    "dataset": "ON-Harmony",
                    "subject": "sub-001",
                    "session": "A",
                    "acquisition_id": "sub-001_ses-A_T1w",
                    "raw_t1w_path": "/raw/sub-001_ses-A_T1w.nii.gz",
                    "cohort": "phase1",
                    "scanner_code": "scanner1",
                    "scanner_name": "Scanner One",
                    "vendor": "Vendor",
                    "scanner_type": "3T",
                    "scanner_typeloc": "Vendor 3T Site",
                    "scan_session_num": "1",
                },
                {
                    "dataset": "ON-Harmony",
                    "subject": "sub-002",
                    "session": "B",
                    "acquisition_id": "sub-002_ses-B_T1w",
                    "raw_t1w_path": "/raw/sub-002_ses-B_T1w.nii.gz",
                    "cohort": "phase2",
                    "scanner_code": "scanner2",
                    "scanner_name": "Scanner Two",
                    "vendor": "Vendor",
                    "scanner_type": "3T",
                    "scanner_typeloc": "Vendor 3T Site",
                    "scan_session_num": "2",
                },
            ]
        )

    def write_embeddings(self, path: Path) -> None:
        # Deliberately reverse the rows and include a toxic age field. The
        # loader must reorder by acquisition and never read the age values.
        pd.DataFrame(
            {
                "file_path": [
                    "/derived/sub-002_ses-B_T1w_ready.nii.gz",
                    "/derived/sub-001_ses-A_T1w_ready.nii.gz",
                ],
                "age": ["MUST_NOT_BE_LOADED", "MUST_NOT_BE_LOADED"],
                "emb_0": [3.0, 1.0],
                "emb_1": [4.0, 2.0],
            }
        ).to_csv(path, index=False)
        self.write_embedding_manifest(
            path.with_name(f"{path.stem}_manifest.json"), path
        )

    def write_locked_model(self, run_dir: Path, fit_code: dict[str, str]) -> None:
        run_dir.mkdir()
        model_path = run_dir / MODEL_FILENAME
        atomic_npz(
            {
                "feature_mean": np.asarray([0.0, 0.0]),
                "feature_scale": np.asarray([1.0, 1.0]),
                "coef": np.asarray([1.0, 2.0]),
                "intercept": np.asarray([0.0]),
                "age_mean": np.asarray([50.0]),
                "age_std": np.asarray([10.0]),
                "alpha": np.asarray([1.0]),
                "n_constant_dimensions": np.asarray([0]),
            },
            model_path,
        )
        (run_dir / "LOCKED.json").write_text(
            json.dumps(
                {
                    "status": "LOCKED_AFTER_TRAIN_VALIDATION",
                    "pipeline_version": "test_pipeline",
                    "arm": "frozen",
                    "model": "TestModel",
                    "variant": "canonical",
                    "readout": "test readout",
                    "embedding_dim": 2,
                    "selected_alpha": 1.0,
                    "model_artifact_sha256": sha256_file(model_path),
                    "fit_code_sha256": fit_code,
                    "manifest": {"manifest_sha256": "training"},
                    "internal_test_accessed": False,
                    "on_harmony_accessed": False,
                }
            ),
            encoding="utf-8",
        )

    def write_embedding_manifest(self, path: Path, embeddings: Path) -> None:
        extractor = path.parent / "extract_test.py"
        extractor.write_text("# immutable test extractor\n", encoding="utf-8")
        path.write_text(
            json.dumps(
                {
                    "manifest_schema_version": ONHARMONY_EMBEDDING_MANIFEST_VERSION,
                    "pipeline_version": "test_extractor_v1",
                    "model": "TestModel",
                    "variant": "canonical",
                    "readout": "test readout",
                    "feature_prefix": "emb_",
                    "embedding_dimensions": 2,
                    "extractor": str(extractor.resolve()),
                    "extractor_sha256": sha256_file(extractor),
                    "output": str(embeddings.resolve()),
                    "output_sha256": sha256_file(embeddings),
                    "n_rows": 2,
                }
            ),
            encoding="utf-8",
        )

    def test_all_internal_test_fitters_use_the_locked_ridge_loader(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        for model_directory in (
            "3DNeuroSimCLR",
            "AnatCL",
            "BrainIAC",
            "BrainSegFounder",
            "Curia2",
            "NeuroVFM",
            "yAware",
        ):
            path = project_root / "FineTuning" / model_directory / "fit_ridge.py"
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports_by_module: dict[str, dict[str, str]] = {}
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module is not None:
                    imports_by_module.setdefault(node.module, {}).update(
                        {alias.name: alias.asname or alias.name for alias in node.names}
                    )
            ridge_imports = imports_by_module.get("FineTuning.shared.ridge", {})
            cohort_imports = imports_by_module.get("FineTuning.shared.cohort", {})
            locked_loader_name = ridge_imports.get("load_locked_ridge_artifact")
            prediction_runner_name = ridge_imports.get(
                "run_frozen_onharmony_prediction"
            )
            source_audit_name = ridge_imports.get("audit_frozen_onharmony_source")
            manifest_hash_name = cohort_imports.get("MANIFEST_SHA256")
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            direct_call_names = {
                node.func.id for node in calls if isinstance(node.func, ast.Name)
            }
            all_call_names = direct_call_names | {
                node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
            }
            lock_dicts = [
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Dict)
                and any(
                    isinstance(target, ast.Name) and target.id == "lock"
                    for target in node.targets
                )
            ]

            def dict_contract(node: ast.Dict) -> dict[str, ast.expr]:
                return {
                    key.value: value
                    for key, value in zip(node.keys, node.values)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }

            lock_contracts = [dict_contract(node) for node in lock_dicts]
            prediction_calls = [
                node
                for node in calls
                if isinstance(node.func, ast.Name)
                and node.func.id == prediction_runner_name
            ]
            with self.subTest(model=model_directory):
                self.assertIsNotNone(locked_loader_name)
                self.assertIn(locked_loader_name, direct_call_names)
                self.assertNotIn("load_npz", all_call_names)
                self.assertTrue(
                    any(
                        isinstance(contract.get("pipeline_version"), ast.Name)
                        and contract["pipeline_version"].id == "PIPELINE_VERSION"
                        and isinstance(contract.get("readout"), ast.Name)
                        and contract["readout"].id == "READOUT"
                        for contract in lock_contracts
                    ),
                    "No lock dict binds PIPELINE_VERSION and READOUT",
                )
                self.assertIsNotNone(prediction_runner_name)
                self.assertIsNotNone(source_audit_name)
                self.assertIn(source_audit_name, direct_call_names)
                self.assertIn(
                    "audit-onharmony",
                    {
                        node.value
                        for node in ast.walk(tree)
                        if isinstance(node, ast.Constant) and isinstance(node.value, str)
                    },
                )
                self.assertIsNotNone(manifest_hash_name)
                self.assertTrue(
                    any(
                        any(
                            keyword.arg == "expected_training_manifest_sha256"
                            and isinstance(keyword.value, ast.Name)
                            and keyword.value.id == manifest_hash_name
                            for keyword in call.keywords
                        )
                        for call in prediction_calls
                    ),
                    "Frozen ON-Harmony call is not bound to MANIFEST_SHA256",
                )

    def test_locked_ridge_loader_rejects_hash_and_alpha_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / MODEL_FILENAME
            atomic_npz(
                {
                    "feature_mean": np.asarray([0.0]),
                    "feature_scale": np.asarray([1.0]),
                    "coef": np.asarray([1.0]),
                    "intercept": np.asarray([0.0]),
                    "age_mean": np.asarray([50.0]),
                    "age_std": np.asarray([10.0]),
                    "alpha": np.asarray([1.0]),
                    "n_constant_dimensions": np.asarray([0]),
                },
                model_path,
            )
            model_sha256 = sha256_file(model_path)
            artifact = load_locked_ridge_artifact(
                model_path,
                expected_sha256=model_sha256,
                expected_alpha=1.0,
                expected_dimension=1,
            )
            self.assertEqual(float(artifact["alpha"][0]), 1.0)

            with self.assertRaisesRegex(ValueError, "hash differs"):
                load_locked_ridge_artifact(
                    model_path,
                    expected_sha256="0" * 64,
                    expected_alpha=1.0,
                    expected_dimension=1,
                )
            with self.assertRaisesRegex(ValueError, "alpha differs"):
                load_locked_ridge_artifact(
                    model_path,
                    expected_sha256=model_sha256,
                    expected_alpha=10.0,
                    expected_dimension=1,
                )

    def test_ridge_artifact_rejects_incomplete_or_invalid_numeric_contracts(self) -> None:
        valid = {
            "feature_mean": np.asarray([0.0, 0.0]),
            "feature_scale": np.asarray([1.0, 1.0]),
            "coef": np.asarray([1.0, 2.0]),
            "intercept": np.asarray([0.0]),
            "age_mean": np.asarray([50.0]),
            "age_std": np.asarray([10.0]),
            "alpha": np.asarray([1.0]),
            "n_constant_dimensions": np.asarray([0], dtype=np.int64),
        }
        self.assertEqual(validate_ridge_artifact(valid, expected_dimension=2), 2)

        invalid_cases = {
            "missing": {key: value for key, value in valid.items() if key != "coef"},
            "unexpected": {**valid, "unused": np.asarray([0.0])},
            "shape": {**valid, "coef": np.asarray([[1.0, 2.0]])},
            "non-finite": {**valid, "feature_mean": np.asarray([0.0, np.nan])},
            "non-positive scale": {**valid, "feature_scale": np.asarray([1.0, 0.0])},
            "non-positive age std": {**valid, "age_std": np.asarray([0.0])},
            "non-positive alpha": {**valid, "alpha": np.asarray([-1.0])},
            "non-integer constants": {
                **valid,
                "n_constant_dimensions": np.asarray([0.0]),
            },
            "too many constants": {
                **valid,
                "n_constant_dimensions": np.asarray([3], dtype=np.int64),
            },
        }
        for label, artifact in invalid_cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_ridge_artifact(artifact, expected_dimension=2)

        with self.assertRaisesRegex(ValueError, "dimension differs"):
            validate_ridge_artifact(valid, expected_dimension=3)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                atomic_npz(invalid_cases["non-positive scale"], Path(directory) / "bad.npz")

    def test_locked_ridge_loader_rejects_mutation_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / MODEL_FILENAME
            atomic_npz(
                {
                    "feature_mean": np.asarray([0.0]),
                    "feature_scale": np.asarray([1.0]),
                    "coef": np.asarray([1.0]),
                    "intercept": np.asarray([0.0]),
                    "age_mean": np.asarray([50.0]),
                    "age_std": np.asarray([10.0]),
                    "alpha": np.asarray([1.0]),
                    "n_constant_dimensions": np.asarray([0]),
                },
                model_path,
            )
            model_sha256 = sha256_file(model_path)
            real_load_npz = ridge_module.load_npz

            def load_then_mutate(path):
                artifact = real_load_npz(path)
                with Path(path).open("ab") as handle:
                    handle.write(b"drift")
                return artifact

            with patch.object(
                ridge_module, "load_npz", side_effect=load_then_mutate
            ), self.assertRaisesRegex(RuntimeError, "changed while it was being read"):
                load_locked_ridge_artifact(
                    model_path,
                    expected_sha256=model_sha256,
                    expected_alpha=1.0,
                    expected_dimension=1,
                )

    def test_atomic_npz_failure_leaves_no_destination_or_temporary_file(self) -> None:
        valid = {
            "feature_mean": np.asarray([0.0]),
            "feature_scale": np.asarray([1.0]),
            "coef": np.asarray([1.0]),
            "intercept": np.asarray([0.0]),
            "age_mean": np.asarray([50.0]),
            "age_std": np.asarray([10.0]),
            "alpha": np.asarray([1.0]),
            "n_constant_dimensions": np.asarray([0], dtype=np.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / MODEL_FILENAME
            with patch.object(
                ridge_module.np,
                "savez_compressed",
                side_effect=OSError("injected NPZ failure"),
            ), self.assertRaisesRegex(OSError, "injected NPZ failure"):
                atomic_npz(valid, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(f".{MODEL_FILENAME}.tmp.*")), [])

    def test_atomic_npz_revalidation_failure_cleans_temporary_artifact(self) -> None:
        valid = {
            "feature_mean": np.asarray([0.0]),
            "feature_scale": np.asarray([1.0]),
            "coef": np.asarray([1.0]),
            "intercept": np.asarray([0.0]),
            "age_mean": np.asarray([50.0]),
            "age_std": np.asarray([10.0]),
            "alpha": np.asarray([1.0]),
            "n_constant_dimensions": np.asarray([0], dtype=np.int64),
        }
        corrupted = {**valid, "feature_scale": np.asarray([0.0])}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / MODEL_FILENAME
            with patch.object(
                ridge_module, "load_npz", return_value=corrupted
            ), self.assertRaisesRegex(ValueError, "strictly positive"):
                atomic_npz(valid, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(f".{MODEL_FILENAME}.tmp.*")), [])

    def run_prediction(
        self,
        *,
        run_dir: Path,
        embeddings: Path,
        output: Path,
        fit_code: dict[str, str],
        source_manifest: Path | None = None,
    ) -> dict[str, object]:
        summary = {
            "status": "PASS",
            "label_blind": True,
            "n_acquisitions": 2,
            "n_subjects": 2,
            "manifest_sha256": "external",
        }
        with patch(
            "FineTuning.shared.ridge.load_onharmony_manifest",
            return_value=(self.frame, summary),
        ):
            return run_frozen_onharmony_prediction(
                run_dir=run_dir,
                embeddings_path=embeddings,
                output_dir=output,
                model_name="TestModel",
                variant="canonical",
                expected_dimension=2,
                feature_prefix="emb_",
                extractor_pipeline_version="test_extractor_v1",
                readout="test readout",
                pipeline_version="test_pipeline",
                expected_training_manifest_sha256="training",
                current_fit_code_sha256=fit_code,
                source_manifest=source_manifest,
            )

    def test_embedding_loader_is_label_blind_and_restores_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.csv"
            self.write_embeddings(path)
            values, provenance = read_label_blind_onharmony_embeddings(
                path,
                expected_manifest=self.frame,
                feature_prefix="emb_",
                expected_dimension=2,
                expected_pipeline_version="test_extractor_v1",
                expected_model="TestModel",
                expected_variant="canonical",
                expected_readout="test readout",
            )
            np.testing.assert_array_equal(values, [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(provenance["source_outcome_columns_present"], ["age"])
            self.assertEqual(provenance["outcome_columns_loaded"], [])
            self.assertNotIn("age", provenance["columns_loaded"])

    def test_source_audit_runs_the_complete_label_blind_embedding_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            embeddings = Path(directory) / "embeddings.csv"
            self.write_embeddings(embeddings)
            summary = {
                "status": "PASS",
                "label_blind": True,
                "n_acquisitions": 2,
                "n_subjects": 2,
                "manifest_sha256": "external",
            }
            with patch(
                "FineTuning.shared.ridge.load_onharmony_manifest",
                return_value=(self.frame, summary),
            ), patch("builtins.print"):
                result = audit_frozen_onharmony_source(
                    embeddings_path=embeddings,
                    model_name="TestModel",
                    variant="canonical",
                    expected_dimension=2,
                    feature_prefix="emb_",
                    extractor_pipeline_version="test_extractor_v1",
                    readout="test readout",
                )
            self.assertEqual(result["status"], "ON_HARMONY_EMBEDDINGS_VALIDATED")
            self.assertEqual(result["embedding_matrix_shape"], [2, 2])
            self.assertTrue(result["label_blind"])
            self.assertFalse(result["outcome_labels_loaded"])
            self.assertEqual(
                result["on_harmony_embeddings"]["extractor_manifest"]["status"],
                "VALIDATED",
            )

    def test_embedding_loader_rejects_csv_changed_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            embeddings = Path(directory) / "embeddings.csv"
            self.write_embeddings(embeddings)
            real_read_csv = ridge_module.pd.read_csv

            def read_then_mutate(*args, **kwargs):
                frame = real_read_csv(*args, **kwargs)
                with embeddings.open("a", encoding="utf-8") as handle:
                    handle.write("\n")
                return frame

            with patch.object(
                ridge_module.pd, "read_csv", side_effect=read_then_mutate
            ), self.assertRaisesRegex(RuntimeError, "changed while it was being read"):
                read_label_blind_onharmony_embeddings(
                    embeddings,
                    expected_manifest=self.frame,
                    feature_prefix="emb_",
                    expected_dimension=2,
                    expected_pipeline_version="test_extractor_v1",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                )

    def test_embedding_manifest_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            embeddings = Path(directory) / "embeddings.csv"
            self.write_embeddings(embeddings)
            embeddings.with_name("embeddings_manifest.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "manifest is required"):
                read_label_blind_onharmony_embeddings(
                    embeddings,
                    expected_manifest=self.frame,
                    feature_prefix="emb_",
                    expected_dimension=2,
                    expected_pipeline_version="test_extractor_v1",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                )

    def test_embedding_loader_rejects_manifest_changed_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            embeddings = Path(directory) / "embeddings.csv"
            self.write_embeddings(embeddings)
            manifest = embeddings.with_name("embeddings_manifest.json")
            real_json_loads = ridge_module.json.loads

            def load_then_mutate(payload, *args, **kwargs):
                result = real_json_loads(payload, *args, **kwargs)
                with manifest.open("a", encoding="utf-8") as handle:
                    handle.write("\n")
                return result

            with patch.object(
                ridge_module.json, "loads", side_effect=load_then_mutate
            ), self.assertRaisesRegex(RuntimeError, "manifest changed while it was being read"):
                read_label_blind_onharmony_embeddings(
                    embeddings,
                    expected_manifest=self.frame,
                    feature_prefix="emb_",
                    expected_dimension=2,
                    expected_pipeline_version="test_extractor_v1",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                )

    def test_embedding_manifest_rejects_model_variant_readout_or_pipeline_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            manifest = root / "manifest.json"
            self.write_embeddings(embeddings)
            self.write_embedding_manifest(manifest, embeddings)
            original = json.loads(manifest.read_text(encoding="utf-8"))
            for field, value in (
                ("manifest_schema_version", "other_schema"),
                ("pipeline_version", "other_pipeline"),
                ("model", "OtherModel"),
                ("variant", "other_variant"),
                ("readout", "other readout"),
                ("feature_prefix", "other_"),
                ("embedding_dimensions", 3),
            ):
                with self.subTest(field=field):
                    payload = dict(original)
                    payload[field] = value
                    manifest.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "semantic contract differs"):
                        read_label_blind_onharmony_embeddings(
                            embeddings,
                            expected_manifest=self.frame,
                            feature_prefix="emb_",
                            expected_dimension=2,
                            expected_pipeline_version="test_extractor_v1",
                            expected_model="TestModel",
                            expected_variant="canonical",
                            expected_readout="test readout",
                            source_manifest=manifest,
                        )

    def test_present_embedding_manifest_requires_all_critical_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            manifest = root / "manifest.json"
            self.write_embeddings(embeddings)
            self.write_embedding_manifest(manifest, embeddings)
            complete = json.loads(manifest.read_text(encoding="utf-8"))

            for field in complete:
                with self.subTest(field=field):
                    payload = dict(complete)
                    payload.pop(field)
                    manifest.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "lacks required fields"):
                        read_label_blind_onharmony_embeddings(
                            embeddings,
                            expected_manifest=self.frame,
                            feature_prefix="emb_",
                            expected_dimension=2,
                            expected_pipeline_version="test_extractor_v1",
                            expected_model="TestModel",
                            expected_variant="canonical",
                            expected_readout="test readout",
                            source_manifest=manifest,
                        )

    def test_embedding_manifest_rejects_invalid_structure_paths_hashes_and_rows(self) -> None:
        cases = (
            ("non-object", None, TypeError, "not a JSON object"),
            ("invalid extractor hash", "extractor_sha256", ValueError, "invalid extractor hash"),
            ("missing extractor", "extractor", FileNotFoundError, "declared by its manifest is missing"),
            ("invalid output hash", "output_sha256", ValueError, "invalid output hash"),
            ("different output", "output", ValueError, "different output CSV"),
            ("wrong row count", "n_rows", ValueError, "row count differs"),
        )
        for name, field, error_type, message in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                embeddings = root / "embeddings.csv"
                manifest = root / "manifest.json"
                self.write_embeddings(embeddings)
                self.write_embedding_manifest(manifest, embeddings)
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if name == "non-object":
                    payload = []
                elif name == "invalid extractor hash":
                    payload[field] = "not-a-sha256"
                elif name == "missing extractor":
                    payload[field] = str(root / "missing_extractor.py")
                elif name == "invalid output hash":
                    payload[field] = "not-a-sha256"
                elif name == "different output":
                    payload[field] = str(root / "other_embeddings.csv")
                elif name == "wrong row count":
                    payload[field] = 3
                manifest.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(error_type, message):
                    read_label_blind_onharmony_embeddings(
                        embeddings,
                        expected_manifest=self.frame,
                        feature_prefix="emb_",
                        expected_dimension=2,
                        expected_pipeline_version="test_extractor_v1",
                        expected_model="TestModel",
                        expected_variant="canonical",
                        expected_readout="test readout",
                        source_manifest=manifest,
                    )

    def test_embedding_csv_rejects_duplicate_or_incomplete_feature_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.csv"
            duplicate.write_text(
                "file_path,emb_0,emb_0\n"
                "/derived/sub-001_ses-A_T1w_ready.nii.gz,1,2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate ON-Harmony CSV columns"):
                read_label_blind_onharmony_embeddings(
                    duplicate,
                    expected_manifest=self.frame,
                    feature_prefix="emb_",
                    expected_dimension=2,
                    expected_pipeline_version="test_extractor_v1",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                )

            incomplete = root / "incomplete.csv"
            pd.DataFrame(
                {
                    "file_path": [
                        "/derived/sub-001_ses-A_T1w_ready.nii.gz",
                        "/derived/sub-002_ses-B_T1w_ready.nii.gz",
                    ],
                    "emb_0": [1.0, 2.0],
                }
            ).to_csv(incomplete, index=False)
            with self.assertRaisesRegex(ValueError, "embedding stream differs"):
                read_label_blind_onharmony_embeddings(
                    incomplete,
                    expected_manifest=self.frame,
                    feature_prefix="emb_",
                    expected_dimension=2,
                    expected_pipeline_version="test_extractor_v1",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                )

    def test_embedding_csv_rejects_nonfinite_duplicate_or_unknown_acquisitions(self) -> None:
        cases = ("non-finite", "duplicate acquisition", "unknown acquisition")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                embeddings = root / "embeddings.csv"
                self.write_embeddings(embeddings)
                frame = pd.read_csv(embeddings)
                if case == "non-finite":
                    frame.loc[0, "emb_0"] = np.nan
                    message = "non-finite feature values"
                elif case == "duplicate acquisition":
                    frame.loc[0, "file_path"] = frame.loc[1, "file_path"]
                    message = "duplicate acquisitions"
                else:
                    frame.loc[0, "file_path"] = "/derived/sub-999_ses-Z_T1w_ready.nii.gz"
                    message = "canonical registry"
                frame.to_csv(embeddings, index=False)
                self.write_embedding_manifest(
                    embeddings.with_name("embeddings_manifest.json"), embeddings
                )

                with self.assertRaisesRegex(ValueError, message):
                    read_label_blind_onharmony_embeddings(
                        embeddings,
                        expected_manifest=self.frame,
                        feature_prefix="emb_",
                        expected_dimension=2,
                        expected_pipeline_version="test_extractor_v1",
                        expected_model="TestModel",
                        expected_variant="canonical",
                        expected_readout="test readout",
                    )

    def test_present_embedding_manifest_rejects_artifact_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            manifest = root / "manifest.json"
            self.write_embeddings(embeddings)
            self.write_embedding_manifest(manifest, embeddings)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["output_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "differs from its extractor manifest"):
                read_label_blind_onharmony_embeddings(
                    embeddings,
                    expected_manifest=self.frame,
                    feature_prefix="emb_",
                    expected_dimension=2,
                    expected_pipeline_version="test_extractor_v1",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                    source_manifest=manifest,
                )

    def test_present_embedding_manifest_rejects_extractor_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            manifest = root / "manifest.json"
            self.write_embeddings(embeddings)
            self.write_embedding_manifest(manifest, embeddings)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            Path(payload["extractor"]).write_text("# changed\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "extractor differs"):
                read_label_blind_onharmony_embeddings(
                    embeddings,
                    expected_manifest=self.frame,
                    feature_prefix="emb_",
                    expected_dimension=2,
                    expected_pipeline_version="test_extractor_v1",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                    source_manifest=manifest,
                )

    def test_complete_embedding_manifest_is_strictly_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            manifest = root / "manifest.json"
            self.write_embeddings(embeddings)
            self.write_embedding_manifest(manifest, embeddings)

            _, provenance = read_label_blind_onharmony_embeddings(
                embeddings,
                expected_manifest=self.frame,
                feature_prefix="emb_",
                expected_dimension=2,
                expected_pipeline_version="test_extractor_v1",
                expected_model="TestModel",
                expected_variant="canonical",
                expected_readout="test readout",
                source_manifest=manifest,
            )
            self.assertEqual(provenance["extractor_manifest"]["status"], "VALIDATED")
            self.assertEqual(
                provenance["extractor_manifest"]["sha256"], sha256_file(manifest)
            )

    def test_frozen_prediction_validates_lock_and_publishes_label_free_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            self.write_embeddings(embeddings)
            fit_code = {"fit_ridge.py": "exact"}
            run_dir = root / "run"
            self.write_locked_model(run_dir, fit_code)

            lock, _, _, integrity = validate_frozen_onharmony_lock(
                run_dir,
                expected_pipeline_version="test_pipeline",
                expected_model="TestModel",
                expected_variant="canonical",
                expected_readout="test readout",
                expected_dimension=2,
                expected_training_manifest_sha256="training",
                current_fit_code_sha256=fit_code,
            )
            self.assertEqual(lock["selected_alpha"], 1.0)
            self.assertEqual(
                integrity["training_lock_sha256"], sha256_file(run_dir / "LOCKED.json")
            )
            with self.assertRaisesRegex(ValueError, "Runtime Ridge code"):
                validate_frozen_onharmony_lock(
                    run_dir,
                    expected_pipeline_version="test_pipeline",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                    expected_dimension=2,
                    expected_training_manifest_sha256="training",
                    current_fit_code_sha256={"fit_ridge.py": "changed"},
                )

            output = root / "output"
            audit = self.run_prediction(
                run_dir=run_dir,
                embeddings=embeddings,
                output=output,
                fit_code=fit_code,
            )
            predictions = pd.read_csv(output / "predictions.tsv", sep="\t")
            self.assertEqual(len(predictions), 2)
            self.assertNotIn("age", predictions.columns)
            self.assertNotIn("age_true", predictions.columns)
            self.assertEqual(
                predictions["acquisition_id"].tolist(),
                self.frame["acquisition_id"].tolist(),
            )
            np.testing.assert_allclose(
                predictions["age_pred"].to_numpy(),
                np.asarray([100.0, 160.0]),
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(audit["status"], "ON_HARMONY_PREDICTED")
            self.assertFalse(audit["outcome_labels_loaded"])
            self.assertEqual(
                audit["predictions_artifact_sha256"],
                sha256_file(output / "predictions.tsv"),
            )
            published_audit = json.loads(
                (output / "prediction_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                published_audit["predictions_artifact_sha256"],
                audit["predictions_artifact_sha256"],
            )

    def test_frozen_lock_rejects_any_semantic_contract_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit_code = {"fit_ridge.py": "exact"}
            for field, observed in (
                ("status", "NOT_LOCKED"),
                ("pipeline_version", "wrong_pipeline"),
                ("arm", "adapted"),
                ("model", "OtherModel"),
                ("variant", "other_variant"),
                ("readout", "wrong readout"),
                ("internal_test_accessed", True),
                ("on_harmony_accessed", True),
                ("embedding_dim", 3),
            ):
                with self.subTest(field=field):
                    run_dir = root / field
                    self.write_locked_model(run_dir, fit_code)
                    lock_path = run_dir / "LOCKED.json"
                    lock = json.loads(lock_path.read_text(encoding="utf-8"))
                    lock[field] = observed
                    lock_path.write_text(json.dumps(lock), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, field):
                        validate_frozen_onharmony_lock(
                            run_dir,
                            expected_pipeline_version="test_pipeline",
                            expected_model="TestModel",
                            expected_variant="canonical",
                            expected_readout="test readout",
                            expected_dimension=2,
                            expected_training_manifest_sha256="training",
                            current_fit_code_sha256=fit_code,
                        )

    def test_frozen_lock_rejects_invalid_alpha_or_non_object_payload(self) -> None:
        for value in (None, True, 0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(alpha=value), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory) / "run"
                fit_code = {"fit_ridge.py": "exact"}
                self.write_locked_model(run_dir, fit_code)
                lock_path = run_dir / "LOCKED.json"
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                lock["selected_alpha"] = value
                lock_path.write_text(json.dumps(lock), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "selected_alpha"):
                    validate_frozen_onharmony_lock(
                        run_dir,
                        expected_pipeline_version="test_pipeline",
                        expected_model="TestModel",
                        expected_variant="canonical",
                        expected_readout="test readout",
                        expected_dimension=2,
                        expected_training_manifest_sha256="training",
                        current_fit_code_sha256=fit_code,
                    )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            fit_code = {"fit_ridge.py": "exact"}
            self.write_locked_model(run_dir, fit_code)
            (run_dir / "LOCKED.json").write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "not a JSON object"):
                validate_frozen_onharmony_lock(
                    run_dir,
                    expected_pipeline_version="test_pipeline",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                    expected_dimension=2,
                    expected_training_manifest_sha256="training",
                    current_fit_code_sha256=fit_code,
                )

    def test_frozen_lock_rejects_model_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            fit_code = {"fit_ridge.py": "exact"}
            self.write_locked_model(run_dir, fit_code)
            with (run_dir / MODEL_FILENAME).open("ab") as handle:
                handle.write(b"drift")
            with self.assertRaisesRegex(ValueError, "model hash differs"):
                validate_frozen_onharmony_lock(
                    run_dir,
                    expected_pipeline_version="test_pipeline",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                    expected_dimension=2,
                    expected_training_manifest_sha256="training",
                    current_fit_code_sha256=fit_code,
                )

    def test_frozen_lock_rejects_mutation_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            fit_code = {"fit_ridge.py": "exact"}
            self.write_locked_model(run_dir, fit_code)
            lock_path = run_dir / "LOCKED.json"
            real_json_loads = ridge_module.json.loads

            def load_then_mutate(payload, *args, **kwargs):
                result = real_json_loads(payload, *args, **kwargs)
                with lock_path.open("a", encoding="utf-8") as handle:
                    handle.write("\n")
                return result

            with patch.object(
                ridge_module.json, "loads", side_effect=load_then_mutate
            ), self.assertRaisesRegex(RuntimeError, "changed while it was being read"):
                validate_frozen_onharmony_lock(
                    run_dir,
                    expected_pipeline_version="test_pipeline",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                    expected_dimension=2,
                    expected_training_manifest_sha256="training",
                    current_fit_code_sha256=fit_code,
                )

    def test_frozen_lock_rejects_training_manifest_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            fit_code = {"fit_ridge.py": "exact"}
            self.write_locked_model(run_dir, fit_code)
            lock_path = run_dir / "LOCKED.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["manifest"]["manifest_sha256"] = "wrong_manifest"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest.manifest_sha256"):
                validate_frozen_onharmony_lock(
                    run_dir,
                    expected_pipeline_version="test_pipeline",
                    expected_model="TestModel",
                    expected_variant="canonical",
                    expected_readout="test readout",
                    expected_dimension=2,
                    expected_training_manifest_sha256="training",
                    current_fit_code_sha256=fit_code,
                )

    def test_lock_drift_before_publication_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            self.write_embeddings(embeddings)
            fit_code = {"fit_ridge.py": "exact"}
            run_dir = root / "run"
            self.write_locked_model(run_dir, fit_code)
            output = root / "output"
            original_atomic_tsv = ridge_module.atomic_tsv

            def write_then_mutate_lock(frame, path) -> None:
                original_atomic_tsv(frame, path)
                lock_path = run_dir / "LOCKED.json"
                lock_path.write_text(
                    lock_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
                )

            with patch(
                "FineTuning.shared.ridge.atomic_tsv", side_effect=write_then_mutate_lock
            ):
                with self.assertRaisesRegex(RuntimeError, "LOCKED.json changed"):
                    self.run_prediction(
                        run_dir=run_dir,
                        embeddings=embeddings,
                        output=output,
                        fit_code=fit_code,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.staging.*")), [])

    def test_model_drift_before_publication_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            self.write_embeddings(embeddings)
            fit_code = {"fit_ridge.py": "exact"}
            run_dir = root / "run"
            self.write_locked_model(run_dir, fit_code)
            output = root / "output"
            original_atomic_tsv = ridge_module.atomic_tsv

            def write_then_mutate_model(frame, path) -> None:
                original_atomic_tsv(frame, path)
                with (run_dir / MODEL_FILENAME).open("ab") as handle:
                    handle.write(b"drift")

            with patch(
                "FineTuning.shared.ridge.atomic_tsv", side_effect=write_then_mutate_model
            ):
                with self.assertRaisesRegex(RuntimeError, "model changed"):
                    self.run_prediction(
                        run_dir=run_dir,
                        embeddings=embeddings,
                        output=output,
                        fit_code=fit_code,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.staging.*")), [])

    def test_embedding_or_manifest_drift_before_publication_leaves_no_output(self) -> None:
        for target in ("embedding", "manifest", "extractor"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                embeddings = root / "embeddings.csv"
                self.write_embeddings(embeddings)
                manifest = embeddings.with_name("embeddings_manifest.json")
                fit_code = {"fit_ridge.py": "exact"}
                run_dir = root / "run"
                self.write_locked_model(run_dir, fit_code)
                output = root / "output"
                original_atomic_tsv = ridge_module.atomic_tsv

                def write_then_mutate(frame, path) -> None:
                    original_atomic_tsv(frame, path)
                    if target == "embedding":
                        artifact = embeddings
                    elif target == "manifest":
                        artifact = manifest
                    else:
                        payload = json.loads(manifest.read_text(encoding="utf-8"))
                        artifact = Path(payload["extractor"])
                    with artifact.open("a", encoding="utf-8") as handle:
                        handle.write("\n")

                with patch(
                    "FineTuning.shared.ridge.atomic_tsv",
                    side_effect=write_then_mutate,
                ), self.assertRaisesRegex(RuntimeError, "embedding"):
                    self.run_prediction(
                        run_dir=run_dir,
                        embeddings=embeddings,
                        output=output,
                        fit_code=fit_code,
                    )
                self.assertFalse(output.exists())
                self.assertEqual(list(root.glob(".output.staging.*")), [])

    def test_serialization_failure_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            self.write_embeddings(embeddings)
            fit_code = {"fit_ridge.py": "exact"}
            run_dir = root / "run"
            self.write_locked_model(run_dir, fit_code)
            output = root / "output"

            with patch(
                "FineTuning.shared.ridge.atomic_json",
                side_effect=OSError("simulated audit write failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated audit write failure"):
                    self.run_prediction(
                        run_dir=run_dir,
                        embeddings=embeddings,
                        output=output,
                        fit_code=fit_code,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.staging.*")), [])

    def test_existing_frozen_prediction_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            self.write_embeddings(embeddings)
            fit_code = {"fit_ridge.py": "exact"}
            run_dir = root / "run"
            self.write_locked_model(run_dir, fit_code)
            output = root / "output"
            output.mkdir()
            sentinel = output / "existing.txt"
            sentinel.write_text("immutable\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "not empty"):
                self.run_prediction(
                    run_dir=run_dir,
                    embeddings=embeddings,
                    output=output,
                    fit_code=fit_code,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "immutable\n")
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["existing.txt"])
            self.assertEqual(list(root.glob(".output.staging.*")), [])

    def test_final_directory_promotion_failure_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            self.write_embeddings(embeddings)
            fit_code = {"fit_ridge.py": "exact"}
            run_dir = root / "run"
            self.write_locked_model(run_dir, fit_code)
            output = root / "output"
            real_replace = ridge_module.os.replace

            def fail_directory_promotion(source, destination) -> None:
                if Path(source).is_dir():
                    raise OSError("simulated final promotion failure")
                real_replace(source, destination)

            with patch.object(
                ridge_module.os, "replace", side_effect=fail_directory_promotion
            ), self.assertRaisesRegex(OSError, "simulated final promotion failure"):
                self.run_prediction(
                    run_dir=run_dir,
                    embeddings=embeddings,
                    output=output,
                    fit_code=fit_code,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.staging.*")), [])


class ModelInputContractTests(unittest.TestCase):
    record = {
        "subject": "sub-001",
        "session": "A",
        "acquisition_id": "sub-001_ses-A_T1w",
        "raw_t1w_path": "/raw/sub-001_ses-A_T1w.nii.gz",
    }

    def test_model_specific_paths_are_acquisition_scoped(self) -> None:
        root = Path("/preproc")
        logs = Path("/logs")
        self.assertTrue(
            str(anatcl.artifact_paths(self.record, root, logs)["input"]).endswith(
                "sub-001_ses-A_T1w/sub-001_ses-A_T1w_anatcl_ready.nii.gz"
            )
        )
        self.assertTrue(
            str(yaware.artifact_paths(self.record, root, logs)["input"]).endswith(
                "sub-001_ses-A_T1w/sub-001_ses-A_T1w_yaware_ready.npy"
            )
        )
        self.assertTrue(
            str(brainiac.artifact_paths(self.record, root, logs)["input"]).endswith(
                "sub-001_ses-A_T1w/sub-001_ses-A_T1w_brainiac_ready.nii.gz"
            )
        )
        self.assertTrue(
            str(brainsegfounder.artifact_paths(self.record, root)["input"]).endswith(
                "sub-001_ses-A_T1w/sub-001_ses-A_T1w_brainsegfounder_ready.nii.gz"
            )
        )
        simclr_paths = neurosimclr.artifact_paths(self.record, root, logs)
        self.assertTrue(
            str(simclr_paths["input"]).endswith(
                "sub-001_ses-A_T1w/sub-001_ses-A_T1w_simclr_ready.nii.gz"
            )
        )
        self.assertTrue(
            str(simclr_paths["mask"]).endswith(
                "sub-001_ses-A_T1w/sub-001_ses-A_T1w_mask_mni.nii.gz"
            )
        )
        self.assertEqual(
            simclr_paths["preprocessing_log"],
            logs / "simclr_sub-001_ses-A_T1w.log",
        )
        self.assertEqual(
            curia2.artifact_paths(self.record)["input"],
            Path(self.record["raw_t1w_path"]),
        )

        neuro_artifact, neuro_sidecar = neurovfm.external_cache_paths(root, self.record)
        self.assertTrue(
            str(neuro_artifact).endswith(
                "sub-001_ses-A_T1w/study_preprocessor.pt"
            )
        )
        self.assertTrue(
            str(neuro_sidecar).endswith(
                "sub-001_ses-A_T1w/study_preprocessor.json"
            )
        )

    def test_neurovfm_cache_does_not_collide_between_sessions(self) -> None:
        first, _ = neurovfm.external_cache_paths("/cache", self.record)
        other = dict(self.record, session="B", acquisition_id="sub-001_ses-B_T1w")
        second, _ = neurovfm.external_cache_paths("/cache", other)
        self.assertNotEqual(first, second)

        unsafe = dict(self.record, acquisition_id="../outside")
        with self.assertRaisesRegex(ValueError, "Unsafe ON-Harmony acquisition"):
            neurovfm.external_cache_paths("/cache", unsafe)

    def test_anatcl_and_yaware_success_logs_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition = self.record["acquisition_id"]
            acquisition_root = root / acquisition
            acquisition_root.mkdir()
            anat_input = acquisition_root / f"{acquisition}_anatcl_ready.nii.gz"
            anat_input.write_bytes(b"anatcl")
            anat_log = root / "anat.log"
            anat_text = (
                "AnatCL preprocessing (IXI Dartel, vox=1.5)\n"
                "Saving CAT12 output (native shape, no resample)\n"
                "Shape: (121, 145, 121) Zooms: (1.5, 1.5, 1.5)\n"
                f"Done: /historical/root/{acquisition}/{anat_input.name}\n"
                "All done.\n"
            )
            anat_log.write_text(anat_text, encoding="utf-8")
            anat_batch = acquisition_root / "cat12_batch.m"
            anat_batch.write_text(
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
            anat_catlog = acquisition_root / "catlog.txt"
            anat_catlog.write_text(
                "CAT12.7 r1742\nDartel registration\n", encoding="utf-8"
            )
            anat_paths = {
                "input": anat_input,
                "preprocessing_log": anat_log,
                "cat12_batch": anat_batch,
                "cat12_runtime_log": anat_catlog,
            }
            result = anatcl.validate_artifact_set(self.record, anat_paths)
            self.assertEqual(result["preprocessing_status"], "PASS")

            yaware_input = acquisition_root / f"{acquisition}_yaware_ready.npy"
            yaware_input.write_bytes(b"yaware")
            yaware_log = root / "yaware.log"
            yaware_text = (
                "y-Aware preprocessing (IXI Dartel, vox=1.5)\n"
                "Z-score normalize + save as .npy\n"
                "shape=(121, 145, 121)\nshape=(1, 121, 145, 121)\n"
                f"Done: /historical/root/{acquisition}/{yaware_input.name}  "
                "shape=(1, 121, 145, 121)\n"
                "All done.\n"
            )
            yaware_log.write_text(yaware_text, encoding="utf-8")
            yaware_paths = {"input": yaware_input, "preprocessing_log": yaware_log}
            result = yaware.validate_artifact_set(self.record, yaware_paths)
            self.assertEqual(result["preprocessing_status"], "PASS")

            for validator, paths, log_path, valid_text in (
                (anatcl.validate_artifact_set, anat_paths, anat_log, anat_text),
                (yaware.validate_artifact_set, yaware_paths, yaware_log, yaware_text),
            ):
                with self.subTest(validator=validator.__module__, failure="wrong acquisition"):
                    log_path.write_text(
                        valid_text.replace(f"/{acquisition}/", "/other_acquisition/"),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "Done:"):
                        validator(self.record, paths)
                with self.subTest(validator=validator.__module__, failure="incomplete"):
                    log_path.write_text(
                        valid_text.replace("All done.", "Not completed."),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "All done"):
                        validator(self.record, paths)

    def test_3dneurosimclr_log_is_bound_to_raw_input_output_and_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisition = self.record["acquisition_id"]
            acquisition_root = root / acquisition
            acquisition_root.mkdir()
            input_path = acquisition_root / f"{acquisition}_simclr_ready.nii.gz"
            input_path.write_bytes(b"simclr")
            mask_path = acquisition_root / f"{acquisition}_mask_mni.nii.gz"
            mask_path.write_bytes(b"mask")
            log_path = root / "simclr.log"
            valid_log = (
                "3D-Neuro-SimCLR preprocessing (TurboPrep-equivalent)\n"
                f"raw={self.record['raw_t1w_path']}\n"
                f"Template SHA-256: {neurosimclr.EXPECTED_TEMPLATE_SHA256}\n"
                "[Preprocessing 5/5] WhiteStripe normalization\n"
                f"Done: /historical/root/{acquisition}/{input_path.name}\n"
                "All done.\n"
                "exit_code: 0\n"
            )
            paths = {
                "input": input_path,
                "mask": mask_path,
                "preprocessing_log": log_path,
            }
            log_path.write_text(valid_log, encoding="utf-8")
            result = neurosimclr.validate_artifact_set(self.record, paths)
            self.assertEqual(result["preprocessing_status"], "PASS")
            self.assertEqual(
                result["template_sha256"], neurosimclr.EXPECTED_TEMPLATE_SHA256
            )

            invalid_logs = {
                "raw": valid_log.replace(
                    str(self.record["raw_t1w_path"]), "/raw/other_T1w.nii.gz"
                ),
                "output": valid_log.replace(input_path.name, "other_simclr_ready.nii.gz"),
                "template": valid_log.replace(
                    neurosimclr.EXPECTED_TEMPLATE_SHA256, "0" * 64
                ),
                "exit": valid_log.replace("exit_code: 0", "exit_code: 1"),
                "whitestripe": valid_log + "WhiteStripe failed\n",
            }
            for failure, invalid_log in invalid_logs.items():
                with self.subTest(failure=failure):
                    log_path.write_text(invalid_log, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "not a successful corrected run"):
                        neurosimclr.validate_artifact_set(self.record, paths)

    def test_neurovfm_cache_metadata_is_bound_to_raw_artifact_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.nii.gz"
            record = dict(
                self.record,
                dataset="ON-Harmony",
                raw_t1w_path=str(raw_path.resolve()),
            )
            official_contract = {
                "cache_version": neurovfm.PREPROCESSING_CACHE_VERSION,
                "contract_sha256": "fixed-contract",
            }

            class FakeModel:
                def preprocessing_cache_paths(self, current_record):
                    return neurovfm.external_cache_paths(root / "cache", current_record)

                def input_path(self, current_record):
                    return Path(str(current_record["raw_t1w_path"])).resolve()

                def load_cached_preprocessing(self, current_record, *, retain_in_memory):
                    self.last_load = (current_record, retain_in_memory)
                    return {"img": np.zeros((7, 1024), dtype=np.float32)}

                def preprocessing_cache_contract(self):
                    return official_contract

            model = FakeModel()
            paths = neurovfm.artifact_paths(model, record)
            paths["input"].parent.mkdir(parents=True)

            def write_valid_cache() -> dict[str, object]:
                raw_path.write_bytes(b"raw-neurovfm")
                paths["input"].write_bytes(b"cached-neurovfm")
                metadata = {
                    "status": "PASS",
                    "cache_version": neurovfm.PREPROCESSING_CACHE_VERSION,
                    "preprocessing_contract": official_contract,
                    "dataset": "ON-Harmony",
                    "subject": record["subject"],
                    "raw_source": {
                        "path": record["raw_t1w_path"],
                        "sha256": sha256_file(raw_path),
                    },
                    "artifact_sha256": sha256_file(paths["input"]),
                }
                paths["cache_metadata"].write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                return metadata

            write_valid_cache()
            result = neurovfm.validate_artifact_set(model, record, paths)
            self.assertEqual(result["preprocessing_status"], "PASS")
            self.assertEqual(result["preprocessing_contract_sha256"], "fixed-contract")
            self.assertEqual(result["n_tokens"], 7)

            for failure in ("subject", "contract", "artifact", "raw"):
                with self.subTest(failure=failure):
                    metadata = write_valid_cache()
                    if failure == "subject":
                        metadata["subject"] = "sub-other"
                    elif failure == "contract":
                        metadata["preprocessing_contract"] = {
                            **official_contract,
                            "contract_sha256": "other-contract",
                        }
                    elif failure == "artifact":
                        paths["input"].write_bytes(b"changed-cache")
                    else:
                        raw_path.write_bytes(b"changed-raw")
                    if failure in {"subject", "contract"}:
                        paths["cache_metadata"].write_text(
                            json.dumps(metadata), encoding="utf-8"
                        )
                    with self.assertRaisesRegex(
                        ValueError, "Invalid ON-Harmony NeuroVFM cache"
                    ):
                        neurovfm.validate_artifact_set(model, record, paths)

    def test_brainiac_qc_is_bound_to_raw_input_and_success_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "ready.nii.gz"
            input_path.write_bytes(b"brainiac")
            qc_path = root / "qc.json"
            qc_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "raw_input": self.record["raw_t1w_path"],
                        "brainiac_shape": [96, 96, 96],
                        "shape_ok": True,
                        "norm_ok": True,
                        "brain_ok": True,
                    }
                ),
                encoding="utf-8",
            )
            log_path = root / "log"
            log_path.write_text(
                f"script   : {brainiac.HISTORICAL_PREPROCESSOR_SHA256}\n"
                "[Preprocessing 4-5/5] Resample 96\nstatus=PASS\n"
                "All done.\nexit_code: 0\n",
                encoding="utf-8",
            )
            result = brainiac.validate_artifact_set(
                self.record,
                {
                    "input": input_path,
                    "qc": qc_path,
                    "preprocessing_log": log_path,
                },
            )
            self.assertEqual(result["preprocessing_status"], "PASS")
            self.assertEqual(result["preprocessing_provenance_schema"], "historical")

            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            qc["raw_input"] = "/raw/other_T1w.nii.gz"
            qc_path.write_text(json.dumps(qc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw_input"):
                brainiac.validate_artifact_set(
                    self.record,
                    {
                        "input": input_path,
                        "qc": qc_path,
                        "preprocessing_log": log_path,
                    },
                )

            qc["raw_input"] = self.record["raw_t1w_path"]
            qc_path.write_text(json.dumps(qc), encoding="utf-8")
            log_path.write_text("status=PASS\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generator SHA-256"):
                brainiac.validate_artifact_set(
                    self.record,
                    {
                        "input": input_path,
                        "qc": qc_path,
                        "preprocessing_log": log_path,
                    },
                )

    def test_brainiac_current_qc_is_bound_to_integrated_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.nii.gz"
            raw_path.write_bytes(b"raw-brainiac")
            record = dict(self.record, raw_t1w_path=str(raw_path.resolve()))
            input_path = root / "ready.nii.gz"
            input_path.write_bytes(b"brainiac-current")
            brain_path = root / "brain.nii.gz"
            brain_path.write_bytes(b"brainiac-brain")
            qc_png = root / "qc.png"
            qc_png.write_bytes(b"brainiac-qc")
            raw_stat = raw_path.stat()
            qc_path = root / "qc.json"
            qc_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "raw_input": record["raw_t1w_path"],
                        "brainiac_shape": [96, 96, 96],
                        "shape_ok": True,
                        "norm_ok": True,
                        "brain_ok": True,
                        "pipeline_version": brainiac.PREPROCESSING_PIPELINE_VERSION,
                        "pipeline_contract_sha256": (
                            brainiac.PREPROCESSING_CONTRACT_SHA256
                        ),
                        "script_sha256": brainiac.EXPECTED_SOURCE_HASHES[
                            "preprocessor"
                        ],
                        "raw_fingerprint": {
                            "path": record["raw_t1w_path"],
                            "size_bytes": raw_stat.st_size,
                            "mtime_ns": raw_stat.st_mtime_ns,
                        },
                        "output_sha256": {
                            "brain": sha256_file(brain_path),
                            "ready": sha256_file(input_path),
                            "qc_png": sha256_file(qc_png),
                        },
                    }
                ),
                encoding="utf-8",
            )
            log_path = root / "log"
            log_path.write_text(
                f"script   : {brainiac.EXPECTED_SOURCE_HASHES['preprocessor']}\n"
                "[Preprocessing 4-5/5] Resample 96\nstatus=PASS\n"
                "All done.\nexit_code: 0\n",
                encoding="utf-8",
            )
            result = brainiac.validate_artifact_set(
                record,
                {
                    "input": input_path,
                    "brain": brain_path,
                    "qc": qc_path,
                    "qc_png": qc_png,
                    "preprocessing_log": log_path,
                },
            )
            self.assertEqual(result["preprocessing_status"], "PASS")
            self.assertEqual(result["preprocessing_provenance_schema"], "current")

            paths = {
                "input": input_path,
                "brain": brain_path,
                "qc": qc_path,
                "qc_png": qc_png,
                "preprocessing_log": log_path,
            }

            qc_png.write_bytes(b"changed-qc")
            with self.assertRaisesRegex(ValueError, "output_sha256"):
                brainiac.validate_artifact_set(record, paths)

            qc_png.write_bytes(b"brainiac-qc")
            input_path.write_bytes(b"changed-ready")
            with self.assertRaisesRegex(ValueError, "output_sha256"):
                brainiac.validate_artifact_set(record, paths)

            input_path.write_bytes(b"brainiac-current")
            valid_log = log_path.read_text(encoding="utf-8")
            log_path.write_text(
                valid_log.replace("All done.", "Not completed."), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "successful audited run"):
                brainiac.validate_artifact_set(record, paths)

            log_path.write_text(valid_log, encoding="utf-8")
            raw_path.write_bytes(b"changed-raw-brainiac")
            with self.assertRaisesRegex(ValueError, "raw_fingerprint"):
                brainiac.validate_artifact_set(record, paths)

    def test_brainsegfounder_qc_uses_actual_v4_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.nii.gz"
            raw_path.write_bytes(b"raw-brainsegfounder")
            record = dict(self.record, raw_t1w_path=str(raw_path.resolve()))
            raw_stat = raw_path.stat()
            input_path = root / "ready.nii.gz"
            input_path.write_bytes(b"brainsegfounder")
            qc_path = root / "qc.json"
            qc_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "pipeline_version": brainsegfounder.INPUT_CONTRACT_VERSION,
                        "pipeline_contract_sha256": brainsegfounder.INPUT_CONTRACT_SHA256,
                        "script_sha256": brainsegfounder.EXPECTED_SOURCE_HASHES[
                            "preprocessor"
                        ],
                        "raw_input": {
                            "path": record["raw_t1w_path"],
                            "size_bytes": raw_stat.st_size,
                            "mtime_ns": raw_stat.st_mtime_ns,
                        },
                        "output_sha256": {"ready": sha256_file(input_path)},
                        "model_view": {"shape": [96, 96, 96]},
                        "registration_recovery": {"used": True},
                    }
                ),
                encoding="utf-8",
            )
            log_path = root / "log"
            log_path.write_text(
                f"pipeline_version={brainsegfounder.INPUT_CONTRACT_VERSION}\n"
                f"contract={brainsegfounder.INPUT_CONTRACT_SHA256}\n"
                "command=/fsl/bin/fnirt args\n"
                "command=/fsl/bin/convertwarp args\n"
                "command=/fsl/bin/applywarp args\n",
                encoding="utf-8",
            )
            result = brainsegfounder.validate_artifact_set(
                record,
                {
                    "input": input_path,
                    "qc": qc_path,
                    "preprocessing_log": log_path,
                },
            )
            self.assertTrue(result["registration_recovery_used"])

            paths = {
                "input": input_path,
                "qc": qc_path,
                "preprocessing_log": log_path,
            }
            input_path.write_bytes(b"changed-brainsegfounder")
            with self.assertRaisesRegex(ValueError, "output_sha256.ready"):
                brainsegfounder.validate_artifact_set(record, paths)

            input_path.write_bytes(b"brainsegfounder")
            valid_qc = json.loads(qc_path.read_text(encoding="utf-8"))
            invalid_qc = dict(valid_qc, pipeline_version="wrong-version")
            qc_path.write_text(json.dumps(invalid_qc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pipeline_version"):
                brainsegfounder.validate_artifact_set(record, paths)

            qc_path.write_text(json.dumps(valid_qc), encoding="utf-8")
            valid_log = log_path.read_text(encoding="utf-8")
            log_path.write_text(
                valid_log.replace("/bin/applywarp ", "/bin/not_applywarp "),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "successful v4 run"):
                brainsegfounder.validate_artifact_set(record, paths)

            log_path.write_text(valid_log, encoding="utf-8")
            raw_path.write_bytes(b"changed-raw-brainsegfounder")
            with self.assertRaisesRegex(ValueError, "raw_input"):
                brainsegfounder.validate_artifact_set(record, paths)


if __name__ == "__main__":
    unittest.main()
