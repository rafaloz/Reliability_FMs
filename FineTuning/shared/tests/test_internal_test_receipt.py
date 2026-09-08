"""Tests for exclusive internal-test evaluation receipts."""

from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from FineTuning.shared.cohort import PROJECT_ROOT, sha256_file
from FineTuning.shared.embedding_bundle import (
    EMBEDDING_BUNDLE_SCHEMA_VERSION,
    publish_embedding_bundle,
)
from FineTuning.shared.metrics import regression_metrics
from FineTuning.shared.test_receipt import (
    COMPLETED_STATUS,
    EMBEDDING_RECEIPT_ROOT,
    EVALUATION_COMPLETION_FIELDS,
    EVALUATION_STARTED_FIELDS,
    EXTRACTION_COMPLETED_STATUS,
    EXTRACTION_COMPLETION_FIELDS,
    EXTRACTION_STARTED_STATUS,
    EXTRACTION_STARTED_FIELDS,
    RESULT_BUNDLE_PROTOCOL,
    STARTED_STATUS,
    complete_internal_test_embedding_receipt,
    complete_internal_test_receipt,
    internal_test_embedding_receipt_path,
    prepare_internal_test_output_dir,
    publish_internal_test_result_bundle,
    reserve_internal_test_embedding_receipt,
    reserve_internal_test_receipt,
    verify_internal_test_input_integrity,
)


MODEL_DIRECTORIES = (
    "AnatCL",
    "BrainIAC",
    "BrainSegFounder",
    "Curia2",
    "NeuroVFM",
    "yAware",
    "3DNeuroSimCLR",
)


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    """Parse one top-level function so tests inspect executable syntax, not text."""

    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {name} function in {path}, found {len(matches)}")
    return matches[0]


def _call_name(node: ast.Call) -> str | None:
    """Return the terminal name of a direct or attribute function call."""

    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _single_call(function: ast.FunctionDef, name: str) -> ast.Call:
    """Require exactly one named call inside the selected function body."""

    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]
    if len(calls) != 1:
        raise AssertionError(
            f"Expected one {name} call in {function.name}, found {len(calls)}"
        )
    return calls[0]


def _literal_dict_items(node: ast.AST) -> dict[str, ast.AST]:
    """Return a literal string-keyed dictionary or reject computed/unpacked fields."""

    if not isinstance(node, ast.Dict):
        raise AssertionError("Receipt payload is not a literal dictionary")
    items: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise AssertionError("Receipt payload contains a computed or unpacked field")
        if key.value in items:
            raise AssertionError(f"Receipt payload repeats field {key.value!r}")
        items[key.value] = value
    return items


def _call_payload(call: ast.Call, position: int) -> dict[str, ast.AST]:
    """Extract and validate one positional literal-dict call argument."""

    if len(call.args) <= position:
        raise AssertionError(f"Call at line {call.lineno} lacks argument {position}")
    return _literal_dict_items(call.args[position])


def _assigned_dict(function: ast.FunctionDef, name: str) -> dict[str, ast.AST]:
    """Return the unique literal dictionary assigned to a local variable."""

    matches = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one assignment to {name} in {function.name}, found {len(matches)}"
        )
    return _literal_dict_items(matches[0])


def _string_constants(node: ast.AST) -> set[str]:
    """Collect string literals from executable syntax while excluding comments."""

    return {
        candidate.value
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
    }


def _is_internal_test_guarded(function: ast.FunctionDef, node: ast.AST) -> bool:
    """Return whether a node is in the true branch of `args.split == internal_test`."""

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(function):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If) and child in parent.body:
            test = parent.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Attribute)
                and isinstance(test.left.value, ast.Name)
                and test.left.value.id == "args"
                and test.left.attr == "split"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "internal_test"
            ):
                return True
        child = parent
    return False


def _uses_composite_subject_identity(node: ast.AST) -> bool:
    """Recognize `predictions[[dataset, subject]].drop_duplicates()` in an expression."""

    for candidate in ast.walk(node):
        if not (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr == "drop_duplicates"
            and isinstance(candidate.func.value, ast.Subscript)
            and isinstance(candidate.func.value.value, ast.Name)
            and candidate.func.value.value.id == "predictions"
        ):
            continue
        selection = candidate.func.value.slice
        if isinstance(selection, ast.List):
            columns = [
                item.value
                for item in selection.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
            if columns == ["dataset", "subject"]:
                return True
    return False


def _prediction_frame(root: Path) -> pd.DataFrame:
    """Return a complete two-subject prediction artifact for publication tests."""

    frame = pd.DataFrame(
        {
            "dataset": ["DatasetA", "DatasetB"],
            "subject": ["sub-01", "sub-01"],
            "split": ["internal_test", "internal_test"],
            "age_true": [40.0, 60.0],
            "sex": ["F", "M"],
            "input_path": [str(root / "raw-a.nii.gz"), str(root / "raw-b.nii.gz")],
            "age_pred": [42.0, 57.0],
        }
    )
    frame["error_years"] = frame["age_pred"] - frame["age_true"]
    frame["absolute_error_years"] = frame["error_years"].abs()
    return frame


def _result_payload(root: Path, predictions: pd.DataFrame) -> dict[str, object]:
    """Create a complete result record tied to four real immutable inputs."""

    artifacts = {
        "training_lock_artifact": root / "LOCKED.json",
        "model_artifact": root / "ridge_model.npz",
        "test_embeddings_artifact": root / "embeddings_internal_test.tsv",
        "test_embeddings_audit_artifact": root
        / "embeddings_internal_test.tsv.audit.json",
    }
    for field, path in artifacts.items():
        path.write_text(f"{field}\n", encoding="utf-8")
    manifest_path = root / "manifest.tsv"
    manifest_path.write_text("dataset\tsubject\n", encoding="utf-8")
    metrics = regression_metrics(predictions["age_true"], predictions["age_pred"])
    metrics.update({"split": "internal_test", "bias_correction": "none"})
    return {
        "pipeline_version": "test_pipeline_v1",
        "arm": "frozen",
        "model": "TestModel",
        "variant": "canonical",
        "metrics": metrics,
        "training_lock_artifact": str(artifacts["training_lock_artifact"]),
        "training_lock_sha256": sha256_file(artifacts["training_lock_artifact"]),
        "model_artifact": str(artifacts["model_artifact"]),
        "model_artifact_sha256": sha256_file(artifacts["model_artifact"]),
        "test_embeddings_artifact": str(artifacts["test_embeddings_artifact"]),
        "test_embeddings_sha256": sha256_file(
            artifacts["test_embeddings_artifact"]
        ),
        "test_embeddings_audit_artifact": str(
            artifacts["test_embeddings_audit_artifact"]
        ),
        "test_embeddings_audit_sha256": sha256_file(
            artifacts["test_embeddings_audit_artifact"]
        ),
        "manifest": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "split_counts": {
                "train": 4,
                "validation": 2,
                "internal_test": len(predictions),
            },
            "on_harmony_firewall": "PASS",
        },
        "n_predictions": len(predictions),
        "n_subjects": len(
            predictions[["dataset", "subject"]].drop_duplicates()
        ),
        "bias_correction": "none",
        "internal_test_accessed": True,
        "on_harmony_accessed": False,
    }


def _evaluation_started_payload(
    root: Path, result: dict[str, object], output_dir: Path
) -> dict[str, object]:
    """Bind an evaluation reservation to its frozen artifacts and destination."""

    manifest = result["manifest"]
    assert isinstance(manifest, dict)
    return {
        "pipeline_version": result["pipeline_version"],
        "arm": "frozen",
        "model": result["model"],
        "variant": result["variant"],
        "run_dir": str(root),
        "training_lock_artifact": result["training_lock_artifact"],
        "training_lock_sha256": result["training_lock_sha256"],
        "model_artifact": result["model_artifact"],
        "model_artifact_sha256": result["model_artifact_sha256"],
        "manifest_artifact": manifest["manifest"],
        "manifest_sha256": manifest["manifest_sha256"],
        "test_embeddings_artifact": result["test_embeddings_artifact"],
        "test_embeddings_audit_artifact": result[
            "test_embeddings_audit_artifact"
        ],
        "output_dir": str(output_dir),
        "on_harmony_accessed": False,
    }


def _evaluation_completion_payload(
    started: dict[str, object], result: dict[str, object], output_dir: Path
) -> dict[str, object]:
    """Return a completion bound to one committed result directory."""

    result_path = output_dir / "metrics_internal_test.json"
    return {
        "arm": started["arm"],
        "model": started["model"],
        "variant": started["variant"],
        "training_lock_artifact": started["training_lock_artifact"],
        "training_lock_sha256": started["training_lock_sha256"],
        "test_embeddings_sha256": result["test_embeddings_sha256"],
        "test_embeddings_audit_sha256": result[
            "test_embeddings_audit_sha256"
        ],
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "on_harmony_accessed": False,
    }


def _embedding_started_payload(root: Path) -> dict[str, object]:
    """Return a complete reservation for one internal-test embedding bundle."""

    manifest = root / "embedding_manifest.tsv"
    manifest.write_text("dataset\tsubject\n", encoding="utf-8")
    return {
        "pipeline_version": "embedding_test_v1",
        "model": "TestModel",
        "variant": "canonical",
        "readout": "test readout",
        "embedding_dim": 1,
        "split": "internal_test",
        "manifest_artifact": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "requested_artifact": str(root / "embeddings.tsv"),
        "requested_sidecar": str(root / "embeddings.tsv.audit.json"),
        "requested_n_subjects": 1,
        "on_harmony_accessed": False,
    }


def _publish_internal_embedding_bundle(
    root: Path, receipt: Path, started: dict[str, object]
) -> dict[str, object]:
    """Publish one complete bundle carrying the active receipt attempt."""

    frame = pd.DataFrame(
        {
            "dataset": ["DatasetA"],
            "subject": ["sub-01"],
            "split": ["internal_test"],
            "age": [50.0],
            "sex": ["F"],
            "input_path": [str(root / "raw.nii.gz")],
            "model": [started["model"]],
            "variant": [started["variant"]],
            "emb_0": [1.25],
        }
    )
    manifest = {
        "manifest": started["manifest_artifact"],
        "manifest_sha256": started["manifest_sha256"],
        "n_subjects": 1,
        "split_counts": {"train": 0, "validation": 0, "internal_test": 1},
        "split_dataset_counts": {
            "train": {},
            "validation": {},
            "internal_test": {"DatasetA": 1},
        },
        "age_min": 50.0,
        "age_max": 50.0,
        "input_contract": {"required_runtime_condition": "test"},
        "raw_input_files_checked": False,
        "missing_raw_input_files": 0,
        "on_harmony_firewall": "PASS",
    }
    identity = {
        "model": started["model"],
        "variant": started["variant"],
        "readout": started["readout"],
        "embedding_dim": started["embedding_dim"],
    }
    payload = {
        "manifest_schema_version": EMBEDDING_BUNDLE_SCHEMA_VERSION,
        "pipeline_version": started["pipeline_version"],
        "arm": "frozen_embedding_extraction",
        **identity,
        "split": "internal_test",
        "n_subjects": 1,
        "artifact": started["requested_artifact"],
        "elapsed_seconds": 1.0,
        "adapter": {
            **identity,
            "input_contract": {"required_runtime_condition": "test"},
        },
        "frozen_boundary": {
            **identity,
            "encoder_total_parameters": 1,
            "encoder_trainable_parameters": 0,
            "encoder_trainable_fraction": 0.0,
            "trainable_components": [],
            "trainable_parameter_names": [],
        },
        "manifest": manifest,
        "release": {"checkpoint_sha256": "b" * 64},
        "code_sha256": {"extract_embeddings.py": "c" * 64},
        "software": {"python": "test"},
        "runtime": {"device": "cpu", "seed": 0, "input_hashes_verified": True},
        "internal_test_accessed": True,
        "internal_test_receipt": {
            "path": str(receipt),
            "attempt_id": started["attempt_id"],
        },
        "on_harmony_accessed": False,
    }
    return publish_embedding_bundle(
        frame,
        started["requested_artifact"],
        started["requested_sidecar"],
        payload,
    )


class InternalTestReceiptTests(unittest.TestCase):
    def test_final_input_integrity_guard_rejects_each_artifact_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "LOCKED.json": root / "LOCKED.json",
                "Ridge model": root / "model.npz",
                "embedding table": root / "embeddings.tsv",
                "embedding sidecar": root / "embeddings.tsv.audit.json",
            }
            for label, path in paths.items():
                path.write_text(f"{label}\n", encoding="utf-8")
            hashes = {label: sha256_file(path) for label, path in paths.items()}

            def verify(**updates: str) -> None:
                verify_internal_test_input_integrity(
                    lock_path=paths["LOCKED.json"],
                    expected_lock_sha256=updates.get(
                        "LOCKED.json", hashes["LOCKED.json"]
                    ),
                    model_path=paths["Ridge model"],
                    expected_model_sha256=updates.get(
                        "Ridge model", hashes["Ridge model"]
                    ),
                    embeddings_path=paths["embedding table"],
                    expected_embeddings_sha256=updates.get(
                        "embedding table", hashes["embedding table"]
                    ),
                    sidecar_path=paths["embedding sidecar"],
                    expected_sidecar_sha256=updates.get(
                        "embedding sidecar", hashes["embedding sidecar"]
                    ),
                )

            verify()
            for label in paths:
                with self.subTest(artifact=label), self.assertRaisesRegex(
                    RuntimeError, label
                ):
                    verify(**{label: "0" * 64})

            with self.assertRaisesRegex(ValueError, "SHA-256 is invalid"):
                verify(**{"LOCKED.json": "ABC"})
            with self.assertRaisesRegex(ValueError, "alias"):
                verify_internal_test_input_integrity(
                    lock_path=paths["LOCKED.json"],
                    expected_lock_sha256=hashes["LOCKED.json"],
                    model_path=paths["LOCKED.json"],
                    expected_model_sha256=hashes["LOCKED.json"],
                    embeddings_path=paths["embedding table"],
                    expected_embeddings_sha256=hashes["embedding table"],
                    sidecar_path=paths["embedding sidecar"],
                    expected_sidecar_sha256=hashes["embedding sidecar"],
                )
            with self.assertRaisesRegex(ValueError, "path must be absolute"):
                verify_internal_test_input_integrity(
                    lock_path="LOCKED.json",
                    expected_lock_sha256=hashes["LOCKED.json"],
                    model_path=paths["Ridge model"],
                    expected_model_sha256=hashes["Ridge model"],
                    embeddings_path=paths["embedding table"],
                    expected_embeddings_sha256=hashes["embedding table"],
                    sidecar_path=paths["embedding sidecar"],
                    expected_sidecar_sha256=hashes["embedding sidecar"],
                )

    def test_internal_test_result_bundle_is_published_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = prepare_internal_test_output_dir(root / "evaluation")
            predictions = _prediction_frame(root)
            payload = _result_payload(root, predictions)
            committed = publish_internal_test_result_bundle(
                predictions,
                destination,
                payload,
            )

            predictions_path = destination / "predictions_internal_test.tsv"
            metrics_path = destination / "metrics_internal_test.json"
            self.assertTrue(predictions_path.is_file())
            self.assertTrue(metrics_path.is_file())
            self.assertEqual(
                committed["predictions_artifact_sha256"],
                sha256_file(predictions_path),
            )
            self.assertEqual(
                committed["publication"],
                {
                    "protocol": RESULT_BUNDLE_PROTOCOL,
                    "status": "COMMITTED",
                    "atomic_directory_promotion": True,
                    "exclusive_destination_reservation": True,
                    "post_publication_verified": True,
                },
            )
            self.assertEqual(
                json.loads(metrics_path.read_text(encoding="utf-8")), committed
            )
            self.assertEqual(list(root.glob(".evaluation.internal-test.*")), [])
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare_internal_test_output_dir(destination)

    def test_internal_test_result_failure_leaves_no_promoted_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = prepare_internal_test_output_dir(root / "evaluation")
            predictions = _prediction_frame(root)
            payload = _result_payload(root, predictions)
            with (
                patch(
                    "FineTuning.shared.test_receipt.json.dumps",
                    side_effect=RuntimeError("simulated metrics failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated metrics failure"),
            ):
                publish_internal_test_result_bundle(
                    predictions,
                    destination,
                    payload,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".evaluation.internal-test.*")), [])

    def test_result_bundle_rejects_prediction_and_payload_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = _prediction_frame(root)
            payload = _result_payload(root, predictions)

            cases = []
            wrong_error = predictions.copy()
            wrong_error.loc[0, "error_years"] = 999.0
            cases.append((wrong_error, payload, "signed errors"))
            duplicate = predictions.copy()
            duplicate.loc[1, ["dataset", "subject"]] = ["DatasetA", "sub-01"]
            cases.append((duplicate, payload, "duplicate subject"))
            wrong_count = {**payload, "n_subjects": 1}
            cases.append((predictions, wrong_count, "subject count"))
            wrong_metric = {
                **payload,
                "metrics": {**payload["metrics"], "mae_years": 999.0},
            }
            cases.append((predictions, wrong_metric, "mae_years"))
            bad_firewall = predictions.copy()
            bad_firewall.loc[0, "dataset"] = "ON-Harmony"
            cases.append((bad_firewall, payload, "firewall"))

            for index, (frame, result, message) in enumerate(cases):
                with self.subTest(case=message), self.assertRaisesRegex(
                    ValueError, message
                ):
                    publish_internal_test_result_bundle(
                        frame, root / f"evaluation-{index}", result
                    )

    def test_result_bundle_reservation_is_exclusive_and_released(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "evaluation"
            reservation = root / ".evaluation.internal-test.lock"
            reservation.write_text("occupied\n", encoding="utf-8")
            predictions = _prediction_frame(root)
            payload = _result_payload(root, predictions)
            with self.assertRaisesRegex(FileExistsError, "already reserved"):
                publish_internal_test_result_bundle(
                    predictions, destination, payload
                )
            self.assertFalse(destination.exists())
            self.assertEqual(reservation.read_text(encoding="utf-8"), "occupied\n")

    def test_post_publication_failure_rolls_back_result_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "evaluation"
            predictions = _prediction_frame(root)
            payload = _result_payload(root, predictions)
            with (
                patch(
                    "FineTuning.shared.test_receipt.pd.read_csv",
                    side_effect=RuntimeError("simulated verification failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "verification failure"),
            ):
                publish_internal_test_result_bundle(
                    predictions, destination, payload
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".evaluation.internal-test.*")), [])

    def test_reservation_is_exclusive_and_completion_preserves_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "INTERNAL_TEST_EVALUATED.json"
            output_dir = root / "evaluation"
            predictions = _prediction_frame(root)
            result = _result_payload(root, predictions)
            started_payload = _evaluation_started_payload(
                root, result, output_dir
            )
            started = reserve_internal_test_receipt(
                receipt,
                started_payload,
            )
            self.assertEqual(started["status"], STARTED_STATUS)
            self.assertTrue(started["internal_test_accessed"])
            with self.assertRaisesRegex(FileExistsError, "already has"):
                reserve_internal_test_receipt(receipt, started_payload)

            publish_internal_test_result_bundle(predictions, output_dir, result)
            completed = complete_internal_test_receipt(
                receipt,
                started,
                _evaluation_completion_payload(started, result, output_dir),
            )
            self.assertEqual(completed["status"], COMPLETED_STATUS)
            self.assertEqual(completed["attempt_id"], started["attempt_id"])
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8")), completed
            )

    def test_tampered_started_receipt_cannot_be_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "INTERNAL_TEST_EVALUATED.json"
            predictions = _prediction_frame(root)
            result = _result_payload(root, predictions)
            output_dir = root / "evaluation"
            started = reserve_internal_test_receipt(
                receipt, _evaluation_started_payload(root, result, output_dir)
            )
            tampered = {**started, "model": "other"}
            receipt.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after exclusive"):
                complete_internal_test_receipt(receipt, started, {})

    def test_embedding_extraction_receipt_is_exclusive_and_operation_specific(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "INTERNAL_TEST_EMBEDDINGS_EXTRACTED.json"
            started_payload = _embedding_started_payload(root)
            started = reserve_internal_test_embedding_receipt(
                receipt,
                started_payload,
            )
            self.assertEqual(started["status"], EXTRACTION_STARTED_STATUS)
            self.assertEqual(started["operation"], "embedding extraction")
            with self.assertRaisesRegex(FileExistsError, "already has"):
                reserve_internal_test_embedding_receipt(receipt, started_payload)

            sidecar = _publish_internal_embedding_bundle(root, receipt, started)
            sidecar_path = Path(started["requested_sidecar"])
            completed = complete_internal_test_embedding_receipt(
                receipt,
                started,
                {
                    "artifact": started["requested_artifact"],
                    "artifact_sha256": sidecar["artifact_sha256"],
                    "sidecar": str(sidecar_path),
                    "sidecar_sha256": sha256_file(sidecar_path),
                    "n_subjects": 1,
                },
            )
            self.assertEqual(completed["status"], EXTRACTION_COMPLETED_STATUS)
            self.assertEqual(completed["attempt_id"], started["attempt_id"])

    def test_receipt_payloads_cannot_override_identity_or_system_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = _prediction_frame(root)
            result = _result_payload(root, predictions)
            output_dir = root / "evaluation"
            receipt = root / "INTERNAL_TEST_EVALUATED.json"
            started_payload = _evaluation_started_payload(root, result, output_dir)
            with self.assertRaisesRegex(ValueError, "protected fields"):
                reserve_internal_test_receipt(
                    receipt, {**started_payload, "attempt_id": "0" * 32}
                )
            with self.assertRaisesRegex(ValueError, "fields differ"):
                incomplete = dict(started_payload)
                incomplete.pop("model_artifact")
                reserve_internal_test_receipt(receipt, incomplete)

            started = reserve_internal_test_receipt(receipt, started_payload)
            publish_internal_test_result_bundle(predictions, output_dir, result)
            completion = _evaluation_completion_payload(started, result, output_dir)
            with self.assertRaisesRegex(ValueError, "changes STARTED identity"):
                complete_internal_test_receipt(
                    receipt, started, {**completion, "model": "OtherModel"}
                )
            with self.assertRaisesRegex(ValueError, "protected fields"):
                complete_internal_test_receipt(
                    receipt, started, {**completion, "status": COMPLETED_STATUS}
                )
            completed = complete_internal_test_receipt(
                receipt, started, completion
            )
            self.assertEqual(completed["model"], started["model"])

    def test_completion_rejects_artifact_drift_and_legacy_temporary_is_harmless(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = _prediction_frame(root)
            result = _result_payload(root, predictions)
            output_dir = root / "evaluation"
            receipt = root / "INTERNAL_TEST_EVALUATED.json"
            started = reserve_internal_test_receipt(
                receipt, _evaluation_started_payload(root, result, output_dir)
            )
            publish_internal_test_result_bundle(predictions, output_dir, result)
            completion = _evaluation_completion_payload(started, result, output_dir)
            result_path = output_dir / "metrics_internal_test.json"
            original = result_path.read_text(encoding="utf-8")
            result_path.write_text(original + " ", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed before receipt"):
                complete_internal_test_receipt(receipt, started, completion)
            result_path.write_text(original, encoding="utf-8")
            legacy_temporary = receipt.with_name(
                receipt.name + f".tmp.{os.getpid()}"
            )
            legacy_temporary.write_text("stale\n", encoding="utf-8")
            completed = complete_internal_test_receipt(
                receipt, started, completion
            )
            self.assertEqual(completed["status"], COMPLETED_STATUS)
            self.assertEqual(legacy_temporary.read_text(encoding="utf-8"), "stale\n")

    def test_evaluation_completion_rejects_each_postreservation_artifact_drift(
        self,
    ) -> None:
        cases = (
            ("training_lock_artifact", "LOCKED.json"),
            ("model_artifact", "Ridge model"),
            ("test_embeddings_artifact", "embedding table"),
            ("test_embeddings_audit_artifact", "embedding sidecar"),
            ("manifest_artifact", "cohort manifest"),
            ("result", "internal-test metrics"),
        )
        for field, label in cases:
            with self.subTest(artifact=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                predictions = _prediction_frame(root)
                result = _result_payload(root, predictions)
                output_dir = root / "evaluation"
                receipt = root / "INTERNAL_TEST_EVALUATED.json"
                started = reserve_internal_test_receipt(
                    receipt, _evaluation_started_payload(root, result, output_dir)
                )
                publish_internal_test_result_bundle(predictions, output_dir, result)
                completion = _evaluation_completion_payload(started, result, output_dir)
                artifact = (
                    Path(completion[field])
                    if field == "result"
                    else Path(started[field])
                )
                artifact.write_bytes(artifact.read_bytes() + b" ")
                with self.assertRaisesRegex(RuntimeError, f"{label} changed before"):
                    complete_internal_test_receipt(receipt, started, completion)
                self.assertEqual(
                    json.loads(receipt.read_text(encoding="utf-8"))["status"],
                    STARTED_STATUS,
                )
                self.assertFalse(
                    (root / f".{receipt.name}.completion.lock").exists()
                )

    def test_completion_lock_is_exclusive_without_consuming_started_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = _prediction_frame(root)
            result = _result_payload(root, predictions)
            output_dir = root / "evaluation"
            receipt = root / "INTERNAL_TEST_EVALUATED.json"
            started = reserve_internal_test_receipt(
                receipt, _evaluation_started_payload(root, result, output_dir)
            )
            publish_internal_test_result_bundle(predictions, output_dir, result)
            completion = _evaluation_completion_payload(started, result, output_dir)
            completion_lock = root / f".{receipt.name}.completion.lock"
            completion_lock.write_text("occupied\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already reserved"):
                complete_internal_test_receipt(receipt, started, completion)
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8"))["status"],
                STARTED_STATUS,
            )
            self.assertEqual(completion_lock.read_text(encoding="utf-8"), "occupied\n")
            completion_lock.unlink()
            completed = complete_internal_test_receipt(
                receipt, started, completion
            )
            self.assertEqual(completed["status"], COMPLETED_STATUS)

    def test_embedding_completion_rejects_artifact_sidecar_and_manifest_drift(
        self,
    ) -> None:
        cases = (
            ("requested_artifact", "embedding artifact"),
            ("requested_sidecar", "embedding sidecar"),
            ("manifest_artifact", "cohort manifest"),
        )
        for field, label in cases:
            with self.subTest(artifact=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt = root / "INTERNAL_TEST_EMBEDDINGS_EXTRACTED.json"
                started = reserve_internal_test_embedding_receipt(
                    receipt, _embedding_started_payload(root)
                )
                sidecar = _publish_internal_embedding_bundle(root, receipt, started)
                sidecar_path = Path(started["requested_sidecar"])
                completion = {
                    "artifact": started["requested_artifact"],
                    "artifact_sha256": sidecar["artifact_sha256"],
                    "sidecar": str(sidecar_path),
                    "sidecar_sha256": sha256_file(sidecar_path),
                    "n_subjects": 1,
                }
                artifact = Path(started[field])
                artifact.write_bytes(artifact.read_bytes() + b" ")
                with self.assertRaisesRegex(RuntimeError, f"{label} changed before"):
                    complete_internal_test_embedding_receipt(
                        receipt, started, completion
                    )
                self.assertEqual(
                    json.loads(receipt.read_text(encoding="utf-8"))["status"],
                    EXTRACTION_STARTED_STATUS,
                )
                self.assertFalse(
                    (root / f".{receipt.name}.completion.lock").exists()
                )

    def test_embedding_receipt_path_is_central_and_variant_scoped(self) -> None:
        fold0 = internal_test_embedding_receipt_path("AnatCL", "fold0")
        fold1 = internal_test_embedding_receipt_path("AnatCL", "fold1")
        self.assertEqual(fold0.parent, EMBEDDING_RECEIPT_ROOT)
        self.assertEqual(fold0.name, "AnatCL__fold0.json")
        self.assertNotEqual(fold0, fold1)
        with self.assertRaisesRegex(ValueError, "Invalid internal-test receipt label"):
            internal_test_embedding_receipt_path("AnatCL", "..")
        with self.assertRaisesRegex(ValueError, "Invalid internal-test receipt label"):
            internal_test_embedding_receipt_path("AnatCL", "fold 0")
        with self.assertRaisesRegex(ValueError, "Invalid internal-test receipt label"):
            internal_test_embedding_receipt_path("AnatCL", 0)  # type: ignore[arg-type]

    def test_all_frozen_evaluators_reserve_before_reading_test_embeddings(self) -> None:
        for model_directory in MODEL_DIRECTORIES:
            with self.subTest(model=model_directory):
                path = PROJECT_ROOT / "FineTuning" / model_directory / "fit_ridge.py"
                function = _function_node(path, "command_evaluate")
                calls = {
                    name: _single_call(function, name)
                    for name in (
                        "load_and_validate_manifest",
                        "fit_code_hashes",
                        "prepare_internal_test_output_dir",
                        "reserve_internal_test_receipt",
                        "read_embedding_table",
                        "verify_internal_test_input_integrity",
                        "publish_internal_test_result_bundle",
                        "complete_internal_test_receipt",
                    )
                }
                ordered = (
                    "load_and_validate_manifest",
                    "fit_code_hashes",
                    "prepare_internal_test_output_dir",
                    "reserve_internal_test_receipt",
                    "read_embedding_table",
                    "verify_internal_test_input_integrity",
                    "publish_internal_test_result_bundle",
                    "complete_internal_test_receipt",
                )
                self.assertEqual(
                    [calls[name].lineno for name in ordered],
                    sorted(calls[name].lineno for name in ordered),
                )

                started = _call_payload(calls["reserve_internal_test_receipt"], 1)
                completion = _call_payload(calls["complete_internal_test_receipt"], 2)
                self.assertEqual(set(started), set(EVALUATION_STARTED_FIELDS))
                self.assertEqual(set(completion), set(EVALUATION_COMPLETION_FIELDS))
                self.assertEqual(ast.unparse(started["pipeline_version"]), "PIPELINE_VERSION")
                self.assertEqual(ast.unparse(started["model_artifact"]), "str(model_path)")
                self.assertEqual(
                    ast.unparse(started["manifest_artifact"]),
                    "str(summary['manifest'])",
                )

                result = _assigned_dict(function, "result")
                self.assertIn("n_predictions", result)
                self.assertIn("n_subjects", result)
                self.assertNotIn("internal_test_subjects", result)
                self.assertTrue(_uses_composite_subject_identity(result["n_subjects"]))
                self.assertIn("training_lock_artifact", result)
                self.assertIn("training_lock_sha256", result)

                hash_function = _function_node(path, "fit_code_hashes")
                hash_call = _single_call(hash_function, "code_hashes")
                self.assertIn("test_receipt.py", _string_constants(hash_call))

    def test_all_frozen_extractors_reserve_before_encoding_test_images(self) -> None:
        for model_directory in MODEL_DIRECTORIES:
            with self.subTest(model=model_directory):
                path = (
                    PROJECT_ROOT
                    / "FineTuning"
                    / model_directory
                    / "extract_embeddings.py"
                )
                function = _function_node(path, "command_extract")
                calls = {
                    name: _single_call(function, name)
                    for name in (
                        "internal_test_embedding_receipt_path",
                        "reserve_internal_test_embedding_receipt",
                        "encode",
                        "publish_embedding_bundle",
                        "complete_internal_test_embedding_receipt",
                    )
                }
                ordered = (
                    "internal_test_embedding_receipt_path",
                    "reserve_internal_test_embedding_receipt",
                    "encode",
                    "publish_embedding_bundle",
                    "complete_internal_test_embedding_receipt",
                )
                self.assertEqual(
                    [calls[name].lineno for name in ordered],
                    sorted(calls[name].lineno for name in ordered),
                )
                self.assertTrue(
                    _is_internal_test_guarded(
                        function, calls["reserve_internal_test_embedding_receipt"]
                    )
                )
                self.assertTrue(
                    _is_internal_test_guarded(
                        function, calls["complete_internal_test_embedding_receipt"]
                    )
                )

                started = _call_payload(
                    calls["reserve_internal_test_embedding_receipt"], 1
                )
                completion = _call_payload(
                    calls["complete_internal_test_embedding_receipt"], 2
                )
                self.assertEqual(set(started), set(EXTRACTION_STARTED_FIELDS))
                self.assertEqual(set(completion), set(EXTRACTION_COMPLETION_FIELDS))
                self.assertEqual(ast.unparse(started["readout"]), "model.readout")
                self.assertEqual(
                    ast.unparse(started["embedding_dim"]), "model.embedding_dim"
                )
                self.assertIsInstance(started["split"], ast.Constant)
                self.assertEqual(started["split"].value, "internal_test")
                self.assertEqual(
                    ast.unparse(started["manifest_artifact"]),
                    "str(manifest['manifest'])",
                )
                self.assertEqual(
                    ast.unparse(started["requested_n_subjects"]),
                    "int(len(selected))",
                )
                self.assertEqual(
                    ast.unparse(completion["sidecar_sha256"]),
                    "sha256_file(embedding_sidecar_path(output))",
                )

                hash_function = _function_node(path, "runtime_code_hashes")
                hash_call = _single_call(hash_function, "code_hashes")
                self.assertIn("test_receipt.py", _string_constants(hash_call))


if __name__ == "__main__":
    unittest.main()
