"""Tests for transactional embedding-table publication."""

from __future__ import annotations

import ast
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from FineTuning.shared.cohort import PROJECT_ROOT, sha256_file
from FineTuning.shared.embedding_bundle import (
    COMMITTED_PUBLICATION,
    EMBEDDING_BUNDLE_SCHEMA_VERSION,
    PUBLICATION_PROTOCOL,
    ensure_new_embedding_bundle,
    load_embedding_bundle_sidecar,
    publish_embedding_bundle,
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
EMBEDDING_BUNDLE_MODULE = "FineTuning.shared.embedding_bundle"
REQUIRED_BUNDLE_FIELDS = {
    "manifest_schema_version",
    "pipeline_version",
    "arm",
    "model",
    "variant",
    "readout",
    "split",
    "n_subjects",
    "embedding_dim",
    "artifact",
    "elapsed_seconds",
    "adapter",
    "frozen_boundary",
    "manifest",
    "release",
    "code_sha256",
    "software",
    "runtime",
    "internal_test_accessed",
    "internal_test_receipt",
    "on_harmony_accessed",
}
REQUIRED_MANIFEST_FIELDS = {
    "manifest",
    "manifest_sha256",
    "n_subjects",
    "split_counts",
    "split_dataset_counts",
    "age_min",
    "age_max",
    "input_contract",
    "raw_input_files_checked",
    "missing_raw_input_files",
    "on_harmony_firewall",
}


def model_script(model_directory: str, filename: str) -> tuple[str, ast.Module]:
    """Read and parse one model entry point so comments cannot satisfy tests."""

    source = (PROJECT_ROOT / "FineTuning" / model_directory / filename).read_text(
        encoding="utf-8"
    )
    return source, ast.parse(source)


def imported_names(tree: ast.AST, module: str) -> set[str]:
    """Return names imported directly from one exact Python module."""

    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def direct_calls(tree: ast.AST, function: str) -> list[ast.Call]:
    """Return executable calls made through an unqualified function name."""

    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function
    ]


def string_literals(tree: ast.AST) -> set[str]:
    """Collect string literals from executable syntax, excluding comments."""

    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def dictionary_key_sets(tree: ast.AST) -> list[set[str]]:
    """Return literal string-key sets for every dictionary in a module."""

    return [
        {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
    ]


def path_literal_sequences(tree: ast.AST) -> list[tuple[str, ...]]:
    """Recover literal components from pathlib-style division expressions."""

    def components(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return components(node.left) + components(node.right)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return (node.value,)
        return ()

    return [
        components(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
    ]


def call_references_name(call: ast.Call, name: str) -> bool:
    """Return whether any positional or keyword argument references a name."""

    arguments: list[ast.AST] = [*call.args, *(item.value for item in call.keywords)]
    return any(
        isinstance(node, ast.Name) and node.id == name
        for argument in arguments
        for node in ast.walk(argument)
    )


def complete_payload(*, pipeline_version: str = "test_pipeline_v1") -> dict:
    """Return the smallest complete frozen-embedding provenance contract."""

    return {
        "manifest_schema_version": EMBEDDING_BUNDLE_SCHEMA_VERSION,
        "pipeline_version": pipeline_version,
        "arm": "frozen_embedding_extraction",
        "model": "test",
        "variant": "canonical",
        "readout": "test readout",
        "split": "train",
        "n_subjects": 1,
        "embedding_dim": 1,
        "artifact": "must-be-replaced",
        "elapsed_seconds": 1.0,
        "adapter": {
            "model": "test",
            "variant": "canonical",
            "readout": "test readout",
            "embedding_dim": 1,
            "input_contract": {"required_runtime_condition": "test"},
        },
        "frozen_boundary": {
            "model": "test",
            "variant": "canonical",
            "readout": "test readout",
            "embedding_dim": 1,
            "encoder_total_parameters": 1,
            "encoder_trainable_parameters": 0,
            "encoder_trainable_fraction": 0.0,
            "trainable_components": [],
            "trainable_parameter_names": [],
        },
        "manifest": {
            "manifest": "/authoritative/manifest.tsv",
            "manifest_sha256": "a" * 64,
            "n_subjects": 1,
            "split_counts": {"train": 1, "validation": 0, "internal_test": 0},
            "split_dataset_counts": {
                "train": {"test": 1},
                "validation": {},
                "internal_test": {},
            },
            "age_min": 50.0,
            "age_max": 50.0,
            "input_contract": {"required_runtime_condition": "test"},
            "raw_input_files_checked": False,
            "missing_raw_input_files": 0,
            "on_harmony_firewall": "PASS",
        },
        "release": {"checkpoint_sha256": "b" * 64},
        "code_sha256": {"extract_embeddings.py": "c" * 64},
        "software": {"python": "test"},
        "runtime": {"device": "cpu", "seed": 0, "input_hashes_verified": True},
        "internal_test_accessed": False,
        "internal_test_receipt": None,
        "on_harmony_accessed": False,
    }


def complete_frame(*, split: str = "train") -> pd.DataFrame:
    """Return the smallest structurally valid frozen-embedding table."""

    return pd.DataFrame(
        {
            "dataset": ["test"],
            "subject": ["sub-1"],
            "split": [split],
            "age": [50.0],
            "sex": ["F"],
            "input_path": ["/inputs/sub-1.nii.gz"],
            "model": ["test"],
            "variant": ["canonical"],
            "emb_0": [1.25],
        }
    )


def internal_test_payload(receipt: object) -> dict:
    """Return a coherent internal-test payload with a configurable receipt."""

    payload = complete_payload()
    payload["split"] = "internal_test"
    payload["internal_test_accessed"] = True
    payload["internal_test_receipt"] = receipt
    payload["manifest"]["split_counts"] = {
        "train": 0,
        "validation": 0,
        "internal_test": 1,
    }
    payload["manifest"]["split_dataset_counts"] = {
        "train": {},
        "validation": {},
        "internal_test": {"test": 1},
    }
    return payload


def load_complete_bundle(output: Path, sidecar: Path, *, pipeline: str) -> dict:
    """Load a test bundle under its exact semantic identity."""

    return load_embedding_bundle_sidecar(
        output,
        sidecar,
        expected_pipeline_version=pipeline,
        expected_model="test",
        expected_variant="canonical",
        expected_readout="test readout",
        expected_split="train",
        expected_dimension=1,
    )


class EmbeddingBundleTests(unittest.TestCase):
    def test_all_extractors_use_and_hash_the_common_publisher(self) -> None:
        for model_directory in MODEL_DIRECTORIES:
            with self.subTest(model=model_directory):
                _, tree = model_script(model_directory, "extract_embeddings.py")
                self.assertIn(
                    "publish_embedding_bundle",
                    imported_names(tree, EMBEDDING_BUNDLE_MODULE),
                )
                self.assertTrue(direct_calls(tree, "publish_embedding_bundle"))
                self.assertIn(
                    ("FineTuning", "shared", "embedding_bundle.py"),
                    path_literal_sequences(tree),
                )
                direct_table_writes = [
                    call
                    for call in direct_calls(tree, "atomic_tsv")
                    if call_references_name(call, "embeddings")
                ]
                self.assertEqual(direct_table_writes, [])

    def test_all_ridge_fitters_use_and_hash_the_common_reader(self) -> None:
        for model_directory in MODEL_DIRECTORIES:
            with self.subTest(model=model_directory):
                _, tree = model_script(model_directory, "fit_ridge.py")
                self.assertIn(
                    "load_embedding_bundle_sidecar",
                    imported_names(tree, EMBEDDING_BUNDLE_MODULE),
                )
                reader_calls = direct_calls(tree, "load_embedding_bundle_sidecar")
                self.assertTrue(reader_calls)
                self.assertTrue(
                    any(
                        keyword.arg == "expected_readout"
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id == "READOUT"
                        for call in reader_calls
                        for keyword in call.keywords
                    )
                )
                self.assertIn(
                    ("FineTuning", "shared", "embedding_bundle.py"),
                    path_literal_sequences(tree),
                )
                self.assertIn(
                    "Embedding bundle changed while its table was being read",
                    string_literals(tree),
                )
                direct_sidecar_json_reads = [
                    call
                    for call in ast.walk(tree)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "json"
                    and call.func.attr == "loads"
                    and call_references_name(call, "sidecar_path")
                ]
                self.assertEqual(direct_sidecar_json_reads, [])

    def test_all_extractors_publish_schema_model_variant_and_readout(self) -> None:
        required_fields = {
            "manifest_schema_version",
            "model",
            "variant",
            "readout",
            "artifact",
            "elapsed_seconds",
            "adapter",
            "frozen_boundary",
            "manifest",
            "release",
            "code_sha256",
            "software",
            "runtime",
            "internal_test_accessed",
            "internal_test_receipt",
            "on_harmony_accessed",
        }
        for model_directory in MODEL_DIRECTORIES:
            with self.subTest(model=model_directory):
                _, tree = model_script(model_directory, "extract_embeddings.py")
                self.assertTrue(
                    any(
                        required_fields <= key_set
                        for key_set in dictionary_key_sets(tree)
                    ),
                    f"No complete embedding sidecar dictionary in {model_directory}",
                )

    def test_successful_bundle_uses_the_sidecar_as_commit_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embeddings.tsv"
            sidecar = Path(str(output) + ".audit.json")
            frame = complete_frame()

            committed = publish_embedding_bundle(
                frame,
                output,
                sidecar,
                complete_payload(),
            )

            observed = json.loads(sidecar.read_text(encoding="utf-8"))
            pd.testing.assert_frame_equal(
                pd.read_csv(output, sep="\t"),
                frame,
            )
            self.assertEqual(observed, committed)
            self.assertEqual(observed["artifact"], str(output.resolve()))
            self.assertEqual(observed["artifact_sha256"], sha256_file(output))
            self.assertEqual(observed["publication"]["protocol"], PUBLICATION_PROTOCOL)
            self.assertEqual(observed["publication"]["status"], "COMMITTED")
            self.assertEqual(
                load_complete_bundle(output, sidecar, pipeline="test_pipeline_v1"),
                committed,
            )

    def test_reader_rejects_uncommitted_wrong_pipeline_or_tampered_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embeddings.tsv"
            sidecar = Path(str(output) + ".audit.json")
            publish_embedding_bundle(
                complete_frame(),
                output,
                sidecar,
                complete_payload(pipeline_version="expected_v1"),
            )
            original = json.loads(sidecar.read_text(encoding="utf-8"))

            invalid_publications = (
                None,
                {**COMMITTED_PUBLICATION, "protocol": "wrong"},
                {**COMMITTED_PUBLICATION, "status": "INCOMPLETE"},
                {**COMMITTED_PUBLICATION, "sidecar_is_commit_marker": False},
            )
            for publication in invalid_publications:
                with self.subTest(publication=publication):
                    sidecar.write_text(
                        json.dumps({**original, "publication": publication}) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "not a committed bundle"):
                        load_complete_bundle(output, sidecar, pipeline="expected_v1")

            sidecar.write_text(json.dumps(original) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pipeline version differs"):
                load_complete_bundle(output, sidecar, pipeline="different_v2")

            output.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                load_complete_bundle(output, sidecar, pipeline="expected_v1")

    def test_reader_rejects_malformed_sidecars_and_artifact_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embeddings.tsv"
            sidecar = Path(str(output) + ".audit.json")
            publish_embedding_bundle(
                complete_frame(), output, sidecar, complete_payload()
            )
            original = json.loads(sidecar.read_text(encoding="utf-8"))

            sidecar.write_text("{\n", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_complete_bundle(output, sidecar, pipeline="test_pipeline_v1")

            sidecar.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "not a JSON object"):
                load_complete_bundle(output, sidecar, pipeline="test_pipeline_v1")

            wrong_artifact = {**original, "artifact": str(output.with_name("other.tsv"))}
            sidecar.write_text(json.dumps(wrong_artifact) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "points to a different table"):
                load_complete_bundle(output, sidecar, pipeline="test_pipeline_v1")

            wrong_digest = {**original, "artifact_sha256": "d" * 64}
            sidecar.write_text(json.dumps(wrong_digest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                load_complete_bundle(output, sidecar, pipeline="test_pipeline_v1")

    def test_reader_rejects_model_variant_readout_split_or_dimension_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embeddings.tsv"
            sidecar = Path(str(output) + ".audit.json")
            publish_embedding_bundle(
                complete_frame(),
                output,
                sidecar,
                complete_payload(),
            )
            original = json.loads(sidecar.read_text(encoding="utf-8"))
            for key, value in (
                ("model", "wrong"),
                ("variant", "wrong"),
                ("readout", "wrong"),
                ("split", "validation"),
                ("embedding_dim", 2),
            ):
                with self.subTest(key=key):
                    mutated = copy.deepcopy(original)
                    mutated[key] = value
                    if key in {"model", "variant", "readout", "embedding_dim"}:
                        mutated["adapter"][key] = value
                        mutated["frozen_boundary"][key] = value
                    if key == "split":
                        mutated["manifest"] = {
                            **original["manifest"],
                            "split_counts": {
                                "train": 0,
                                "validation": 1,
                                "internal_test": 0,
                            },
                            "split_dataset_counts": {
                                "train": {},
                                "validation": {"test": 1},
                                "internal_test": {},
                            },
                        }
                    sidecar.write_text(
                        json.dumps(mutated) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "semantic identity differs"):
                        load_complete_bundle(
                            output, sidecar, pipeline="test_pipeline_v1"
                        )

    def test_publisher_rejects_incomplete_semantics_and_row_count_drift(self) -> None:
        frame = complete_frame()
        invalid_payloads: list[tuple[dict, str]] = []
        for key in REQUIRED_BUNDLE_FIELDS:
            payload = complete_payload()
            del payload[key]
            invalid_payloads.append((payload, "lacks required semantic fields"))
        payload = complete_payload()
        payload["manifest"] = {"manifest_sha256": "a" * 64}
        invalid_payloads.append((payload, "cohort manifest is incomplete"))
        payload = complete_payload()
        payload["manifest"]["manifest_sha256"] = "not-a-hash"
        invalid_payloads.append((payload, "manifest SHA-256 is invalid"))
        payload = complete_payload()
        payload["n_subjects"] = 2
        payload["manifest"]["split_counts"]["train"] = 2
        payload["manifest"]["split_dataset_counts"]["train"]["test"] = 2
        payload["manifest"]["n_subjects"] = 2
        invalid_payloads.append((payload, "table row count differs"))

        for index, (payload, message) in enumerate(invalid_payloads):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "embeddings.tsv"
                sidecar = Path(str(output) + ".audit.json")
                with self.assertRaisesRegex(ValueError, message):
                    publish_embedding_bundle(frame, output, sidecar, payload)
                self.assertFalse(output.exists())
                self.assertFalse(sidecar.exists())

    def test_publisher_rejects_false_frozen_identity_and_malformed_receipts(self) -> None:
        frame = complete_frame()
        invalid_payloads: list[tuple[pd.DataFrame, dict, str]] = []

        payload = complete_payload()
        payload["code_sha256"]["extract_embeddings.py"] = int("1" * 64)
        invalid_payloads.append((frame, payload, "code hashes are incomplete"))

        for container in ("adapter", "frozen_boundary"):
            for key, value in (
                ("model", "wrong"),
                ("variant", "wrong"),
                ("readout", "wrong"),
                ("embedding_dim", 2),
            ):
                payload = complete_payload()
                payload[container][key] = value
                invalid_payloads.append(
                    (frame, payload, "semantic identity differs")
                )

        for key, value, message in (
            ("encoder_total_parameters", 0, "invalid encoder size"),
            ("encoder_trainable_parameters", 1, "contains trainable parameters"),
            ("encoder_trainable_fraction", 0.1, "has a trainable fraction"),
            ("trainable_components", ["encoder.layer4"], "non-empty"),
            (
                "trainable_parameter_names",
                ["encoder.layer4.weight"],
                "non-empty",
            ),
        ):
            payload = complete_payload()
            payload["frozen_boundary"][key] = value
            invalid_payloads.append((frame, payload, message))

        for receipt in (
            {"path": 1, "attempt_id": "a" * 32},
            {"path": "/receipt.json", "attempt_id": 1},
            {"path": "relative/receipt.json", "attempt_id": "a" * 32},
            {"path": "/receipt.json", "attempt_id": "not-a-uuid"},
        ):
            invalid_payloads.append(
                (
                    complete_frame(split="internal_test"),
                    internal_test_payload(receipt),
                    "invalid reservation",
                )
            )

        for index, (case_frame, payload, message) in enumerate(invalid_payloads):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "embeddings.tsv"
                sidecar = Path(str(output) + ".audit.json")
                with self.assertRaisesRegex(ValueError, message):
                    publish_embedding_bundle(case_frame, output, sidecar, payload)
                self.assertFalse(output.exists())
                self.assertFalse(sidecar.exists())

    def test_publisher_enforces_runtime_and_dataset_access_firewalls(self) -> None:
        valid_receipt = {"path": "/receipt.json", "attempt_id": "a" * 32}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "internal_test.tsv"
            sidecar = Path(str(output) + ".audit.json")
            committed = publish_embedding_bundle(
                complete_frame(split="internal_test"),
                output,
                sidecar,
                internal_test_payload(valid_receipt),
            )
            self.assertEqual(committed["internal_test_receipt"], valid_receipt)

        invalid_cases: list[tuple[pd.DataFrame, dict, str]] = []

        payload = complete_payload()
        payload["runtime"]["input_hashes_verified"] = False
        invalid_cases.append(
            (complete_frame(), payload, "did not verify its model inputs")
        )

        payload = complete_payload()
        payload["internal_test_accessed"] = "false"
        invalid_cases.append((complete_frame(), payload, "flag is invalid"))

        payload = complete_payload()
        payload["internal_test_accessed"] = True
        invalid_cases.append((complete_frame(), payload, "flag differs from its split"))

        payload = complete_payload()
        payload["on_harmony_accessed"] = True
        invalid_cases.append((complete_frame(), payload, "must not access ON-Harmony"))

        payload = complete_payload()
        payload["internal_test_receipt"] = valid_receipt
        invalid_cases.append(
            (complete_frame(), payload, "contains an internal-test receipt")
        )

        invalid_cases.append(
            (
                complete_frame(split="internal_test"),
                internal_test_payload(None),
                "lacks its reservation",
            )
        )

        for index, (frame, payload, message) in enumerate(invalid_cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "embeddings.tsv"
                sidecar = Path(str(output) + ".audit.json")
                with self.assertRaisesRegex(ValueError, message):
                    publish_embedding_bundle(frame, output, sidecar, payload)
                self.assertFalse(output.exists())
                self.assertFalse(sidecar.exists())

    def test_publisher_rejects_incomplete_or_inconsistent_cohort_manifest(self) -> None:
        frame = complete_frame()
        invalid_payloads: list[tuple[dict, str]] = []

        for key in REQUIRED_MANIFEST_FIELDS:
            payload = complete_payload()
            del payload["manifest"][key]
            invalid_payloads.append((payload, "cohort manifest is incomplete"))

        for key, value, message in (
            ("manifest", "relative/manifest.tsv", "manifest path is invalid"),
            ("manifest_sha256", "not-a-hash", "manifest SHA-256 is invalid"),
            ("on_harmony_firewall", "FAIL", "firewall did not pass"),
            ("n_subjects", True, "subject count is invalid"),
            ("n_subjects", 0, "subject count is invalid"),
            ("age_min", "50", "age range is invalid"),
            ("age_max", True, "age range is invalid"),
            ("age_min", float("nan"), "age range is invalid"),
            ("age_min", 51.0, "age range is invalid"),
            ("raw_input_files_checked", "false", "audit flag is invalid"),
            ("missing_raw_input_files", -1, "audit count is invalid"),
        ):
            payload = complete_payload()
            payload["manifest"][key] = value
            invalid_payloads.append((payload, message))

        for split_counts, message in (
            ({"train": 1}, "does not contain exactly three splits"),
            (
                {"train": True, "validation": 0, "internal_test": 0},
                "split counts are invalid",
            ),
            (
                {"train": -1, "validation": 1, "internal_test": 1},
                "split counts are invalid",
            ),
            (
                {"train": 2, "validation": 0, "internal_test": 0},
                "split counts do not sum to its total",
            ),
        ):
            payload = complete_payload()
            payload["manifest"]["split_counts"] = split_counts
            invalid_payloads.append((payload, message))

        payload = complete_payload()
        payload["manifest"]["split_dataset_counts"] = []
        invalid_payloads.append((payload, "dataset counts are invalid"))

        payload = complete_payload()
        payload["manifest"]["split_dataset_counts"] = {"train": {"test": 1}}
        invalid_payloads.append((payload, "split/dataset keys differ"))

        for dataset_counts, message in (
            (
                {
                    "train": {"": 1},
                    "validation": {},
                    "internal_test": {},
                },
                "dataset counts are invalid",
            ),
            (
                {
                    "train": {"test": 0},
                    "validation": {},
                    "internal_test": {},
                },
                "dataset counts are invalid",
            ),
            (
                {
                    "train": {"test": 2},
                    "validation": {},
                    "internal_test": {},
                },
                "dataset counts do not sum to their split",
            ),
        ):
            payload = complete_payload()
            payload["manifest"]["split_dataset_counts"] = dataset_counts
            invalid_payloads.append((payload, message))

        for contract in (
            {},
            {"required_runtime_condition": ""},
            {"unknown_contract": "test"},
            {
                "required_runtime_condition": "test",
                "required_offline_group": "test",
            },
        ):
            payload = complete_payload()
            payload["manifest"]["input_contract"] = contract
            invalid_payloads.append((payload, "input contract is invalid"))

        payload = complete_payload()
        payload["adapter"]["input_contract"] = {
            "required_runtime_condition": "different"
        }
        invalid_payloads.append((payload, "input contracts differ"))

        payload = complete_payload()
        payload["manifest"]["raw_input_files_checked"] = True
        payload["manifest"]["missing_raw_input_files"] = 1
        invalid_payloads.append((payload, "audit count is invalid"))

        payload = complete_payload()
        payload["manifest"]["n_subjects"] = 2
        payload["manifest"]["split_counts"]["train"] = 2
        payload["manifest"]["split_dataset_counts"]["train"]["test"] = 2
        invalid_payloads.append((payload, "row count differs from its cohort split"))

        for index, (payload, message) in enumerate(invalid_payloads):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "embeddings.tsv"
                sidecar = Path(str(output) + ".audit.json")
                with self.assertRaisesRegex(ValueError, message):
                    publish_embedding_bundle(frame, output, sidecar, payload)
                self.assertFalse(output.exists())
                self.assertFalse(sidecar.exists())

    def test_incomplete_or_existing_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "embeddings.tsv"
            sidecar = Path(str(output) + ".audit.json")
            with self.assertRaisesRegex(ValueError, "different paths"):
                ensure_new_embedding_bundle(output, output)
            with self.assertRaisesRegex(ValueError, "share one directory"):
                ensure_new_embedding_bundle(output, root / "elsewhere" / sidecar.name)

            output.write_text("orphan", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Incomplete embedding bundle"):
                ensure_new_embedding_bundle(output, sidecar)

            output.unlink()
            sidecar.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Incomplete embedding bundle"):
                ensure_new_embedding_bundle(output, sidecar)

            output.write_text("orphan", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                ensure_new_embedding_bundle(output, sidecar)

            nested_output = root / "new" / "nested" / "embeddings.tsv"
            nested_sidecar = Path(str(nested_output) + ".audit.json")
            self.assertEqual(
                ensure_new_embedding_bundle(nested_output, nested_sidecar),
                nested_output.resolve(),
            )
            self.assertTrue(nested_output.parent.is_dir())

    def test_serialization_failure_leaves_no_bundle_or_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embeddings.tsv"
            sidecar = Path(str(output) + ".audit.json")
            payload = complete_payload()
            payload["release"]["not_json_serializable"] = object()

            with self.assertRaisesRegex(TypeError, "JSON serializable"):
                publish_embedding_bundle(complete_frame(), output, sidecar, payload)

            self.assertFalse(output.exists())
            self.assertFalse(sidecar.exists())
            self.assertEqual(list(Path(directory).glob(".*.bundle.*")), [])

    def test_sidecar_publication_failure_rolls_back_the_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embeddings.tsv"
            sidecar = Path(str(output) + ".audit.json")
            frame = complete_frame()
            real_link = os.link
            calls = 0

            def fail_second_link(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected sidecar publication failure")
                real_link(source, destination)

            with (
                patch(
                    "FineTuning.shared.embedding_bundle.os.link",
                    side_effect=fail_second_link,
                ),
                self.assertRaisesRegex(OSError, "injected sidecar"),
            ):
                publish_embedding_bundle(frame, output, sidecar, complete_payload())

            self.assertFalse(output.exists())
            self.assertFalse(sidecar.exists())
            self.assertEqual(list(Path(directory).glob(".*.bundle.*")), [])

    def test_postpublication_verification_failure_rolls_back_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embeddings.tsv"
            sidecar = Path(str(output) + ".audit.json")

            with (
                patch(
                    "FineTuning.shared.embedding_bundle.load_embedding_bundle_sidecar",
                    side_effect=RuntimeError("injected verification failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected verification"),
            ):
                publish_embedding_bundle(
                    complete_frame(), output, sidecar, complete_payload()
                )

            self.assertFalse(output.exists())
            self.assertFalse(sidecar.exists())
            self.assertEqual(list(Path(directory).glob(".*.bundle.*")), [])

    def test_reader_rejects_nonadjacent_paths_and_invalid_expected_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "embeddings.tsv"
            sidecar = Path(str(output) + ".audit.json")
            publish_embedding_bundle(
                complete_frame(),
                output,
                sidecar,
                complete_payload(),
            )

            remote_sidecar = root / "elsewhere" / sidecar.name
            remote_sidecar.parent.mkdir()
            remote_sidecar.write_bytes(sidecar.read_bytes())
            with self.assertRaisesRegex(ValueError, "share one directory"):
                load_embedding_bundle_sidecar(
                    output,
                    remote_sidecar,
                    expected_pipeline_version="test_pipeline_v1",
                    expected_model="test",
                    expected_variant="canonical",
                    expected_readout="test readout",
                    expected_split="train",
                    expected_dimension=1,
                )

            with self.assertRaisesRegex(ValueError, "different paths"):
                load_embedding_bundle_sidecar(
                    output,
                    output,
                    expected_pipeline_version="test_pipeline_v1",
                    expected_model="test",
                    expected_variant="canonical",
                    expected_readout="test readout",
                    expected_split="train",
                    expected_dimension=1,
                )

            invalid_expectations = (
                ({"expected_pipeline_version": ""}, "identity fields"),
                ({"expected_model": None}, "identity fields"),
                ({"expected_variant": ""}, "identity fields"),
                ({"expected_readout": 1}, "identity fields"),
                ({"expected_split": "unknown"}, "expected embedding split"),
                ({"expected_dimension": True}, "expected embedding dimension"),
                ({"expected_dimension": 0}, "expected embedding dimension"),
            )
            defaults = {
                "expected_pipeline_version": "test_pipeline_v1",
                "expected_model": "test",
                "expected_variant": "canonical",
                "expected_readout": "test readout",
                "expected_split": "train",
                "expected_dimension": 1,
            }
            for override, message in invalid_expectations:
                with self.subTest(override=override), self.assertRaisesRegex(
                    ValueError, message
                ):
                    load_embedding_bundle_sidecar(
                        output,
                        sidecar,
                        **{**defaults, **override},
                    )

    def test_publisher_rejects_malformed_embedding_tables(self) -> None:
        invalid_cases: list[tuple[pd.DataFrame, dict, str]] = []

        missing_metadata = complete_frame().drop(columns=["sex"])
        invalid_cases.append((missing_metadata, complete_payload(), "metadata columns"))

        invalid_cases.append(
            (complete_frame().iloc[0:0], complete_payload(), "table is empty")
        )

        no_embeddings = complete_frame().drop(columns=["emb_0"])
        invalid_cases.append((no_embeddings, complete_payload(), "no emb_"))

        noncontiguous = complete_frame().rename(columns={"emb_0": "emb_1"})
        invalid_cases.append((noncontiguous, complete_payload(), "contiguous"))

        malformed_suffix = complete_frame().rename(columns={"emb_0": "emb_bad"})
        invalid_cases.append((malformed_suffix, complete_payload(), "suffixes"))

        nonfinite = complete_frame()
        nonfinite.loc[0, "emb_0"] = float("nan")
        invalid_cases.append((nonfinite, complete_payload(), "non-finite"))

        nonfinite_age = complete_frame()
        nonfinite_age.loc[0, "age"] = float("nan")
        invalid_cases.append((nonfinite_age, complete_payload(), "non-finite ages"))

        duplicated = pd.concat([complete_frame(), complete_frame()], ignore_index=True)
        invalid_cases.append((duplicated, complete_payload(), "duplicate"))

        wrong_dimension = complete_frame()
        wrong_dimension["emb_1"] = 2.0
        invalid_cases.append((wrong_dimension, complete_payload(), "dimension differs"))

        wrong_split = complete_frame()
        wrong_split.loc[0, "split"] = "validation"
        invalid_cases.append((wrong_split, complete_payload(), "split differs"))

        wrong_model = complete_frame()
        wrong_model.loc[0, "model"] = "other"
        invalid_cases.append((wrong_model, complete_payload(), "exclusively model"))

        wrong_variant = complete_frame()
        wrong_variant.loc[0, "variant"] = "other"
        invalid_cases.append(
            (wrong_variant, complete_payload(), "exclusively variant")
        )

        for column in (
            "dataset",
            "subject",
            "split",
            "sex",
            "input_path",
            "model",
            "variant",
        ):
            for value in ("", None):
                invalid_metadata = complete_frame()
                invalid_metadata.loc[0, column] = value
                invalid_cases.append(
                    (invalid_metadata, complete_payload(), "invalid text metadata")
                )

        for index, (frame, payload, message) in enumerate(invalid_cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "embeddings.tsv"
                sidecar = Path(str(output) + ".audit.json")
                with self.assertRaisesRegex(ValueError, message):
                    publish_embedding_bundle(frame, output, sidecar, payload)
                self.assertFalse(output.exists())
                self.assertFalse(sidecar.exists())


if __name__ == "__main__":
    unittest.main()
