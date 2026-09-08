#!/usr/bin/env python3
"""Shared guards and provenance for locked internal-test evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from FineTuning.shared.cohort import sha256_file
from FineTuning.shared.protocol import (
    ADAPTED_FIXED_HYPERPARAMETERS,
    adapted_learning_rates,
)


SELECTION_NAME = "SELECTED_LR.json"


def validate_selected_evaluation_run(
    *,
    selection_dir: Path,
    run_dir: Path,
    lock: dict[str, Any],
    manifest_sha256: str,
    expected_model: str,
    evaluation_variant: str,
    checkpoint_name: str = "selected_model.pt",
    lock_name: str = "LOCKED.json",
    selection_variant: str | None = None,
) -> dict[str, Any]:
    """Bind an evaluation run to the LR chosen using validation only."""

    selected_variant = selection_variant or evaluation_variant
    selection_path = selection_dir.resolve() / SELECTION_NAME
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_selection = {
        "status": "LR_SELECTED_AFTER_TRAIN_VALIDATION",
        "model": expected_model,
        "variant": selected_variant,
        "manifest_sha256": manifest_sha256,
        "selection_seed": 0,
        "internal_test_accessed": False,
        "on_harmony_accessed": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": selection.get(key)}
        for key, expected in expected_selection.items()
        if selection.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Invalid validation-only LR selection artifact: {mismatches}")

    if (lock.get("model"), lock.get("variant")) != (
        expected_model,
        evaluation_variant,
    ):
        raise ValueError("Evaluation lock has the wrong model or variant")
    arguments = lock.get("run_config", {}).get("arguments", {})
    selected_config = selection.get("selected_lr_config")
    if arguments.get("lr_config") != selected_config:
        raise ValueError(
            "Evaluation run does not use the validation-selected LR configuration: "
            f"selected={selected_config!r}, run={arguments.get('lr_config')!r}"
        )
    expected_encoder, expected_head = adapted_learning_rates(str(selected_config))
    try:
        observed_encoder = float(arguments.get("lr_encoder"))
        observed_head = float(arguments.get("lr_head"))
    except (TypeError, ValueError) as error:
        raise ValueError("Evaluation run lacks valid locked learning rates") from error
    if not math.isclose(observed_encoder, expected_encoder):
        raise ValueError("Evaluation run encoder LR differs from the selected configuration")
    if not math.isclose(observed_head, expected_head):
        raise ValueError("Evaluation run head LR differs from the selected configuration")
    for key, expected in ADAPTED_FIXED_HYPERPARAMETERS.items():
        observed = arguments.get(key)
        try:
            valid = (
                math.isclose(float(observed), expected)
                if isinstance(expected, float)
                else observed == expected
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(
                f"Evaluation run {key}={observed!r}, expected locked value {expected!r}"
            )

    selected_run_dir = Path(str(selection.get("selected_run_dir", ""))).resolve()
    selected_lock_path = selected_run_dir / lock_name
    selected_checkpoint_path = selected_run_dir / checkpoint_name
    if not selected_lock_path.is_file() or not selected_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Selection anchor is incomplete: {selected_lock_path}, {selected_checkpoint_path}"
        )
    selected_lock = json.loads(selected_lock_path.read_text(encoding="utf-8"))
    if (
        selected_lock.get("status") != "LOCKED_AFTER_TRAIN_VALIDATION"
        or selected_lock.get("internal_test_accessed") is not False
        or selected_lock.get("on_harmony_accessed") is not False
        or (selected_lock.get("model"), selected_lock.get("variant"))
        != (expected_model, selected_variant)
        or selected_lock.get("manifest", {}).get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError("Selection anchor is not the expected validation-only locked run")
    selected_checkpoint_sha256 = sha256_file(selected_checkpoint_path)
    if (
        selected_checkpoint_sha256 != selection.get("selected_checkpoint_sha256")
        or selected_checkpoint_sha256 != selected_lock.get("checkpoint_sha256")
    ):
        raise ValueError("Validation-selected checkpoint hash differs from its selection artifact")

    if lock.get("code_sha256") != selected_lock.get("code_sha256"):
        raise ValueError("Evaluation run code differs from the LR-selection anchor")
    if evaluation_variant == selected_variant:
        for key in ("release", "adapter"):
            if lock.get(key) != selected_lock.get(key):
                raise ValueError(f"Evaluation run differs from the selection anchor in {key}")
    if lock.get("manifest", {}).get("manifest_sha256") != manifest_sha256:
        raise ValueError("Evaluation run manifest differs from the selection artifact")

    try:
        run_seed = int(arguments.get("seed"))
    except (TypeError, ValueError) as error:
        raise ValueError("Evaluation run lacks a valid seed") from error
    if (
        run_seed == int(selection["selection_seed"])
        and evaluation_variant == selected_variant
        and lock.get("checkpoint_sha256") != selected_checkpoint_sha256
    ):
        raise ValueError("Selection-seed evaluation must use the exact selected checkpoint")
    return {
        "selection_artifact": str(selection_path),
        "selection_artifact_sha256": sha256_file(selection_path),
        "selected_lr_config": str(selected_config),
        "selection_model": expected_model,
        "selection_variant": selected_variant,
        "selection_seed": int(selection["selection_seed"]),
        "selection_checkpoint_sha256": selected_checkpoint_sha256,
        "evaluation_run": str(run_dir.resolve()),
        "evaluation_variant": evaluation_variant,
        "evaluation_seed": run_seed,
        "cross_variant_lr_transfer": evaluation_variant != selected_variant,
    }


def validate_evaluation_runtime_code(
    *,
    locked_code: dict[str, str],
    current_code: dict[str, str],
    script_key: str,
    evaluation_guard_key: str,
    approved_legacy_script_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    """Accept exact code or one explicitly scoped legacy evaluation migration."""

    if current_code == locked_code:
        return {"mode": "exact_training_runtime", "code_sha256": current_code}

    locked_keys = set(locked_code)
    current_keys = set(current_code)
    changed = sorted(
        key
        for key in locked_keys & current_keys
        if locked_code[key] != current_code[key]
    )
    added = sorted(current_keys - locked_keys)
    removed = sorted(locked_keys - current_keys)
    approved = set(approved_legacy_script_hashes)
    valid_migration = (
        changed == [script_key]
        and added == [evaluation_guard_key]
        and not removed
        and locked_code.get(script_key) in approved
    )
    if not valid_migration:
        raise ValueError(
            "Runtime code differs from the selected checkpoint outside the approved "
            f"evaluation-only migration: changed={changed}, added={added}, removed={removed}"
        )
    return {
        "mode": "approved_evaluation_guard_migration",
        "change_scope": "internal-test gating and result provenance only",
        "changed_files": changed,
        "added_files": added,
        "locked_code_sha256": locked_code,
        "evaluation_code_sha256": current_code,
    }
