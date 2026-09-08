"""Shared provenance helpers for adapted-model internal-test evaluation.

The locked-run validators are imported from ``shared.onharmony`` for historical
compatibility only. Calling these helpers does not load or inspect ON-Harmony;
the external prediction entry points remain isolated in ``predict_onharmony.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from FineTuning.shared.cohort import PROJECT_ROOT, sha256_file
from FineTuning.shared.onharmony import (
    RuntimeMigration,
    adapted_runtime_migration,
    adapted_selection_runtime_migration,
    load_adapted_model,
    validate_prediction_runtime_code,
    validate_run_binding,
)
from FineTuning.shared.runtime import code_hashes


SCRIPT_PATH = Path(__file__).resolve()


def internal_test_runtime_migration(
    model_name: str,
    locked_code: Mapping[str, str],
) -> RuntimeMigration | None:
    """Return the exact approved locked-run to current-runtime transition."""

    return adapted_runtime_migration(model_name, locked_code)


def internal_test_selection_runtime_migration(
    model_name: str,
) -> RuntimeMigration | None:
    """Return a separately pinned selection-anchor to final-run transition."""

    return adapted_selection_runtime_migration(model_name)


def validate_internal_test_runtime_code(
    *,
    model_name: str,
    locked_code: Mapping[str, str],
    current_code: Mapping[str, str],
) -> dict[str, Any]:
    """Validate exact runtime identity or one registered post-training migration."""

    migration = internal_test_runtime_migration(model_name, locked_code)
    provenance = validate_prediction_runtime_code(
        locked_code, current_code, migration
    )
    provenance["internal_test_guard_code"] = {
        str(SCRIPT_PATH.relative_to(PROJECT_ROOT)): sha256_file(SCRIPT_PATH)
    }
    return provenance


def validate_internal_test_binding(**kwargs: Any) -> dict[str, Any]:
    """Bind an internal-test run to its validation-selected locked checkpoint."""

    return validate_run_binding(**kwargs)


def load_internal_test_model(**kwargs: Any) -> dict[str, Any]:
    """Restore the adapted boundary and age head without external-data access."""

    return load_adapted_model(**kwargs)


def internal_test_orchestrator_hashes(model_script: Path) -> dict[str, str]:
    """Hash the unified evaluator and its evaluation-only provenance helpers."""

    return code_hashes(
        [
            Path(model_script).resolve(),
            SCRIPT_PATH,
            PROJECT_ROOT / "FineTuning" / "shared" / "onharmony.py",
        ]
    )
