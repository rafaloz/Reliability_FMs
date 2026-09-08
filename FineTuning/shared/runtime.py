"""Runtime helpers shared by frozen extraction and controlled adaptation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from FineTuning.shared.cohort import PROJECT_ROOT, sha256_file


CUBLAS_DETERMINISTIC_CONFIGS = {":4096:8", ":16:8"}


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy; GPU entry points additionally seed PyTorch."""

    random.seed(seed)
    np.random.seed(seed)


def atomic_json(payload: object, path: Path | str) -> None:
    """Write JSON through a same-directory temporary file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def atomic_tsv(frame: pd.DataFrame, path: Path | str) -> None:
    """Write a tabular artifact atomically to avoid partial promoted outputs."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, destination)


def package_versions() -> dict[str, str]:
    """Capture cross-model packages without importing their GPU modules."""

    versions = {"python": platform.python_version()}
    packages = (
        "torch",
        "torchvision",
        "monai",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "nibabel",
        "transformers",
        "huggingface-hub",
        "neurovfm",
        "anatcl",
    )
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    return versions


def display_code_path(path: Path) -> str:
    """Prefer project-relative provenance keys while retaining external package paths."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def code_hashes(paths: Iterable[Path]) -> dict[str, str]:
    """Hash each unique runtime source file and reject missing dependencies."""

    result: dict[str, str] = {}
    for path in paths:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Runtime code dependency not found: {resolved}")
        result[display_code_path(resolved)] = sha256_file(resolved)
    return dict(sorted(result.items()))


def import_torch() -> Any:
    """Import PyTorch only for GPU commands, producing a direct environment error."""

    # CuBLAS reads this setting when its CUDA workspace is first initialized.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "This command requires the model's PyTorch environment; use myLocalEnv for "
            "six models and the neurovfm environment for NeuroVFM"
        ) from error
    return torch


def resolve_device(torch: Any, requested: str) -> Any:
    """Resolve auto/cuda/cpu and fail before model loading if CUDA is unavailable."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def seed_torch(torch: Any, seed: int) -> None:
    """Set strict deterministic execution and disable silent TF32 approximations."""

    set_global_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in CUBLAS_DETERMINISTIC_CONFIGS:
            raise RuntimeError(
                "Deterministic CUDA execution requires CUBLAS_WORKSPACE_CONFIG="
                ":4096:8 or :16:8 before importing torch"
            )
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)


def autocast_context(torch: Any, device: Any, amp: str) -> Any:
    """Create the explicitly requested CUDA autocast context."""

    if amp not in {"off", "float16", "bfloat16"}:
        raise ValueError(f"Unknown AMP mode: {amp}")
    enabled = device.type == "cuda" and amp != "off"
    dtype = torch.float16 if amp == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def ensure_new_output_dir(path: Path | str) -> Path:
    """Refuse to mix a new experiment with any pre-existing artifact."""

    directory = Path(path).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_new_file(path: Path | str) -> Path:
    """Refuse to overwrite an existing promoted artifact."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def missing_adapter_inputs(adapter: Any, frame: Any) -> list[str]:
    """Return missing model inputs for selected manifest rows without loading a model."""

    missing: list[str] = []
    for record in frame.to_dict(orient="records"):
        for path in adapter.required_input_paths(record):
            if not Path(path).is_file():
                missing.append(str(path))
    return missing


def atomic_torch_save(torch: Any, payload: object, path: Path | str) -> None:
    """Save a tensor checkpoint through a same-directory temporary path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def capture_trainable_state(adapter: Any) -> dict[str, Any]:
    """Store only trainable encoder parameters; immutable base weights stay referenced."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in adapter.trainable_named_parameters()
    }


def restore_trainable_state(adapter: Any, state: dict[str, Any]) -> None:
    """Restore the exact controlled encoder parameters by stable parameter name."""

    current = dict(adapter.trainable_named_parameters())
    if set(current) != set(state):
        raise ValueError(
            "Controlled encoder state keys differ: "
            f"missing={sorted(set(current) - set(state))[:10]}, "
            f"unexpected={sorted(set(state) - set(current))[:10]}"
        )
    for name, parameter in current.items():
        source = state[name]
        if tuple(source.shape) != tuple(parameter.shape):
            raise ValueError(
                f"Controlled parameter shape drift for {name}: "
                f"checkpoint={tuple(source.shape)}, current={tuple(parameter.shape)}"
            )
        parameter.data.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
