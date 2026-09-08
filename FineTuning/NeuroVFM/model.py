#!/usr/bin/env python3
"""Pinned NeuroVFM release, readout, and controlled-adaptation boundary."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Mapping, Sequence


from FineTuning.shared.cohort import (
    DEFAULT_MANIFEST,
    EXPECTED_SPLIT_DATASET_COUNTS,
    MANIFEST_SHA256,
    PROJECT_ROOT,
    load_and_validate_manifest as _load_and_validate_manifest,
    sha256_file,
)
from FineTuning.shared.runtime import atomic_json, atomic_torch_save


MODEL_NAME = "NeuroVFM"
VARIANT = "canonical"
ENVIRONMENT = "neurovfm"
EMBEDDING_DIM = 768
READOUT = "mean over variable-length final-normalized foreground tokens"
ADAPTATION_BOUNDARY = "final volumetric ViT block + final layer norm + linear age head"

NEUROVFM_MODEL_REPO = "mlinslab/neurovfm-encoder"
NEUROVFM_MODEL_REVISION = "d5194fc70a162185f8ef062e362bd522a35312a9"
NEUROVFM_GITHUB_COMMIT = "9240021d4ef5c262b21cee5d219c2adf65f4d42f"
PREPROCESSING_CACHE_VERSION = "neurovfm_study_preprocessor_cache_v1"
DEFAULT_PREPROCESSING_CACHE_ROOT = (
    PROJECT_ROOT / "preprocessings" / "FineTunningDataset" / "preproc_NeuroVFM"
)
CACHE_BATCH_REQUIRED_KEYS = {
    "img",
    "coords",
    "series_masks_indices",
    "series_cu_seqlens",
    "series_max_len",
    "mode",
    "path",
    "size",
}
CACHE_PREPROCESSOR_CODE_FILES = (
    "pipelines/preprocessor.py",
    "data/preprocess.py",
    "data/utils.py",
)

# Inference-path hashes from the clean upstream tree at NEUROVFM_GITHUB_COMMIT.
NEUROVFM_CODE_SHA256 = {
    "pipelines/__init__.py": "260c455f898cf1eedc3aa378bc845087b0f595f7e66508d8a132f7b8f2a0208a",
    "pipelines/encoder.py": "20fd919712e3702275d7520087eefd24cf6aaa5f577659e628268b45922ce2ee",
    "pipelines/preprocessor.py": "4de998e7b7bc877f62a436c1d6c363642f4703c45b542d09520be5deaede7192",
    "data/preprocess.py": "ccc015bf528838ff6e947e7180d174477364f74f19c06b9d9d48186f92dd0edd",
    "data/utils.py": "c56e9e34c1af8b81a85c1ee82c999868197cca8a05c9a8ea0733b0e1e502d9a9",
    "systems/utils.py": "92aa9b6150d4102d01bbd4712a9340371590384fa4c70ae99c248c020e8d7f60",
    "models/__init__.py": "9079eb45c796c9120b2c6b5f0b027f308cbb87e1baa698f53a39664ed50c4952",
    "models/patch_embed.py": "87bc8264dc3158817aadca389058c831895685ecf26089e79829b443b3deb74d",
    "models/pos_embed.py": "0c855f48fc78f4e6d55f2683311d8c83874b2f6cb233250497888aff56015790",
    "models/vit.py": "826fa39b4aa6b583637599ed7bacee481e8fc78a2991d006c72b93936286f71e",
}


@dataclass(frozen=True)
class NeuroVFMOptions:
    """Immutable upstream source options for one NeuroVFM execution."""

    model_source: str = NEUROVFM_MODEL_REPO
    model_revision: str | None = NEUROVFM_MODEL_REVISION
    allow_download: bool = False
    preprocessing_cache_root: Path | None = None
    require_preprocessing_cache: bool = False
    memory_cache: bool = True

    def validate(self) -> None:
        if not str(self.model_source).strip():
            raise ValueError("model_source cannot be empty")
        local = Path(self.model_source).expanduser()
        if not local.exists() and not self.model_revision:
            raise ValueError("A remote NeuroVFM source requires an immutable revision")
        if self.require_preprocessing_cache and self.preprocessing_cache_root is None:
            raise ValueError(
                "require_preprocessing_cache=True requires preprocessing_cache_root"
            )


def _record_value(record: Any, key: str) -> Any:
    """Read a manifest value from a mapping, pandas row, or named tuple."""

    if isinstance(record, Mapping):
        return record[key]
    try:
        return record[key]
    except (KeyError, TypeError, IndexError):
        return getattr(record, key)


def _safe_cache_component(value: object, label: str) -> str:
    """Validate a manifest identifier before using it in a cache path."""

    component = str(value).strip()
    if not component or component in {".", ".."} or Path(component).name != component:
        raise ValueError(f"Unsafe {label} value for NeuroVFM cache: {value!r}")
    return component


def preprocessing_cache_paths(
    root: Path | str, record: Any
) -> tuple[Path, Path]:
    """Return the tensor artifact and JSON sidecar for one manifest record."""

    dataset = _safe_cache_component(_record_value(record, "dataset"), "dataset")
    subject = _safe_cache_component(_record_value(record, "subject"), "subject")
    directory = Path(root).expanduser().resolve() / dataset / subject
    return directory / "study_preprocessor.pt", directory / "study_preprocessor.json"


def _source_signature(path: Path, *, include_sha256: bool) -> dict[str, Any]:
    """Describe the immutable raw source used to produce one cache artifact."""

    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        result["sha256"] = sha256_file(path)
    return result


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible contract independently of display formatting."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution_versions(names: Sequence[str]) -> dict[str, str]:
    """Capture packages that can change released preprocessing numerics."""

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return versions


def _cpu_cache_value(torch: Any, value: Any, *, location: str) -> Any:
    """Copy a released batch value into the restricted cache payload schema."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Number):
        return value.item() if hasattr(value, "item") else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"Non-string mapping key in cached NeuroVFM batch at {location}")
        return {
            key: _cpu_cache_value(torch, item, location=f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        converted = [
            _cpu_cache_value(torch, item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
        return tuple(converted) if isinstance(value, tuple) else converted
    raise TypeError(
        f"Unsupported NeuroVFM cache value at {location}: {type(value).__name__}"
    )


def _assert_cache_value_equal(
    torch: Any,
    expected: Any,
    observed: Any,
    *,
    location: str,
) -> None:
    """Require a bit-exact round trip for every cached batch field."""

    if isinstance(expected, torch.Tensor):
        if not isinstance(observed, torch.Tensor):
            raise TypeError(f"Cache round-trip changed tensor type at {location}")
        if (
            expected.dtype != observed.dtype
            or tuple(expected.shape) != tuple(observed.shape)
            or not torch.equal(expected, observed)
        ):
            raise ValueError(f"Cache round-trip changed tensor values at {location}")
        return
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(expected) != set(observed):
            raise ValueError(f"Cache round-trip changed mapping keys at {location}")
        for key in expected:
            _assert_cache_value_equal(
                torch,
                expected[key],
                observed[key],
                location=f"{location}.{key}",
            )
        return
    if isinstance(expected, Sequence) and not isinstance(
        expected, (str, bytes, bytearray)
    ):
        if not isinstance(observed, type(expected)) or len(expected) != len(observed):
            raise ValueError(f"Cache round-trip changed sequence structure at {location}")
        for index, (expected_item, observed_item) in enumerate(zip(expected, observed)):
            _assert_cache_value_equal(
                torch,
                expected_item,
                observed_item,
                location=f"{location}[{index}]",
            )
        return
    if type(expected) is not type(observed) or expected != observed:
        raise ValueError(f"Cache round-trip changed scalar value at {location}")


def load_and_validate_manifest(
    path: Path | str = DEFAULT_MANIFEST,
    *,
    expected_sha256: str | None = MANIFEST_SHA256,
    expected_counts: dict[str, dict[str, int]] | None = EXPECTED_SPLIT_DATASET_COUNTS,
    check_input_files: bool = False,
) -> tuple[Any, dict[str, object]]:
    """Validate the shared cohort and its raw-runtime permission for NeuroVFM."""

    return _load_and_validate_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_counts=expected_counts,
        required_runtime_condition=MODEL_NAME,
        check_raw_input_files=check_input_files,
    )


def validate_neurovfm_code(package_root: Path | str) -> dict[str, Any]:
    """Require the installed inference code to match the audited Git commit."""

    root = Path(package_root).resolve()
    missing: list[str] = []
    mismatched: dict[str, dict[str, str]] = {}
    observed: dict[str, str] = {}
    for relative, expected in NEUROVFM_CODE_SHA256.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        value = sha256_file(path)
        observed[relative] = value
        if value != expected:
            mismatched[relative] = {"expected": expected, "observed": value}
    if missing or mismatched:
        raise RuntimeError(
            "Installed NeuroVFM code differs from the audited release "
            f"{NEUROVFM_GITHUB_COMMIT}: missing={missing}, mismatched={mismatched}"
        )
    return {
        "github_repository": "https://github.com/MLNeurosurg/neurovfm",
        "github_commit": NEUROVFM_GITHUB_COMMIT,
        "package_root": str(root),
        "files_sha256": observed,
    }


def resolve_neurovfm_snapshot(
    model_source: str,
    revision: str | None,
    *,
    allow_download: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a local checkpoint or the exact cached/downloaded HF revision."""

    local = Path(model_source).expanduser()
    if local.exists():
        snapshot = local.resolve()
        source_kind = "local_directory"
        resolved_revision = revision
    else:
        if not revision:
            raise ValueError("A Hugging Face NeuroVFM source requires an immutable revision")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("NeuroVFM checkpoint resolution requires huggingface-hub") from error

        options = {
            "repo_id": model_source,
            "revision": revision,
            "local_files_only": not allow_download,
        }
        config = Path(hf_hub_download(filename="config.json", **options)).expanduser()
        weights = Path(hf_hub_download(filename="pytorch_model.bin", **options)).expanduser()
        config_snapshot = Path(os.path.abspath(config.parent))
        weights_snapshot = Path(os.path.abspath(weights.parent))
        if config_snapshot != weights_snapshot:
            raise RuntimeError("NeuroVFM config and weights resolved to different snapshots")
        # Hugging Face snapshot entries are symlinks into blobs/. Resolving the
        # file symlink here would lose the snapshot directory and its filenames.
        snapshot = config_snapshot
        source_kind = "huggingface_snapshot"
        resolved_revision = revision

    required = [snapshot / "config.json", snapshot / "pytorch_model.bin"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return snapshot, {
        "model_source": model_source,
        "model_revision": resolved_revision,
        "source_kind": source_kind,
        "resolved_snapshot": str(snapshot),
        "artifacts_sha256": {path.name: sha256_file(path) for path in required},
    }


def validate_neurovfm_device(torch: Any, device: Any, amp: str) -> dict[str, Any]:
    """Reject unsupported BF16 GPUs before compiling NeuroVFM fused kernels."""

    if device.type != "cuda":
        raise RuntimeError("NeuroVFM experiments require a CUDA GPU in the neurovfm environment")
    if amp != "bfloat16":
        raise ValueError("The audited NeuroVFM runtime requires --amp bfloat16")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "NeuroVFM BF16 fused kernels require compute capability sm_80 or newer; "
            f"observed sm_{capability[0]}{capability[1]}"
        )
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": f"sm_{capability[0]}{capability[1]}",
        "amp": amp,
        "bf16_supported": True,
    }


class NeuroVFMModel:
    """Official frozen path plus a differentiable version of the same forward."""

    model_name = MODEL_NAME
    variant = VARIANT
    environment = ENVIRONMENT
    embedding_dim = EMBEDDING_DIM
    readout = READOUT
    adaptation_boundary = ADAPTATION_BOUNDARY

    def __init__(self, options: NeuroVFMOptions) -> None:
        options.validate()
        self.options = options
        self.pipeline: Any | None = None
        self.preprocessor: Any | None = None
        self.model: Any | None = None
        self.device: Any | None = None
        self.last_token_count: int | None = None
        self._controlled = False
        self._snapshot: Path | None = None
        self._release_details: dict[str, Any] | None = None
        self._preprocessing_contract: dict[str, Any] | None = None
        self._preprocessing_memory_cache: dict[Path, dict[str, Any]] = {}
        self._cache_disk_loads = 0
        self._cache_memory_hits = 0
        self._raw_preprocessing_calls = 0

    def input_path(self, record: Any) -> Path:
        return Path(str(_record_value(record, "raw_t1w_path")))

    def required_input_paths(self, record: Any) -> list[Path]:
        if self.options.require_preprocessing_cache:
            artifact, sidecar = self.preprocessing_cache_paths(record)
            return [artifact, sidecar]
        return [self.input_path(record)]

    def preprocessing_cache_paths(self, record: Any) -> tuple[Path, Path]:
        """Return canonical cache paths and reject an unconfigured cache."""

        root = self.options.preprocessing_cache_root
        if root is None:
            raise RuntimeError("NeuroVFM preprocessing cache root is not configured")
        return preprocessing_cache_paths(root, record)

    def _resolve_release(self) -> tuple[Path, dict[str, Any]]:
        if self._snapshot is not None and self._release_details is not None:
            return self._snapshot, self._release_details
        try:
            import neurovfm
        except ImportError as error:
            raise RuntimeError("Activate the dedicated neurovfm environment") from error

        snapshot, snapshot_details = resolve_neurovfm_snapshot(
            self.options.model_source,
            self.options.model_revision,
            allow_download=self.options.allow_download,
        )
        package_root = Path(neurovfm.__file__).resolve().parent
        code_details = validate_neurovfm_code(package_root)
        self._snapshot = snapshot
        self._release_details = {"snapshot": snapshot_details, "code": code_details}
        return snapshot, self._release_details

    def load(self, device: Any) -> None:
        """Load through the verified wrapper used by the ON-Harmony extractor."""

        from Emb_extractors.extract_NeuroVFM import load_encoder_verified

        snapshot, _ = self._resolve_release()
        self.pipeline, self.preprocessor = load_encoder_verified(str(snapshot), str(device))
        self.model = self.pipeline.model
        self.device = device

    def load_preprocessor(self) -> None:
        """Load only the audited released preprocessor for CPU cache creation."""

        self._resolve_release()
        from neurovfm.pipelines.preprocessor import StudyPreprocessor

        self.preprocessor = StudyPreprocessor()

    def preprocessing_cache_contract(self) -> dict[str, Any]:
        """Bind cached batches to the exact released preprocessing implementation."""

        if self._preprocessing_contract is not None:
            return self._preprocessing_contract
        _, details = self._resolve_release()
        snapshot = details["snapshot"]
        code = details["code"]
        contract: dict[str, Any] = {
            "cache_version": PREPROCESSING_CACHE_VERSION,
            "model_repository": snapshot["model_source"],
            "model_revision": snapshot["model_revision"],
            "config_sha256": snapshot["artifacts_sha256"]["config.json"],
            "github_commit": code["github_commit"],
            "preprocessor_files_sha256": {
                relative: code["files_sha256"][relative]
                for relative in CACHE_PREPROCESSOR_CODE_FILES
            },
            "study_preprocessor": {
                "patch_size": [4, 16, 16],
                "target_spacing": [1.0, 1.0, 4.0],
                "remove_background": True,
            },
            "preprocessing_software": _distribution_versions(
                ("neurovfm", "torch", "numpy", "nibabel", "SimpleITK")
            ),
            "modality": "mri",
            "batch_keys": sorted(CACHE_BATCH_REQUIRED_KEYS),
        }
        self._preprocessing_contract = {
            **contract,
            "contract_sha256": _canonical_json_sha256(contract),
        }
        return self._preprocessing_contract

    def _require_loaded(self) -> Any:
        if self.model is None or self.pipeline is None or self.preprocessor is None:
            raise RuntimeError("NeuroVFM has not loaded its checkpoint and preprocessor")
        return self.model

    def _require_preprocessor(self) -> Any:
        """Return the released preprocessor without requiring encoder weights."""

        if self.preprocessor is None:
            raise RuntimeError("NeuroVFM StudyPreprocessor has not been loaded")
        return self.preprocessor

    @staticmethod
    def _validate_preprocessed_batch(torch: Any, batch: Any) -> dict[str, Any]:
        """Validate the released StudyPreprocessor output used by both model paths."""

        if not isinstance(batch, dict):
            raise TypeError(f"NeuroVFM preprocessor returned {type(batch).__name__}, not dict")
        missing = sorted(CACHE_BATCH_REQUIRED_KEYS - set(batch))
        if missing:
            raise ValueError(f"NeuroVFM preprocessor omitted fields: {missing}")
        for key in ("img", "coords", "series_cu_seqlens"):
            if not isinstance(batch[key], torch.Tensor):
                raise TypeError(f"NeuroVFM batch field {key!r} is not a tensor")
            if batch[key].device.type != "cpu":
                raise ValueError(f"Cached NeuroVFM field {key!r} must remain on CPU")
        if batch["img"].ndim != 2 or int(batch["img"].shape[1]) != 1024:
            raise ValueError(
                f"NeuroVFM cached patch shape must be (N, 1024), observed "
                f"{tuple(batch['img'].shape)}"
            )
        if batch["coords"].ndim != 2 or int(batch["coords"].shape[1]) != 3:
            raise ValueError(
                f"NeuroVFM cached coordinate shape must be (N, 3), observed "
                f"{tuple(batch['coords'].shape)}"
            )
        token_count = int(batch["img"].shape[0])
        if token_count <= 0:
            raise ValueError("NeuroVFM preprocessor returned no foreground tokens")
        if int(batch["coords"].shape[0]) != token_count:
            raise ValueError(
                "NeuroVFM token/coordinate count differs: "
                f"{token_count} versus {int(batch['coords'].shape[0])}"
            )
        cu_seqlens = batch["series_cu_seqlens"]
        if int(cu_seqlens.numel()) != 2:
            raise ValueError("Expected exactly one T1w series per manifest subject")
        endpoints = cu_seqlens.detach().cpu().reshape(-1)
        if int(endpoints[0].item()) != 0 or int(endpoints[-1].item()) != token_count:
            raise ValueError(
                "NeuroVFM cumulative sequence lengths do not span all foreground tokens"
            )
        if int(batch["series_max_len"]) != token_count:
            raise ValueError("NeuroVFM series_max_len differs from the only series length")
        for key in ("mode", "path", "size"):
            if not isinstance(batch.get(key), (list, tuple)) or len(batch[key]) != 1:
                raise ValueError(f"Expected exactly one NeuroVFM {key!r} entry")
        if str(batch["mode"][0]).lower() != "mri":
            raise ValueError(f"Unexpected NeuroVFM modality: {batch['mode']!r}")
        masks = batch.get("series_masks_indices")
        if masks is not None and (
            not isinstance(masks, torch.Tensor) or int(masks.numel()) != 0
        ):
            raise ValueError(
                "Released NeuroVFM cache expects background tokens to be removed"
            )
        for key in ("img", "coords"):
            tensor = batch[key]
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"NeuroVFM batch field {key!r} contains non-finite values")
        return batch

    @staticmethod
    def _batch_summary(batch: Mapping[str, Any]) -> dict[str, Any]:
        """Describe tensor geometry without embedding large values in JSON provenance."""

        summary: dict[str, Any] = {"keys": sorted(batch)}
        for key, value in batch.items():
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                summary[key] = {
                    "shape": [int(item) for item in value.shape],
                    "dtype": str(value.dtype),
                }
            elif key == "series_max_len":
                try:
                    summary[key] = int(value)
                except (TypeError, ValueError):
                    summary[key] = str(value)
            elif key in {"mode", "path", "size"}:
                summary[key] = json.loads(json.dumps(value, allow_nan=False))
        summary["n_tokens"] = int(batch["img"].shape[0])
        return summary

    def build_preprocessing_cache(
        self,
        record: Any,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Run StudyPreprocessor once and atomically persist its CPU batch."""

        import torch

        preprocessor = self._require_preprocessor()
        artifact, sidecar = self.preprocessing_cache_paths(record)
        if artifact.is_file() and sidecar.is_file() and not overwrite:
            batch = self._load_preprocessing_cache(record, retain_in_memory=False)
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            return {
                "status": "SKIPPED",
                "artifact": str(artifact),
                "sidecar": str(sidecar),
                "artifact_sha256": metadata["artifact_sha256"],
                "artifact_size_bytes": int(artifact.stat().st_size),
                "n_tokens": int(batch["img"].shape[0]),
            }

        raw_path = self.input_path(record)
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw_signature = _source_signature(raw_path, include_sha256=True)
        released_batch = preprocessor.load_study(str(raw_path), modality="mri")
        self._raw_preprocessing_calls += 1
        selected_keys = CACHE_BATCH_REQUIRED_KEYS
        batch = {
            key: _cpu_cache_value(torch, value, location=f"batch.{key}")
            for key, value in released_batch.items()
            if key in selected_keys
        }
        self._validate_preprocessed_batch(torch, batch)
        contract = self.preprocessing_cache_contract()
        dataset = str(_record_value(record, "dataset"))
        subject = str(_record_value(record, "subject"))
        payload = {
            "cache_version": PREPROCESSING_CACHE_VERSION,
            "preprocessing_contract_sha256": contract["contract_sha256"],
            "dataset": dataset,
            "subject": subject,
            "raw_source": raw_signature,
            "batch": batch,
        }
        atomic_torch_save(torch, payload, artifact)
        artifact_sha256 = sha256_file(artifact)
        metadata = {
            "status": "PASS",
            "cache_version": PREPROCESSING_CACHE_VERSION,
            "preprocessing_contract": contract,
            "dataset": dataset,
            "subject": subject,
            "raw_source": raw_signature,
            "artifact": str(artifact),
            "artifact_sha256": artifact_sha256,
            "artifact_size_bytes": int(artifact.stat().st_size),
            "batch": self._batch_summary(batch),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(metadata, sidecar)
        roundtrip = self._load_preprocessing_cache(record, retain_in_memory=False)
        _assert_cache_value_equal(torch, batch, roundtrip, location="batch")
        if self._batch_summary(roundtrip) != metadata["batch"]:
            raise RuntimeError(f"NeuroVFM cache round-trip changed batch metadata: {artifact}")
        return {
            "status": "PASS",
            "artifact": str(artifact),
            "sidecar": str(sidecar),
            "artifact_sha256": artifact_sha256,
            "n_tokens": int(batch["img"].shape[0]),
            "artifact_size_bytes": int(artifact.stat().st_size),
        }

    def _load_preprocessing_cache(
        self,
        record: Any,
        *,
        retain_in_memory: bool,
    ) -> dict[str, Any]:
        """Load and validate one cache artifact, optionally retaining it in RAM."""

        import torch

        artifact, sidecar = self.preprocessing_cache_paths(record)
        if retain_in_memory and artifact in self._preprocessing_memory_cache:
            self._cache_memory_hits += 1
            return self._preprocessing_memory_cache[artifact]
        if not artifact.is_file() or not sidecar.is_file():
            missing = artifact if not artifact.is_file() else sidecar
            raise FileNotFoundError(missing)
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        contract = self.preprocessing_cache_contract()
        expected_identity = {
            "cache_version": PREPROCESSING_CACHE_VERSION,
            "preprocessing_contract_sha256": contract["contract_sha256"],
            "dataset": str(_record_value(record, "dataset")),
            "subject": str(_record_value(record, "subject")),
        }
        observed_identity = {
            "cache_version": metadata.get("cache_version"),
            "preprocessing_contract_sha256": metadata.get(
                "preprocessing_contract", {}
            ).get("contract_sha256"),
            "dataset": metadata.get("dataset"),
            "subject": metadata.get("subject"),
        }
        if observed_identity != expected_identity:
            raise ValueError(
                "NeuroVFM preprocessing-cache contract drift: "
                f"expected={expected_identity}, observed={observed_identity}"
            )
        raw_path = self.input_path(record)
        observed_source = _source_signature(raw_path, include_sha256=False)
        cached_source = metadata.get("raw_source", {})
        for key in ("path", "size_bytes", "mtime_ns"):
            if observed_source.get(key) != cached_source.get(key):
                raise ValueError(
                    f"NeuroVFM raw source changed after caching ({key}): {raw_path}"
                )

        encoded = artifact.read_bytes()
        observed_artifact_sha256 = hashlib.sha256(encoded).hexdigest()
        if observed_artifact_sha256 != metadata.get("artifact_sha256"):
            raise ValueError(f"NeuroVFM cache artifact SHA-256 mismatch: {artifact}")
        payload = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
        payload_identity = {
            "cache_version": payload.get("cache_version"),
            "preprocessing_contract_sha256": payload.get(
                "preprocessing_contract_sha256"
            ),
            "dataset": payload.get("dataset"),
            "subject": payload.get("subject"),
        }
        if payload_identity != expected_identity:
            raise ValueError(
                "NeuroVFM cache payload identity differs from its contract: "
                f"{payload_identity}"
            )
        batch = self._validate_preprocessed_batch(torch, payload.get("batch"))
        if payload.get("raw_source") != cached_source:
            raise ValueError(
                f"NeuroVFM cache payload and sidecar disagree on raw source: {artifact}"
            )
        if list(batch["path"]) != [str(raw_path)]:
            raise ValueError(
                f"NeuroVFM cached batch points to a different raw input: {artifact}"
            )
        if self._batch_summary(batch) != metadata.get("batch"):
            raise ValueError(f"NeuroVFM cache batch metadata mismatch: {artifact}")
        self._cache_disk_loads += 1
        if retain_in_memory:
            self._preprocessing_memory_cache[artifact] = batch
        return batch

    def load_cached_preprocessing(
        self,
        record: Any,
        *,
        retain_in_memory: bool = True,
    ) -> dict[str, Any]:
        """Validate one mandatory cache entry without loading encoder weights."""

        if not self.options.require_preprocessing_cache:
            raise RuntimeError("This NeuroVFM execution does not require preprocessing cache")
        return self._load_preprocessing_cache(
            record,
            retain_in_memory=retain_in_memory,
        )

    def preprocessing_cache_statistics(self) -> dict[str, Any]:
        """Report disk loads, RAM hits, and retained tensor bytes for provenance."""

        tensor_bytes = 0
        tensor_count = 0
        for batch in self._preprocessing_memory_cache.values():
            for value in batch.values():
                if hasattr(value, "numel") and hasattr(value, "element_size"):
                    tensor_count += 1
                    tensor_bytes += int(value.numel() * value.element_size())
        return {
            "configured_root": (
                str(Path(self.options.preprocessing_cache_root).expanduser().resolve())
                if self.options.preprocessing_cache_root is not None
                else None
            ),
            "required": bool(self.options.require_preprocessing_cache),
            "memory_cache_enabled": bool(self.options.memory_cache),
            "memory_items": len(self._preprocessing_memory_cache),
            "memory_tensor_count": tensor_count,
            "memory_tensor_bytes": tensor_bytes,
            "memory_tensor_gib": float(tensor_bytes / 1024**3),
            "disk_loads": self._cache_disk_loads,
            "memory_hits": self._cache_memory_hits,
            "raw_preprocessing_calls": self._raw_preprocessing_calls,
        }

    def _load_batch(self, record: Any) -> dict[str, Any]:
        self._require_loaded()
        import torch

        if self.options.preprocessing_cache_root is not None:
            artifact, sidecar = self.preprocessing_cache_paths(record)
            if artifact.is_file() and sidecar.is_file():
                batch = self._load_preprocessing_cache(
                    record,
                    retain_in_memory=self.options.memory_cache,
                )
            elif self.options.require_preprocessing_cache:
                missing = artifact if not artifact.is_file() else sidecar
                raise FileNotFoundError(missing)
            else:
                batch = self.preprocessor.load_study(
                    str(self.input_path(record)), modality="mri"
                )
                self._raw_preprocessing_calls += 1
                batch = self._validate_preprocessed_batch(torch, batch)
        else:
            batch = self.preprocessor.load_study(str(self.input_path(record)), modality="mri")
            self._raw_preprocessing_calls += 1
            batch = self._validate_preprocessed_batch(torch, batch)
        token_count = int(batch["img"].shape[0])
        self.last_token_count = token_count
        return batch

    def _validate_tokens(self, tokens: Any) -> Any:
        import torch

        if tokens.ndim != 2 or int(tokens.shape[1]) != self.embedding_dim:
            raise ValueError(
                f"NeuroVFM produced token shape {tuple(tokens.shape)}; "
                f"expected (N, {self.embedding_dim})"
            )
        if int(tokens.shape[0]) <= 0 or not torch.isfinite(tokens).all():
            raise ValueError("NeuroVFM returned empty or non-finite tokens")
        return tokens

    def _differentiable_tokens(self, batch: dict[str, Any]) -> Any:
        """Repeat EncoderPipeline.embed without its inference_mode decorator."""

        import torch

        self._require_loaded()
        tokens = batch["img"].to(self.device)
        coords = batch["coords"].to(self.device)
        cu_seqlens = batch["series_cu_seqlens"].to(self.device)
        max_seqlen = batch["series_max_len"]
        masks_source = batch.get("series_masks_indices")
        masks = (
            masks_source.to(self.device)
            if masks_source is not None and masks_source.numel() > 0
            else None
        )
        # EncoderPipeline.embed normalizes before entering its BF16 autocast.
        # Preserve that boundary in the differentiable path used for adaptation.
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            tokens = self.pipeline.norm_module.normalize(
                tokens.float(),
                batch["mode"],
                batch["path"],
                cu_seqlens=cu_seqlens,
                sizes=batch.get("size"),
            )
        return self._require_loaded()(
            tokens,
            coords,
            masks=masks,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            use_flash_attn=False,
        )

    def encode(self, record: Any, *, training: bool, epoch: int, seed: int) -> Any:
        """Return one study-defined mean-pooled 768-D vector per raw T1w volume."""

        del epoch, seed
        batch = self._load_batch(record)
        if training or self._controlled:
            tokens = self._differentiable_tokens(batch)
        else:
            # This is the exact released frozen path used for the canonical extraction.
            tokens = self.pipeline.embed(batch, use_amp=True)
        tokens = self._validate_tokens(tokens)
        embedding = tokens.float().mean(dim=0)
        if embedding.ndim != 1 or int(embedding.shape[0]) != self.embedding_dim:
            raise ValueError(f"Invalid NeuroVFM pooled shape: {tuple(embedding.shape)}")
        return embedding

    def encode_official(self, record: Any) -> Any:
        """Run the released inference-mode path irrespective of adaptation state."""

        batch = self._load_batch(record)
        tokens = self._validate_tokens(self.pipeline.embed(batch, use_amp=True))
        embedding = tokens.float().mean(dim=0)
        if embedding.ndim != 1 or int(embedding.shape[0]) != self.embedding_dim:
            raise ValueError(f"Invalid NeuroVFM official pooled shape: {tuple(embedding.shape)}")
        return embedding

    def set_frozen(self) -> dict[str, Any]:
        model = self._require_loaded()
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in self.pipeline.norm_module.parameters():
            parameter.requires_grad = False
        model.eval()
        self.pipeline.norm_module.eval()
        self._controlled = False
        return self._parameter_summary([])

    def controlled_modules(self) -> list[tuple[str, Any]]:
        model = self._require_loaded()
        if not hasattr(model, "blocks") or len(model.blocks) != 12:
            raise TypeError("Expected the released 12-block NeuroVFM ViT-Base")
        if not hasattr(model, "norm") or model.norm is None:
            raise TypeError("The loaded NeuroVFM encoder has no final norm module")
        return [("blocks.11", model.blocks[-1]), ("norm", model.norm)]

    def set_controlled_adaptation(self) -> dict[str, Any]:
        model = self._require_loaded()
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in self.pipeline.norm_module.parameters():
            parameter.requires_grad = False
        components = self.controlled_modules()
        permitted: set[int] = set()
        counts: dict[str, int] = {}
        for label, module in components:
            count = 0
            for parameter in module.parameters():
                parameter.requires_grad = True
                permitted.add(id(parameter))
                count += int(parameter.numel())
            if count == 0:
                raise RuntimeError(f"Controlled module {label!r} has no parameters")
            counts[label] = count
        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in permitted
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable NeuroVFM parameters: {unexpected[:10]}")
        model.eval()
        self.pipeline.norm_module.eval()
        self._controlled = True
        summary = self._parameter_summary(components)
        summary["component_parameter_counts"] = counts
        return summary

    def controlled_train_mode(self) -> None:
        """Keep the whole encoder deterministic while retaining target gradients."""

        self._require_loaded().eval()
        self.pipeline.norm_module.eval()

    def trainable_named_parameters(self) -> list[tuple[str, Any]]:
        return [
            (name, parameter)
            for name, parameter in self._require_loaded().named_parameters()
            if parameter.requires_grad
        ]

    def _parameter_summary(self, components: list[tuple[str, Any]]) -> dict[str, Any]:
        model = self._require_loaded()
        total = int(sum(parameter.numel() for parameter in model.parameters()))
        trainable = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
        return {
            "model": self.model_name,
            "variant": self.variant,
            "embedding_dim": self.embedding_dim,
            "readout": self.readout,
            "adaptation_boundary": self.adaptation_boundary,
            "encoder_total_parameters": total,
            "encoder_trainable_parameters": trainable,
            "encoder_trainable_fraction": float(trainable / total) if total else 0.0,
            "trainable_components": [label for label, _ in components],
            "trainable_parameter_names": [
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            ],
            "mode_during_adaptation": "encoder eval; gradients only through blocks.11 and norm",
            "input_normalization_trainable": False,
        }

    def release_provenance(self) -> dict[str, Any]:
        snapshot, details = self._resolve_release()
        return {
            **details,
            "resolved_snapshot": str(snapshot),
        }

    def code_paths(self) -> list[Path]:
        paths = [
            PROJECT_ROOT / "Emb_extractors" / "extract_NeuroVFM.py",
            Path(__file__).resolve(),
        ]
        if self._release_details is not None:
            package_root = Path(self._release_details["code"]["package_root"])
            paths.extend(package_root / relative for relative in NEUROVFM_CODE_SHA256)
        return paths

    def audit_description(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "variant": self.variant,
            "environment": self.environment,
            "embedding_dim": self.embedding_dim,
            "readout": self.readout,
            "readout_status": "study-defined untrained token mean",
            "adaptation_boundary": self.adaptation_boundary,
            "input_contract": {"required_runtime_condition": self.model_name},
            "model_source": self.options.model_source,
            "model_revision": self.options.model_revision,
            "allow_download": self.options.allow_download,
            "preprocessing_cache": {
                "version": PREPROCESSING_CACHE_VERSION,
                "root": (
                    str(Path(self.options.preprocessing_cache_root).expanduser().resolve())
                    if self.options.preprocessing_cache_root is not None
                    else None
                ),
                "required": bool(self.options.require_preprocessing_cache),
                "memory_reuse": bool(self.options.memory_cache),
                "stores_encoder_outputs": False,
            },
        }
