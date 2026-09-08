#!/usr/bin/env python3
"""BrainSegFounder checkpoint, readout, and frozen model-input contract."""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FineTuning.shared.cohort import (  # noqa: E402
    DEFAULT_MANIFEST,
    EXPECTED_SPLIT_DATASET_COUNTS,
    MANIFEST_SHA256,
    load_and_validate_manifest as _load_and_validate_manifest,
    sha256_file,
)


MODEL_NAME = "BrainSegFounder"
VARIANT = "bottleneck_mean"
ENVIRONMENT = "myLocalEnv"
EMBEDDING_DIM = 768
READOUT = "spatial mean of the final 3x3x3 Swin bottleneck"
ADAPTATION_BOUNDARY = "swinViT.layers4 (final Swin stage)"
OFFLINE_INPUT_GROUP = "preproc_brainsegfounder"

OFFICIAL_REPOSITORY = "https://github.com/lab-smile/BrainSegFounder"
OFFICIAL_CODE_COMMIT = "0bbf43282447b8f998e4ed47afe26b84e8f05419"
CHECKPOINT_RELEASE = "64-gpu-model_bestValRMSE.pt"
CHECKPOINT_FILENAME = "64-gpu-model_bestValRMSE.pt"
CHECKPOINT_SHA256 = "4be92492ae4f55e934e278700a13a6c18ef8406daba2177c5f9f157dcff5f341"

PREPROCESSING_PIPELINE_VERSION = "brainsegfounder-ukb-gdcoff-finetuning-v4"
PREPROCESSING_CONTRACT_SHA256 = (
    "cdcf47bab70947585cc16578eb02cf1b67a4ece862d1137835802889c4b5f2ff"
)
COMPATIBLE_PREPROCESSING_CONTRACTS = {
    "brainsegfounder-ukb-gdcoff-finetuning-v3": (
        "9aa88a3f0ee244725b6d6326cb77b5996062250bfc3513f0ce0140957fb23e3e"
    ),
    PREPROCESSING_PIPELINE_VERSION: PREPROCESSING_CONTRACT_SHA256,
}
MODEL_INPUT_SHAPE = (96, 96, 96)
BOTTLENECK_SHAPE = (1, EMBEDDING_DIM, 3, 3, 3)
ALLOWED_NON_ENCODER_PREFIXES = ("rotation_head.", "contrastive_head.", "conv.")


@dataclass(frozen=True)
class BrainSegFounderOptions:
    """Paths that identify the checkpoint and offline preprocessing outputs."""

    checkpoint: Path | None = None
    preproc_root: Path | None = None


def _record_value(record: Any, key: str) -> Any:
    """Read a manifest field from a mapping, pandas row, or named tuple."""

    if isinstance(record, Mapping):
        return record[key]
    try:
        return record[key]
    except (KeyError, TypeError, IndexError):
        return getattr(record, key)


def _first_existing(candidates: Iterable[Path]) -> Path:
    """Resolve the first existing path while retaining a useful absent default."""

    paths = [Path(path).expanduser() for path in candidates]
    if not paths:
        raise ValueError("No path candidates were supplied")
    for path in paths:
        if path.exists():
            return path.resolve()
    return paths[0].resolve()


def _input_file_signature(paths: Iterable[Path]) -> tuple[tuple[str, int, int, int], ...]:
    """Return cheap file identities used to invalidate provenance caches."""

    signature: list[tuple[str, int, int, int]] = []
    for path in paths:
        try:
            metadata = path.stat()
        except FileNotFoundError:
            raise FileNotFoundError(path) from None
        if not stat.S_ISREG(metadata.st_mode):
            raise FileNotFoundError(path)
        signature.append(
            (str(path), int(metadata.st_size), int(metadata.st_mtime_ns), int(metadata.st_ctime_ns))
        )
    return tuple(signature)


def default_checkpoint(override: Path | None = None) -> Path:
    """Resolve the pinned public BrainSegFounder checkpoint."""

    if override is not None:
        return Path(override).expanduser().resolve()
    candidates: list[Path] = []
    configured = os.environ.get("BRAINSEGFOUNDER_CHECKPOINT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            PROJECT_ROOT / "Models" / "BrainSegFounder" / CHECKPOINT_FILENAME,
            Path("/home/rafa/workRafa/FoundationModels_OH_T1w/Models/BrainSegFounder")
            / CHECKPOINT_FILENAME,
            Path("/datos/work/rnavgon/FoundationModels_OH_T1w/Models/BrainSegFounder")
            / CHECKPOINT_FILENAME,
        ]
    )
    return _first_existing(candidates)


def default_preproc_root(override: Path | None = None) -> Path:
    """Resolve dataset-scoped BrainSegFounder preprocessing outputs."""

    if override is not None:
        return Path(override).expanduser().resolve()
    candidates: list[Path] = []
    configured = os.environ.get("BRAINSEGFOUNDER_PREPROC_ROOT")
    if configured:
        candidates.append(Path(configured))
    relative = Path("preprocessings/FineTunningDataset/preproc_BrainSegFounder")
    candidates.extend(
        [
            PROJECT_ROOT / relative,
            Path("/home/rafa/workRafa/FoundationModels_OH_T1w") / relative,
            Path("/datos/work/rnavgon/FoundationModels_OH_T1w") / relative,
        ]
    )
    return _first_existing(candidates)


def load_and_validate_manifest(
    path: Path | str = DEFAULT_MANIFEST,
    *,
    expected_sha256: str | None = MANIFEST_SHA256,
    expected_counts: dict[str, dict[str, int]] | None = EXPECTED_SPLIT_DATASET_COUNTS,
    check_input_files: bool = False,
) -> tuple[Any, dict[str, object]]:
    """Validate the cohort and permission to use offline BSF inputs."""

    return _load_and_validate_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_counts=expected_counts,
        required_offline_group=OFFLINE_INPUT_GROUP,
        check_raw_input_files=check_input_files,
    )


def clean_swin_state_dict(
    state_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int], int]:
    """Isolate the Swin encoder and account for every excluded SSL tensor."""

    cleaned: dict[str, Any] = {}
    dropped_counts = {prefix.rstrip("."): 0 for prefix in ALLOWED_NON_ENCODER_PREFIXES}
    dropped_parameters = 0
    unexpected: list[str] = []
    for original_key, value in state_dict.items():
        key = str(original_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("swinViT."):
            encoder_key = key[len("swinViT.") :]
            if encoder_key in cleaned:
                raise ValueError(f"Duplicate Swin key: {encoder_key}")
            cleaned[encoder_key] = value
            continue
        matched = next((prefix for prefix in ALLOWED_NON_ENCODER_PREFIXES if key.startswith(prefix)), None)
        if matched is None:
            unexpected.append(str(original_key))
            continue
        if not hasattr(value, "numel"):
            raise TypeError(f"Non-tensor checkpoint value: {original_key}")
        dropped_counts[matched.rstrip(".")] += 1
        dropped_parameters += int(value.numel())
    if unexpected:
        raise ValueError(f"Unexpected BrainSegFounder checkpoint keys; first 10: {unexpected[:10]}")
    if not cleaned:
        raise ValueError("Checkpoint contains no swinViT encoder weights")
    if any(count == 0 for count in dropped_counts.values()):
        raise ValueError(f"Expected SSL branches are absent: {dropped_counts}")
    return cleaned, dropped_counts, dropped_parameters


class BrainSegFounderModel:
    """Released SwinUNETR encoder with a study-defined bottleneck mean readout."""

    model_name = MODEL_NAME
    variant = VARIANT
    environment = ENVIRONMENT
    embedding_dim = EMBEDDING_DIM
    readout = READOUT
    adaptation_boundary = ADAPTATION_BOUNDARY

    def __init__(self, options: BrainSegFounderOptions) -> None:
        self.options = options
        self.checkpoint = default_checkpoint(options.checkpoint)
        self.preproc_root = default_preproc_root(options.preproc_root)
        self.model: Any | None = None
        self.device: Any | None = None
        self.last_input_provenance: dict[str, Any] | None = None
        self._release_details: dict[str, Any] | None = None
        self._load_details: dict[str, Any] | None = None
        self._ssl_parameters: int | None = None
        self._input_provenance_cache: dict[
            tuple[str, str, str],
            tuple[tuple[tuple[str, int, int, int], ...], dict[str, Any]],
        ] = {}

    def subject_dir(self, record: Any) -> Path:
        return self.preproc_root / str(_record_value(record, "dataset")) / str(
            _record_value(record, "subject")
        )

    def input_path(self, record: Any) -> Path:
        subject = str(_record_value(record, "subject"))
        return self.subject_dir(record) / f"{subject}_brainsegfounder_ready.nii.gz"

    def qc_path(self, record: Any) -> Path:
        return self.subject_dir(record) / "preprocessing_qc.json"

    def required_input_paths(self, record: Any) -> list[Path]:
        return [self.input_path(record), self.qc_path(record)]

    def release_provenance(self) -> dict[str, Any]:
        """Require the exact public checkpoint used by the reliability analysis."""

        if self._release_details is not None:
            return self._release_details
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        observed = sha256_file(self.checkpoint)
        if observed != CHECKPOINT_SHA256:
            raise ValueError(
                "BrainSegFounder checkpoint drift: "
                f"expected {CHECKPOINT_SHA256}, observed {observed}"
            )
        self._release_details = {
            "repository": OFFICIAL_REPOSITORY,
            "code_commit": OFFICIAL_CODE_COMMIT,
            "checkpoint_release": CHECKPOINT_RELEASE,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": observed,
        }
        return self._release_details

    def validate_input_provenance(self, record: Any) -> dict[str, Any]:
        """Verify PASS QC and the hash of the exact model-ready volume."""

        image_path = self.input_path(record)
        qc_path = self.qc_path(record)
        paths = (image_path, qc_path)
        signature = _input_file_signature(paths)
        cache_key = tuple(
            str(_record_value(record, key)) for key in ("dataset", "subject", "split")
        )
        cached = self._input_provenance_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return dict(cached[1])

        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        expected_fields = {
            "status": "PASS",
            "dataset": str(_record_value(record, "dataset")),
            "subject": str(_record_value(record, "subject")),
            "split": str(_record_value(record, "split")),
        }
        mismatches = {
            key: {"expected": expected, "observed": qc.get(key)}
            for key, expected in expected_fields.items()
            if qc.get(key) != expected
        }
        observed_version = qc.get("pipeline_version")
        observed_contract = qc.get("pipeline_contract_sha256")
        if (
            observed_version not in COMPATIBLE_PREPROCESSING_CONTRACTS
            or COMPATIBLE_PREPROCESSING_CONTRACTS[observed_version]
            != observed_contract
        ):
            mismatches["preprocessing_contract"] = {
                "expected": COMPATIBLE_PREPROCESSING_CONTRACTS,
                "observed": {
                    "pipeline_version": observed_version,
                    "pipeline_contract_sha256": observed_contract,
                },
            }
        if mismatches:
            raise ValueError(f"Stale BrainSegFounder preprocessing QC: {mismatches}")
        output_hashes = qc.get("output_sha256")
        if not isinstance(output_hashes, dict):
            raise ValueError(f"Missing output hashes in {qc_path}")
        observed_hash = sha256_file(image_path)
        if output_hashes.get("ready") != observed_hash:
            raise ValueError("BrainSegFounder model-ready input hash differs from QC")
        provenance = {
            "qc_path": str(qc_path),
            "qc_sha256": sha256_file(qc_path),
            "image_path": str(image_path),
            "image_sha256": observed_hash,
            "pipeline_version": qc["pipeline_version"],
            "pipeline_contract_sha256": qc["pipeline_contract_sha256"],
        }
        final_signature = _input_file_signature(paths)
        if final_signature != signature:
            raise RuntimeError("BrainSegFounder input changed during provenance validation")
        self._input_provenance_cache[cache_key] = (final_signature, provenance)
        return dict(provenance)

    def load_input(self, record: Any) -> Any:
        """Load one already normalized 96-cube without another transformation."""

        import nibabel as nib
        import torch

        provenance = self.validate_input_provenance(record)
        image = nib.load(str(self.input_path(record)))
        if image.ndim != 3 or tuple(image.shape) != MODEL_INPUT_SHAPE:
            raise ValueError(
                f"BrainSegFounder requires an exact 3D {MODEL_INPUT_SHAPE} volume; "
                f"got {image.shape}"
            )
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("BrainSegFounder input contains non-finite values")
        if not np.any(data != 0):
            raise ValueError("BrainSegFounder input is empty")
        self.last_input_provenance = provenance
        return torch.from_numpy(np.ascontiguousarray(data[None, None, ...]))

    def load(self, device: Any) -> None:
        """Strictly load the released swinViT encoder and exclude SSL heads."""

        import torch
        from monai.networks.nets import SwinUNETR

        release = self.release_provenance()
        checkpoint = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, Mapping):
            raise ValueError("BrainSegFounder checkpoint lacks a tensor mapping")
        cleaned, dropped_counts, dropped_parameters = clean_swin_state_dict(state_dict)
        model = SwinUNETR(
            img_size=MODEL_INPUT_SHAPE,
            in_channels=1,
            out_channels=1,
            feature_size=48,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
        )
        probe_key = next(iter(cleaned))
        before = model.swinViT.state_dict()[probe_key].detach().clone()
        info = model.swinViT.load_state_dict(cleaned, strict=True)
        if info.missing_keys or info.unexpected_keys:
            raise RuntimeError("Strict swinViT load returned missing or unexpected keys")
        after = model.swinViT.state_dict()[probe_key].detach()
        changed = float((after.cpu() - before.cpu()).norm().item())
        bit_exact = bool(torch.equal(after.cpu(), cleaned[probe_key].detach().cpu()))
        if changed <= 0 or not bit_exact:
            raise RuntimeError("BrainSegFounder checkpoint proof-of-load failed")
        patch = model.swinViT.state_dict().get("patch_embed.proj.weight")
        if patch is None or tuple(patch.shape) != (48, 1, 2, 2, 2):
            raise ValueError(f"Unexpected single-channel patch embed: {getattr(patch, 'shape', None)}")
        model.eval()
        model.to(device)
        self.model = model
        self.device = device
        self._ssl_parameters = dropped_parameters
        self._load_details = {
            "strict": True,
            "swin_encoder_keys": len(cleaned),
            "probe_key": probe_key,
            "probe_changed_l2": changed,
            "probe_bit_exact": bit_exact,
            "excluded_ssl_tensor_counts": dropped_counts,
            "excluded_ssl_parameters": dropped_parameters,
            "ssl_heads_used": False,
            "checkpoint_sha256": release["checkpoint_sha256"],
        }
        print(
            "BrainSegFounder loaded strictly: "
            f"{len(cleaned)} swinViT keys; SSL heads/decoder excluded; "
            f"embedding_dim={EMBEDDING_DIM}",
            flush=True,
        )

    def _require_loaded(self) -> Any:
        if self.model is None or self.device is None:
            raise RuntimeError("BrainSegFounder has not loaded its checkpoint")
        return self.model

    def _validate_embedding(self, tensor: Any) -> Any:
        """Require one finite 768-dimensional bottleneck representation."""

        import torch

        if tensor.ndim == 2 and int(tensor.shape[0]) == 1:
            tensor = tensor[0]
        if tuple(tensor.shape) != (EMBEDDING_DIM,):
            raise RuntimeError(
                f"Unexpected BrainSegFounder embedding shape: {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise RuntimeError("BrainSegFounder produced non-finite embedding values")
        return tensor

    def set_frozen(self) -> dict[str, Any]:
        """Freeze the entire encoder for deterministic embedding extraction."""

        model = self._require_loaded()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        return self._parameter_summary([])

    def set_controlled_adaptation(self) -> dict[str, Any]:
        """Freeze SwinUNETR, then unfreeze only its final encoder stage."""

        model = self._require_loaded()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        layer4 = getattr(model.swinViT, "layers4", None)
        if layer4 is None or len(layer4) != 1:
            raise ValueError("Expected the single final layers4 container in MONAI Swin")
        for parameter in layer4.parameters():
            parameter.requires_grad_(True)
        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("swinViT.layers4.")
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable BrainSegFounder parameters: {unexpected[:10]}")
        model.eval()
        summary = self._parameter_summary(["swinViT.layers4"])
        summary["component_parameter_counts"] = {
            "swinViT.layers4": int(sum(p.numel() for p in layer4.parameters()))
        }
        return summary

    def controlled_train_mode(self) -> None:
        """Keep the whole SwinUNETR wrapper in evaluation mode while adapting."""

        self._require_loaded().eval()

    def trainable_named_parameters(self) -> list[tuple[str, Any]]:
        return [
            (name, parameter)
            for name, parameter in self._require_loaded().named_parameters()
            if parameter.requires_grad
        ]

    def _parameter_summary(self, components: list[str]) -> dict[str, Any]:
        """Describe absolute and relative Swin encoder adaptation cost."""

        model = self._require_loaded()
        encoder_total = int(sum(parameter.numel() for parameter in model.swinViT.parameters()))
        trainable = int(sum(parameter.numel() for _, parameter in self.trainable_named_parameters()))
        return {
            "model": self.model_name,
            "variant": self.variant,
            "embedding_dim": self.embedding_dim,
            "readout": self.readout,
            "adaptation_boundary": self.adaptation_boundary,
            "encoder_total_parameters": encoder_total,
            "encoder_trainable_parameters": trainable,
            "encoder_trainable_fraction": float(trainable / encoder_total) if encoder_total else 0.0,
            "trainable_components": components,
            "trainable_parameter_names": [name for name, _ in self.trainable_named_parameters()],
            "excluded_ssl_parameters": self._ssl_parameters,
            "ssl_heads_loaded": False,
            "ssl_heads_used": False,
            "mode_during_adaptation": (
                "full wrapper eval; gradients only through swinViT.layers4"
                if components
                else "full wrapper eval; no trainable parameters"
            ),
        }

    def encode(self, record: Any, *, training: bool, epoch: int, seed: int) -> Any:
        """Spatially average the final 3x3x3 Swin feature map."""

        del training, epoch, seed
        model = self._require_loaded()
        model.eval()
        volume = self.load_input(record).to(self.device, non_blocking=True)
        hidden_states = model.swinViT(volume, model.normalize)
        if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) != 5:
            raise RuntimeError("BrainSegFounder swinViT did not return five feature scales")
        bottleneck = hidden_states[4]
        if tuple(bottleneck.shape) != BOTTLENECK_SHAPE:
            raise RuntimeError(f"Unexpected Swin bottleneck shape: {tuple(bottleneck.shape)}")
        return self._validate_embedding(bottleneck.mean(dim=(2, 3, 4)))

    def audit_description(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "variant": self.variant,
            "environment": self.environment,
            "embedding_dim": self.embedding_dim,
            "readout": self.readout,
            "readout_status": "study-defined; the public release has no canonical global embedding",
            "adaptation_boundary": self.adaptation_boundary,
            "checkpoint": str(self.checkpoint),
            "input_root": str(self.preproc_root),
            "input_contract": {"required_offline_group": OFFLINE_INPUT_GROUP},
            "preprocessing_pipeline_version": PREPROCESSING_PIPELINE_VERSION,
            "preprocessing_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
            "compatible_preprocessing_contracts": dict(
                COMPATIBLE_PREPROCESSING_CONTRACTS
            ),
            "model_input_shape": list(MODEL_INPUT_SHAPE),
            "bottleneck_shape": list(BOTTLENECK_SHAPE),
        }

    def load_description(self) -> dict[str, Any]:
        if self._load_details is None:
            raise RuntimeError("Checkpoint has not been loaded")
        return dict(self._load_details)

    def code_paths(self) -> list[Path]:
        return [
            SCRIPT_PATH,
            PROJECT_ROOT
            / "preprocessings"
            / "FineTunningDataset"
            / "preprocessing_BrainSegFounder.py",
        ]
