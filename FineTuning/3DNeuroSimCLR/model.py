#!/usr/bin/env python3
"""3D-Neuro-SimCLR checkpoint and frozen model-input contract."""

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


MODEL_NAME = "3D-Neuro-SimCLR"
VARIANT = "canonical"
ENVIRONMENT = "myLocalEnv"
EMBEDDING_DIM = 512
READOUT = "global-average-pooled encoder output before the contrastive projector"
ADAPTATION_BOUNDARY = "layer4 (final 3D ResNet-18 stage)"
OFFLINE_INPUT_GROUP = "preproc_3d_neuro_simclr"

OFFICIAL_REPOSITORY = "https://github.com/emilykaczmarek/3D-Neuro-SimCLR"
OFFICIAL_CODE_COMMIT = "e061f3a19d20755d45361cb206b0803a6de804ff"
CHECKPOINT_RELEASE = "v1.0.0"
CHECKPOINT_FILENAME = "simclr_3d_brain_foundation.tar"
CHECKPOINT_SHA256 = "f046dafe009de0cfa113a022c1ef9603d123834f217074eb86d26037e72b98f5"

PREPROCESSING_PIPELINE_VERSION = "3d-neuro-simclr-turboprep-finetuning-v5"
PREPROCESSING_CONTRACT_SHA256 = (
    "b19576f63cb90ca270591631610cda23b79b44fbc48645a9424da568316204fd"
)
REGISTERED_SHAPE = (193, 229, 193)
MODEL_INPUT_SHAPE = (1, 150, 192, 192)
CROP = (slice(None), slice(10, 160), slice(19, 211), slice(1, None))


@dataclass(frozen=True)
class NeuroSimCLROptions:
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
    """Resolve the first existing path while retaining a useful default if absent."""

    paths = [Path(path).expanduser() for path in candidates]
    if not paths:
        raise ValueError("No path candidates were supplied")
    for path in paths:
        if path.exists():
            return path.resolve()
    return paths[0].resolve()


def _input_file_signature(paths: Iterable[Path]) -> tuple[tuple[str, int, int, int], ...]:
    """Return cheap file identities used to invalidate cached provenance checks."""

    signature: list[tuple[str, int, int, int]] = []
    for path in paths:
        try:
            metadata = path.stat()
        except FileNotFoundError:
            raise FileNotFoundError(path) from None
        if not stat.S_ISREG(metadata.st_mode):
            raise FileNotFoundError(path)
        signature.append(
            (
                str(path),
                int(metadata.st_size),
                int(metadata.st_mtime_ns),
                int(metadata.st_ctime_ns),
            )
        )
    return tuple(signature)


def default_checkpoint(override: Path | None = None) -> Path:
    """Resolve the pinned public checkpoint under local and cluster layouts."""

    if override is not None:
        return Path(override).expanduser().resolve()
    candidates: list[Path] = []
    configured = os.environ.get("NEUROSIMCLR_CHECKPOINT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            PROJECT_ROOT / "Models" / "3D-Neuro-SimCLR" / CHECKPOINT_FILENAME,
            Path("/home/rafa/workRafa/FoundationModels_OH_T1w/Models/3D-Neuro-SimCLR")
            / CHECKPOINT_FILENAME,
            Path("/datos/work/rnavgon/FoundationModels_OH_T1w/Models/3D-Neuro-SimCLR")
            / CHECKPOINT_FILENAME,
        ]
    )
    return _first_existing(candidates)


def default_preproc_root(override: Path | None = None) -> Path:
    """Resolve dataset-scoped 3D-Neuro-SimCLR preprocessing outputs."""

    if override is not None:
        return Path(override).expanduser().resolve()
    candidates: list[Path] = []
    configured = os.environ.get("NEUROSIMCLR_PREPROC_ROOT")
    if configured:
        candidates.append(Path(configured))
    relative = Path("preprocessings/FineTunningDataset/preproc_3DNeuroSimCLR")
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
    """Validate the cohort and its permission to use the offline SimCLR inputs."""

    return _load_and_validate_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_counts=expected_counts,
        required_offline_group=OFFLINE_INPUT_GROUP,
        check_raw_input_files=check_input_files,
    )


def transpose_crop(array: np.ndarray) -> np.ndarray:
    """Apply the exact transpose and fixed crop in the released dataset loader."""

    if tuple(array.shape) != (1, *REGISTERED_SHAPE):
        raise ValueError(
            f"Unexpected pre-crop shape {tuple(array.shape)}; "
            f"expected {(1, *REGISTERED_SHAPE)}"
        )
    cropped = np.transpose(array, (0, 3, 2, 1))[CROP]
    if tuple(cropped.shape) != MODEL_INPUT_SHAPE:
        raise RuntimeError(f"Unexpected model-input shape after crop: {cropped.shape}")
    return cropped


def masked_standardize(
    images: np.ndarray,
    masks: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply the released per-volume z-score over masked voxels only."""

    if images.shape != masks.shape or images.ndim != 5:
        raise ValueError("images and masks must be equally shaped [N,C,D,H,W] arrays")
    if not np.isfinite(images).all() or not np.isfinite(masks).all():
        raise ValueError("Masked standardization requires finite arrays")
    images = images.astype(np.float32, copy=False) * masks.astype(np.float32, copy=False)
    masks = masks.astype(np.float32, copy=False)
    axes = (2, 3, 4)
    counts = masks.sum(axis=axes, keepdims=True)
    if np.any(counts < 1):
        raise ValueError("The cropped brain mask is empty")
    means = images.sum(axis=axes, keepdims=True) / counts
    variances = np.square(images).sum(axis=axes, keepdims=True) / counts - np.square(means)
    if np.any(variances + eps <= 0):
        raise ValueError("The cropped masked image has zero variance")
    standardized = (images - means) / np.sqrt(variances + eps)
    standardized = standardized * masks
    if not np.isfinite(standardized).all():
        raise ValueError("Masked standardization produced non-finite values")
    return standardized.astype(np.float32, copy=False)


def clean_encoder_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Select encoder weights and remove only the released DDP/wrapper prefixes."""

    cleaned: dict[str, Any] = {}
    unexpected: list[str] = []
    projector_keys = 0
    for original_key, value in state_dict.items():
        key = str(original_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("projector."):
            projector_keys += 1
            continue
        if not key.startswith("encoder."):
            unexpected.append(str(original_key))
            continue
        encoder_key = key[len("encoder.") :]
        if encoder_key in cleaned:
            raise ValueError(f"Duplicate encoder key after prefix removal: {encoder_key}")
        cleaned[encoder_key] = value
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys; first 10: {unexpected[:10]}")
    if not cleaned:
        raise ValueError("Checkpoint contains no encoder weights")
    if projector_keys == 0:
        raise ValueError("Checkpoint contains no contrastive-projector weights")
    return cleaned


class NeuroSimCLRModel:
    """Public 3D ResNet-18 encoder with the exact released input loader."""

    model_name = MODEL_NAME
    variant = VARIANT
    environment = ENVIRONMENT
    embedding_dim = EMBEDDING_DIM
    readout = READOUT
    adaptation_boundary = ADAPTATION_BOUNDARY

    def __init__(self, options: NeuroSimCLROptions) -> None:
        self.options = options
        self.checkpoint = default_checkpoint(options.checkpoint)
        self.preproc_root = default_preproc_root(options.preproc_root)
        self.model: Any | None = None
        self.device: Any | None = None
        self.last_input_provenance: dict[str, Any] | None = None
        self._release_details: dict[str, Any] | None = None
        self._load_details: dict[str, Any] | None = None
        self._projector_parameters: int | None = None
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
        return self.subject_dir(record) / f"{subject}_simclr_ready.nii.gz"

    def mask_path(self, record: Any) -> Path:
        subject = str(_record_value(record, "subject"))
        return self.subject_dir(record) / f"{subject}_mask_mni.nii.gz"

    def qc_path(self, record: Any) -> Path:
        return self.subject_dir(record) / "preprocessing_qc.json"

    def required_input_paths(self, record: Any) -> list[Path]:
        return [self.input_path(record), self.mask_path(record), self.qc_path(record)]

    def release_provenance(self) -> dict[str, Any]:
        """Require the exact public checkpoint used by the reliability analysis."""

        if self._release_details is not None:
            return self._release_details
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        observed = sha256_file(self.checkpoint)
        if observed != CHECKPOINT_SHA256:
            raise ValueError(
                "3D-Neuro-SimCLR checkpoint drift: "
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
        """Verify PASS QC and hashes for the exact offline image/mask pair consumed."""

        image_path = self.input_path(record)
        mask_path = self.mask_path(record)
        qc_path = self.qc_path(record)
        paths = (image_path, mask_path, qc_path)
        signature = _input_file_signature(paths)
        cache_key = (
            str(_record_value(record, "dataset")),
            str(_record_value(record, "subject")),
            str(_record_value(record, "split")),
        )
        cached = self._input_provenance_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return dict(cached[1])

        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        expected_fields = {
            "status": "PASS",
            "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
            "pipeline_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
            "dataset": str(_record_value(record, "dataset")),
            "subject": str(_record_value(record, "subject")),
            "split": str(_record_value(record, "split")),
        }
        mismatches = {
            key: {"expected": expected, "observed": qc.get(key)}
            for key, expected in expected_fields.items()
            if qc.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Stale 3D-Neuro-SimCLR preprocessing QC: {mismatches}")
        output_hashes = qc.get("output_sha256")
        if not isinstance(output_hashes, dict):
            raise ValueError(f"Missing output hashes in {qc_path}")
        expected_hashes = {"ready": sha256_file(image_path), "mask_mni": sha256_file(mask_path)}
        hash_mismatches = {
            key: {"expected": output_hashes.get(key), "observed": observed}
            for key, observed in expected_hashes.items()
            if output_hashes.get(key) != observed
        }
        if hash_mismatches:
            raise ValueError(f"Preprocessed input hash mismatch: {hash_mismatches}")
        provenance = {
            "qc_path": str(qc_path),
            "qc_sha256": sha256_file(qc_path),
            "image_path": str(image_path),
            "image_sha256": expected_hashes["ready"],
            "mask_path": str(mask_path),
            "mask_sha256": expected_hashes["mask_mni"],
            "pipeline_version": qc["pipeline_version"],
            "pipeline_contract_sha256": qc["pipeline_contract_sha256"],
        }
        final_signature = _input_file_signature(paths)
        if final_signature != signature:
            raise RuntimeError("Preprocessed inputs changed during provenance validation")
        self._input_provenance_cache[cache_key] = (final_signature, provenance)
        return dict(provenance)

    def load_input(self, record: Any) -> Any:
        """Load, validate, crop, and standardize one image/mask pair."""

        import nibabel as nib
        import torch

        provenance = self.validate_input_provenance(record)
        image = nib.load(str(self.input_path(record)))
        mask_image = nib.load(str(self.mask_path(record)))
        if tuple(image.shape) != REGISTERED_SHAPE or tuple(mask_image.shape) != REGISTERED_SHAPE:
            raise ValueError(
                f"Unexpected image/mask grid: image={image.shape}, mask={mask_image.shape}"
            )
        if not np.allclose(image.affine, mask_image.affine, rtol=0.0, atol=1e-6):
            raise ValueError("Image and mask affines differ")
        data = image.get_fdata(dtype=np.float32)
        mask_values = mask_image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all() or not np.isfinite(mask_values).all():
            raise ValueError("Preprocessed image or mask contains non-finite values")
        binary = np.isclose(mask_values, 0.0, rtol=0.0, atol=1e-6) | np.isclose(
            mask_values, 1.0, rtol=0.0, atol=1e-6
        )
        if not binary.all():
            raise ValueError("Registered mask is not binary")
        mask = (mask_values > 0.5).astype(np.float32)
        cropped_data = transpose_crop(data[np.newaxis, ...])
        cropped_mask = transpose_crop(mask[np.newaxis, ...])
        standardized = masked_standardize(
            cropped_data[np.newaxis, ...], cropped_mask[np.newaxis, ...]
        )
        self.last_input_provenance = provenance
        return torch.from_numpy(np.ascontiguousarray(standardized))

    def load(self, device: Any) -> None:
        """Build MONAI ResNet-18 and strictly load only public encoder weights."""

        import torch
        from monai.networks.nets import resnet18

        release = self.release_provenance()
        checkpoint = torch.load(
            str(self.checkpoint), map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model_state_dict"), Mapping
        ):
            raise ValueError("Checkpoint lacks a model_state_dict mapping")
        released_state = checkpoint["model_state_dict"]
        cleaned = clean_encoder_state_dict(released_state)
        projector_parameters = 0
        for original_key, value in released_state.items():
            key = str(original_key)
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("projector."):
                if not hasattr(value, "numel"):
                    raise TypeError(
                        f"Projector checkpoint value is not tensor-like: {original_key}"
                    )
                projector_parameters += int(value.numel())
        if projector_parameters <= 0:
            raise ValueError("Checkpoint contains no projector parameters")
        model = resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False)
        probe_key = next(iter(cleaned))
        before = model.state_dict()[probe_key].detach().clone()
        model.load_state_dict(cleaned, strict=True)
        after = model.state_dict()[probe_key].detach()
        changed = float((after.cpu() - before.cpu()).norm().item())
        bit_exact = bool(torch.equal(after.cpu(), cleaned[probe_key].detach().cpu()))
        if changed <= 0 or not bit_exact:
            raise RuntimeError("Strict checkpoint proof-of-load failed")
        model.eval()
        model.to(device)
        self.model = model
        self.device = device
        self._projector_parameters = projector_parameters
        self._load_details = {
            "strict": True,
            "encoder_keys": len(cleaned),
            "probe_key": probe_key,
            "probe_changed_l2": changed,
            "probe_bit_exact": bit_exact,
            "projector_retained": False,
            "projector_parameters": projector_parameters,
            "checkpoint_sha256": release["checkpoint_sha256"],
        }
        print(
            "3D-Neuro-SimCLR loaded strictly: "
            f"{len(cleaned)} encoder keys; projector excluded; embedding_dim={EMBEDDING_DIM}",
            flush=True,
        )

    def _require_loaded(self) -> Any:
        if self.model is None or self.device is None:
            raise RuntimeError("3D-Neuro-SimCLR has not loaded its checkpoint")
        return self.model

    def _validate_embedding(self, tensor: Any) -> Any:
        """Require one finite 512-dimensional encoder representation."""

        import torch

        if tensor.ndim == 2 and int(tensor.shape[0]) == 1:
            tensor = tensor[0]
        if tuple(tensor.shape) != (EMBEDDING_DIM,):
            raise RuntimeError(f"Unexpected embedding shape: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Encoder produced non-finite embedding values")
        return tensor

    def set_frozen(self) -> dict[str, Any]:
        """Freeze the entire encoder for deterministic embedding extraction."""

        model = self._require_loaded()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        return self._parameter_summary([])

    def controlled_modules(self) -> list[tuple[str, Any]]:
        """Expose exactly the final two-block ResNet stage for adaptation."""

        layer4 = getattr(self._require_loaded(), "layer4", None)
        if layer4 is None or len(layer4) != 2:
            raise ValueError("Expected the two-block layer4 of a 3D ResNet-18")
        return [("layer4", layer4)]

    def set_controlled_adaptation(self) -> dict[str, Any]:
        """Freeze the encoder, then unfreeze only its declared final stage."""

        model = self._require_loaded()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        components = self.controlled_modules()
        permitted: set[int] = set()
        counts: dict[str, int] = {}
        for label, module in components:
            count = 0
            for parameter in module.parameters():
                parameter.requires_grad_(True)
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
            raise RuntimeError(
                f"Unexpected trainable 3D-Neuro-SimCLR parameters: {unexpected[:10]}"
            )
        model.eval()
        summary = self._parameter_summary(components)
        summary["component_parameter_counts"] = counts
        return summary

    def controlled_train_mode(self) -> None:
        """Keep all BatchNorm running statistics fixed while optimizing layer4."""

        self._require_loaded().eval()

    def trainable_named_parameters(self) -> list[tuple[str, Any]]:
        """Return only parameters admitted by the controlled boundary."""

        return [
            (name, parameter)
            for name, parameter in self._require_loaded().named_parameters()
            if parameter.requires_grad
        ]

    def _parameter_summary(self, components: list[tuple[str, Any]]) -> dict[str, Any]:
        """Describe absolute and relative encoder adaptation cost."""

        model = self._require_loaded()
        encoder_total = int(sum(parameter.numel() for parameter in model.parameters()))
        encoder_trainable = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
        return {
            "model": self.model_name,
            "variant": self.variant,
            "embedding_dim": self.embedding_dim,
            "readout": self.readout,
            "adaptation_boundary": self.adaptation_boundary,
            "encoder_total_parameters": encoder_total,
            "encoder_trainable_parameters": encoder_trainable,
            "encoder_trainable_fraction": (
                float(encoder_trainable / encoder_total) if encoder_total else 0.0
            ),
            "trainable_components": [label for label, _ in components],
            "trainable_parameter_names": [
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            ],
            "contrastive_projection_parameters": self._projector_parameters,
            "contrastive_projection_loaded": False,
            "contrastive_projection_used": False,
            "mode_during_adaptation": (
                "full encoder eval; gradients only through layer4 parameters"
            ),
        }

    def encode(self, record: Any, *, training: bool, epoch: int, seed: int) -> Any:
        """Return one 512-dimensional global-pooled encoder representation."""

        del training, epoch, seed

        model = self._require_loaded()
        model.eval()
        volume = self.load_input(record).to(self.device, non_blocking=True)
        return self._validate_embedding(model(volume))

    def audit_description(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "variant": self.variant,
            "environment": self.environment,
            "embedding_dim": self.embedding_dim,
            "readout": self.readout,
            "adaptation_boundary": self.adaptation_boundary,
            "checkpoint": str(self.checkpoint),
            "input_root": str(self.preproc_root),
            "input_contract": {"required_offline_group": OFFLINE_INPUT_GROUP},
            "preprocessing_pipeline_version": PREPROCESSING_PIPELINE_VERSION,
            "preprocessing_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
            "registered_shape": list(REGISTERED_SHAPE),
            "model_input_shape": list(MODEL_INPUT_SHAPE),
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
            / "preprocessing_3DNeuroSimCLR.py",
        ]
