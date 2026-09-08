#!/usr/bin/env python3
"""Curia-2 checkpoint, volume readout, and controlled-adaptation boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from FineTuning.shared.cohort import (
    DEFAULT_MANIFEST,
    EXPECTED_SPLIT_DATASET_COUNTS,
    MANIFEST_SHA256,
    PROJECT_ROOT,
    load_and_validate_manifest as _load_and_validate_manifest,
    sha256_file,
)


MODEL_NAME = "Curia-2"
VARIANT = "canonical"
EMBEDDING_DIM = 768
READOUT = "mean CLS token across all axial slices"
ADAPTATION_BOUNDARY = "final DINOv2 block + final layer norm + linear age head"

CURIA2_RELEASE_REVISION = "25d164965f10ceaf6b394f803a4704598cb3ab1a"
CURIA2_RELEASE_HASHES = {
    "config.json": "9b22dcc40c81d8e3d536978cb81db049e9940ea6b02b3bfc815e1e781d57f252",
    "curia_image_processor.py": "74a4eb5e27539686965327c4e8293648a552777452a51eb1ccac956067cc236b",
    "preprocessor_config.json": "469b3bed97af2b8923002a11a6fe4500f34bcae39a49c5822ae15cefa15702e5",
    "model.safetensors": "403a02e27531d2858ecd1e9b1ec2d5ea7bfa909f10ff0d9e8416090a6a8c96ef",
}
TARGET_AXCODES = ("L", "A", "S")


@dataclass(frozen=True)
class Curia2Options:
    """Paths and slice settings that define one Curia-2 execution."""

    model_dir: Path | None = None
    slice_batch_size: int = 8

    def validate(self) -> None:
        if self.slice_batch_size <= 0:
            raise ValueError("slice_batch_size must be positive")


def _record_value(record: Any, key: str) -> Any:
    """Read a manifest value from a mapping, pandas row, or named tuple."""

    if isinstance(record, Mapping):
        return record[key]
    try:
        return record[key]
    except (KeyError, TypeError, IndexError):
        return getattr(record, key)


def _first_existing(candidates: Iterable[Path]) -> Path:
    """Return the first existing candidate, retaining a useful missing path otherwise."""

    paths = [Path(path).expanduser() for path in candidates]
    if not paths:
        raise ValueError("No Curia-2 model directory candidates were supplied")
    for path in paths:
        if path.exists():
            return path.resolve()
    return paths[0].resolve()


def default_model_dir(override: Path | None = None) -> Path:
    """Resolve the pinned Curia-2 release under known local and cluster layouts."""

    candidates: list[Path] = []
    if override is not None:
        candidates.append(override)
    configured = os.environ.get("CURIA2_MODEL_DIR")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            PROJECT_ROOT / "Models" / "Curia-2",
            PROJECT_ROOT.parent / "FoundationModels_Paper" / "Models" / "Curia-2",
            Path("/home/rafa/workRafa/FoundationModels_Paper/Models/Curia-2"),
            Path("/datos/work/rnavgon/FoundationModels_Paper/Models/Curia-2"),
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
    """Validate the shared cohort and its permission to use raw T1w with Curia-2."""

    return _load_and_validate_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_counts=expected_counts,
        required_runtime_condition=MODEL_NAME,
        check_raw_input_files=check_input_files,
    )


def validate_curia2_release(model_dir: Path | str) -> dict[str, object]:
    """Require the exact public Curia-2 checkpoint and processor used upstream."""

    directory = Path(model_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    observed: dict[str, str] = {}
    for filename, expected in CURIA2_RELEASE_HASHES.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        observed[filename] = digest
        if digest != expected:
            raise ValueError(
                f"Curia-2 release drift for {filename}: expected {expected}, observed {digest}"
            )
    return {
        "model_dir": str(directory),
        "huggingface_repo": "raidium/curia-2",
        "downloaded_revision": CURIA2_RELEASE_REVISION,
        "files_sha256": observed,
    }


def _load_release_model(model_dir: Path, device: Any) -> tuple[Any, Any]:
    """Load the pinned Curia-2 backbone and its released bicubic processor."""

    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)

    expected_config = {
        "model_type": "dinov2",
        "hidden_size": 768,
        "image_size": 512,
        "patch_size": 16,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_channels": 1,
    }
    config_mismatches = {
        key: (getattr(model.config, key, None), expected)
        for key, expected in expected_config.items()
        if getattr(model.config, key, None) != expected
    }
    if config_mismatches:
        raise ValueError(f"Curia-2 model configuration drift: {config_mismatches}")

    expected_processor = {
        "crop_size": 512,
        "do_resize": True,
        "do_normalize": True,
        "clip_below_air": True,
        "eps": 1e-6,
    }
    processor_mismatches = {
        key: (getattr(processor, key, None), expected)
        for key, expected in expected_processor.items()
        if getattr(processor, key, None) != expected
    }
    if processor_mismatches:
        raise ValueError(f"Curia-2 processor configuration drift: {processor_mismatches}")

    model.eval()
    model.to(device)
    config = model.config
    print(
        f"Curia-2 loaded from {model_dir}  "
        f"(hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
        f"image_size={config.image_size}, patch_size={config.patch_size}, "
        "resize_interp=bicubic)"
    )
    return model, processor


def _load_las_volume(path: Path) -> np.ndarray:
    """Load a raw NIfTI and reorient its voxel array to deterministic LAS."""

    import nibabel as nib
    from nibabel.orientations import (
        apply_orientation,
        axcodes2ornt,
        io_orientation,
        ornt_transform,
    )

    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(
            "Curia-2 requires an exactly 3D NIfTI volume; "
            f"got shape {tuple(image.shape)} for {path}"
        )
    volume = image.get_fdata().astype(np.float32)
    axcodes = nib.aff2axcodes(image.affine)
    if axcodes != TARGET_AXCODES:
        current = io_orientation(image.affine)
        target = axcodes2ornt(TARGET_AXCODES)
        volume = apply_orientation(volume, ornt_transform(current, target))
    return volume


def las_slice_to_pl(volume: np.ndarray, index: int) -> np.ndarray:
    """Rotate one axial LAS slice from LA array axes to the required PL axes."""

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D LAS volume, got shape {volume.shape}")
    if not 0 <= index < volume.shape[-1]:
        raise IndexError(index)
    return np.rot90(volume[..., index].astype(np.float32, copy=False), k=1).copy()


class Curia2Model:
    """Curia-2 inference readout and exact last-block adaptation boundary."""

    model_name = MODEL_NAME
    variant = VARIANT
    embedding_dim = EMBEDDING_DIM
    readout = READOUT
    adaptation_boundary = ADAPTATION_BOUNDARY

    def __init__(self, options: Curia2Options) -> None:
        options.validate()
        self.options = options
        self.model_dir = default_model_dir(options.model_dir)
        self.model: Any | None = None
        self.processor: Any | None = None
        self.device: Any | None = None

    def input_path(self, record: Any) -> Path:
        return Path(str(_record_value(record, "raw_t1w_path")))

    def required_input_paths(self, record: Any) -> list[Path]:
        return [self.input_path(record)]

    def load(self, device: Any) -> None:
        """Verify and load the pinned Curia-2 release for downstream modelling."""

        validate_curia2_release(self.model_dir)
        self.model, self.processor = _load_release_model(self.model_dir, device)
        self.device = device

    def _require_loaded(self) -> Any:
        if self.model is None:
            raise RuntimeError("Curia-2 has not loaded its checkpoint")
        return self.model

    def _validate_embedding(self, tensor: Any) -> Any:
        if tensor.ndim == 2 and int(tensor.shape[0]) == 1:
            tensor = tensor[0]
        if tensor.ndim != 1 or int(tensor.shape[0]) != self.embedding_dim:
            raise ValueError(
                f"Curia-2 produced shape {tuple(tensor.shape)}; "
                f"expected ({self.embedding_dim},)"
            )
        return tensor

    def encode(self, record: Any, *, training: bool, epoch: int, seed: int) -> Any:
        """Return the mean CLS vector over the selected axial slices."""

        del epoch, seed
        if self.processor is None:
            raise RuntimeError("Curia-2 processor is unavailable before load()")
        import torch

        volume = _load_las_volume(self.input_path(record))
        selected = range(volume.shape[-1])

        chunk_sums = []
        for start in range(0, len(selected), self.options.slice_batch_size):
            indices = selected[start : start + self.options.slice_batch_size]
            slices = [torch.from_numpy(las_slice_to_pl(volume, index)) for index in indices]
            pixels = self.processor(slices, return_tensors="pt")["pixel_values"]
            if tuple(pixels.shape[1:]) != (1, 512, 512):
                raise ValueError(f"Unexpected Curia-2 processor output: {tuple(pixels.shape)}")
            if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
                raise ValueError("Curia-2 processor returned invalid pixel values")
            pixels = pixels.to(self.device, non_blocking=True)
            hidden = self._require_loaded()(pixel_values=pixels).last_hidden_state
            chunk_sums.append(hidden[:, 0, :].float().sum(dim=0))
        embedding = torch.stack(chunk_sums).sum(dim=0) / len(selected)
        return self._validate_embedding(embedding)

    def set_frozen(self) -> dict[str, Any]:
        model = self._require_loaded()
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        return self._parameter_summary([])

    def controlled_modules(self) -> list[tuple[str, Any]]:
        model = self._require_loaded()
        if len(model.encoder.layer) != 12:
            raise ValueError(f"Expected 12 Curia-2 blocks, observed {len(model.encoder.layer)}")
        return [("encoder.layer.11", model.encoder.layer[-1]), ("layernorm", model.layernorm)]

    def set_controlled_adaptation(self) -> dict[str, Any]:
        model = self._require_loaded()
        for parameter in model.parameters():
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
            raise RuntimeError(f"Unexpected trainable Curia-2 parameters: {unexpected[:10]}")
        model.eval()
        summary = self._parameter_summary(components)
        summary["component_parameter_counts"] = counts
        return summary

    def controlled_train_mode(self) -> None:
        self._require_loaded().eval()

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
            "mode_during_adaptation": "encoder eval; gradients only through declared modules",
        }

    def release_provenance(self) -> dict[str, object]:
        return validate_curia2_release(self.model_dir)

    def code_paths(self) -> list[Path]:
        return [Path(__file__).resolve()]

    def audit_description(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "variant": self.variant,
            "environment": "myLocalEnv",
            "embedding_dim": self.embedding_dim,
            "readout": self.readout,
            "adaptation_boundary": self.adaptation_boundary,
            "input_contract": {"required_runtime_condition": self.model_name},
            "model_dir": str(self.model_dir),
        }
