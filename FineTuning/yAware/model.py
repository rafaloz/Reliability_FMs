#!/usr/bin/env python3
"""y-Aware checkpoint, CAT12 input, and controlled-adaptation contract."""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
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


MODEL_NAME = "y-Aware"
VARIANT = "canonical"
ENVIRONMENT = "myLocalEnv"
EMBEDDING_DIM = 1024
READOUT = "global-average-pooled 3D DenseNet-121 encoder output before the projection head"
ADAPTATION_BOUNDARY = "features.denseblock4 (final 3D DenseNet-121 block)"
OFFLINE_INPUT_GROUP = "cat12_ixi555_gm"

OFFICIAL_REPOSITORY = "https://github.com/Duplums/yAwareContrastiveLearning"
OFFICIAL_CODE_COMMIT = "7f6b3c604419d10f5f6628938e1506c85d42fbe3"
OFFICIAL_DENSENET_SHA256 = "9192eb5a33dc67c9be8f96c55d05eec76d703ec0062b74505b1c8613496d4b39"
CHECKPOINT_FILENAME = "DenseNet121_BHB-10K_yAwareContrastive.pth"
CHECKPOINT_SHA256 = "83f97b8ce677974bb3f49384ba519ed5523d7fdff887e6f6a2115d9fe1d595f5"

PREPROCESSING_PIPELINE_VERSION = "cat12-ixi555-anatcl-yaware-dual-output-v2"
PREPROCESSING_CONTRACT_SHA256 = (
    "17c310d6943d1d54628422594aa1ddd28dfb40a78a2d2482aa0cf2c7c00cbc37"
)
MODEL_INPUT_SHAPE = (1, 121, 145, 121)
EXPECTED_NORMALIZATION = "whole-volume z-score"


@dataclass(frozen=True)
class YAwareOptions:
    """Paths that fully identify the single public y-Aware condition."""

    checkpoint: Path | None = None
    preproc_root: Path | None = None
    qc_root: Path | None = None


def _record_value(record: Any, key: str) -> Any:
    """Read a manifest field from a mapping, pandas row, or named tuple."""

    if isinstance(record, Mapping):
        return record[key]
    try:
        return record[key]
    except (KeyError, TypeError, IndexError):
        return getattr(record, key)


def _first_existing(candidates: Iterable[Path]) -> Path:
    """Resolve the first existing path while preserving an informative default."""

    paths = [Path(path).expanduser() for path in candidates]
    if not paths:
        raise ValueError("No path candidates were supplied")
    for path in paths:
        if path.exists():
            return path.resolve()
    return paths[0].resolve()


def default_checkpoint(override: Path | None = None) -> Path:
    """Resolve the one pinned BHB-10K y-Aware checkpoint."""

    if override is not None:
        return Path(override).expanduser().resolve()
    candidates: list[Path] = []
    configured = os.environ.get("YAWARE_CHECKPOINT")
    if configured:
        candidates.append(Path(configured))
    relative = Path("Models/y-Aware") / CHECKPOINT_FILENAME
    candidates.extend(
        [
            PROJECT_ROOT / relative,
            Path("/home/rafa/workRafa/FoundationModels_OH_T1w") / relative,
            Path("/datos/work/rnavgon/FoundationModels_OH_T1w") / relative,
        ]
    )
    return _first_existing(candidates)


def default_preproc_root(override: Path | None = None) -> Path:
    """Resolve the dataset-scoped z-scored NumPy inputs used by y-Aware."""

    if override is not None:
        return Path(override).expanduser().resolve()
    candidates: list[Path] = []
    configured = os.environ.get("YAWARE_PREPROC_ROOT")
    if configured:
        candidates.append(Path(configured))
    relative = Path("preprocessings/FineTunningDataset/preproc_yAware")
    candidates.extend(
        [
            PROJECT_ROOT / relative,
            Path("/home/rafa/workRafa/FoundationModels_OH_T1w") / relative,
            Path("/datos/work/rnavgon/FoundationModels_OH_T1w") / relative,
        ]
    )
    return _first_existing(candidates)


def default_qc_root(preproc_root: Path, override: Path | None = None) -> Path:
    """Resolve shared CAT12 QC, which is retained beside the AnatCL view."""

    if override is not None:
        return Path(override).expanduser().resolve()
    configured = os.environ.get("YAWARE_QC_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    sibling = preproc_root.parent / "preproc_AnatCL"
    relative = Path("preprocessings/FineTunningDataset/preproc_AnatCL")
    return _first_existing(
        [
            sibling,
            PROJECT_ROOT / relative,
            Path("/home/rafa/workRafa/FoundationModels_OH_T1w") / relative,
            Path("/datos/work/rnavgon/FoundationModels_OH_T1w") / relative,
        ]
    )


def load_and_validate_manifest(
    path: Path | str = DEFAULT_MANIFEST,
    *,
    expected_sha256: str | None = MANIFEST_SHA256,
    expected_counts: dict[str, dict[str, int]] | None = EXPECTED_SPLIT_DATASET_COUNTS,
    check_input_files: bool = False,
) -> tuple[Any, dict[str, object]]:
    """Validate the cohort and permission to use the shared CAT12 inputs."""

    return _load_and_validate_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_counts=expected_counts,
        required_offline_group=OFFLINE_INPUT_GROUP,
        check_raw_input_files=check_input_files,
    )


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    """Extract the released y-Aware state dict from supported layouts."""

    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("model"), Mapping):
        return checkpoint["model"]
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("state_dict"), Mapping):
        return checkpoint["state_dict"]
    if hasattr(checkpoint, "state_dict") and callable(checkpoint.state_dict):
        state = checkpoint.state_dict()
        if isinstance(state, Mapping):
            return state
    if isinstance(checkpoint, Mapping) and checkpoint and all(
        hasattr(value, "shape") for value in checkpoint.values()
    ):
        return checkpoint
    raise ValueError(
        "Unexpected y-Aware checkpoint layout: "
        f"type={type(checkpoint).__name__}, "
        f"keys={list(checkpoint)[:10] if isinstance(checkpoint, Mapping) else 'n/a'}"
    )


def _strip_data_parallel_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the released DataParallel prefix and reject key collisions."""

    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        target = key.removeprefix("module.")
        if target in cleaned:
            raise ValueError(f"Duplicate y-Aware checkpoint key after prefix removal: {target}")
        cleaned[target] = value
    return cleaned


def make_yaware_input(data: np.ndarray) -> np.ndarray:
    """Validate the already normalized release-shaped input without transforming it."""

    array = np.asarray(data, dtype=np.float32)
    if tuple(array.shape) != MODEL_INPUT_SHAPE:
        raise ValueError(f"y-Aware input must be {MODEL_INPUT_SHAPE}, got {array.shape}")
    if not np.isfinite(array).all() or float(np.std(array)) <= 0.0:
        raise ValueError("y-Aware input is non-finite or has zero variance")
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array))
    if abs(mean) > 5e-4 or abs(standard_deviation - 1.0) > 5e-4:
        raise ValueError(
            "y-Aware input is not the expected whole-volume z-score: "
            f"mean={mean}, std={standard_deviation}"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def _build_pretraining_model(torch: Any) -> Any:
    """Build the exact released encoder-mode 3D DenseNet-121 architecture."""

    nn = torch.nn
    functional = torch.nn.functional

    class DenseLayer(nn.Module):
        def __init__(self, num_input_features: int, growth_rate: int, bn_size: int) -> None:
            super().__init__()
            self.norm1 = nn.BatchNorm3d(num_input_features)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv3d(
                num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False
            )
            self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        def forward(self, *previous_features: Any) -> Any:
            concatenated = torch.cat(previous_features, 1)
            bottleneck = self.conv1(self.relu1(self.norm1(concatenated)))
            return self.conv2(self.relu2(self.norm2(bottleneck)))

    class DenseBlock(nn.Module):
        def __init__(
            self, num_layers: int, num_input_features: int, growth_rate: int, bn_size: int
        ) -> None:
            super().__init__()
            for index in range(num_layers):
                self.add_module(
                    f"denselayer{index + 1}",
                    DenseLayer(num_input_features + index * growth_rate, growth_rate, bn_size),
                )

        def forward(self, initial_features: Any) -> Any:
            features = [initial_features]
            for layer in self.children():
                features.append(layer(*features))
            return torch.cat(features, 1)

    class YAwarePretrainingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            growth_rate = 32
            block_config = (6, 12, 24, 16)
            num_features = 64
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            nn.Conv3d(
                                1, num_features, kernel_size=7, stride=2, padding=3, bias=False
                            ),
                        ),
                        ("norm0", nn.BatchNorm3d(num_features)),
                        ("relu0", nn.ReLU(inplace=True)),
                        ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
                    ]
                )
            )
            for block_index, num_layers in enumerate(block_config, start=1):
                block = DenseBlock(num_layers, num_features, growth_rate, bn_size=4)
                self.features.add_module(f"denseblock{block_index}", block)
                num_features += num_layers * growth_rate
                if block_index != len(block_config):
                    transition = nn.Sequential(
                        OrderedDict(
                            [
                                ("norm", nn.BatchNorm3d(num_features)),
                                ("relu", nn.ReLU(inplace=True)),
                                (
                                    "conv",
                                    nn.Conv3d(
                                        num_features,
                                        num_features // 2,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                ),
                                ("pool", nn.AvgPool3d(kernel_size=2, stride=2)),
                            ]
                        )
                    )
                    self.features.add_module(f"transition{block_index}", transition)
                    num_features //= 2
            if num_features != EMBEDDING_DIM:
                raise RuntimeError(f"Unexpected y-Aware feature width: {num_features}")
            self.hidden_representation = nn.Linear(num_features, 512)
            self.head_projection = nn.Linear(512, 128)

        def preprojection(self, volume: Any) -> Any:
            features = self.features(volume)
            pooled = functional.adaptive_avg_pool3d(functional.relu(features, inplace=True), 1)
            return torch.flatten(pooled, 1)

        def forward(self, volume: Any) -> Any:
            representation = self.preprojection(volume)
            hidden = functional.relu(self.hidden_representation(representation), inplace=True)
            return self.head_projection(hidden)

    return YAwarePretrainingModel()


class YAwareModel:
    """The pinned y-Aware checkpoint with strict input and readout contracts."""

    model_name = MODEL_NAME
    variant = VARIANT
    environment = ENVIRONMENT
    embedding_dim = EMBEDDING_DIM
    readout = READOUT
    adaptation_boundary = ADAPTATION_BOUNDARY

    def __init__(self, options: YAwareOptions) -> None:
        self.options = options
        self.checkpoint = default_checkpoint(options.checkpoint)
        self.preproc_root = default_preproc_root(options.preproc_root)
        self.qc_root = default_qc_root(self.preproc_root, options.qc_root)
        self.model: Any | None = None
        self.device: Any | None = None
        self.last_input_provenance: dict[str, Any] | None = None
        self._release_details: dict[str, Any] | None = None
        self._load_details: dict[str, Any] | None = None

    def subject_dir(self, record: Any) -> Path:
        return self.preproc_root / str(_record_value(record, "dataset")) / str(
            _record_value(record, "subject")
        )

    def input_path(self, record: Any) -> Path:
        subject = str(_record_value(record, "subject"))
        return self.subject_dir(record) / f"{subject}_yaware_ready.npy"

    def qc_path(self, record: Any) -> Path:
        return self.qc_root / str(_record_value(record, "dataset")) / str(
            _record_value(record, "subject")
        ) / "preprocessing_qc.json"

    def required_input_paths(self, record: Any) -> list[Path]:
        return [self.input_path(record), self.qc_path(record)]

    def release_provenance(self) -> dict[str, Any]:
        """Require the exact public y-Aware checkpoint used by the benchmark."""

        if self._release_details is not None:
            return dict(self._release_details)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        observed = sha256_file(self.checkpoint)
        if observed != CHECKPOINT_SHA256:
            raise ValueError(
                f"y-Aware checkpoint drift: expected {CHECKPOINT_SHA256}, observed {observed}"
            )
        self._release_details = {
            "repository": OFFICIAL_REPOSITORY,
            "code_commit": OFFICIAL_CODE_COMMIT,
            "official_densenet_sha256": OFFICIAL_DENSENET_SHA256,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": observed,
        }
        return dict(self._release_details)

    def validate_input_provenance(self, record: Any) -> dict[str, Any]:
        """Verify shared CAT12 QC and the exact consumed y-Aware NumPy hash."""

        input_path = self.input_path(record)
        qc_path = self.qc_path(record)
        for path in (input_path, qc_path):
            if not path.is_file():
                raise FileNotFoundError(path)
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
            raise ValueError(f"Stale y-Aware preprocessing QC: {mismatches}")
        observed_hash = sha256_file(input_path)
        if qc.get("yaware_output_sha256") != observed_hash:
            raise ValueError(
                "y-Aware preprocessed input hash differs from preprocessing_qc.json: "
                f"{input_path}"
            )
        runtime = qc.get("runtime_views", {}).get("yaware", {})
        if runtime.get("shape") != list(MODEL_INPUT_SHAPE):
            raise ValueError(f"y-Aware runtime-view shape drift in {qc_path}")
        if runtime.get("normalization") != EXPECTED_NORMALIZATION:
            raise ValueError(f"y-Aware normalization drift in {qc_path}")
        return {
            "qc_path": str(qc_path),
            "qc_sha256": sha256_file(qc_path),
            "input_path": str(input_path),
            "input_sha256": observed_hash,
            "pipeline_version": qc["pipeline_version"],
            "pipeline_contract_sha256": qc["pipeline_contract_sha256"],
        }

    def load_input(self, record: Any) -> Any:
        """Load the release-shaped z-scored NumPy input without another transform."""

        import torch

        provenance = self.validate_input_provenance(record)
        data = np.load(self.input_path(record), allow_pickle=False)
        array = make_yaware_input(data)
        tensor = torch.from_numpy(array).unsqueeze(0)
        if tuple(tensor.shape) != (1, *MODEL_INPUT_SHAPE):
            raise RuntimeError(f"Unexpected y-Aware tensor shape: {tuple(tensor.shape)}")
        self.last_input_provenance = provenance
        return tensor

    def load(self, device: Any) -> None:
        """Build encoder mode, strictly load all backbone and projection parameters."""

        import torch

        release = self.release_provenance()
        model = _build_pretraining_model(torch)
        checkpoint = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
        state = _strip_data_parallel_prefix(_checkpoint_state_dict(checkpoint))
        expected_keys = set(model.state_dict())
        observed_keys = set(state)
        if observed_keys != expected_keys:
            raise ValueError(
                "y-Aware checkpoint key mismatch before strict load: "
                f"missing={sorted(expected_keys - observed_keys)[:10]}, "
                f"unexpected={sorted(observed_keys - expected_keys)[:10]}"
            )
        probe_key = "features.conv0.weight"
        before = model.state_dict()[probe_key].detach().clone()
        info = model.load_state_dict(state, strict=True)
        after = model.state_dict()[probe_key].detach()
        changed = float((after.cpu() - before.cpu()).norm().item())
        bit_exact = bool(torch.equal(after.cpu(), state[probe_key].detach().cpu()))
        if info.missing_keys or info.unexpected_keys or changed <= 0.0 or not bit_exact:
            raise RuntimeError(
                "Strict y-Aware checkpoint proof-of-load failed: "
                f"missing={info.missing_keys}, unexpected={info.unexpected_keys}, "
                f"changed_l2={changed}, bit_exact={bit_exact}"
            )
        model.eval()
        model.to(device)
        self.model = model
        self.device = device
        self._load_details = {
            "strict": True,
            "state_keys": len(state),
            "probe_key": probe_key,
            "probe_changed_l2": changed,
            "probe_bit_exact": bit_exact,
            "projection_head_loaded": True,
            "projection_head_used": False,
            "fresh_norm5_added": False,
            "checkpoint_sha256": release["checkpoint_sha256"],
        }
        print(
            f"y-Aware loaded strictly: {len(state)} keys; projection head excluded from "
            f"readout; embedding_dim={EMBEDDING_DIM}",
            flush=True,
        )

    def _require_loaded(self) -> Any:
        if self.model is None or self.device is None:
            raise RuntimeError("y-Aware has not loaded its checkpoint")
        return self.model

    def _validate_embedding(self, tensor: Any) -> Any:
        import torch

        if tensor.ndim == 2 and int(tensor.shape[0]) == 1:
            tensor = tensor[0]
        if tuple(tensor.shape) != (EMBEDDING_DIM,):
            raise ValueError(f"y-Aware produced shape {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("y-Aware produced non-finite embedding values")
        return tensor

    def encode(self, record: Any, *, training: bool, epoch: int, seed: int) -> Any:
        """Return the 1024-D encoder representation before both projection layers."""

        del training, epoch, seed
        volume = self.load_input(record).to(self.device, non_blocking=True)
        embedding = self._require_loaded().preprojection(volume)
        return self._validate_embedding(embedding)

    def set_frozen(self) -> dict[str, Any]:
        """Freeze encoder and the loaded but unused contrastive projection head."""

        model = self._require_loaded()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        return self._parameter_summary([])

    def controlled_modules(self) -> list[tuple[str, Any]]:
        """Expose exactly the final DenseNet block as the controlled boundary."""

        denseblock4 = getattr(self._require_loaded().features, "denseblock4", None)
        if denseblock4 is None or len(list(denseblock4.children())) != 16:
            raise ValueError("Expected the 16-layer denseblock4 of 3D DenseNet-121")
        return [("features.denseblock4", denseblock4)]

    def set_controlled_adaptation(self) -> dict[str, Any]:
        """Freeze everything, then unfreeze only the declared final DenseNet block."""

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
            raise RuntimeError(f"Unexpected trainable y-Aware parameters: {unexpected[:10]}")
        model.eval()
        summary = self._parameter_summary(components)
        summary["component_parameter_counts"] = counts
        return summary

    def controlled_train_mode(self) -> None:
        """Keep all BatchNorm running statistics fixed during controlled adaptation."""

        self._require_loaded().eval()

    def trainable_named_parameters(self) -> list[tuple[str, Any]]:
        return [
            (name, parameter)
            for name, parameter in self._require_loaded().named_parameters()
            if parameter.requires_grad
        ]

    def _parameter_summary(self, components: list[tuple[str, Any]]) -> dict[str, Any]:
        model = self._require_loaded()
        encoder_total = int(sum(parameter.numel() for parameter in model.features.parameters()))
        encoder_trainable = int(
            sum(
                parameter.numel()
                for parameter in model.features.parameters()
                if parameter.requires_grad
            )
        )
        projector_total = int(
            sum(parameter.numel() for parameter in model.hidden_representation.parameters())
            + sum(parameter.numel() for parameter in model.head_projection.parameters())
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
            "contrastive_projection_parameters": projector_total,
            "contrastive_projection_used": False,
            "mode_during_adaptation": (
                "full encoder eval; gradients only through denseblock4 parameters"
            ),
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
            / "preprocessing_CAT12_AnatCL_yAware.py",
        ]

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
            "qc_root": str(self.qc_root),
            "input_contract": {"required_offline_group": OFFLINE_INPUT_GROUP},
            "preprocessing_pipeline_version": PREPROCESSING_PIPELINE_VERSION,
            "preprocessing_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
            "model_input_shape": list(MODEL_INPUT_SHAPE),
            "normalization": EXPECTED_NORMALIZATION,
            "projection_head_used": False,
            "fresh_norm5_added": False,
        }
