#!/usr/bin/env python3
"""BrainIAC checkpoint, readout, and frozen model-input contract."""

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


MODEL_NAME = "BrainIAC"
VARIANT = "canonical"
ENVIRONMENT = "myLocalEnv"
EMBEDDING_DIM = 768
READOUT = "first patch token from the released 216-token ViT output"
ADAPTATION_BOUNDARY = "blocks.11 plus final layer normalization"
OFFLINE_INPUT_GROUP = "preproc_brainiac"

OFFICIAL_REPOSITORY = "https://github.com/AIM-KannLab/BrainIAC"
OFFICIAL_CODE_COMMIT = "ba60f45bed832ee0e5683c678caeaeefe2072f0d"
CHECKPOINT_RELEASE = "BrainIAC.ckpt"
CHECKPOINT_FILENAME = "BrainIAC.ckpt"
CHECKPOINT_SHA256 = "f22bdbcae26823a9d9e8aee883c6f24386ba4617339c12269848b6666cc62693"

PREPROCESSING_PIPELINE_VERSION = "brainiac-release-aligned-finetuning-v1"
PREPROCESSING_CONTRACT_SHA256 = (
    "2eea43f6031355e2ff4d131c549637ac633bff195989e2d0dec30b5499f231ca"
)
MODEL_INPUT_SHAPE = (96, 96, 96)
PATCH_SIZE = (16, 16, 16)
N_PATCH_TOKENS = 216


@dataclass(frozen=True)
class BrainIACOptions:
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
    """Resolve the pinned BrainIAC checkpoint under local and cluster layouts."""

    if override is not None:
        return Path(override).expanduser().resolve()
    candidates: list[Path] = []
    configured = os.environ.get("BRAINIAC_CHECKPOINT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            PROJECT_ROOT / "Models" / "BrainIAC" / CHECKPOINT_FILENAME,
            Path("/home/rafa/workRafa/FoundationModels_OH_T1w/Models/BrainIAC")
            / CHECKPOINT_FILENAME,
            Path("/datos/work/rnavgon/FoundationModels_OH_T1w/Models/BrainIAC")
            / CHECKPOINT_FILENAME,
        ]
    )
    return _first_existing(candidates)


def default_preproc_root(override: Path | None = None) -> Path:
    """Resolve the dataset-scoped BrainIAC preprocessing outputs."""

    if override is not None:
        return Path(override).expanduser().resolve()
    candidates: list[Path] = []
    configured = os.environ.get("BRAINIAC_PREPROC_ROOT")
    if configured:
        candidates.append(Path(configured))
    relative = Path("preprocessings/FineTunningDataset/preproc_BrainIAC")
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
    """Validate the cohort and permission to use offline BrainIAC inputs."""

    return _load_and_validate_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_counts=expected_counts,
        required_offline_group=OFFLINE_INPUT_GROUP,
        check_raw_input_files=check_input_files,
    )


def clean_backbone_state_dict(
    state_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], int, int]:
    """Select the released ViT backbone and reject unknown checkpoint branches."""

    cleaned: dict[str, Any] = {}
    projector_tensors = 0
    projector_parameters = 0
    unexpected: list[str] = []
    for original_key, value in state_dict.items():
        key = str(original_key)
        if key.startswith("backbone."):
            backbone_key = key[len("backbone.") :]
            if backbone_key in cleaned:
                raise ValueError(f"Duplicate backbone key: {backbone_key}")
            cleaned[backbone_key] = value
        elif key.startswith("projection_head."):
            if not hasattr(value, "numel"):
                raise TypeError(f"Non-tensor projection-head value: {original_key}")
            projector_tensors += 1
            projector_parameters += int(value.numel())
        else:
            unexpected.append(key)
    if unexpected:
        raise ValueError(f"Unexpected BrainIAC checkpoint keys; first 10: {unexpected[:10]}")
    if not cleaned or projector_tensors == 0:
        raise ValueError("Checkpoint must contain both backbone and projection-head tensors")
    return cleaned, projector_tensors, projector_parameters


class BrainIACModel:
    """Released 3D ViT-B encoder with first-patch representation."""

    model_name = MODEL_NAME
    variant = VARIANT
    environment = ENVIRONMENT
    embedding_dim = EMBEDDING_DIM
    readout = READOUT
    adaptation_boundary = ADAPTATION_BOUNDARY

    def __init__(self, options: BrainIACOptions) -> None:
        self.options = options
        self.checkpoint = default_checkpoint(options.checkpoint)
        self.preproc_root = default_preproc_root(options.preproc_root)
        self.model: Any | None = None
        self.device: Any | None = None
        self.last_input_provenance: dict[str, Any] | None = None
        self._release_details: dict[str, Any] | None = None
        self._load_details: dict[str, Any] | None = None
        self._loaded_parameter_names: set[str] = set()
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
        return self.subject_dir(record) / f"{subject}_brainiac_ready.nii.gz"

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
                f"BrainIAC checkpoint drift: expected {CHECKPOINT_SHA256}, observed {observed}"
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
            raise ValueError(f"Stale BrainIAC preprocessing QC: {mismatches}")
        output_hashes = qc.get("output_sha256")
        if not isinstance(output_hashes, dict):
            raise ValueError(f"Missing output hashes in {qc_path}")
        observed_hash = sha256_file(image_path)
        if output_hashes.get("ready") != observed_hash:
            raise ValueError("BrainIAC model-ready input hash differs from preprocessing QC")
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
            raise RuntimeError("BrainIAC input changed during provenance validation")
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
                f"BrainIAC requires an exact 3D {MODEL_INPUT_SHAPE} volume; got {image.shape}"
            )
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("BrainIAC input contains non-finite values")
        if not np.any(data != 0):
            raise ValueError("BrainIAC input is empty")
        self.last_input_provenance = provenance
        return torch.from_numpy(np.ascontiguousarray(data[None, None, ...]))

    def load(self, device: Any) -> None:
        """Build MONAI ViT-B and load every released backbone tensor."""

        import torch
        from monai.networks.nets import ViT

        release = self.release_provenance()
        checkpoint = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, Mapping):
            raise ValueError("BrainIAC checkpoint lacks a state_dict mapping")
        cleaned, projector_tensors, projector_parameters = clean_backbone_state_dict(state_dict)
        model = ViT(
            in_channels=1,
            img_size=MODEL_INPUT_SHAPE,
            patch_size=PATCH_SIZE,
            hidden_size=EMBEDDING_DIM,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=True,
        )
        model_state = model.state_dict()
        unexpected = sorted(set(cleaned) - set(model_state))
        if unexpected:
            raise ValueError(f"Unexpected BrainIAC backbone keys: {unexpected[:10]}")
        missing = sorted(set(model_state) - set(cleaned))
        disallowed_missing = [name for name in missing if "cross_attn" not in name]
        if disallowed_missing:
            raise ValueError(f"Missing non-cross-attention keys: {disallowed_missing[:10]}")
        probe_key = next(iter(cleaned))
        before = model_state[probe_key].detach().clone()
        info = model.load_state_dict(cleaned, strict=False)
        if sorted(info.unexpected_keys) != unexpected or sorted(info.missing_keys) != missing:
            raise RuntimeError("MONAI load report differs from the prevalidated key contract")
        after = model.state_dict()[probe_key].detach()
        changed = float((after.cpu() - before.cpu()).norm().item())
        bit_exact = bool(torch.equal(after.cpu(), cleaned[probe_key].detach().cpu()))
        if changed <= 0 or not bit_exact:
            raise RuntimeError("BrainIAC checkpoint proof-of-load failed")
        position = model.state_dict().get("patch_embedding.position_embeddings")
        if position is None or tuple(position.shape) != (1, N_PATCH_TOKENS, EMBEDDING_DIM):
            raise ValueError(f"Unexpected BrainIAC positional embedding: {getattr(position, 'shape', None)}")
        if hasattr(model, "cls_token") and getattr(model, "cls_token") is not None:
            raise ValueError("BrainIAC checkpoint contract expects no CLS token")
        model.eval()
        model.to(device)
        self.model = model
        self.device = device
        self._loaded_parameter_names = set(cleaned)
        self._projector_parameters = projector_parameters
        self._load_details = {
            "contract_strict": True,
            "pytorch_strict": not missing,
            "backbone_keys": len(cleaned),
            "compatibility_missing_keys": missing,
            "compatibility_reason": (
                "newer MONAI instantiates unused cross-attention modules absent from the release"
                if missing
                else None
            ),
            "probe_key": probe_key,
            "probe_changed_l2": changed,
            "probe_bit_exact": bit_exact,
            "projection_head_tensors": projector_tensors,
            "projection_head_parameters": projector_parameters,
            "projection_head_used": False,
            "n_patch_tokens": N_PATCH_TOKENS,
            "cls_token_present": False,
            "checkpoint_sha256": release["checkpoint_sha256"],
        }
        print(
            "BrainIAC loaded with a strict release contract: "
            f"{len(cleaned)} backbone keys; {len(missing)} unused MONAI compatibility keys; "
            f"projection head excluded; embedding_dim={EMBEDDING_DIM}",
            flush=True,
        )

    def _require_loaded(self) -> Any:
        if self.model is None or self.device is None:
            raise RuntimeError("BrainIAC has not loaded its checkpoint")
        return self.model

    def _validate_embedding(self, tensor: Any) -> Any:
        """Require one finite 768-dimensional first-patch representation."""

        import torch

        if tensor.ndim == 2 and int(tensor.shape[0]) == 1:
            tensor = tensor[0]
        if tuple(tensor.shape) != (EMBEDDING_DIM,):
            raise RuntimeError(f"Unexpected BrainIAC embedding shape: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise RuntimeError("BrainIAC produced non-finite embedding values")
        return tensor

    def set_frozen(self) -> dict[str, Any]:
        """Freeze the entire encoder for deterministic embedding extraction."""

        model = self._require_loaded()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        return self._parameter_summary([])

    def _controlled_parameter(self, name: str) -> bool:
        """Admit only released parameters in the final block and final norm."""

        return name in self._loaded_parameter_names and (
            name.startswith("blocks.11.") or name.startswith("norm.")
        )

    def set_controlled_adaptation(self) -> dict[str, Any]:
        """Freeze ViT-B, then unfreeze its final pretrained block and norm."""

        model = self._require_loaded()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(self._controlled_parameter(name))
        trainable = self.trainable_named_parameters()
        if not trainable:
            raise RuntimeError("BrainIAC controlled boundary selected no parameters")
        if any("cross_attn" in name for name, _ in trainable):
            raise RuntimeError("Unreleased MONAI cross-attention parameters became trainable")
        model.eval()
        summary = self._parameter_summary(["blocks.11", "norm"])
        summary["component_parameter_counts"] = {
            label: int(
                sum(
                    parameter.numel()
                    for name, parameter in trainable
                    if name.startswith(f"{label}.")
                )
            )
            for label in ("blocks.11", "norm")
        }
        return summary

    def controlled_train_mode(self) -> None:
        """Keep the whole ViT in evaluation mode during controlled adaptation."""

        self._require_loaded().eval()

    def trainable_named_parameters(self) -> list[tuple[str, Any]]:
        return [
            (name, parameter)
            for name, parameter in self._require_loaded().named_parameters()
            if parameter.requires_grad
        ]

    def _parameter_summary(self, components: list[str]) -> dict[str, Any]:
        """Describe absolute cost against the released, forward-active encoder."""

        model = self._require_loaded()
        named = dict(model.named_parameters())
        active_total = int(
            sum(named[name].numel() for name in self._loaded_parameter_names if name in named)
        )
        trainable = int(sum(parameter.numel() for _, parameter in self.trainable_named_parameters()))
        return {
            "model": self.model_name,
            "variant": self.variant,
            "embedding_dim": self.embedding_dim,
            "readout": self.readout,
            "adaptation_boundary": self.adaptation_boundary,
            "encoder_total_parameters": active_total,
            "encoder_architecture_parameters": int(sum(p.numel() for p in model.parameters())),
            "encoder_trainable_parameters": trainable,
            "encoder_trainable_fraction": float(trainable / active_total) if active_total else 0.0,
            "trainable_components": components,
            "trainable_parameter_names": [name for name, _ in self.trainable_named_parameters()],
            "contrastive_projection_parameters": self._projector_parameters,
            "contrastive_projection_loaded": False,
            "contrastive_projection_used": False,
            "mode_during_adaptation": (
                "full encoder eval; gradients only through the final pretrained block and norm"
            ),
        }

    def encode(self, record: Any, *, training: bool, epoch: int, seed: int) -> Any:
        """Return the first token from the released 216-patch ViT sequence."""

        del training, epoch, seed
        model = self._require_loaded()
        model.eval()
        volume = self.load_input(record).to(self.device, non_blocking=True)
        output = model(volume)
        tokens = output[0] if isinstance(output, (tuple, list)) else output
        if tuple(tokens.shape) != (1, N_PATCH_TOKENS, EMBEDDING_DIM):
            raise RuntimeError(f"Unexpected BrainIAC token output: {tuple(tokens.shape)}")
        return self._validate_embedding(tokens[:, 0, :])

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
            "model_input_shape": list(MODEL_INPUT_SHAPE),
            "patch_size": list(PATCH_SIZE),
            "n_patch_tokens": N_PATCH_TOKENS,
            "cls_token_present": False,
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
            / "preprocessing_BrainIAC.py",
        ]
