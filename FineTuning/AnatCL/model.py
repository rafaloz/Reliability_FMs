#!/usr/bin/env python3
"""AnatCL checkpoint, CAT12 input, and controlled-adaptation contract."""

from __future__ import annotations

import importlib
import json
import os
import pickle
import sys
import types
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


MODEL_NAME = "AnatCL"
ENVIRONMENT = "myLocalEnv"
EMBEDDING_DIM = 512
READOUT = "global-average-pooled 3D ResNet-18 encoder output before the contrastive head"
ADAPTATION_BOUNDARY = "backbone.encoder.layer4 (final 3D ResNet-18 stage)"
OFFLINE_INPUT_GROUP = "cat12_ixi555_gm"

OFFICIAL_REPOSITORY = "https://github.com/EIDOSLAB/AnatCL"
OFFICIAL_CODE_COMMIT = "2701acaadcd1a49960a835a8bfb9ba9708d29632"
CHECKPOINT_RELEASE = "anatcl-g3/global"
CHECKPOINT_FILENAMES = tuple(f"fold{fold}_weights.pth" for fold in range(5))
CHECKPOINT_SHA256 = {
    0: "77f6d44d0317ffc704ffc751db7eab349a8109e2cd596ab06070f1aec7a55d64",
    1: "d49a38859526b437ced02a25eef1ea7d1bc579637b8bc9b0def9b7f21ca5640f",
    2: "36237efc84469285f99c533e0d22247c74c20f00fb324bbdb398551a177c9f7a",
    3: "7a350609b770f2b25ce71ef90f2f00210d68b405c952e3d9ffe77f378f303de4",
    4: "e8f458ebdc1326259e656772165deb0b24215d9158094db61f291370d952a8c7",
}
OFFICIAL_SOURCE_SHA256 = {
    "anatcl/anatcl.py": "505dd542452ec91db74effc8d9333ae0347f618669dacf6827952d09c3cf9633",
    "anatcl/models/resnet3d.py": "bca448a5e9e9e4ec5f86ecb5e5077640aa50b90721959701472c0068de1a6bf5",
}

PREPROCESSING_PIPELINE_VERSION = "cat12-ixi555-anatcl-yaware-dual-output-v2"
PREPROCESSING_CONTRACT_SHA256 = (
    "17c310d6943d1d54628422594aa1ddd28dfb40a78a2d2482aa0cf2c7c00cbc37"
)
SHARED_INPUT_SHAPE = (121, 145, 121)
EXPECTED_ZOOMS_MM = (1.5, 1.5, 1.5)
MODEL_INPUT_SHAPE = (1, 121, 128, 121)
CROP_Y = (8, 136)


def variant_for_fold(fold: int) -> str:
    """Return the stable variant label used in tables and audit artifacts."""

    if fold not in range(5):
        raise ValueError(f"AnatCL fold must be 0..4, got {fold}")
    return f"fold{fold}"


@dataclass(frozen=True)
class AnatCLOptions:
    """Paths and fold that fully identify one AnatCL condition."""

    fold: int = 0
    checkpoint_dir: Path | None = None
    preproc_root: Path | None = None

    def validate(self) -> None:
        variant_for_fold(self.fold)


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


def default_checkpoint_dir(override: Path | None = None) -> Path:
    """Resolve the corrected five-fold anatcl-g3 checkpoint directory."""

    if override is not None:
        return Path(override).expanduser().resolve()
    candidates: list[Path] = []
    configured = os.environ.get("ANATCL_CHECKPOINT_DIR")
    if configured:
        candidates.append(Path(configured))
    relative = Path("Models/AnatCL/NEW_anatcl-g3")
    candidates.extend(
        [
            PROJECT_ROOT / relative,
            Path("/home/rafa/workRafa/FoundationModels_OH_T1w") / relative,
            Path("/datos/work/rnavgon/FoundationModels_OH_T1w") / relative,
        ]
    )
    return _first_existing(candidates)


def default_preproc_root(override: Path | None = None) -> Path:
    """Resolve the dataset-scoped shared CAT12 outputs used by AnatCL."""

    if override is not None:
        return Path(override).expanduser().resolve()
    candidates: list[Path] = []
    configured = os.environ.get("ANATCL_PREPROC_ROOT")
    if configured:
        candidates.append(Path(configured))
    relative = Path("preprocessings/FineTunningDataset/preproc_AnatCL")
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
    """Validate the cohort and permission to use the shared CAT12 inputs."""

    return _load_and_validate_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_counts=expected_counts,
        required_offline_group=OFFLINE_INPUT_GROUP,
        check_raw_input_files=check_input_files,
    )


def semantic_release_contract(details: Mapping[str, Any]) -> dict[str, Any]:
    """Compare release identity independently of the checkpoint's filesystem location."""

    return {key: value for key, value in details.items() if key != "checkpoint"}


def semantic_adapter_contract(details: Mapping[str, Any]) -> dict[str, Any]:
    """Compare adapter behavior independently of relocatable input/model roots."""

    ignored = {"checkpoint_dir", "input_root"}
    return {key: value for key, value in details.items() if key not in ignored}


class _AnatCLUnpickler(pickle.Unpickler):
    """Redirect legacy ``models.*`` checkpoint objects into the AnatCL package."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "models" or module.startswith("models."):
            redirected = "anatcl." + module
            try:
                return super().find_class(redirected, name)
            except (ModuleNotFoundError, AttributeError):
                stub = type(
                    name,
                    (),
                    {
                        "__init__": lambda self, *args, **kwargs: None,
                        "__setstate__": lambda self, state: (
                            self.__dict__.update(state) if isinstance(state, dict) else None
                        ),
                    },
                )
                return stub
        return super().find_class(module, name)


_CUSTOM_PICKLE = types.ModuleType("_anatcl_checkpoint_pickle")
for _attribute in dir(pickle):
    setattr(_CUSTOM_PICKLE, _attribute, getattr(pickle, _attribute))
_CUSTOM_PICKLE.Unpickler = _AnatCLUnpickler


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    """Extract the public backbone state dict from either released layout."""

    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("model"), Mapping):
        return checkpoint["model"]
    if hasattr(checkpoint, "state_dict") and callable(checkpoint.state_dict):
        state = checkpoint.state_dict()
        if isinstance(state, Mapping):
            return state
    raise ValueError(
        "Unexpected AnatCL checkpoint layout: "
        f"type={type(checkpoint).__name__}, "
        f"keys={list(checkpoint)[:10] if isinstance(checkpoint, Mapping) else 'n/a'}"
    )


def make_anatcl_input(data: np.ndarray) -> np.ndarray:
    """Derive the release-aligned channel-first crop without intensity scaling."""

    array = np.asarray(data, dtype=np.float32)
    if tuple(array.shape) != SHARED_INPUT_SHAPE:
        raise ValueError(
            f"AnatCL shared input must be {SHARED_INPUT_SHAPE}, got {array.shape}"
        )
    if not np.isfinite(array).all() or float(np.std(array)) <= 0.0:
        raise ValueError("AnatCL shared input is non-finite or has zero variance")
    cropped = np.ascontiguousarray(array[:, CROP_Y[0] : CROP_Y[1], :], dtype=np.float32)
    output = cropped[np.newaxis, ...]
    if tuple(output.shape) != MODEL_INPUT_SHAPE:
        raise RuntimeError(f"Unexpected AnatCL runtime-view shape: {output.shape}")
    return output


class AnatCLModel:
    """One corrected AnatCL fold with a strictly validated CAT12 input path."""

    model_name = MODEL_NAME
    environment = ENVIRONMENT
    embedding_dim = EMBEDDING_DIM
    readout = READOUT
    adaptation_boundary = ADAPTATION_BOUNDARY

    def __init__(self, options: AnatCLOptions) -> None:
        options.validate()
        self.options = options
        self.fold = int(options.fold)
        self.variant = variant_for_fold(self.fold)
        self.checkpoint_dir = default_checkpoint_dir(options.checkpoint_dir)
        self.checkpoint = self.checkpoint_dir / CHECKPOINT_FILENAMES[self.fold]
        self.preproc_root = default_preproc_root(options.preproc_root)
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
        return self.subject_dir(record) / f"{subject}_anatcl_ready.nii.gz"

    def qc_path(self, record: Any) -> Path:
        return self.subject_dir(record) / "preprocessing_qc.json"

    def required_input_paths(self, record: Any) -> list[Path]:
        return [self.input_path(record), self.qc_path(record)]

    def release_provenance(self) -> dict[str, Any]:
        """Require the exact corrected checkpoint selected for this fold."""

        if self._release_details is not None:
            return dict(self._release_details)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        observed = sha256_file(self.checkpoint)
        expected = CHECKPOINT_SHA256[self.fold]
        if observed != expected:
            raise ValueError(
                f"AnatCL {self.variant} checkpoint drift: expected {expected}, observed {observed}"
            )
        self._release_details = {
            "repository": OFFICIAL_REPOSITORY,
            "code_commit": OFFICIAL_CODE_COMMIT,
            "checkpoint_release": CHECKPOINT_RELEASE,
            "fold": self.fold,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": observed,
            "expected_source_sha256": OFFICIAL_SOURCE_SHA256,
        }
        return dict(self._release_details)

    def validate_input_provenance(self, record: Any) -> dict[str, Any]:
        """Verify PASS QC, contract, subject identity, and the consumed NIfTI hash."""

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
            raise ValueError(f"Stale AnatCL preprocessing QC: {mismatches}")
        observed_hash = sha256_file(input_path)
        if qc.get("anatcl_output_sha256") != observed_hash:
            raise ValueError(
                "AnatCL preprocessed input hash differs from preprocessing_qc.json: "
                f"{input_path}"
            )
        runtime = qc.get("runtime_views", {}).get("anatcl", {})
        if runtime.get("shape") != list(MODEL_INPUT_SHAPE):
            raise ValueError(f"AnatCL runtime-view shape drift in {qc_path}")
        if runtime.get("crop_y") != list(CROP_Y) or runtime.get("normalization") != "none":
            raise ValueError(f"AnatCL runtime-view operation drift in {qc_path}")
        return {
            "qc_path": str(qc_path),
            "qc_sha256": sha256_file(qc_path),
            "input_path": str(input_path),
            "input_sha256": observed_hash,
            "pipeline_version": qc["pipeline_version"],
            "pipeline_contract_sha256": qc["pipeline_contract_sha256"],
        }

    def load_input(self, record: Any) -> Any:
        """Load the full mwp1 volume and derive the exact 121x128x121 view."""

        import nibabel as nib
        import torch

        provenance = self.validate_input_provenance(record)
        image = nib.load(str(self.input_path(record)))
        if tuple(int(value) for value in image.shape) != SHARED_INPUT_SHAPE:
            raise ValueError(f"Unexpected AnatCL shared input shape: {image.shape}")
        zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
        if not np.allclose(zooms, EXPECTED_ZOOMS_MM, rtol=0.0, atol=1e-3):
            raise ValueError(f"Unexpected AnatCL voxel spacing: {zooms}")
        affine = np.asarray(image.affine)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("AnatCL input has an invalid affine")
        if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-8:
            raise ValueError("AnatCL input affine is degenerate")
        data = image.get_fdata(dtype=np.float32)
        tensor = torch.from_numpy(make_anatcl_input(data)).unsqueeze(0)
        if tuple(tensor.shape) != (1, *MODEL_INPUT_SHAPE):
            raise RuntimeError(f"Unexpected AnatCL tensor shape: {tuple(tensor.shape)}")
        self.last_input_provenance = provenance
        return tensor

    def _validate_installed_source(self) -> dict[str, str]:
        """Require installed AnatCL architecture files to match the audited commit."""

        modules = {
            "anatcl/anatcl.py": importlib.import_module("anatcl.anatcl"),
            "anatcl/models/resnet3d.py": importlib.import_module("anatcl.models.resnet3d"),
        }
        observed: dict[str, str] = {}
        for relative, module in modules.items():
            source = Path(str(module.__file__)).resolve()
            digest = sha256_file(source)
            if digest != OFFICIAL_SOURCE_SHA256[relative]:
                raise ValueError(
                    f"Installed AnatCL source drift for {relative}: "
                    f"expected {OFFICIAL_SOURCE_SHA256[relative]}, observed {digest}"
                )
            observed[relative] = digest
        return observed

    def load(self, device: Any) -> None:
        """Build AnatCL and strictly load the selected corrected backbone."""

        import torch
        from anatcl import AnatCL

        self.release_provenance()
        installed_source = self._validate_installed_source()
        if self._release_details is None:
            raise RuntimeError("AnatCL release provenance was not initialized")
        self._release_details["installed_source_sha256"] = installed_source
        release = self.release_provenance()
        model = AnatCL(
            model="resnet18",
            descriptor="global",
            fold=self.fold,
            use_head=False,
            pretrained=False,
        )
        checkpoint = torch.load(
            str(self.checkpoint),
            map_location="cpu",
            pickle_module=_CUSTOM_PICKLE,
            weights_only=False,
        )
        state = _checkpoint_state_dict(checkpoint)
        probe_key = next(iter(state))
        if probe_key not in model.backbone.state_dict():
            raise ValueError(f"Checkpoint probe key is absent from AnatCL backbone: {probe_key}")
        before = model.backbone.state_dict()[probe_key].detach().clone()
        info = model.backbone.load_state_dict(state, strict=True)
        after = model.backbone.state_dict()[probe_key].detach()
        changed = float((after.cpu() - before.cpu()).norm().item())
        bit_exact = bool(torch.equal(after.cpu(), state[probe_key].detach().cpu()))
        if info.missing_keys or info.unexpected_keys or changed <= 0.0 or not bit_exact:
            raise RuntimeError(
                "Strict AnatCL checkpoint proof-of-load failed: "
                f"missing={info.missing_keys}, unexpected={info.unexpected_keys}, "
                f"changed_l2={changed}, bit_exact={bit_exact}"
            )
        model.eval()
        model.to(device)
        self.model = model
        self.device = device
        self._load_details = {
            "strict": True,
            "backbone_keys": len(state),
            "probe_key": probe_key,
            "probe_changed_l2": changed,
            "probe_bit_exact": bit_exact,
            "projection_head_used": False,
            "checkpoint_sha256": release["checkpoint_sha256"],
            "installed_source_sha256": installed_source,
        }
        print(
            f"AnatCL {self.variant} loaded strictly: {len(state)} backbone keys; "
            f"projection head excluded; embedding_dim={EMBEDDING_DIM}",
            flush=True,
        )

    def _require_loaded(self) -> Any:
        if self.model is None or self.device is None:
            raise RuntimeError("AnatCL has not loaded its checkpoint")
        return self.model

    def _validate_embedding(self, tensor: Any) -> Any:
        import torch

        if tensor.ndim == 2 and int(tensor.shape[0]) == 1:
            tensor = tensor[0]
        if tuple(tensor.shape) != (EMBEDDING_DIM,):
            raise ValueError(f"AnatCL produced shape {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("AnatCL produced non-finite embedding values")
        return tensor

    def encode(self, record: Any, *, training: bool, epoch: int, seed: int) -> Any:
        """Return the encoder representation; no projection head is applied."""

        del training, epoch, seed
        volume = self.load_input(record).to(self.device, non_blocking=True)
        embedding = self._require_loaded()(volume)
        return self._validate_embedding(embedding)

    def set_frozen(self) -> dict[str, Any]:
        """Freeze both encoder and unused contrastive projection head."""

        model = self._require_loaded()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        return self._parameter_summary([])

    def controlled_modules(self) -> list[tuple[str, Any]]:
        """Expose only the final ResNet stage as the controlled encoder boundary."""

        encoder = self._require_loaded().backbone.encoder
        layer4 = getattr(encoder, "layer4", None)
        if layer4 is None or len(layer4) != 2:
            raise ValueError("Expected the two-block layer4 of a 3D ResNet-18")
        return [("backbone.encoder.layer4", layer4)]

    def set_controlled_adaptation(self) -> dict[str, Any]:
        """Freeze everything, then unfreeze exactly the declared final stage."""

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
            raise RuntimeError(f"Unexpected trainable AnatCL parameters: {unexpected[:10]}")
        model.eval()
        summary = self._parameter_summary(components)
        summary["component_parameter_counts"] = counts
        return summary

    def controlled_train_mode(self) -> None:
        """Keep all BatchNorm statistics fixed while optimizing layer4 weights."""

        self._require_loaded().eval()

    def trainable_named_parameters(self) -> list[tuple[str, Any]]:
        return [
            (name, parameter)
            for name, parameter in self._require_loaded().named_parameters()
            if parameter.requires_grad
        ]

    def _parameter_summary(self, components: list[tuple[str, Any]]) -> dict[str, Any]:
        model = self._require_loaded()
        encoder = model.backbone.encoder
        encoder_total = int(sum(parameter.numel() for parameter in encoder.parameters()))
        encoder_trainable = int(
            sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad)
        )
        projector_total = int(sum(parameter.numel() for parameter in model.backbone.head.parameters()))
        return {
            "model": self.model_name,
            "variant": self.variant,
            "fold": self.fold,
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
                "full wrapper eval; gradients only through declared layer4 parameters"
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
            "fold": self.fold,
            "environment": self.environment,
            "embedding_dim": self.embedding_dim,
            "readout": self.readout,
            "adaptation_boundary": self.adaptation_boundary,
            "checkpoint_dir": str(self.checkpoint_dir),
            "input_root": str(self.preproc_root),
            "input_contract": {"required_offline_group": OFFLINE_INPUT_GROUP},
            "preprocessing_pipeline_version": PREPROCESSING_PIPELINE_VERSION,
            "preprocessing_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
            "shared_input_shape": list(SHARED_INPUT_SHAPE),
            "model_input_shape": list(MODEL_INPUT_SHAPE),
            "crop_y": list(CROP_Y),
            "normalization": "none (release Normalize(mean=0,std=1) is identity)",
        }
