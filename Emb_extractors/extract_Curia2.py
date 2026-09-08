#!/usr/bin/env python3
"""
Embedding extractor for Curia-2 (ViT-B via DINOv2, 2D slice-based).

Input:  A .txt file with one path per line to RAW 3D T1w volumes (no preproc).
        In the V5 reliability paper this extractor is fed ON-Harmony raw T1w:
            /datos/BD/MRI/public_datasets_brain_MRI/ON_Harmony/original_data/
                sub-*/ses-*/anat/sub-*_ses-*_T1w.nii.gz
        Curia-2 does its own preproc internally via AutoImageProcessor
        (resize 512x512 + normalization). Skull-stripping is NOT required
        and NOT applied here — the `_brain.nii.gz` output of the BrainIAC
        pipeline is NOT a Curia-2 input in this project.
Output: A .csv in DATA/embeddings/ with columns:
          file_path, stem, n_all_slices, n_core_slices,
          cls_all_0   ... cls_all_767     # mean(CLS)   over all axial slices
          patch_all_0 ... patch_all_767   # mean(patch) over all axial slices
          cls_core_0   ... cls_core_767   # mean(CLS)   over central `core_fraction` band
          patch_core_0 ... patch_core_767 # mean(patch) over central `core_fraction` band

Internal preprocessing from raw T1w:
  1. Reading the RAW T1w volume and reorienting to canonical LAS so  # audit M-Curia-21: doc said skull-stripped; extractor takes RAW T1w
     the slice-axis logic is deterministic regardless of NIfTI storage order.
  2. Iterating over every axial slice and rotating each slice 90 deg
     counter-clockwise to obtain the
     PL orientation required by Curia-2's README (axis 0 = P, axis 1 = L).
     This step matters: without it, cosine(LA, PL) ~ 0.78 across subjects,
     below the between-subject baseline — the model treats a 90 deg rotation
     as a bigger perturbation than a different brain.
  3. Passing the reoriented 2D array to Curia-2's `AutoImageProcessor`
     (handles resize to 512x512 + normalization).

Embedding extraction:
  1. Building the Curia-2 ViT-B backbone and official image processor.
  2. Forward pass through the Curia-2 backbone. From the last hidden state
     we keep BOTH per-slice representations:
       - the [CLS] token (index 0)                       (768-dim)
       - the mean over patch tokens (indices 1 onwards)  (768-dim)
  3. From the per-slice streams we compute TWO subject-level aggregations
     each, differing only in how many slices are averaged:
       - "all":  mean over every axial slice
       - "core": mean over the CENTRAL `core_fraction` of axial slices
                 (default 0.20). Approximates the Curia-paper §4.5
                 protocol on IXI ("we extracted 20% of the brain's axial
                 images from the T1-weighted MR images"), which biases the
                 embedding toward mid-brain slices (basal ganglia, corpus
                 callosum, ventricles) that carry higher anatomical
                 information density.
     All four streams (CLS-all, patch-all, CLS-core, patch-core) are shipped
     side-by-side in the CSV, so downstream code can pick the aggregation
     that performs best on a held-out validation set per task, without
     re-running the (expensive) forward pass.

History:
  - Prior to 2026-07-09 this extractor omitted canonical LAS reorientation
    and the 90-degree LA-to-PL slice rotation, which produced LA-oriented
    slices. See `code/check_curia2_orientation.py` for the diagnostic that
    surfaced this.
  - 2026-07-09: extractor also emits the patch-mean stream alongside CLS.
  - 2026-07-09: extractor also emits the central-20% aggregation (in
    addition to the all-content one) to approximate the Curia paper's
    IXI slice-selection protocol. Earlier CSVs only carry `emb_*` columns.

Requirements:
  pip install torch transformers nibabel numpy pandas

Usage:
  python extract_Curia2.py \\
      --input_list /path/to/curia2_file_list.txt \\
      --model_dir /path/to/Models/Curia-2 \\
      --output ../DATA/embeddings_Curia2.csv
"""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
    ornt_transform,
)
from transformers import AutoImageProcessor, AutoModel


TARGET_AXCODES = ("L", "A", "S")  # canonical layout after reorientation
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_ACQUISITIONS = 165
EXPECTED_SUBJECTS = 20
EXPECTED_EMBEDDING_DIM = 768
RAW_ROOT_MARKER = "/ON_Harmony/original_data/"
RAW_PATTERN = re.compile(
    r"^(?P<acquisition>sub-(?P<participant>[0-9]+)_ses-"
    r"(?P<session>[A-Za-z0-9]+)_T1w)\.nii\.gz$"
)
CURIA2_RELEASE_REVISION = "25d164965f10ceaf6b394f803a4704598cb3ab1a"
CURIA2_RELEASE_HASHES = {
    "config.json": "9b22dcc40c81d8e3d536978cb81db049e9940ea6b02b3bfc815e1e781d57f252",
    "curia_image_processor.py": "74a4eb5e27539686965327c4e8293648a552777452a51eb1ccac956067cc236b",
    "preprocessor_config.json": "469b3bed97af2b8923002a11a6fe4500f34bcae39a49c5822ae15cefa15702e5",
    "model.safetensors": "403a02e27531d2858ecd1e9b1ec2d5ea7bfa909f10ff0d9e8416090a6a8c96ef",
}


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of one release or output artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_curia2_release(model_dir):
    """Require the exact downloaded Curia-2 model and processor release."""
    model_dir = Path(model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    observed = {}
    for filename, expected in CURIA2_RELEASE_HASHES.items():
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        observed[filename] = digest
        if digest != expected:
            raise ValueError(
                f"Curia-2 release drift for {filename}: "
                f"expected {expected}, observed {digest}"
            )
    return model_dir, observed


def build_model(model_dir, device="cpu", interpolation="bicubic"):
    """Load Curia-2 backbone (SSL-pretrained ViT-B) and its image processor.

    The processor shipped in `raidium/curia-2` hard-codes bicubic interpolation
    inside its `_resize`. Curia paper (v1) §4.1 said bilinear, but the Curia-2
    paper (v2) is silent on the resize interpolation choice. We therefore keep
    the shipped processor's default (bicubic) as the canonical inference path.
    Pass `interpolation="bilinear"` to monkey-patch _resize with bilinear
    (matching v1 paper explicitly) if you want to test that specific config.
    """
    if interpolation not in ("bilinear", "bicubic"):
        raise ValueError(f"interpolation must be 'bilinear' or 'bicubic', got {interpolation!r}")
    model_dir, _ = validate_curia2_release(model_dir)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)

    # Config sanity — canonical raidium/curia-2 ViT-B (Dinov2Model):
    #   Dinov2, hidden_size=768, image_size=512, patch_size=16, layers=12, heads=12,
    #   num_channels=1.  If any of these drift, the pipeline is not talking to the
    #   published Curia-2 architecture we designed against.
    cfg = model.config
    expected = {
        "model_type": "dinov2",
        "hidden_size": 768,
        "image_size": 512,
        "patch_size": 16,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_channels": 1,
    }
    mismatches = {k: (getattr(cfg, k, None), v)
                  for k, v in expected.items() if getattr(cfg, k, None) != v}
    assert not mismatches, (
        f"Curia-2 config drift vs canonical raidium/curia-2 (Dinov2 ViT-B): {mismatches}. "
        f"Wrong checkpoint or wrong repo?")

    # Processor sanity — the shipped CuriaImageProcessor.
    # audit M-Curia-129: assert clip_below_air + eps to mirror check_extractor_curia2 sanity dict
    proc_expected = {"crop_size": 512, "do_resize": True, "do_normalize": True,
                     "clip_below_air": True, "eps": 1e-6}
    proc_mismatches = {k: (getattr(processor, k, None), v)
                       for k, v in proc_expected.items()
                       if getattr(processor, k, None) != v}
    assert not proc_mismatches, (
        f"Curia-2 processor config drift: {proc_mismatches}")

    if interpolation == "bilinear":
        _patch_processor_interpolation(processor, mode="bilinear")
    model.eval()
    model.to(device)
    print(f"Curia-2 loaded from {model_dir}  (hidden_size={cfg.hidden_size}, "
          f"layers={cfg.num_hidden_layers}, image_size={cfg.image_size}, "
          f"patch_size={cfg.patch_size}, resize_interp={interpolation})")
    return model, processor


def _patch_processor_interpolation(processor, mode="bilinear"):
    """Override the Curia processor's _resize method to use a chosen interp mode.
    Keeps `align_corners=False` and `antialias=True` from the original impl.
    """
    def _resize_patched(self, tensor):
        if not self.do_resize:
            return tensor
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor (H, W), got shape {tensor.shape}")
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(self.crop_size, self.crop_size),
            mode=mode,
            align_corners=False,
            antialias=True,
        )
        return tensor[0, 0]
    processor._resize = _resize_patched.__get__(processor, type(processor))


@torch.no_grad()
def extract_volume_embedding(volume_3d, model, processor, device,
                             core_fraction=0.20, core_n_slices=None,
                             slice_batch_size=32):
    """Return four subject-level embeddings + slice counts per volume.

    For each axial slice we compute two per-slice representations:
      - CLS token from last_hidden_state[:, 0, :]           (768-dim)
      - Mean over patch tokens: last_hidden_state[:, 1:, :] (768-dim)

    We then mean-pool each stream across TWO different slice subsets:
      - "all":  every axial slice
      - "core": the central `core_n_slices` slices (if given) or the central
                `core_fraction` of all axial slices otherwise.
                Approximates the Curia paper §4.5 IXI protocol
                ("20% of the brain's axial images"; paper actually reports
                ~11 slices/volume on IXI, so pass core_n_slices=11 to match
                the paper's absolute count exactly).

    The four output vectors correspond to the four (stream x subset)
    combinations. Computing all four costs zero extra forward passes since
    the patch stream is already produced by the same forward, and the core
    aggregation is a slice of the per-slice arrays.

    The input volume is assumed to be in LAS orientation (see load_volume).
    Each axial slice `vol[..., z]` therefore has axis 0 = L and axis 1 = A.
    Curia-2's README requires axial slices in PL orientation, i.e. axis 0 = P
    and axis 1 = L, so we apply a single 90 deg CCW rotation per slice:

        LA (rows=L, cols=A)  --np.rot90(k=1)-->  PL (rows=P, cols=L)
    """
    kept = list(range(volume_3d.shape[-1]))
    if not kept:
        raise RuntimeError("Input volume has no axial slices")

    if slice_batch_size <= 0:
        raise ValueError("slice_batch_size must be positive")
    cls_chunks = []
    patch_chunks = []
    for batch_start in range(0, len(kept), slice_batch_size):
        indices = kept[batch_start:batch_start + slice_batch_size]
        slices = []
        for z in indices:
            slc_LA = volume_3d[..., z].astype(np.float32, copy=False)
            slc_PL = np.rot90(slc_LA, k=1).copy()
            # The torch branch stays float32; the released NumPy branch casts to
            # int16 and corrupts high-intensity ON-Harmony acquisitions.
            slices.append(torch.from_numpy(slc_PL))

        inputs = processor(slices, return_tensors="pt")
        pixels = inputs.get("pixel_values")
        if pixels is None or tuple(pixels.shape[1:]) != (1, 512, 512):
            raise ValueError(
                f"Unexpected Curia-2 processor output: "
                f"{None if pixels is None else tuple(pixels.shape)}"
            )
        if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
            raise ValueError("Curia-2 processor returned invalid pixel values")
        outputs = model(pixel_values=pixels.to(device, non_blocking=True))
        hidden = outputs.last_hidden_state
        cls_chunks.append(hidden[:, 0, :].float().cpu().numpy())
        patch_chunks.append(hidden[:, 1:, :].float().mean(dim=1).cpu().numpy())

    cls_arr = np.concatenate(cls_chunks, axis=0)      # [n_all, 768]
    patch_arr = np.concatenate(patch_chunks, axis=0)  # [n_all, 768]

    n_all = cls_arr.shape[0]
    if core_n_slices is not None:
        n_core = max(1, min(n_all, int(core_n_slices)))
    else:
        n_core = max(1, int(round(n_all * core_fraction)))
    start = (n_all - n_core) // 2
    end = start + n_core

    cls_all = cls_arr.mean(axis=0)
    patch_all = patch_arr.mean(axis=0)
    cls_core = cls_arr[start:end].mean(axis=0)
    patch_core = patch_arr[start:end].mean(axis=0)

    return cls_all, patch_all, cls_core, patch_core, n_all, n_core


def load_volume(file_path):
    """Load an exactly 3D NIfTI and reorient it to deterministic canonical LAS.
    Returns a float32 numpy array of shape (nL, nA, nS).
    """
    img = nib.load(file_path)
    if len(img.shape) != 3:
        raise ValueError(
            f"Curia-2 requires an exactly 3D NIfTI; got {img.shape}: {file_path}"
        )
    data = img.get_fdata().astype(np.float32)
    if not np.isfinite(data).all() or not np.any(data != 0):
        raise ValueError(f"Curia-2 input is empty or non-finite: {file_path}")
    axcodes = nib.aff2axcodes(img.affine)
    if axcodes != TARGET_AXCODES:
        current = io_orientation(img.affine)
        target = axcodes2ornt(TARGET_AXCODES)
        data = apply_orientation(data, ornt_transform(current, target))
    return data


def stem_from_path(file_path):
    """Strip .nii or .nii.gz and the _brain / _brainiac_ready suffix."""
    base = os.path.basename(file_path)
    base = re.sub(r"\.nii(\.gz)?$", "", base)
    base = re.sub(r"_brain$|_brainiac_ready$", "", base)
    return base


def _atomic_write_csv(df, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = output_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, output_path)


def _atomic_write_json(payload, output_path):
    """Publish JSON only after its complete, standards-compliant serialization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, output_path)


def _manifest_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def _acquisition_set(paths, label):
    """Extract and validate unique ON-Harmony raw acquisition identifiers."""
    acquisitions = []
    participants = []
    for path in paths:
        match = RAW_PATTERN.fullmatch(Path(path).name)
        if match is None:
            raise ValueError(f"Unexpected {label} filename: {Path(path).name}")
        acquisitions.append(match.group("acquisition"))
        participants.append(match.group("participant"))
    if len(acquisitions) != len(set(acquisitions)):
        raise ValueError(f"{label} contains duplicate ON-Harmony acquisitions")
    return set(acquisitions), set(participants)


def _load_and_validate_inputs(input_list, canonical_raw_list, expected_count):
    """Require the exact 165-acquisition raw ON-Harmony cohort."""
    input_list = Path(input_list).expanduser().resolve()
    canonical_raw_list = Path(canonical_raw_list).expanduser().resolve()
    if not input_list.is_file() or not canonical_raw_list.is_file():
        raise FileNotFoundError(
            f"Missing ON-Harmony registry: {input_list}, {canonical_raw_list}"
        )
    file_paths = [
        line.strip() for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(file_paths) != expected_count:
        raise ValueError(f"Expected {expected_count} Curia-2 inputs, found {len(file_paths)}")
    if len(file_paths) != len(set(file_paths)):
        raise ValueError("Curia-2 input list contains duplicate paths")
    for path in file_paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            raise ValueError(f"Input path is not absolute: {path}")
        if RAW_ROOT_MARKER not in path:
            raise ValueError(f"Curia-2 input is not an ON-Harmony raw T1w: {path}")
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty Curia-2 raw input: {path}")

    input_acquisitions, participants = _acquisition_set(file_paths, "Curia-2 filelist")
    raw_paths = [
        line.strip()
        for line in canonical_raw_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    canonical_acquisitions, canonical_participants = _acquisition_set(
        raw_paths, "canonical raw filelist"
    )
    if input_acquisitions != canonical_acquisitions:
        missing = sorted(canonical_acquisitions - input_acquisitions)
        extra = sorted(input_acquisitions - canonical_acquisitions)
        raise ValueError(
            f"Curia-2 cohort differs from canonical raw registry: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    if len(participants) != EXPECTED_SUBJECTS or participants != canonical_participants:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} canonical subjects, found {len(participants)}"
        )
    return file_paths, {
        "input_list": str(input_list),
        "input_list_sha256": sha256_file(input_list),
        "canonical_raw_list": str(canonical_raw_list),
        "canonical_raw_list_sha256": sha256_file(canonical_raw_list),
        "n_acquisitions": len(file_paths),
        "n_subjects": len(participants),
    }


def _autodetect_paths():
    """Return (input_list, model_dir) defaults based on where the script sits.

    Convention on cluster: /datos/work/rnavgon/FoundationModels_OH_T1w/Emb_extractors/…
    Convention locally (sshfs): /home/rafa/workRafa/FoundationModels_OH_T1w/Emb_extractors/…
    Curia-2 model lives in the sibling FoundationModels_Paper/Models/Curia-2 tree.
    """
    base = PROJECT_ROOT
    proj_parent = base.parent
    input_list = base / "DATA" / "filelists" / "filelist_Curia2.txt"
    model_dir = proj_parent / "FoundationModels_Paper" / "Models" / "Curia-2"
    return input_list, model_dir


def main():
    _def_input_list, _def_model_dir = _autodetect_paths()
    parser = argparse.ArgumentParser(description="Curia-2 embedding extractor")
    parser.add_argument("--input_list", type=str,
                        default=_def_input_list,
                        help=f".txt file with one RAW T1w NIfTI path per line "
                             f"(default auto-detected: {_def_input_list})")
    parser.add_argument("--model_dir", type=str,
                        default=_def_model_dir,
                        help=f"Path to Models/Curia-2/ (default auto-detected: {_def_model_dir})")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "DATA" / "embeddings" /
                                    "embeddings_Curia2.csv"),
                        help="Output CSV path")
    parser.add_argument(
        "--canonical_raw_list", type=str,
        default=str(PROJECT_ROOT / "DATA" / "filelists" / "filelist_BrainIAC_raw.txt"),
        help="Canonical 165-acquisition ON-Harmony raw registry used for identity checks",
    )
    parser.add_argument("--expected_count", type=int, default=EXPECTED_ACQUISITIONS)
    parser.add_argument("--slice_batch_size", type=int, default=32,
                        help="Axial slices forwarded together (default 32)")
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing CSV/manifest intentionally")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--core_fraction", type=float, default=0.20,
                        help="Fraction of all axial slices used for "
                             "the 'core' aggregation (default 0.20, matching "
                             "the Curia paper §4.5 wording; ignored if "
                             "--core_n_slices is set).")
    parser.add_argument("--core_n_slices", type=int, default=None,
                        help="Override --core_fraction with an ABSOLUTE count "
                             "of central slices. Curia v1 §4.5 IXI reports "
                             "~11 slices/volume; pass --core_n_slices 11 for a "
                             "paper-strict match regardless of total slice "
                             "count in each subject.")
    parser.add_argument("--interpolation", choices=("bilinear", "bicubic"),
                        default="bicubic",
                        help="Interpolation for the 512x512 resize. Default "
                             "'bicubic' matches the processor shipped with "
                             "raidium/curia-2. The Curia paper (v1) §4.1 "
                             "mentions bilinear, but Curia-2 paper does NOT "
                             "specify — so we defer to the shipped processor. "
                             "Choose 'bilinear' to monkey-patch that override.")
    args = parser.parse_args()

    if args.expected_count <= 0 or args.slice_batch_size <= 0:
        parser.error("--expected_count and --slice_batch_size must be positive")
    if not 0 < args.core_fraction <= 1:
        parser.error("--core_fraction must lie in (0, 1]")
    if args.core_n_slices is not None and args.core_n_slices <= 0:
        parser.error("--core_n_slices must be positive")
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = _manifest_path(output_path)
    existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing final artifacts; pass --force intentionally: "
            + ", ".join(existing)
        )
    file_paths, input_registry = _load_and_validate_inputs(
        args.input_list, args.canonical_raw_list, args.expected_count
    )

    # Startup banner — makes it unambiguous which process is which and where
    # it runs (CPU vs GPU) when multiple extractors run in parallel and their
    # stdouts get inspected together via tail/logs.
    device_str = args.device
    device_detail = "unknown"
    if args.device.startswith("cuda"):
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            device_detail = f"cuda:{idx} — {torch.cuda.get_device_name(idx)}"
        else:
            device_detail = "CUDA REQUESTED BUT torch.cuda.is_available()=False"
    elif args.device == "cpu":
        import multiprocessing
        device_detail = (f"cpu (torch threads={torch.get_num_threads()}, "
                         f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','unset')}, "
                         f"machine cores={multiprocessing.cpu_count()})")

    print("=" * 64, flush=True)
    print("  Curia-2 extractor", flush=True)
    print(f"  PID:        {os.getpid()}", flush=True)
    print(f"  Input list: {args.input_list}", flush=True)
    print(f"  N files:    {len(file_paths)}", flush=True)
    print(f"  Output CSV: {args.output}", flush=True)
    print(f"  Device:     {device_detail}", flush=True)
    print(f"  Interp:     {args.interpolation}", flush=True)
    print(f"  Core frac:  {args.core_fraction}"
          + (f"  (or core_n_slices={args.core_n_slices})" if args.core_n_slices else ""),
          flush=True)
    print(f"  Slice batch:{args.slice_batch_size}", flush=True)
    print("=" * 64, flush=True)

    # Extraction 1/3: build Curia-2 ViT-B and its official image processor.
    print(f"[Extraction 1/3] Build Curia-2 ViT-B (HF) + AutoImageProcessor")
    model, processor = build_model(args.model_dir, args.device,
                                   interpolation=args.interpolation)

    rows = []
    failures = []
    for i, fpath in enumerate(file_paths):
        try:
            # Preprocessing 1/3: load raw T1w and reorient to LAS.
            vol = load_volume(fpath)
            # Preprocessing 2/3: select axial slices and rotate LA→PL.
            # Preprocessing 3/3: AutoImageProcessor resize/intensity preparation.
            # Extraction 2/3: per-slice forward through the ViT.
            # Extraction 3/3: aggregate the four subject-level streams
            #         (cls_all, cls_core, patch_all, patch_core), each 768-d
            cls_all, patch_all, cls_core, patch_core, n_all, n_core = (
                extract_volume_embedding(vol, model, processor, args.device,
                                         core_fraction=args.core_fraction,
                                         core_n_slices=args.core_n_slices,
                                         slice_batch_size=args.slice_batch_size))
        except Exception as e:
            print(f"  [{i+1}/{len(file_paths)}] FAIL {stem_from_path(fpath)}: {e}")
            failures.append((fpath, repr(e)))
            continue

        row = {"file_path": fpath, "stem": stem_from_path(fpath),
               "n_all_slices": n_all, "n_core_slices": n_core}
        for d, val in enumerate(cls_all):
            row[f"cls_all_{d}"] = val
        for d, val in enumerate(patch_all):
            row[f"patch_all_{d}"] = val
        for d, val in enumerate(cls_core):
            row[f"cls_core_{d}"] = val
        for d, val in enumerate(patch_core):
            row[f"patch_core_{d}"] = val
        rows.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == len(file_paths):
            print(f"  [{i+1}/{len(file_paths)}] {stem_from_path(fpath)}  "
                  f"(n_all={n_all}, n_core={n_core})")

    # Output validation/serialization is not an extraction step.
    print(f"[Output] Validate and save CSV → {args.output}")
    df = pd.DataFrame(rows)
    stream_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ("cls_all_", "patch_all_", "cls_core_", "patch_core_"))]
    nonfinite = 0 if not stream_cols else int(
        (~np.isfinite(df[stream_cols].to_numpy(dtype=np.float64))).sum())
    if failures or len(df) != len(file_paths) or nonfinite:
        partial_path = str(output_path) + ".partial.csv"
        if not df.empty:
            _atomic_write_csv(df, partial_path)
        raise RuntimeError(
            f"Refusing to replace {args.output}: extracted {len(df)}/{len(file_paths)} "
            f"rows, failures={len(failures)}, nonfinite_values={nonfinite}. "
            f"Partial output: {partial_path}")
    if df["file_path"].duplicated().any() or df["stem"].duplicated().any():
        raise RuntimeError("Refusing to save Curia-2 CSV with duplicate rows")
    counts = {p: sum(1 for c in df.columns if c.startswith(p))
              for p in ("cls_all_", "patch_all_", "cls_core_", "patch_core_")}
    if any(count != EXPECTED_EMBEDDING_DIM for count in counts.values()):
        raise RuntimeError(f"Unexpected Curia-2 stream dimensions: {counts}")
    if (df[["n_all_slices", "n_core_slices"]] <= 0).any().any():
        raise RuntimeError("Curia-2 output contains invalid slice counts")
    _atomic_write_csv(df, str(output_path))
    model_dir, release_hashes = validate_curia2_release(args.model_dir)
    manifest = {
        "manifest_schema_version": "onharmony_embedding_manifest_v1",
        "pipeline_version": "curia2_onharmony_embedding_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Curia-2",
        "variant": "canonical",
        "readout": "mean CLS token across all axial slices",
        "primary_readout": "mean CLS token across all axial slices",
        "feature_prefix": "cls_all_",
        "embedding_dimensions": EXPECTED_EMBEDDING_DIM,
        "streams": counts,
        "device": args.device,
        "slice_batch_size": args.slice_batch_size,
        "interpolation": args.interpolation,
        "core_fraction": args.core_fraction,
        "core_n_slices": args.core_n_slices,
        "registry": input_registry,
        "release": {
            "model_dir": str(model_dir),
            "huggingface_repo": "raidium/curia-2",
            "downloaded_revision": CURIA2_RELEASE_REVISION,
            "files_sha256": release_hashes,
        },
        "extractor": str(SCRIPT_PATH),
        "extractor_sha256": sha256_file(SCRIPT_PATH),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "n_rows": len(df),
    }
    _atomic_write_json(manifest, manifest_path)
    print(f"Saved: {output_path}  ({len(df)} rows)")
    for p, n in counts.items():
        print(f"    {p}*: {n} dims")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
