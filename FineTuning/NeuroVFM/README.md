# NeuroVFM brain-age workflow

NeuroVFM has three model-specific entry points:

- `extract_embeddings.py`: audit the exact upstream code/checkpoint and extract
  frozen volume embeddings.
- `fit_ridge.py`: select and lock the common StandardScaler + Ridge head using
  train and validation embeddings.
- `finetune.py`: train the final volumetric ViT block, final encoder norm, and a
  scalar age head; select among the three predeclared learning-rate pairs; and
  perform the separately unlocked internal-test evaluation.

`model.py` owns the NeuroVFM-specific contract. It uses the released
`StudyPreprocessor`, verifies the installed upstream source and checkpoint
revision, preserves the complete variable-length foreground-token sequence,
and averages final-normalized tokens into one 768-D vector.
This mean pooling is a simple study-defined readout, not the supervised
attentive probe used in the NeuroVFM paper.

All commands require the dedicated `neurovfm` environment and a BF16-capable
GPU with compute capability `sm_80` or newer. The RTX A5000 (`sm_86`) is
compatible. A T4 (`sm_75`) is not.

For installation, use the [official upstream instructions at the pinned commit](https://github.com/MLNeurosurg/neurovfm/tree/9240021d4ef5c262b21cee5d219c2adf65f4d42f#installation).
The Python/CUDA dependency notes and distinction between the upstream
specification and the study runtime are in
[`../../INSTALLATION.md`](../../INSTALLATION.md#neurovfm).

## Audit and smoke test

Run this before any extraction or training:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python FineTuning/NeuroVFM/extract_embeddings.py audit \
  --split train \
  --check-input-files \
  --inspect-model \
  --smoke-forward \
  --device cuda \
  --amp bfloat16 \
  --seed 0
```

The default checkpoint source is `mlinslab/neurovfm-encoder` at revision
`d5194fc70a162185f8ef062e362bd522a35312a9`. Resolution is cache-only unless
`--allow-download` is supplied.

## Persistent preprocessing cache

The released `StudyPreprocessor` is deterministic and independent of the
trainable encoder weights. Materialize its output once before extraction or
training. The cache contains input patches, coordinates, sequence metadata,
modality, path, and preprocessed volume size. It does **not** contain encoder
features or predictions.

One-subject pilot:

```bash
python FineTuning/NeuroVFM/extract_embeddings.py cache \
  --splits train validation \
  --dataset CamCAN \
  --subject sub-CC110033
```

Complete train/validation cache:

```bash
python FineTuning/NeuroVFM/extract_embeddings.py cache \
  --splits train validation
```

Artifacts are written by default to
`preprocessings/FineTunningDataset/preproc_NeuroVFM/<dataset>/<subject>/`.
Each tensor artifact has a JSON sidecar binding it to the raw-file identity,
the pinned NeuroVFM revision, and the exact upstream preprocessing code. A
new artifact must also survive a bit-exact serialization round trip; a changed
or corrupt artifact is rejected. Training forbids raw preprocessing:
all train and validation cache entries must exist before the output directory
is created. Before loading the encoder onto the GPU, training validates and
reads every train/validation batch from disk once, then retains those CPU
tensors in process memory for every epoch and locked validation pass. The run
records cache build, disk-read, RAM-hit, raw-call, and retained-byte counts.

Use `--overwrite` only to rebuild a complete existing pair after an intentional
code or software change. An interrupted one-file partial pair is rebuilt
automatically on the next cache run.

## Frozen arm

```bash
python FineTuning/NeuroVFM/extract_embeddings.py extract \
  --split train \
  --device cuda \
  --amp bfloat16 \
  --seed 0 \
  --output Results/brain_age/embeddings_FT/NeuroVFM/canonical/train.tsv

python FineTuning/NeuroVFM/extract_embeddings.py extract \
  --split validation \
  --device cuda \
  --amp bfloat16 \
  --seed 0 \
  --output Results/brain_age/embeddings_FT/NeuroVFM/canonical/validation.tsv

python FineTuning/NeuroVFM/fit_ridge.py fit \
  --train-embeddings Results/brain_age/embeddings_FT/NeuroVFM/canonical/train.tsv \
  --validation-embeddings Results/brain_age/embeddings_FT/NeuroVFM/canonical/validation.tsv \
  --output-dir Results/brain_age/frozen/NeuroVFM/canonical
```

Frozen extraction calls the official `EncoderPipeline.embed` path. No token
truncation, random crop, or task-specific attentive pooling is introduced.
It consumes the same verified cache as controlled adaptation.

The locked Ridge head is applied label-blind to the canonical ON-Harmony
embedding CSV by the common launcher. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for the frozen output
contract and sequencing.

## Controlled adaptation

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python FineTuning/NeuroVFM/finetune.py train \
  --device cuda \
  --amp bfloat16 \
  --seed 0 \
  --lr-config conservative \
  --output-dir Results/brain_age/adapted/NeuroVFM/selection_seed0_conservative
```

`EncoderPipeline.embed` is decorated with `torch.inference_mode()`. The adapted
arm therefore repeats the same normalization and standard model forward
without that decorator. It keeps the encoder in evaluation mode and enables
gradients only for `blocks.11`, `norm`, and the linear age head.

Internal-test preprocessing and frozen embeddings may be materialized in
advance because they are label-free. Their predictions and metrics remain
locked until learning-rate and seed selection have been frozen. Adapted
internal-test evaluation requires both `--acknowledge-locked-test` and
`--selection-dir` pointing to the immutable LR-selection artifact. ON-Harmony
remains reserved for final external evaluation.

## Adapted ON-Harmony inference

`predict_onharmony.py cache` materializes the released `StudyPreprocessor`
output once per acquisition, including repeated sessions. Its `audit` and
`predict` commands then restore the locked final block, norm, and age head and
never rerun raw preprocessing inside an epoch. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for the retained runs,
label-blind output contract, and required command sequence.
