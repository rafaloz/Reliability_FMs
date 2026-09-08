# 3D-Neuro-SimCLR brain-age workflow

This directory implements both prespecified brain-age arms: frozen embeddings
with a common StandardScaler + Ridge regressor, and controlled adaptation of
the final ResNet stage plus a linear age head. Both arms share one model
contract, so crop, masked standardization, checkpoint loading, and readout are
not reimplemented by the individual entry points.

The extractor consumes the retained files produced by
`preprocessings/FineTunningDataset/preprocessing_3DNeuroSimCLR.py`:

- `<subject>_simclr_ready.nii.gz`: WhiteStripe-normalized registered T1w;
- `<subject>_mask_mni.nii.gz`: registered binary brain mask;
- `preprocessing_qc.json`: PASS status, input contract, and output hashes.

At model input, it reproduces the public release: transpose and fixed crop to
`1 x 150 x 192 x 192`, followed by per-volume z-score normalization over the
mask. The public 3D ResNet-18 is loaded strictly, its contrastive projector is
discarded, and the 512-dimensional global-pooled encoder output is used. The
adapted arm freezes the complete encoder except `layer4`; all BatchNorm modules
remain in evaluation mode so their pretrained running statistics cannot drift.

## Cluster audit and smoke test

```bash
cd /datos/work/rnavgon/FoundationModels_OH_T1w
conda activate myLocalEnv

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/3DNeuroSimCLR/extract_embeddings.py audit \
  --split train \
  --check-input-files \
  --inspect-model \
  --smoke-forward \
  --device cuda \
  --amp bfloat16 \
  --seed 0
```

## Frozen train and validation embeddings

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/3DNeuroSimCLR/extract_embeddings.py extract \
  --split train \
  --device cuda \
  --amp bfloat16 \
  --seed 0 \
  --output Results/brain_age/embeddings_FT/3DNeuroSimCLR/canonical/train.tsv

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/3DNeuroSimCLR/extract_embeddings.py extract \
  --split validation \
  --device cuda \
  --amp bfloat16 \
  --seed 0 \
  --output Results/brain_age/embeddings_FT/3DNeuroSimCLR/canonical/validation.tsv
```

Fit the Ridge grid using validation only:

```bash
python -u FineTuning/3DNeuroSimCLR/fit_ridge.py fit \
  --train-embeddings Results/brain_age/embeddings_FT/3DNeuroSimCLR/canonical/train.tsv \
  --validation-embeddings Results/brain_age/embeddings_FT/3DNeuroSimCLR/canonical/validation.tsv \
  --output-dir Results/brain_age/frozen/3DNeuroSimCLR/canonical
```

The locked Ridge head is applied label-blind to the canonical ON-Harmony
embedding CSV with the common sequential launcher. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for the 11 frozen
representations, firewall, and output convention.

## Controlled adaptation

Run `conservative`, `intermediate`, and `high` with selection seed 0. Each
output directory must be new and is locked after train/validation:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/3DNeuroSimCLR/finetune.py train \
  --device cuda \
  --amp bfloat16 \
  --seed 0 \
  --lr-config high \
  --output-dir Results/brain_age/adapted/3DNeuroSimCLR/selection_seed0_high
```

After the three runs finish, select the learning rate without reading the
internal test:

```bash
python -u FineTuning/3DNeuroSimCLR/finetune.py select \
  --run-dirs \
    Results/brain_age/adapted/3DNeuroSimCLR/selection_seed0_conservative \
    Results/brain_age/adapted/3DNeuroSimCLR/selection_seed0_intermediate \
    Results/brain_age/adapted/3DNeuroSimCLR/selection_seed0_high \
  --output-dir Results/brain_age/adapted/3DNeuroSimCLR/lr_selection_seed0
```

The selected configuration is then repeated with seeds 1 and 2. Adapted-model
evaluation reads the validated preprocessed images directly; it does not use
the frozen embedding bundles. Evaluation requires both
`--acknowledge-locked-test` and `--selection-dir` pointing to the immutable
LR-selection artifact, and creates a separate test-access receipt.

## Label-blind ON-Harmony prediction

`predict_onharmony.py` applies one adapted checkpoint directly to the 165
ON-Harmony acquisitions. It verifies the validation-only lock, the selected
learning rate, unchanged training code, the canonical raw filelist, the fixed
ICBM152 2009c template contract, and each successful preprocessing log. This
stage never loads chronological age and therefore writes predictions without
`age_true`, errors, or metrics.

Audit one acquisition before a full run:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/3DNeuroSimCLR/predict_onharmony.py audit \
  --run-dir Results/brain_age/adapted/3DNeuroSimCLR/selection_seed0_intermediate \
  --selection-dir Results/brain_age/adapted/3DNeuroSimCLR/lr_selection_seed0 \
  --subject sub-03286 \
  --session NOT1ACH001 \
  --check-input-files \
  --inspect-model \
  --smoke-forward \
  --device cuda
```

After the audit passes, materialize the full immutable prediction table:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/3DNeuroSimCLR/predict_onharmony.py predict \
  --run-dir Results/brain_age/adapted/3DNeuroSimCLR/selection_seed0_intermediate \
  --selection-dir Results/brain_age/adapted/3DNeuroSimCLR/lr_selection_seed0 \
  --device cuda \
  --output-dir Results/brain_age/onharmony/adapted/3DNeuroSimCLR/selection_seed0_intermediate
```
