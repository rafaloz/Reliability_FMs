# BrainIAC brain-age workflow

This directory implements the two common downstream arms without changing the
released BrainIAC input or representation:

- frozen `768`-dimensional first-patch embeddings followed by the shared
  `StandardScaler + Ridge` model;
- controlled adaptation of ViT block 12 and the final layer normalization,
  followed by one linear age head.

The input is the `96 x 96 x 96` NIfTI produced by
`preprocessings/FineTunningDataset/preprocessing_BrainIAC.py`. Each input is
accepted only when its PASS QC, preprocessing contract, and SHA-256 hash match.
The contrastive projection head is never used. BrainIAC has no CLS token: the
canonical released readout is token zero of the 216-patch sequence.

## Audit

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/BrainIAC/extract_embeddings.py audit \
  --split train \
  --check-input-files \
  --inspect-model \
  --smoke-forward \
  --device cuda \
  --amp bfloat16 \
  --seed 0
```

## Frozen arm

Run `extract` separately for `train` and `validation`, then fit Ridge:

```bash
python -u FineTuning/BrainIAC/extract_embeddings.py extract \
  --split train --device cuda --amp bfloat16 --seed 0 \
  --output Results/brain_age/embeddings_FT/BrainIAC/canonical/train.tsv

python -u FineTuning/BrainIAC/extract_embeddings.py extract \
  --split validation --device cuda --amp bfloat16 --seed 0 \
  --output Results/brain_age/embeddings_FT/BrainIAC/canonical/validation.tsv

python -u FineTuning/BrainIAC/fit_ridge.py fit \
  --train-embeddings Results/brain_age/embeddings_FT/BrainIAC/canonical/train.tsv \
  --validation-embeddings Results/brain_age/embeddings_FT/BrainIAC/canonical/validation.tsv \
  --output-dir Results/brain_age/frozen/BrainIAC/canonical
```

The locked Ridge head is applied label-blind to the canonical ON-Harmony
embedding CSV by the common launcher. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for the frozen output
contract and sequencing.

Internal-test extraction is label-free, but it is still a protected access.
Both extraction and the later frozen evaluation require the explicit
`--acknowledge-locked-test` flag and create persistent receipts. The test split
is used only after all train/validation selection has finished.

## Controlled adaptation

Run seed 0 once for each locked LR configuration:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/BrainIAC/finetune.py train \
  --device cuda --amp bfloat16 --seed 0 \
  --lr-config conservative \
  --output-dir Results/brain_age/adapted/BrainIAC/selection_seed0_conservative
```

Replace `conservative` with `intermediate` and `high`, then select using only
validation MAE:

```bash
python -u FineTuning/BrainIAC/finetune.py select \
  --run-dirs \
    Results/brain_age/adapted/BrainIAC/selection_seed0_conservative \
    Results/brain_age/adapted/BrainIAC/selection_seed0_intermediate \
    Results/brain_age/adapted/BrainIAC/selection_seed0_high \
  --output-dir Results/brain_age/adapted/BrainIAC/lr_selection_seed0
```

After selection, repeat the selected LR configuration with seeds 1 and 2. The
encoder stays in evaluation mode during adaptation; gradients are enabled only
for the declared final block and final normalization.

## Adapted ON-Harmony inference

`predict_onharmony.py` validates each ON-Harmony NIfTI against its PASS QC and
preprocessing log, restores the locked adapted checkpoint, and writes
label-blind predictions in canonical acquisition order. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for retained runs,
audit command, and output contract.
