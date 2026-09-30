# y-Aware brain-age workflow

This directory implements the two common downstream arms for the single public
BHB-10K y-Aware checkpoint:

- `extract_embeddings.py`: audits the release and extracts the frozen
  1024-dimensional representation before the contrastive projection head;
- `fit_ridge.py`: fits `StandardScaler + Ridge` using train only and selects
  its alpha by uncorrected validation MAE;
- `finetune.py`: adapts only `features.denseblock4` plus a new linear age head;
- `predict_onharmony.py`: applies a retained adapted checkpoint to ON-Harmony
  without loading chronological age or other outcomes;
- `model.py`: owns checkpoint, input, readout, and trainable-boundary contracts;

The model consumes the complete `1 x 121 x 145 x 121` z-scored NumPy view
created by the shared CAT12/IXI555 preprocessing. It does not crop or normalize
the volume again. The released encoder-mode DenseNet-121 is reconstructed
without adding the classifier-only `norm5`. All checkpoint keys, including the
`1024 -> 512 -> 128` projection head, are loaded strictly; that projection head
is retained only for provenance and is never used by either downstream arm.

## Audit and smoke test

Run this before extraction or training:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/yAware/extract_embeddings.py audit \
  --split train \
  --check-input-files \
  --inspect-model \
  --smoke-forward \
  --device cuda \
  --amp bfloat16 \
  --seed 0
```

The command must report the pinned checkpoint SHA-256, a strict proof of load,
no fresh `norm5`, a 1024-dimensional readout, and gradients only in
`features.denseblock4` plus the temporary smoke-test head.

## Frozen arm

Extract train and validation separately:

```bash
python -u FineTuning/yAware/extract_embeddings.py extract \
  --split train --device cuda --amp bfloat16 --seed 42 \
  --output Results/brain_age/embeddings_FT/yAware/canonical/train.tsv

python -u FineTuning/yAware/extract_embeddings.py extract \
  --split validation --device cuda --amp bfloat16 --seed 42 \
  --output Results/brain_age/embeddings_FT/yAware/canonical/validation.tsv

python -u FineTuning/yAware/fit_ridge.py fit \
  --train-embeddings Results/brain_age/embeddings_FT/yAware/canonical/train.tsv \
  --validation-embeddings Results/brain_age/embeddings_FT/yAware/canonical/validation.tsv \
  --output-dir Results/brain_age/frozen/yAware/canonical
```

The locked Ridge model is applied label-blind to the canonical ON-Harmony
embedding CSV by the common launcher. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for the frozen output
contract and sequencing.

Internal-test embedding extraction remains label-free, but it requires
`--acknowledge-locked-test` and reserves a one-time receipt before reading the
first test input. Prediction and evaluation remain locked until model selection
is complete.

## Controlled adaptation: five runs

First train the three learning-rate configurations with seed 0:

```bash
for CONFIG in conservative intermediate high; do
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u FineTuning/yAware/finetune.py train \
    --device cuda --amp bfloat16 --seed 0 \
    --lr-config "$CONFIG" \
    --output-dir "Results/brain_age/adapted/yAware/selection_seed0_${CONFIG}"
done
```

Select the configuration using validation only:

```bash
python -u FineTuning/yAware/finetune.py select \
  --run-dirs \
    Results/brain_age/adapted/yAware/selection_seed0_conservative \
    Results/brain_age/adapted/yAware/selection_seed0_intermediate \
    Results/brain_age/adapted/yAware/selection_seed0_high \
  --output-dir Results/brain_age/adapted/yAware/lr_selection_seed0
```

Read `SELECTED_LR.json`, then run that one configuration with seeds 1 and 2.
Together with the three seed-0 candidates, this gives exactly five trainings.
The primary adapted summary uses the selected configuration at seeds 0, 1, and
2; the two rejected seed-0 candidates are hyperparameter-selection runs, not
additional models.

Internal-test evaluation is a separate one-time command and requires both
`--acknowledge-locked-test` and `--selection-dir` pointing to the immutable
seed-0 LR-selection artifact. ON-Harmony is rejected by the shared cohort
firewall throughout development.

## Definitive cohort

`CamCAN/sub-CC620821` was removed from the definitive common cohort after its
shared CAT12 affine/Dartel preprocessing failed on two independent attempts.
The candidate-cohort exclusion ledger retains the case and reason, while every
model consumes the same 1,411-subject manifest (`1130/141/140`). Train and
validation remain unchanged.

## Adapted ON-Harmony inference

`predict_onharmony.py` validates the full-volume CAT12/y-Aware view and its
shared preprocessing log, restores `denseblock4` and the locked age head, and
generates label-blind predictions for all 165 acquisitions. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for retained runs,
audit command, and output contract.
