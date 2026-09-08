# BrainSegFounder brain-age workflow

This directory implements the two common downstream arms for the released
single-channel BrainSegFounder checkpoint:

- frozen `768`-dimensional embeddings followed by the shared
  `StandardScaler + Ridge` model;
- controlled adaptation of `swinViT.layers4`, followed by one linear age head.

The input is the `96 x 96 x 96` NIfTI produced by
`preprocessings/FineTunningDataset/preprocessing_BrainSegFounder.py`. Each input
is accepted only when its PASS QC, preprocessing contract, and SHA-256 hash
match. Preprocessing contract `v4` first runs the released UKB translation and
FNIRT chain. If that chain produces an objectively empty MNI image, it repeats
FNIRT after recovering the crop-to-raw transform from the NIfTI geometry; the
trigger and both attempts are retained in the subject QC. Validated `v3` PASS
artifacts remain accepted with their original provenance because they executed
the identical primary branch; they are not relabelled or recomputed. The
release does not define a global embedding. Consistent with the
reliability benchmark, this study spatially averages the final
`[768, 3, 3, 3]` bottleneck and labels this study-defined variant
`bottleneck_mean`. Rotation, contrastive, and decoder heads are
excluded from both downstream arms.

## Audit

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/BrainSegFounder/extract_embeddings.py audit \
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
python -u FineTuning/BrainSegFounder/extract_embeddings.py extract \
  --split train --device cuda --amp bfloat16 --seed 0 \
  --output Results/brain_age/embeddings_FT/BrainSegFounder/canonical/train.tsv

python -u FineTuning/BrainSegFounder/extract_embeddings.py extract \
  --split validation --device cuda --amp bfloat16 --seed 0 \
  --output Results/brain_age/embeddings_FT/BrainSegFounder/canonical/validation.tsv

python -u FineTuning/BrainSegFounder/fit_ridge.py fit \
  --train-embeddings Results/brain_age/embeddings_FT/BrainSegFounder/canonical/train.tsv \
  --validation-embeddings Results/brain_age/embeddings_FT/BrainSegFounder/canonical/validation.tsv \
  --output-dir Results/brain_age/frozen/BrainSegFounder/bottleneck_mean
```

The locked `bottleneck_mean` Ridge head is applied label-blind to the canonical
ON-Harmony embedding CSV by the common launcher. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for the frozen output
contract and sequencing.

Internal-test embeddings may be extracted with the other splits because the
operation is label-free. Internal-test prediction and evaluation retain their
explicit acknowledgement gate and are performed only after model selection.

## Controlled adaptation

Run seed 0 once for each locked LR configuration:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/BrainSegFounder/finetune.py train \
  --device cuda --amp bfloat16 --seed 0 \
  --lr-config conservative \
  --output-dir Results/brain_age/adapted/BrainSegFounder/selection_seed0_conservative
```

Replace `conservative` with `intermediate` and `high`, then select using only
validation MAE:

```bash
python -u FineTuning/BrainSegFounder/finetune.py select \
  --run-dirs \
    Results/brain_age/adapted/BrainSegFounder/selection_seed0_conservative \
    Results/brain_age/adapted/BrainSegFounder/selection_seed0_intermediate \
    Results/brain_age/adapted/BrainSegFounder/selection_seed0_high \
  --output-dir Results/brain_age/adapted/BrainSegFounder/lr_selection_seed0
```

After selection, repeat the selected LR configuration with seeds 1 and 2. The
full wrapper stays in evaluation mode; gradients are enabled only for the final
Swin stage.

The August 2026 seed-0 runs straddled a selector-only correction: the original
selector requested a nonexistent `registered_shape` field. The training loop was
unchanged, and the repeated high run reproduced every validation prediction and
every non-timing history value bit-exactly. The archived audit utility
`Supplementary/Analyses/select_lr_migration_BrainSegFounder.py` records and
allowlists only those two runtime hashes before writing the standard selection
artifact; it is not part of the encoder training runtime.

## Adapted ON-Harmony inference

`predict_onharmony.py` accepts only the audited ON-Harmony preprocessing
contract and records the effective preprocessing-script hash per acquisition,
including the bit-equivalent pilot produced before the console-only v4 change.
It restores the locked `bottleneck_mean` model and emits label-blind
predictions. See [`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for
the retained runs, audit command, and output contract.
