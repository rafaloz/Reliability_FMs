# AnatCL brain-age workflow

This directory implements the two downstream arms for the five corrected
`anatcl-g3/global` folds:

- `extract_embeddings.py` audits one fold and extracts frozen 512-dimensional
  encoder representations from the shared CAT12 `mwp1` output.
- `fit_ridge.py` selects and locks the common `StandardScaler + Ridge` head
  using train and validation embeddings only.
- `finetune.py` trains the final 3D ResNet-18 stage (`layer4`) plus a linear age
  head, selects the learning-rate pair using validation MAE, and keeps the
  internal test behind an explicit lock.

`model.py` is not an entry point. It pins all five checkpoint hashes, verifies
the installed AnatCL architecture against release commit
`2701acaadcd1a49960a835a8bfb9ba9708d29632`, strictly loads the selected
checkpoint into `model.backbone`, excludes the contrastive projection head,
and validates every offline preprocessing QC record before reading its NIfTI.

The model input is derived at runtime from the full `121 x 145 x 121` CAT12
modulated-GM image by cropping `Y[8:136]`. No intensity transform is added:
the release's `Normalize(mean=0, std=1)` operation is the identity transform.

## Audit and smoke test

Run fold 0 before any extraction or training:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python FineTuning/AnatCL/extract_embeddings.py audit \
  --fold 0 \
  --checkpoint-dir Models/AnatCL/NEW_anatcl-g3 \
  --preproc-root preprocessings/FineTunningDataset/preproc_AnatCL \
  --split train \
  --check-input-files \
  --inspect-model \
  --smoke-forward \
  --device cuda \
  --amp bfloat16 \
  --seed 0
```

Repeat the audit with `--fold 1` through `--fold 4` to verify each checkpoint
hash and strict load. One forward/gradient smoke per fold is sufficient; it is
not a training run.

## Frozen arm

Extract train and validation separately for each fold. For example, fold 0:

```bash
python FineTuning/AnatCL/extract_embeddings.py extract \
  --fold 0 \
  --checkpoint-dir Models/AnatCL/NEW_anatcl-g3 \
  --preproc-root preprocessings/FineTunningDataset/preproc_AnatCL \
  --split train --device cuda --amp bfloat16 --seed 42 \
  --output Results/brain_age/embeddings_FT_bfloat16_v3/AnatCL/fold0/train.tsv

python FineTuning/AnatCL/extract_embeddings.py extract \
  --fold 0 \
  --checkpoint-dir Models/AnatCL/NEW_anatcl-g3 \
  --preproc-root preprocessings/FineTunningDataset/preproc_AnatCL \
  --split validation --device cuda --amp bfloat16 --seed 42 \
  --output Results/brain_age/embeddings_FT_bfloat16_v3/AnatCL/fold0/validation.tsv

python FineTuning/AnatCL/fit_ridge.py fit \
  --fold 0 \
  --train-embeddings Results/brain_age/embeddings_FT_bfloat16_v3/AnatCL/fold0/train.tsv \
  --validation-embeddings Results/brain_age/embeddings_FT_bfloat16_v3/AnatCL/fold0/validation.tsv \
  --output-dir Results/brain_age/frozen_bfloat16_v3/AnatCL/fold0
```

The same frozen protocol is applied independently to folds 1-4. Fold identity
is embedded in each TSV, sidecar, Ridge lock, and test receipt.

BF16 is fixed for the frozen arm so that numerical precision matches the
adapted arm. The authoritative frozen outputs are the `bfloat16_v3` embedding,
Ridge, internal-test, and ON-Harmony prediction trees. The former FP32 outputs
and the BF16 v2 validation checkpoint are historical audit artifacts and must
not be mixed with the v3 chain.
Current Figures 2--4 consume the ON-Harmony BF16 embeddings and the frozen BF16
v3 internal-test/ON-Harmony predictions; they do not read the historical FP32
frozen directories for AnatCL.

The internal test was re-extracted only after all five v3 Ridge heads had been
selected using train and validation. The fixed revision identifier
`anatcl-frozen-bfloat16-alignment-v1` requires BF16, the canonical v3 output,
the corresponding validation-only `LOCKED.json`, and an intact original FP32
receipt. It writes a separate revision receipt and never modifies the original
receipt. For one fold, the authorized command is:

```bash
python FineTuning/AnatCL/extract_embeddings.py extract \
  --fold 0 --device cuda --amp bfloat16 --seed 42 \
  --split internal_test \
  --output Results/brain_age/embeddings_FT_bfloat16_v3/AnatCL/fold0/internal_test.tsv \
  --acknowledge-locked-test \
  --internal-test-revision anatcl-frozen-bfloat16-alignment-v1 \
  --internal-test-revision-lock \
    Results/brain_age/frozen_bfloat16_v3/AnatCL/fold0/LOCKED.json
```

The matching ON-Harmony embeddings are generated with the same fixed
precision. First reconstruct and validate the retained per-acquisition CAT12
evidence; the current Python preprocessor is recorded only as a reference
implementation because the historical wrapper source is no longer available:

```bash
python preprocessings/OnHarmonyDataset/audit_AnatCL_provenance.py \
  --project-root "$PWD" \
  --workers 16 \
  --hash-raw \
  --output-dir ExcelsChecks/paper_actual/audits/AnatCL
```

Then extract the five fold-specific embedding sets:

```bash
python Emb_extractors/extract_AnatCL.py \
  --input_list DATA/filelists/filelist_AnatCL.txt \
  --checkpoint_dir Models/AnatCL/NEW_anatcl-g3 \
  --participants DATA/participants.tsv \
  --output_dir DATA/embeddings_bfloat16_v2 \
  --output_stem embeddings_AnatCL \
  --preprocessing_provenance \
    ExcelsChecks/paper_actual/audits/AnatCL/anatcl_preprocessing_provenance.json \
  --device cuda --amp bfloat16 --seed 42
```

The five locked fold-specific Ridge heads are each applied label-blind to the
matching ON-Harmony fold embedding CSV by the common launcher. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for the 11 frozen
representations, firewall, and output convention.

## Controlled adaptation: exactly nine runs

The nine unique training runs are defined before internal-test or ON-Harmony
evaluation:

| Stage | Fold | Seeds | LR configurations | Runs |
|---|---:|---|---|---:|
| LR selection | 0 | 0 | conservative, intermediate, high | 3 |
| Seed repetitions | 0 | 1, 2 | selected LR only | 2 |
| Fold sensitivity | 1, 2, 3, 4 | 0 | selected LR only | 4 |
| **Total** | | | | **9** |

One selection candidate is launched as follows:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python FineTuning/AnatCL/finetune.py train \
  --fold 0 \
  --checkpoint-dir Models/AnatCL/NEW_anatcl-g3 \
  --preproc-root preprocessings/FineTunningDataset/preproc_AnatCL \
  --device cuda --amp bfloat16 --seed 0 \
  --lr-config conservative \
  --output-dir Results/brain_age/adapted/AnatCL/fold0_seed0_conservative
```

After all three fold-0/seed-0 candidates finish:

```bash
python FineTuning/AnatCL/finetune.py select \
  --fold 0 \
  --run-dirs \
    Results/brain_age/adapted/AnatCL/fold0_seed0_conservative \
    Results/brain_age/adapted/AnatCL/fold0_seed0_intermediate \
    Results/brain_age/adapted/AnatCL/fold0_seed0_high \
  --output-dir Results/brain_age/adapted/AnatCL/fold0_lr_selection
```

Read `SELECTED_LR.json`, then run only that LR for fold 0 seeds 1 and 2 and
for folds 1-4 seed 0. The primary adapted summary gives equal weight to folds
0-4 at seed 0. Fold 0 seeds 0/1/2 provide a separate initialization
sensitivity; the two sources of variability must not be pooled.
Every adapted internal-test evaluation, including folds 1-4, must pass
`--selection-dir Results/brain_age/adapted/AnatCL/fold0_lr_selection`; the
result records whether the fold-0 LR was transferred across folds.

No training augmentations are applied. Every model uses the same train and
validation subjects, standardized-age MSE, validation-MAE selection rule,
30-epoch ceiling, early stopping, optimizer family, and gradient-accumulation
contract as the other controlled-adaptation workflows.

## Internal-test gate

`CamCAN/sub-CC620821` was removed from the definitive common cohort after its
CAT12 affine/Dartel registration failed on two independent attempts. The
candidate-cohort exclusion ledger retains the case and reason, while every
model consumes the same 1,411-subject manifest (`1130/141/140`). Train and
validation are unchanged; no missing input is skipped implicitly.

## Adapted ON-Harmony inference

`predict_onharmony.py` restores one locked fold/seed checkpoint and validates
the shared CAT12 input, historical MATLAB batch, CAT12 runtime log, and wrapper
success log for every acquisition. The current Python preprocessor is recorded
as a reference implementation, not misidentified as the deleted historical
generator. Run it for the five folds at seed 0 and separately for fold 0 seeds
1 and 2; do not pool fold and seed sensitivity. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for the seven retained
runs, audit command, and label-blind output contract.

## Tests

```bash
MPLCONFIGDIR=/tmp/matplotlib-finetuning \
python -m unittest discover -s FineTuning/AnatCL/tests -v
```
