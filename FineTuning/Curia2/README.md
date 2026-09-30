# Curia-2 brain-age workflow

Curia-2 has three explicit entry points:

- `extract_embeddings.py`: audit the release/input contract and extract frozen
  volume embeddings.
- `fit_ridge.py`: select and lock the common StandardScaler + Ridge head using
  train and validation embeddings.
- `finetune.py`: train the final DINOv2 block, final layer normalization, and a
  scalar age head; select among the three predeclared learning-rate pairs; and
  perform the separately unlocked internal-test evaluation.

`model.py` is not launched directly. It independently pins and loads the public
Curia-2 release, reads raw T1w NIfTI files, converts LAS axial slices to PL
orientation, calls the released processor through its float-tensor branch,
averages the final-layer CLS tokens from every axial slice, and declares the
controlled trainable boundary. It does not import the ON-Harmony reliability
extractor.

## Audit and smoke test

Run this before any extraction or training:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python FineTuning/Curia2/extract_embeddings.py audit \
  --model-dir /datos/work/rnavgon/FoundationModels_Paper/Models/Curia-2 \
  --split train \
  --check-input-files \
  --inspect-model \
  --smoke-forward \
  --device cuda \
  --amp bfloat16 \
  --seed 0 \
  --slice-batch-size 8
```

## Frozen arm

```bash
python FineTuning/Curia2/extract_embeddings.py extract \
  --model-dir /datos/work/rnavgon/FoundationModels_Paper/Models/Curia-2 \
  --split train \
  --device cuda \
  --amp off \
  --seed 42 \
  --slice-batch-size 8 \
  --output Results/brain_age/embeddings_FT/Curia2/canonical/train.tsv

python FineTuning/Curia2/extract_embeddings.py extract \
  --model-dir /datos/work/rnavgon/FoundationModels_Paper/Models/Curia-2 \
  --split validation \
  --device cuda \
  --amp off \
  --seed 42 \
  --slice-batch-size 8 \
  --output Results/brain_age/embeddings_FT/Curia2/canonical/validation.tsv

python FineTuning/Curia2/fit_ridge.py fit \
  --train-embeddings Results/brain_age/embeddings_FT/Curia2/canonical/train.tsv \
  --validation-embeddings Results/brain_age/embeddings_FT/Curia2/canonical/validation.tsv \
  --output-dir Results/brain_age/frozen/Curia2/canonical
```

Frozen extraction uses every axial slice in float32 to match the reliability
extractor's numerical precision. The previous embeddings and Ridge artifacts
used a study-defined 5% nonzero-content filter; they are incompatible with this
contract and must be regenerated.

Frozen ON-Harmony prediction uses only the canonical `cls_all_` stream and the
locked validation-selected Ridge head. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for the label firewall
and common launcher.

## Controlled adaptation

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python FineTuning/Curia2/finetune.py train \
  --model-dir /datos/work/rnavgon/FoundationModels_Paper/Models/Curia-2 \
  --device cuda \
  --amp bfloat16 \
  --seed 0 \
  --lr-config conservative \
  --output-dir Results/brain_age/adapted/Curia2/selection_seed0_conservative
```

Training, validation, and frozen inference all use every axial slice.

Run the conservative, intermediate, and high configurations with seed 0 and
select their learning rate using validation MAE. Then repeat only the selected
configuration with seeds 1 and 2. The three retained seeds are repetitions; do
not choose among them using internal-test or ON-Harmony performance.

The definitive common manifest contains 1,411 subjects (`1130/141/140`). All
1,411 raw inputs were audited as exactly 3D; the loader rejects any future
non-3D input instead of silently selecting an axis.

Internal-test embeddings may be materialized in advance, but their predictions
and metrics must not be inspected until every learning-rate and seed decision
has been frozen. ON-Harmony remains reserved for final external evaluation.

## Adapted ON-Harmony inference

`predict_onharmony.py` restores the locked final block, normalization, and age
head and generates label-blind predictions for the canonical 165 acquisitions.
It does not apply the training-cohort loader or access chronological age. See
[`../ONHARMONY_INFERENCE.md`](../ONHARMONY_INFERENCE.md) for the retained runs,
audit command, output contract, and sequencing rules.
