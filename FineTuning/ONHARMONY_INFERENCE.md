# ON-Harmony inference for frozen and adapted brain-age models

The seven model-specific `predict_onharmony.py` entry points apply the locked
controlled-adaptation checkpoints to all 165 ON-Harmony acquisitions. This is
an external, label-blind inference step: the loader reads acquisition and
scanner identifiers but never age or prediction-error columns.

All seven entry points delegate the complete inference lifecycle to
`FineTuning/shared/onharmony.py`: canonical raw manifest loading, immutable
lock/checkpoint/LR-selection binding, model restoration, ordered prediction,
and atomic publication. Model-specific files define only their input-artifact
resolver and validator, release adapter, and forward pass. The common tests
fail if any model reintroduces a private copy of the lifecycle.

The frozen arm follows the same firewall. Its model-specific `fit_ridge.py`
entry points fit `StandardScaler + Ridge` using only the brain-age train and
validation embeddings, lock the selected alpha, and then apply that immutable
linear model to the frozen ON-Harmony embeddings.

## Frozen Ridge representations

The frozen analysis produces 11 prediction artifacts: one for each of
3D-Neuro-SimCLR, BrainIAC, BrainSegFounder, Curia-2, NeuroVFM, and y-Aware,
plus one for each of the five released AnatCL folds. Curia-2 uses its canonical
`cls_all_` stream (mean CLS over every axial slice); the other models use their
canonical `emb_` stream.

Some historical ON-Harmony embedding CSVs contain an `age` column. The frozen
prediction loader inspects the header but uses column-restricted loading for
only `file_path` and the selected embedding stream, so no outcome value enters
the resulting table or estimator. It verifies the canonical 165-acquisition
set and order, dimensionality, finite values, the locked Ridge artifact and
code hashes, and the mandatory extractor manifest paired with the CSV. Each
output contains predicted age but no chronological age, error, or performance
metric:

```text
Results/brain_age/onharmony/frozen/<Model>/<variant>/
```

Fit all 11 train/validation Ridge heads and create the corresponding
label-blind ON-Harmony predictions, then evaluate all 11 internal-test
representations with:

```bash
cd /datos/work/rnavgon/FoundationModels_OH_T1w
bash FineTuning/launch_frozen_ridge_onharmony.sh
```

The launcher uses the `myLocalEnv` Python for six model families and the
dedicated `neurovfm` Python for NeuroVFM. It first locks every Ridge model,
then freezes every ON-Harmony prediction, and only after both phases have
finished invokes the existing `evaluate --acknowledge-locked-test` command.
Internal-test predictions and uncorrected metrics are written separately:

```text
Results/brain_age/internal_test/frozen/<Model>/<variant>/
```

Each Ridge run receives an immutable `INTERNAL_TEST_EVALUATED.json` receipt.
Before fitting anything, a global preflight requires the 11 complete
train/validation/internal-test embedding bundles, all 11 ON-Harmony embedding
CSVs and extractor manifests, unique targets, and no pre-existing Ridge,
ON-Harmony, internal-test, or receipt output. Outputs are never skipped, reused,
or overwritten. For every representation it also invokes the model-specific
`audit-onharmony` command, which applies the same label-blind loader used for
prediction and verifies the manifest schema and semantic identity, CSV and
extractor hashes, feature stream, dimensionality, finite values, canonical
165-acquisition set, and row order. The launcher passes the definitive cohort
manifest and the ON-Harmony filelist, metadata, and extractor manifest paths
explicitly to every relevant phase; it does not depend on entry-point defaults.

## Retained adapted models

Run the selected learning-rate configuration for seeds 0, 1, and 2 for
Curia-2, NeuroVFM, BrainIAC, BrainSegFounder, y-Aware, and 3D-Neuro-SimCLR.
For AnatCL, additionally run folds 1-4 at seed 0. This gives 25 prediction
artifacts across all seven families:

| Model | Adapted runs evaluated on ON-Harmony |
|---|---|
| Curia-2 | `selection_seed0_high`, `final_seed1_high`, `final_seed2_high` |
| NeuroVFM | `selection_seed0_high`, `final_seed1_high`, `final_seed2_high` |
| BrainIAC | `selection_seed0_high`, `final_seed1_high`, `final_seed2_high` |
| BrainSegFounder | `selection_seed0_high`, `final_seed1_high`, `final_seed2_high` |
| y-Aware | `selection_seed0_high`, `final_seed1_high`, `final_seed2_high` |
| 3D-Neuro-SimCLR | `selection_seed0_intermediate`, `final_seed1_intermediate`, `final_seed2_intermediate` |
| AnatCL | `fold0_seed0_high`, `fold0_seed1_high`, `fold0_seed2_high`, and `fold1_seed0_high` through `fold4_seed0_high` |

Rejected seed-0 learning-rate candidates and the two BrainIAC learning-rate
sensitivity runs are not part of the final external evaluation.

## Required sequence

1. Finish and validate each model's ON-Harmony preprocessing.
2. For NeuroVFM, materialize its released `StudyPreprocessor` cache with the
   model-specific `cache` command.
3. Run `audit --check-input-files --inspect-model --smoke-forward` on exactly
   one acquisition for each model.
4. Run `predict` once for every retained adapted checkpoint.
5. After all 25 ON-Harmony prediction bundles are frozen, run the adapted
   internal-test launcher. Its global preflight validates every training lock,
   checkpoint, validation-only LR selection and ON-Harmony prerequisite before
   any test evaluation. A completed evaluation is resumed only when its metrics
   and receipt remain bound to the expected model, variant, seed, checkpoint,
   cohort manifest and LR-selection artifact. AnatCL is the sole declared
   cross-variant LR transfer: fold 0 selects the LR used by folds 1-4.

Every command verifies the adapted checkpoint, immutable training lock,
validation-only LR selection artifact, runtime options, model boundary, input
provenance, and canonical 165-acquisition order. Output directories must not
already exist; the launcher never skips or reuses a previous prediction bundle.
A successful run atomically publishes:

- `predictions.tsv`: identifiers, scanner descriptors, predicted age, training
  seed/LR/checkpoint identity, and input provenance; no chronological age;
- `prediction_audit.json`: hashes and full model, input, runtime, and selection
  provenance.

Use this output convention:

```text
Results/brain_age/onharmony/adapted/<Model>/<adapted-run>/
```

## Example

The command shape is the same for all models except NeuroVFM's preceding
cache step. For example, Curia-2 seed 0 is audited and then evaluated with:

```bash
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/Curia2/predict_onharmony.py audit \
  --run-dir Results/brain_age/adapted/Curia2/selection_seed0_high \
  --selection-dir Results/brain_age/adapted/Curia2/lr_selection_seed0 \
  --subject sub-03286 --session NOT1ACH001 \
  --check-input-files --inspect-model --smoke-forward \
  --device cuda --amp bfloat16

CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u FineTuning/Curia2/predict_onharmony.py predict \
  --run-dir Results/brain_age/adapted/Curia2/selection_seed0_high \
  --selection-dir Results/brain_age/adapted/Curia2/lr_selection_seed0 \
  --device cuda --amp bfloat16 \
  --output-dir Results/brain_age/onharmony/adapted/Curia2/selection_seed0_high
```

Do not calculate MAE, bias correction, repeatability, or reproducibility until
all model predictions are frozen. Those analyses belong to a separate step
that can then join the predictions to the outcome data.

## Sequential seven-family launcher

All seven families can be audited and evaluated sequentially with
`launch_onharmony_adapted.sh`. It runs the 25 retained checkpoints and builds
the NeuroVFM cache before its three inference runs. Before starting any model,
its global preflight requires all 25 checkpoints and immutable training locks,
all seven validation-only LR selections, all prediction entry points, 25 unique
targets, and no pre-existing output directory.
