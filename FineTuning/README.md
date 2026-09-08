# Controlled brain-age adaptation

This directory keeps model-specific workflows separate while sharing only the
cohort and statistical rules that must remain identical across models.

`shared/` contains the common cohort, statistical, runtime, and provenance
guards. The model-specific numerical implementations remain in their model
directories. `shared/onharmony.py` contains the single hash-pinned registry for
post-training runtime migrations, and `shared/internal_test.py` delegates to
that same registry. All seven model families use it for both adapted
ON-Harmony inference and internal-test evaluation.

Historical `LOCKED.json` files and checkpoints are never rewritten. Each
accepted migration specifies the exact old and new SHA-256 values for every
changed dependency, the exact hashes of any added dependency, a stable
migration identifier, and its reason. Unregistered changes, removed files,
unknown historical hashes, or edits to a pinned migration target abort before
inference. New prediction and internal-test provenance records preserve both
the locked training map and the current evaluation map. BrainIAC's separately
pinned seed-0 selection-anchor transition remains distinct from these
checkpoint-to-current-runtime migrations.

Model-specific end-to-end workflows are present for Curia-2, NeuroVFM,
3D-Neuro-SimCLR, AnatCL, y-Aware, BrainIAC, and BrainSegFounder. Each directory
contains its own model contract, frozen embedding extractor, Ridge entry point,
controlled-adaptation entry point, README, and tests. Runtime readiness remains
model-specific: an implementation is not promoted to a full execution until
its checkpoint audit and differentiability smoke test pass in the declared
cluster environment.

The development manifest contains train, validation, and an internal test.
Frozen encoders may materialize embeddings for all three splits because that
operation is label-free and does not fit parameters. Ridge selection remains
strictly train/validation-only, and internal-test metrics remain explicitly
locked. ON-Harmony is rejected by every training, selection, embedding, and
internal-test entry point. It is accepted only by each model's dedicated,
label-blind `predict_onharmony.py` interface.

Frozen embedding artifacts use the common layout
`Results/brain_age/embeddings_FT/<model>/<variant>/<split>.tsv`, with an adjacent
`.audit.json` sidecar. The three materialized splits are `train`, `validation`,
and `internal_test`; only the first two may be passed to Ridge fitting or model
selection.

## Full regression suite

Run the suites explicitly; plain `unittest discover` does not descend into all
model test directories and therefore reports only a partial run:

```bash
MPLCONFIGDIR=/tmp/mpl-finetuning-tests python -m unittest -v \
  FineTuning.3DNeuroSimCLR.tests.test_3dneurosimclr \
  FineTuning.AnatCL.tests.test_anatcl \
  FineTuning.BrainIAC.tests.test_brainiac \
  FineTuning.BrainSegFounder.tests.test_brainsegfounder \
  FineTuning.Curia2.tests.test_curia2 \
  FineTuning.NeuroVFM.tests.test_neurovfm \
  FineTuning.yAware.tests.test_yaware \
  FineTuning.shared.tests.test_embedding_bundle \
  FineTuning.shared.tests.test_internal_test_receipt \
  FineTuning.shared.tests.test_onharmony
```

The audited baseline is 289 passing tests.

## Locked internal-test evaluation

`launch_internal_test_adapted.sh` is the single entry point for opening the
internal test after LR selection and label-blind ON-Harmony inference are
complete. Its preflight validates all 25 retained checkpoints: three seeds for
each model, plus the four additional seed-0 AnatCL folds. It refuses partial
outputs and records one immutable receipt beside every adapted checkpoint.
For all seven model families, the launcher invokes the same model-local command
shape: `finetune.py evaluate`. External ON-Harmony inference remains separate
in `predict_onharmony.py` and is never routed through `finetune.py`.

Run the non-destructive audit first:

```bash
PREFLIGHT_ONLY=1 ./FineTuning/launch_internal_test_adapted.sh 0
```

The actual one-time evaluation requires an explicit acknowledgement:

```bash
ACKNOWLEDGE_LOCKED_TEST=YES ./FineTuning/launch_internal_test_adapted.sh 0
```

Per-run predictions and metrics are written under
`Results/brain_age/internal_test/adapted/<model>/<run>/`. The completed launcher
also publishes the immutable `internal_test_metrics_all_runs.tsv` and its
`internal_test_metrics_all_runs.audit.json` provenance manifest. The audit binds
the exact 25-target matrix to the cohort manifest, launcher, per-run metrics,
predictions, validation-only LR selection, and retained checkpoint hashes. A
later run verifies byte-identical artifacts and refuses to overwrite drifted
ones. Seeds and folds are not averaged at this stage.
