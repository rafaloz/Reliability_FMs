# Recalculate the published results

`Analyses/reproduce_published.py` reads the compact CSVs in `data/` and calls
the statistical functions in the existing analysis scripts. It does not
require MRI images, model weights, GPU inference or the internal training
bundles. It does not retrain or select any model.

Install the CPU dependencies using [`INSTALLATION.md`](INSTALLATION.md).
Run the commands below from the repository root.

## Cross-scanner results without demographics

```bash
python Analyses/reproduce_published.py \
  --data-root data \
  --output-root Results/published_onharmony
```

This calculates full-embedding dbICC, predicted-age dbICC, mean absolute
pairwise cross-scanner differences, and participant-adjusted scanner shifts
relative to NOT1. It also calculates the seven paired embedding-to-prediction
contrasts and their BH-adjusted p-values.

Internal-test accuracy is unavailable without true ages. The global BH
adjustment across 21 fine-tuning contrasts is therefore left unavailable,
rather than replaced by an adjustment across the 14 available contrasts.
The separate seven-model adjustments within the two cross-scanner endpoints
are still reported.

## Add internal-test accuracy

Obtain the chronological ages from the original dataset sources linked in
[`data/sources.csv`](data/sources.csv). Prepare a **local, private CSV** with
these columns:

```csv
dataset,subject,age
```

Use the dataset names and original subject IDs in `data/cohort_splits.csv`.
The file must contain all 140 `internal_test` subjects, each exactly once.
Rows for the other selected development subjects are allowed but not used.
Duplicate IDs within a dataset, missing test subjects, nonfinite values and
ages outside the study's 18-88-year range are rejected. Sex and ON-Harmony
ages are not needed for these calculations.

```bash
python Analyses/reproduce_published.py \
  --data-root data \
  --ages /private/path/internal_test_ages.csv \
  --output-root Results/published
```

This additionally calculates internal-test MAE and R-squared with bootstrap
intervals, paired MAE gains, and the global BH adjustment across seven models
and three endpoints. Ages are joined by **dataset and subject**, never by row
position. They are not copied into the output files.

## Include variance decomposition

Install R, `lme4` and `jsonlite` as described in the installation guide, then
add `--reml` to either command. For example:

```bash
python Analyses/reproduce_published.py \
  --data-root data \
  --ages /private/path/internal_test_ages.csv \
  --reml --workers 2 \
  --output-root Results/published_with_reml
```

This runs the existing `Analyses/figure2_variance_components.R` on each
representation/fold. It writes the R inputs and results under the output
directory. Variance decomposition is omitted unless `--reml` is requested.
The FreeSurfer-ComBat sensitivity is not included in the compact deposit.

## Outputs and figures

The output directory must be new. Existing results are never overwritten.

| Output | Contents |
| --- | --- |
| `figure2_reliability_results.json` | Nine representation conditions, fold-level dbICC results and the AnatCL-SimCLR comparison. Variance components are included with `--reml`. |
| `figure3_brain_age_results.json` | Frozen prediction metrics, embedding-to-prediction contrasts and scanner shifts. Accuracy fields are unavailable without `--ages`. |
| `figure4_adaptation_results.json` | Per-run results, primary metric averages, paired gains and multiplicity adjustments. |
| `metrics.csv` | Tidy summary of estimates, intervals and available p/q values. |
| `reproduction_audit.json` | Successful-completion marker, input/code/output hashes and bootstrap settings. |

Without a `PASS` audit, the output directory is incomplete. The audit checks
that inputs and statistical source files did not change during calculation.
Hashes verify identity, not the correctness of ages obtained from an external
source.

The primary calculation uses 1,000 participant-level bootstrap replicates
with the original analysis seeds. Each seed/fold is evaluated separately;
metrics, not predictions, are averaged. The two additional AnatCL fold-0
seeds remain separate sensitivity runs. The 120 primary ON-Harmony scans are
selected before computing cross-scanner metrics.

With ages and REML included, the JSONs can be passed directly to the existing
figure generators. These script names refer to analysis identifiers:

```bash
python Figures/figure2_reliability.py \
  --results Results/published_with_reml/figure2_reliability_results.json \
  --output-dir Results/published_with_reml/figures \
  --manifest Results/published_with_reml/figure2_figure_manifest.json

python Figures/figure3_brain_age.py \
  --results Results/published_with_reml/figure3_brain_age_results.json \
  --output-dir Results/published_with_reml/figures \
  --manifest Results/published_with_reml/figure3_figure_manifest.json

python Figures/figure4_adaptation.py \
  --results Results/published_with_reml/figure4_adaptation_results.json \
  --output-dir Results/published_with_reml/figures \
  --manifest Results/published_with_reml/figure4_figure_manifest.json
```

The two prediction figures do not require REML, but do require ages. The
representation figure requires REML. The renderers expect the primary
1,000-replicate results; a reduced `--n-bootstrap` value is for smoke checks,
not for reproducing the reported figures.
