# Cross-Scanner Reliability of Brain MRI Foundation Models

Source code for **Cross-Scanner Reliability of Brain MRI Foundation Models:
A Travelling-Heads Study**.

## Layout

- `data/`: derived CSVs organised by dataset; sources and attribution are in
  `data/sources.csv`. The compact layout is described below.
- `preprocessings/`: model-specific preprocessing and visual QC.
- `Emb_extractors/`: ON-Harmony frozen extraction, including the separate
  BrainSegFounder BF16 extractor used for precision alignment.
- `FineTuning/`: seven model adapters, Ridge, fine-tuning, evaluation and guards.
- `Analyses/`: current representation, prediction and adaptation analyses.
- `Figures/`: main and supplementary
  figure generators; analysis artifacts must be generated separately.
- `Tables/`: model descriptions and dbICC scaling
  sensitivity. The manuscript's comparative results table is not generated here.
- `Supplementary/`: AnatCL-SimCLR extractor and the BrainSegFounder selection
  migration helper.

Models: y-Aware, AnatCL, 3D-Neuro-SimCLR, BrainIAC, BrainSegFounder, NeuroVFM,
and Curia-2. Model READMEs describe training interfaces. Historical execution
counts and paths in those documents are study records, not files in this clone.

## Statistical workflow

Installation commands and model-environment requirements are in
[`INSTALLATION.md`](INSTALLATION.md). The exact development splits and
ON-Harmony acquisition selection are in
[`data/SELECTIONS.md`](data/SELECTIONS.md).

The cohort contains 1,130 train, 141 validation and 140 internal-test subjects.
Selection uses validation only. ON-Harmony inference produces 165 acquisitions;
primary analyses retain the first acquisition in each of 120 participant-scanner
cells (20 participants, six scanners each). Repeats are **not averaged**.

1. Generate/audit model inputs and frozen embeddings.
2. Fit standardizers and Ridge with train; select alpha with validation.
3. Select adaptation LR with seed 0 and validation; evaluate retained runs.

Primary analyses use one pretrained checkpoint per model. AnatCL uses fold 0,
with three fine-tuning seeds as for the other models. Its remaining pretrained
folds are retained for sensitivity analyses. The AnatCL-SimCLR primary contrast
uses fold 0 of both encoders; all five paired folds are also reported separately.
   ON-Harmony must not influence training or selection.
4. Run `Analyses/figure2_reliability.py`, `figure3_brain_age.py`, then
   `figure4_adaptation.py` once the required artifacts are available.
5. Run corresponding figure generators; rendering does not train models.

Figure 2 uses native squared-distance dbICC and crossed REML variance components.
Figure 3 separates test MAE, predicted-age dbICC and absolute cross-scanner change.
Figure 4 uses paired gains and global BH correction over seven models and three
endpoints. Metrics are averaged across retained runs, not predictions.

In both prediction figures, scanner shifts are adjusted for participant and
reported relative to NOT1 (Philips Achieva), which was used in both ON-Harmony
phases. Positive values mean older predictions than NOT1, not overestimation
relative to chronological age. NOT1 is zero by definition, not a gold standard.
Scanner contrasts are recalculated within every participant-bootstrap replicate.

## Result data

The deposit contains 22 data CSVs, plus inventories and dictionaries:

- Each development dataset has one `predictions.csv`, with model, run, fold,
  seed and split recorded in columns. Internal-test predictions are retained
  for every model and run; CamCAN also retains validation predictions.
- `data/ON-Harmony/` contains nine embedding CSVs (seven FMs, AnatCL-SimCLR
  and FreeSurfer) and one prediction CSV. All 165 acquisitions are retained;
  `primary_acquisition` identifies the 120 used in primary analyses.
- `data/CamCAN/embeddings/` retains seven CSVs containing all frozen features
  from train, validation and internal test as part of the CamCAN derivative
  deposit. Folds remain separate rows, not averaged vectors.

True ages are not redistributed. MAE, R-squared and age-scatter plots require
demographics obtained from the original datasets and joined by dataset and
participant ID. Cross-scanner dbICC and mean absolute pairwise differences
can be calculated without ages. Seeds and folds must be analysed separately
before averaging their metrics; predictions are not ensembled.

Inventories map each original export to its compact file; `rows` refers to
that original export, not the whole compact file.
`Analyses/reproduce_published.py` connects these CSVs directly to the existing
statistical functions. See [`REPRODUCING_RESULTS.md`](REPRODUCING_RESULTS.md)
for cross-scanner calculations without ages, optional local age input for
internal-test accuracy, REML and figure-generation commands. The original
analysis entry points remain available for the audited study bundles.
`tools/compact_derivatives.py` creates the compact layout without changing
retained numeric values.

## Environment

Install the CPU analysis dependencies with `python -m pip install -r requirements.txt`.
See [`INSTALLATION.md`](INSTALLATION.md) for R packages and separate GPU setup.

GPU execution needs model-specific dependencies and official checkpoint/code
revisions checked by each `FineTuning/<model>/model.py`. NeuroVFM requires its
dedicated upstream environment. REML requires R, `lme4` and `jsonlite`.
Preprocessing needs external tools according to model: CAT12/SPM with MATLAB,
FSL, ANTs, SynthStrip/SynthSeg or HD-BET. Tools, templates and weights are not bundled.

## Data sources

Data sources are ON-Harmony (`ds004712`, version 2.0.1), NeuroCog (`ds003592`,
with demographics on OSF `yhzxe`), NIMH-RV (`ds004215`), MR-ART (`ds004173`),
[WAND](https://doi.org/10.12751/g-node.5mv3bf), and
[CamCAN](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/).
The four OpenNeuro datasets are distributed under CC0; WAND uses CC BY 4.0.
CamCAN requires approved access and its own data-use agreement.

## License

The source code is licensed under Apache-2.0. Dataset-specific terms and
attribution for `data/` are listed in `data/sources.csv`; CamCAN retains its
[data-use terms](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/).
External implementations, weights, software and datasets retain their own
licenses and access terms.
