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
- `Tables/`: generators for the model descriptions, sensitivity analyses and
  main comparison of frozen and fine-tuned predictions (Table 4).
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
4. Run the embedding, frozen prediction and adaptation analyses once the required
   inputs are available. ON-Harmony is not used for training or selection.
5. Generate figures and tables from the analysis outputs; rendering does not train models.

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

The published CSVs can be used to reproduce the cross-scanner results.
Reproducing internal-test accuracy also requires chronological ages from the
original datasets. See [Reproducing the results](REPRODUCING_RESULTS.md) for commands.

## Environment

Install the CPU analysis dependencies with `python -m pip install -r requirements.txt`.
See [`INSTALLATION.md`](INSTALLATION.md) for R packages and separate GPU setup.

GPU execution needs model-specific dependencies and official checkpoint/code
revisions checked by each `FineTuning/<model>/model.py`. NeuroVFM requires its
dedicated upstream environment. REML requires R, `lme4` and `jsonlite`.
Preprocessing needs external tools according to model: CAT12/SPM with MATLAB,
FSL, ANTs, SynthStrip/SynthSeg or HD-BET. Tools, templates and weights are not bundled.

## Data sources

The travelling-heads dataset was [ON-Harmony (OpenNeuro ds004712, version 2.0.1)](https://openneuro.org/datasets/ds004712/versions/2.0.1).
The brain-age development cohort combined five datasets:

- [CamCAN](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/)
- [NeuroCog (OpenNeuro ds003592)](https://openneuro.org/datasets/ds003592), with [accompanying demographic and behavioural data on OSF](https://osf.io/yhzxe/)
- [WAND (G-Node)](https://doi.org/10.12751/g-node.5mv3bf)
- [NIMH-RV (OpenNeuro ds004215)](https://openneuro.org/datasets/ds004215)
- [MR-ART (OpenNeuro ds004173)](https://openneuro.org/datasets/ds004173)

The four OpenNeuro datasets are distributed under CC0; WAND uses CC BY 4.0.
CamCAN requires approved access and its own data-use agreement.

## License

The source code is licensed under Apache-2.0. Dataset-specific terms and
attribution for `data/` are listed in `data/sources.csv`; CamCAN retains its
[data-use terms](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/).
External implementations, weights, software and datasets retain their own
licenses and access terms.
