# Cross-Scanner Reliability of Brain MRI Foundation Models

Source code for **Cross-Scanner Reliability of Brain MRI Foundation Models:
A Travelling-Heads Study**.

## Publication candidate

This branch contains the reviewed implementation of the seven-model benchmark
and its brain-age extension. It is a source-code release, not a distribution of
participant data, pretrained weights, or a turnkey reproduction environment.
The earlier implementation remains under the
[`legacy-original-code`](https://github.com/rafaloz/Reliability_FMs/tree/legacy-original-code)
tag for transparency; it is not the current analysis.

Scientific sources are copied without rewriting their contents.
`source_manifest.json` records their SHA-256 values, preserving code identities
used by the study's audit contracts. Study-specific path defaults and historical
migration records remain in the code. Supply explicit paths through command-line
options where supported; check paths and environments before using any launcher.

## Layout

- `preprocessings/`: model-specific preprocessing and visual QC.
- `Emb_extractors/`: ON-Harmony frozen extraction, including the separate
  BrainSegFounder BF16 extractor used for precision alignment.
- `FineTuning/`: seven model adapters, Ridge, fine-tuning, evaluation and guards.
- `Analyses/`: current representation, prediction and adaptation analyses/tests.
- `Paper/ImagingNeuroscience/Figures/FiguresScripts/`: main and supplementary
  figure generators; analysis artifacts must be generated separately.
- `Paper/ImagingNeuroscience/Tables/`: model descriptions and dbICC scaling
  sensitivity. The manuscript's comparative results table is not generated here.
- `Supplementary/`: AnatCL-SimCLR extractor and the BrainSegFounder selection
  migration helper required by the model tests.
- `ExcelsChecks/`: source-only BrainIAC precision audit imported by its tests;
  no participant spreadsheets or audit data are included.

Models: y-Aware, AnatCL, 3D-Neuro-SimCLR, BrainIAC, BrainSegFounder, NeuroVFM,
and Curia-2. Model READMEs describe training interfaces. Historical execution
counts and paths in those documents are study records, not files in this clone.

## Statistical workflow

The cohort contains 1,130 train, 141 validation and 140 internal-test subjects.
Selection uses validation only. ON-Harmony inference produces 165 acquisitions;
primary analyses retain the first acquisition in each of 120 participant-scanner
cells (20 participants, six scanners each). Repeats are **not averaged**.

1. Generate/audit model inputs and frozen embeddings.
2. Fit standardizers and Ridge with train; select alpha with validation.
3. Select adaptation LR with seed 0 and validation; evaluate retained runs.
   ON-Harmony must not influence training or selection.
4. Run `Analyses/figure2_reliability.py`, `figure3_brain_age.py`, then
   `figure4_adaptation.py` once the required artifacts are available.
5. Run corresponding figure generators; rendering does not train models.

Figure 2 uses native squared-distance dbICC and crossed REML variance components.
Figure 3 separates test MAE, predicted-age dbICC and absolute cross-scanner change.
Figure 4 uses paired gains and global BH correction over seven models and three
endpoints. Metrics are averaged across retained runs, not predictions.

## Environment and tests

`requirements-tests.txt` pins local Python versions for CPU tests, not the GPU
training environment. Install in a dedicated environment:

```bash
python -m pip install -r requirements-tests.txt
python -m pytest Analyses/tests FineTuning/shared/tests -q
```

Checked from this data-free source copy on 2026-09-08: **142 tests and 279
subtests passed**, using local Conda Python 3.10 and PyTorch 2.5.1 CPU.
This was a run in the existing environment, not a fresh-install verification
of `requirements-tests.txt`. Only dependency deprecation warnings were emitted.

GPU execution needs model-specific dependencies and official checkpoint/code
revisions checked by each `FineTuning/<model>/model.py`. NeuroVFM requires its
dedicated upstream environment. REML requires R, `lme4` and `jsonlite`.
Preprocessing needs external tools according to model: CAT12/SPM with MATLAB,
FSL, ANTs, SynthStrip/SynthSeg or HD-BET. Tools, templates and weights are not bundled.

Some model tests intentionally read the definitive study manifest or templates:
they cannot all pass in a data-free clone. Synthetic tests do not rerun MRI
preprocessing or establish actual checkpoint loading.

## Data and reproducibility limits

MRI, demographics, participant manifests, embeddings, individual predictions,
training artifacts, QC exports and manuscript PDFs are excluded. Obtain data
and model releases from their owners under their access conditions and licenses.

Validators pin the study cohort and artifact hashes. Arbitrary replacement
manifests are rejected. A different cohort requires a separately versioned
contract, not disabling integrity checks. Exact paper reproduction requires
input/provenance bundles not released with this source-only candidate.

## License

The repository retains Apache-2.0. External implementations, weights, software
and datasets retain their own licenses and access terms; they are not
relicensed or redistributed here.
