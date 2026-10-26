# Installation

Run commands from the repository root. Use separate environments for statistical
analysis and GPU model execution.

## Statistical analyses and figures

The Python dependencies in `requirements.txt` match the local Python 3.10
environment used to check the analysis and rendering code. They are not a
lockfile for GPU training or MRI preprocessing.

```bash
conda create -n reliability-analysis python=3.10 pip
conda activate reliability-analysis
python -m pip install -r requirements.txt
python tools/prepare_reproduction_selections.py verify --data-root data
```

REML variance decomposition additionally requires R and the `lme4` and
`jsonlite` packages. The recorded study versions were R 4.5.3, `lme4` 1.1.37
and `jsonlite` 2.0.0. Install these in a separate R library if needed:

```r
install.packages("remotes", repos = "https://cloud.r-project.org")
remotes::install_version("lme4", version = "1.1.37",
                        repos = "https://cloud.r-project.org", upgrade = "never")
remotes::install_version("jsonlite", version = "2.0.0",
                        repos = "https://cloud.r-project.org", upgrade = "never")
```

Check the active R installation with:

```bash
Rscript --vanilla -e 'print(R.version.string); print(packageVersion("lme4")); print(packageVersion("jsonlite"))'
```

The MRI-input illustration and model-specific QC scripts also need the
corresponding model environment and actual images. They are not required to
calculate reliability from the published embeddings and predictions.

See [`REPRODUCING_RESULTS.md`](REPRODUCING_RESULTS.md) for runnable analysis
and figure-generation commands using the published CSVs.

## Model execution

Acquire images and demographics from the original datasets, and templates and
weights from the original model releases. Dataset links are in
[`data/sources.csv`](data/sources.csv). Each model's adapter in
`FineTuning/<model>/model.py` records the expected checkpoint and code revision;
the adjacent README gives its extraction, Ridge and fine-tuning commands.

PyTorch, its CUDA build, and model-specific libraries must be installed in the
model environment, not by extending the CPU analysis environment indiscriminately.
The runtime audits record these dependencies. For example, the y-Aware and
Curia-2 runs recorded PyTorch 2.5.0, torchvision 0.20.0, MONAI 1.4.0, NumPy
1.26.4 and Transformers 4.46.3. These are recorded runtime versions, not a
complete environment lockfile.

MRI preprocessing also uses external software depending on the model:
CAT12/SPM with MATLAB, FSL, ANTs, SynthStrip/SynthSeg or HD-BET. Python
requirements do not install these tools or their templates.

### NeuroVFM

Use the [official installation instructions at the study's pinned revision](https://github.com/MLNeurosurg/neurovfm/tree/9240021d4ef5c262b21cee5d219c2adf65f4d42f#installation),
not the moving upstream default branch. The
[dependency specification at that revision](https://github.com/MLNeurosurg/neurovfm/blob/9240021d4ef5c262b21cee5d219c2adf65f4d42f/pyproject.toml)
declares Python 3.10.14 and PyTorch 2.5.0. It also requires the source-built
FlashAttention 2.6.3 kernels described in the upstream instructions.

The study's verified NeuroVFM runtime used Python 3.10.18,
PyTorch 2.5.0+cu124 and torchvision 0.20.0+cu124. This differs from upstream's
strict Python declaration; the recorded runtime should not be mistaken for
an unmodified upstream installation recipe.

The model adapter pins:

- Code: `MLNeurosurg/neurovfm`, commit
  `9240021d4ef5c262b21cee5d219c2adf65f4d42f`.
- Weights: `mlinslab/neurovfm-encoder`, revision
  `d5194fc70a162185f8ef062e362bd522a35312a9`.
- Hardware: BF16-capable NVIDIA GPU, compute capability `sm_80` or newer.

Follow [`FineTuning/NeuroVFM/README.md`](FineTuning/NeuroVFM/README.md) for
checkpoint verification and the forward/backward smoke test before extraction
or training. Installing the upstream package alone does not validate weights
or model inputs.
