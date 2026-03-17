# Cross-Scanner Reliability of Brain MRI Foundation Model Embeddings

Code and data for the paper:

> **Cross-Scanner Reliability of Brain MRI Foundation Model Embeddings: A Travelling-Heads Study**
> Rafael Navarro-Gonzalez, Santiago Aja-Fernandez, Alvaro Planchuelo-Gomez, Rodrigo de Luis-Garcia
> *Medical Image Analysis, 2026*

## Overview

This repository contains the preprocessing, embedding extraction, analysis, and figure generation code used to evaluate the cross-scanner reliability of five brain MRI foundation model embeddings and a FreeSurfer morphometric baseline, using the [ON-Harmony travelling-heads dataset](https://openneuro.org/datasets/ds004712/versions/2.0.1).

## Models evaluated

| Model | Architecture | Pretraining | Embedding dim |
|-------|-------------|-------------|---------------|
| BrainIAC | ViT-B | SimCLR | 768 |
| 3D-Neuro-SimCLR | ResNet-18 | SimCLR | 512 |
| y-Aware | DenseNet-121 | y-Aware InfoNCE | 1,024 |
| AnatCL | ResNet-18 | Anatomy-aware CL | 512 |
| BrainSegFounder | Swin-T | Recon+rot+CL | 768 |
| FreeSurfer (DKT) | — | — | 203 |

## Repository structure

```
Reliability_FMs/
├── preprocessing/          # Model-specific preprocessing pipelines
│   ├── preprocessing_BrainIAC.py
│   ├── preprocessing_3DNeuroSimCLR.py
│   ├── preprocessing_AnatCL.py
│   ├── preprocessing_yAware.py
│   ├── preprocessing_BrainSegFounder.py
│   └── fix_anatcl_shape.sh
├── extraction/             # Embedding extraction scripts
│   ├── extract_BrainIAC.py
│   ├── extract_3DNeuroSimCLR.py
│   ├── extract_AnatCL.py
│   ├── extract_yAware.py
│   ├── extract_BrainSegFounder.py
│   └── Launch_extractors.sh
├── analysis/               # Reliability analysis scripts
│   ├── utils_reliability.py          # Core: ICC, variance decomposition, CoV, fingerprinting
│   ├── analyse_BrainIAC.py
│   ├── analyse_3DNeuroSimCLR.py
│   ├── analyse_AnatCL.py
│   ├── analyse_yAware.py
│   ├── analyse_BrainSegFounder.py
│   ├── analyse_FreeSurfer.py
│   ├── supplementary_analyses.py     # Sex classification, PCA, pairwise distances
│   └── pca_icc_robustness.py         # Dimensionality-normalised ICC check
├── figures/                # Figure generation
│   ├── generate_figures.py           # Main figures (Fig 1-4)
│   └── generate_supplementary_figures.py  # Supplementary figures (Fig S1-S4)
├── data/
│   ├── participants.tsv              # ON-Harmony participant demographics
│   ├── embeddings/                   # Extracted embeddings (CSV)
│   └── results/                      # Analysis results (JSON)
└── README.md
```

## Data

- **Raw MRI data**: The ON-Harmony dataset is publicly available at [OpenNeuro ds004712 v2.0.1](https://openneuro.org/datasets/ds004712/versions/2.0.1).
- **Extracted embeddings**: Provided in `data/embeddings/` as CSV files. These allow reproducing all analyses and figures without requiring model weights or GPU access.
- **Analysis results**: Provided in `data/results/` as JSON files. These allow reproducing all figures directly.

## Reproducing the analyses

### Prerequisites

- Python 3.10+
- NumPy, SciPy, scikit-learn, pandas, matplotlib, seaborn

### Steps

1. **Preprocessing** (requires raw ON-Harmony data + model-specific tools):
   ```bash
   # See individual scripts in preprocessing/ for dependencies
   # (ANTs, FSL, FreeSurfer, CAT12, HD-BET, SynthStrip)
   ```

2. **Embedding extraction** (requires model weights + GPU):
   ```bash
   cd extraction/
   bash Launch_extractors.sh
   ```

3. **Analysis** (requires only the CSV embeddings):
   ```bash
   cd analysis/
   python analyse_BrainIAC.py
   python analyse_3DNeuroSimCLR.py
   python analyse_AnatCL.py
   python analyse_yAware.py
   python analyse_BrainSegFounder.py
   python analyse_FreeSurfer.py
   python supplementary_analyses.py
   python pca_icc_robustness.py
   ```

4. **Figure generation** (requires only the JSON results):
   ```bash
   cd figures/
   python generate_figures.py
   python generate_supplementary_figures.py
   ```

## Citation

If you use this code or data, please cite:

```bibtex
@article{NavarroGonzalez2026_reliability,
  title={Cross-Scanner Reliability of Brain MRI Foundation Model Embeddings: A Travelling-Heads Study},
  author={Navarro-Gonz{\'a}lez, Rafael and Aja-Fern{\'a}ndez, Santiago and Planchuelo-G{\'o}mez, {\'A}lvaro and de Luis-Garc{\'{\i}}a, Rodrigo},
  journal={Medical Image Analysis},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
