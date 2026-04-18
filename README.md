# Anonymization and Visualization of Health Data and Biomarkers

**Minh H. Vu, Daniel Edler, Carl Wibom, Martin Rosvall, Beatrice Melin**  
Department of Diagnostics and Intervention, Umeå University, Sweden  
Corresponding author: minh.vu@umu.se

---

## Overview

This repository contains the full codebase for the paper *"Anonymization and Visualization of Health Data and Biomarkers"*. It provides an end-to-end framework for privacy-preserving synthetic health data generation, validated on the **PREDICT cohort** — a multi-cohort biobank from Västerbotten County, Sweden (50,274 individuals, 26 datasets across three GDPR regulatory levels).

The framework integrates advanced deep generative models (DGMs) with robust preprocessing, formal differential privacy (DP), systematic data-sufficiency analysis, domain-guided quality control, and interactive visualization — all released as open-source, containerized software.

This work builds on our prior **TabGen-Framework** ([github.com/vuhoangminh/TabGen-Framework](https://github.com/vuhoangminh/TabGen-Framework), *TMLR* 2026), which introduced the `CorrDst` loss functions and IORBO hyperparameter optimization. The present paper adds the biomedical preprocessing pipeline, data-sufficiency analysis, rejection sampling, and visualization tooling.

---

## Key Contributions

### 1. Privacy-Aware Synthesis Framework
Open-source, Docker-containerized pipeline. A single configuration file specifies dataset attributes, preprocessing rules, and privacy constraints, enabling reproducible deployment across institutions.

### 2. Biomedical Data-Handling & Evaluation Framework
Four integrated components:

- **Robust preprocessing** — handles missingness rates from 0.4% to 90% via median imputation, quantile transformations, and explicit missingness indicators; substantially outperforms standard constant-value imputation.
- **Formal DP training** — formal differential privacy (Rényi DP accounting via opacus) for select DGMs, combined with empirical privacy risk evaluation (heuristic matching, SDMetrics Disclosure Protection score).
- **Data-sufficiency analysis** — identifies minimum sample sizes for reliable training across models and dataset levels (TabSyn: ~30,000 records; CTGAN/TVAE: ~40,000–60,000).
- **Postprocessing & rejection sampling** — enforces expert-defined correlations and logical constraints (e.g., no male patients assigned breast cancer), ensuring clinical plausibility of synthetic outputs.

### 3. Interactive Visualization Tool
Privacy-preserving exploration of health data with:
- Adaptive anonymization via k-anonymity (counts suppressed below 10 individuals)
- Hierarchical ICD-10 disease code navigation
- Interactive filtering and stratification by demographics, diagnoses, and biomarkers

---

## Results

Across 26 biobank datasets spanning three regulatory levels:

| Model | Loss | Avg. Rank | Notes |
| --- | --- | --- | --- |
| TabSyn | CorrDst | 3.4 | Best overall — recommended default |
| TabSyn | BaseFn | ~4.0 | Strong across all levels |
| CTGAN | CorrDst | ~7–9 | Good statistical fidelity; weaker ML utility |
| TVAE | CorrDst | ~12 | Localized strengths |
| CTAB-GAN | BaseFn/CorrDst | >19 | Frequent OOM/timeout failures |

`TabSyn + CorrDst` consistently achieved the highest ranks across statistical fidelity, ML utility (TSTR), and differential privacy metrics. The proposed preprocessing pipeline substantially improved distributional accuracy and clinical plausibility compared to standard approaches.

---

## Dataset Structure

The PREDICT cohort datasets are organized into three GDPR regulatory levels:

| Level | Access | Datasets | Rows |
| --- | --- | --- | --- |
| Level 1 | Cancer registration only (legal restrictions) | 7 record-based datasets | ~123,420 |
| Level 2 | GDPR approval + patient journals + 25% metabolomics | 8 patient-based datasets | ~52,439 |
| Level 3 | Full GDPR approval + 100% metabolomics | 11 datasets (cancer cohorts + metabolomics) | ~50,087 |

---

## Framework Pipeline

```
Sensitive Data + Configuration File
        │
        ▼  Preprocessing
        │  • Median imputation, quantile transforms, missingness indicators
        │  • ICD code labeling, event aggregation, pseudonymization
        │
        ▼  Generative Model Training (DGM)
        │  • Models: CTGAN, TVAE, CopulaGAN, DP-CGANS, CTAB-GAN, TabDDPM, TabSyn
        │  • Loss: BaseFn (vanilla) or CorrDst (correlation- and distribution-aware)
        │  • Hyperparameter optimization via IORBO
        │  • Optional: formal DP training (Rényi DP + opacus)
        │
        ▼  Evaluation & Model Ranking
        │  • Statistical similarity (KL divergence, Wasserstein, correlation)
        │  • ML utility: Train-Synthetic-Test-Real (TSTR), AUROC
        │  • Privacy: (ε, δ) accounting, empirical disclosure risk
        │
        ▼  Quality Control
        │  • Domain-informed sanity checks (sex-specific cancer patterns, etc.)
        │  • Rejection sampling to enforce expert-defined constraints
        │
        ▼  Synthetic Data (safe to share) + Visualization
```

---

## Models

| Model | Type | DP Support |
| --- | --- | --- |
| CTGAN | GAN | Yes (opacus) |
| CTGAN* | GAN (unconditional) | Yes |
| DP-CGANS | GAN | Yes (built-in) |
| CopulaGAN | GAN | Yes |
| TVAE | VAE | — |
| CTAB-GAN | GAN | — |
| TabDDPM-MLP | Diffusion | — |
| TabDDPM-ResNet | Diffusion | — |
| TabSyn | Diffusion (transformer) | — |

---

## Environment & Setup

**Hardware used:** NVIDIA A100 (40GB VRAM), Intel Xeon Gold 6338, 256GB DDR4 RAM, ~700 GPU runtime days.

```bash
# Set up conda environment (downloads Anaconda, installs CUDA 11.2, PyTorch 1.13.1)
bash scripts/envs/setup.sh

# For HPC clusters (Alvis/Chalmers)
bash scripts/envs/biobank_alvis.sh

# Or use the provided Apptainer/Docker container
# Container definitions: scripts/apptainer/biobank.def
```

Key packages: `ctgan`, `sdv`, `opacus`, `anonymeter`, `hyperopt`, `xgboost==1.7.6`, `scipy`, `cuML`, `scikit-posthocs`, `synthcity`

---

## Usage

```bash
# Step 1: Preprocess raw biobank data
python scripts/biobank/preprocess_biobank_phase1.py   # merge & clean TSVs
python scripts/biobank/preprocess_biobank_phase2.py   # cancer type labeling

# Step 2: Train a generative model (example: CTGAN with DP)
python scripts/biobank/main_ctgan_dp_biobank.py \
    --dataset biobank_record_vital \
    --private --dp_sigma 1.0 \
    --is_loss_corr --is_loss_dwp

# Step 3: Hyperparameter optimization (IORBO)
python scripts/main_optimize_technical_paper.py --dataset adult --model ctgan
python scripts/main_optimize_technical_paper_tabsyn.py --dataset biobank_patient_dead

# Step 4: Post-process synthetic data
python scripts/biobank/main_postprocess_biobank.py

# Step 5: Generate paper figures
python scripts/plot/plot_nature_ranking.py
```

---

## Repository Structure

```
├── engine/                  # Core library: datasets, evaluation, custom losses, DP accounting
│   ├── config.py            # Model/dataset registries, plot config
│   ├── datasets.py          # Dataset factory
│   ├── custom_loss.py       # CorrDst loss (correlation + distribution aware)
│   ├── rdp_accountant.py    # Rényi DP budget tracking
│   └── evaluate_technical_paper.py  # Statistical, ML, and DP metrics
├── models/                  # Generative model implementations
│   ├── ctgan.py             # CTGAN with DP support
│   ├── tvae.py, copulagan.py, dpcgans.py, autoencoder.py
│   ├── tab_ddpm/            # TabDDPM (third-party)
│   ├── tabsyn/              # TabSyn (third-party)
│   └── CTAB/                # CTAB-GAN (third-party)
├── scripts/
│   ├── biobank/             # Preprocess → train → postprocess pipeline
│   ├── bianca/              # BIANCA study analysis and distribution comparison
│   ├── plot/                # Figure generation for paper
│   ├── latex/               # LaTeX table generation
│   ├── envs/                # Conda and HPC environment setup
│   ├── apptainer/           # Container definitions
│   └── jobs/                # HPC batch job scripts
├── docs/
│   └── paper.pdf            # Published paper
└── database/                # Data directory (gitignored)
    ├── dataset/             # Raw and processed datasets
    ├── gan/                 # GAN training outputs
    └── synthetic/           # Final synthetic data
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{vu2026b,
  title={Anonymization and Visualization of Health Data and Biomarkers},
  author={Vu, Minh H. and Edler, Daniel and Wibom, Carl and Rosvall, Martin and Melin, Beatrice},
  institution={Umeå University},
  year={2026}
}
```

For the foundational loss functions and IORBO optimization, please also cite the prior work:

```bibtex
@article{vu2026a,
  title={A Unified Framework for Tabular Generative Modeling},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=RPZ0EW0lz0}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Ethics approval was obtained through an amendment to the regional ethics committee (Umeå University). Raw biobank data is not distributed with this repository.
