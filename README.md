# Anonymization and Visualization of Health Data and Biomarkers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-green)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13.1-orange)](https://pytorch.org/)
[![Prior Work: TabGen](https://img.shields.io/badge/Prior%20Work-TabGen--Framework-blue)](https://github.com/vuhoangminh/TabGen-Framework)

**Minh H. Vu, Daniel Edler, Carl Wibom, Martin Rosvall, Beatrice Melin**  
Department of Diagnostics and Intervention, Umeå University, Sweden  
Corresponding author: [minh.vu@umu.se](mailto:minh.vu@umu.se)

---

## Overview

This repository contains the full codebase for the paper *"Anonymization and Visualization of Health Data and Biomarkers"*. It provides an end-to-end framework for privacy-preserving synthetic health data generation, validated on the **PREDICT cohort** — a multi-cohort biobank from Västerbotten County, Sweden (50,274 individuals, 26 datasets across three GDPR regulatory levels).

The framework integrates advanced deep generative models (DGMs) with robust preprocessing, formal differential privacy (DP), systematic data-sufficiency analysis, domain-guided quality control, and interactive visualization — all released as open-source, containerized software.

This work builds on our prior **TabGen-Framework** ([github.com/vuhoangminh/TabGen-Framework](https://github.com/vuhoangminh/TabGen-Framework), *TMLR* 2026), which introduced the `CorrDst` loss functions and IORBO hyperparameter optimization. The present paper adds the biomedical preprocessing pipeline, data-sufficiency analysis, rejection sampling, and visualization tooling.

> **Looking for the foundational framework?**
> The `CorrDst` loss functions and IORBO hyperparameter optimization used in this work were introduced in our prior paper. See **TabGen-Framework** for the benchmarking suite across 20 public datasets and 10 DGM baselines.
> **→ [github.com/vuhoangminh/TabGen-Framework](https://github.com/vuhoangminh/TabGen-Framework)**

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

```text
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

**Local setup (conda):**

```bash
# Python 3.9, CUDA 11.2 (stable, recommended for reproducibility)
bash scripts/envs/setup.sh

# Python 3.10+, CUDA 12 (newer hardware)
bash scripts/envs/setup_py312.sh
```

**HPC setup (Apptainer container):**

```bash
# Build container from definition file
apptainer build biobank.sif scripts/apptainer/biobank.def

# Run any script inside the container (--nv enables GPU)
apptainer exec --nv biobank.sif python scripts/biobank/preprocess_biobank_phase1.py
```

Key packages: `ctgan==0.8.0`, `sdv==1.8.0`, `opacus`, `anonymeter==1.0.0`, `hyperopt`, `xgboost==1.7.6`, `torch==1.13.1`, `scipy`, `cuML`, `scikit-posthocs`, `imbalanced-learn`

---

## Usage

```bash
# Step 1: Preprocess raw biobank data
python scripts/biobank/preprocess_biobank_phase1.py   # merge TSVs, bin dates, pseudonymize
python scripts/biobank/preprocess_biobank_phase2.py   # label cancer types, aggregate events
# Note: Phase 3 preprocessing (median imputation, missingness indicators, quantile transforms,
# binary/date encoding) is applied automatically by the dataset loader on every run.
# See engine/dataset_helper/preprocessing.py for implementation details.

# Step 2: Train a generative model
# CTGAN — baseline (no DP)
python scripts/tabgen/main_tabgen.py --dataset biobank_patient_dead --arch ctgan

# CTGAN + Differential Privacy (opacus, Rényi DP accounting)
python scripts/biobank/main_ctgan_dp_biobank.py \
    --dataset biobank_record_vital \
    --private 1 --dp_sigma 1.0 \
    --is_loss_corr 1 --is_loss_dwp 1

# CopulaGAN
python scripts/tabgen/main_tabgen.py --dataset biobank_patient_dead --arch copulagan

# TVAE (Tabular VAE)
python scripts/tabgen/main_tabgen.py --dataset biobank_patient_dead --arch tvae

# DP-CGANS (alternative DP-GAN)
python scripts/tabgen/main_tabgen.py --dataset biobank_patient_dead --arch dpcgans

# CTAB-GAN
python scripts/tabgen/main_tabgen.py --dataset biobank_patient_dead --arch ctab

# TabDDPM (single run)
python scripts/tabgen/main_optimize_tabgen_tabddpm_single.py --dataset biobank_patient_dead

# TabSyn — best overall; runs IORBO hyperparameter optimization
python scripts/tabgen/main_optimize_tabgen_tabsyn.py --dataset biobank_patient_dead

# Step 3: Hyperparameter optimization (IORBO) for GAN/VAE models
python scripts/tabgen/main_optimize_tabgen.py --dataset biobank_patient_dead --arch ctgan
python scripts/tabgen/main_optimize_tabgen_tabddpm.py --dataset biobank_patient_dead

# Step 4: Data-sufficiency analysis
python scripts/main_data_sufficient.py --dataset biobank_record_vital

# Step 5: Post-process synthetic data
python scripts/biobank/main_postprocess_biobank.py

# Step 6: Statistical significance testing
python scripts/perform_friedman_nemenyi_biobank.py
```

---

## Repository Structure

```text
├── engine/                  # Core library: datasets, evaluation, custom losses, DP accounting
│   ├── config.py            # Model/dataset registries, plot config
│   ├── datasets.py          # Dataset factory
│   ├── custom_loss.py       # CorrDst loss (correlation + distribution aware)
│   ├── rdp_accountant.py    # Rényi DP budget tracking
│   ├── evaluate_technical_paper.py  # Statistical, ML, and DP metrics
│   └── dataset_helper/
│       └── preprocessing.py # Phase 3: MissingValueEncoder, DateEncoder, BinaryColumnEncoder, FlexiblePipeline
├── models/                  # Generative model implementations
│   ├── ctgan.py             # CTGAN with DP support
│   ├── tvae.py, copulagan.py, dpcgans.py, autoencoder.py
│   ├── tab_ddpm/            # TabDDPM (third-party)
│   ├── tabsyn/              # TabSyn (third-party)
│   └── CTAB/                # CTAB-GAN (third-party)
├── scripts/
│   ├── biobank/             # Preprocess → train → postprocess pipeline
│   ├── bianca/              # BIANCA study analysis and distribution comparison
│   ├── tabgen/              # Optimization & statistical test entry points (shared across papers)
│   │   ├── main_tabgen.py                       # Train a single generative model
│   │   ├── main_optimize_tabgen.py              # IORBO hyperparameter optimization (GAN/VAE)
│   │   ├── main_optimize_tabgen_tabddpm.py      # IORBO for TabDDPM
│   │   ├── main_optimize_tabgen_tabddpm_single.py  # Single TabDDPM run
│   │   ├── main_optimize_tabgen_tabsyn.py       # IORBO for TabSyn
│   │   ├── check_optimize_tabgen.py             # Verify optimization outputs
│   │   ├── perform_friedman_nemenyi_tabgen.py   # Statistical tests (TabGen paper)
│   │   ├── perform_friedman_nemenyi_ablation.py # Ablation tests
│   │   └── perform_friedman_nemenyi_bo.py       # BO comparison tests
│   ├── apptainer/           # Container definitions (biobank.def, biobank_tabddpm.def)
│   ├── envs/                # Conda environment setup (setup.sh, setup_py312.sh)
│   ├── plot/                # Figure generation (private)
│   ├── latex/               # LaTeX table generation (private)
│   ├── main_data_sufficient*.py             # Data-sufficiency analysis
│   ├── main_optimize_ml_model_*.py          # ML model tuning
│   └── perform_friedman_nemenyi_biobank.py  # Statistical tests (biobank paper)
├── docs/                    # Research notes, paper PDF (private)
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
  title       = {Anonymization and Visualization of Health Data and Biomarkers},
  author      = {Vu, Minh H. and Edler, Daniel and Wibom, Carl and Rosvall, Martin and Melin, Beatrice},
  institution = {Umeå University},
  year        = {2026},
  url         = {https://github.com/vuhoangminh/SecureSynth-Framework},
}
```

For the foundational `CorrDst` loss functions and IORBO optimization introduced in the prior work, please also cite:

```bibtex
@article{vu2026a,
  title   = {A Unified Framework for Tabular Generative Modeling: Loss Functions,
             Benchmarks, and Improved Multi-objective Bayesian Optimization Approaches},
  author  = {Minh Hoang Vu and Daniel Edler and Carl Wibom and Tommy L{\"o}fstedt
             and Beatrice Melin and Martin Rosvall},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=RPZ0EW0lz0},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Ethics approval was obtained through an amendment to the regional ethics committee (Umeå University). Raw biobank data is not distributed with this repository.
