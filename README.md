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

Open-source, Docker-containerized pipeline. A single configuration file specifies dataset attributes, preprocessing rules, and privacy constraints, enabling reproducible deployment across institutions. See [`configs/README.md`](configs/README.md) for a full field-by-field reference.

### 2. Biomedical Data-Handling & Evaluation Framework

Four integrated components:

- **Robust preprocessing** — handles missingness rates from 0.4% to 90% via median imputation, quantile transformations, and explicit missingness indicators; substantially outperforms standard constant-value imputation.
- **Formal DP training** — formal differential privacy (Rényi DP accounting via opacus) for select DGMs, combined with empirical privacy risk evaluation (heuristic matching, SDMetrics Disclosure Protection score).
- **Data-sufficiency analysis** — identifies minimum sample sizes for reliable training across models and dataset levels (TabSyn: ~30,000 records; CTGAN/TVAE: ~40,000–60,000).
- **Postprocessing & rejection sampling** — enforces expert-defined correlations and logical constraints (e.g., no female patients assigned prostate cancer, no male patients assigned ovarian cancer), ensuring clinical plausibility of synthetic outputs.

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

## Quick Start

### 1. Environment setup

```bash
bash scripts/envs/biobank.sh
conda activate biobank312
```

### 2. Prepare a config file

Drop your dataset in `database/raw/` and create a config in `configs/`. See [`configs/README.md`](configs/README.md) for all fields. A minimal example:

```toml
[data]
path   = "database/raw/clinical.csv"
format = "csv"
drop_columns = ["patient_id"]

[columns]
continuous = ["age", "bmi", "glucose"]
discrete   = ["sex", "diagnosis"]
target     = "mortality"
task       = "classification"

[training]
gms    = ["CTGAN", "TVAE", "TabSyn"]
losses = ["cd"]
epochs = 10000

[output]
output_dir = "output/clinical/synthetic/"
```

### 3. Validate config (no files written)

```bash
PYTHONPATH=. python run.py --config configs/clinical.toml --dry-run
```

### 4. Smoke test (fast, ~5 min)

Runs 20 epochs and 2 IORBO trials per model. All outputs go to
`database/prepared/<dataset>/test/` so they never overwrite production results.

```bash
PYTHONPATH=. python run.py --config configs/clinical.toml --test
```

### 5. Full production run

```bash
PYTHONPATH=. python run.py --config configs/clinical.toml
```

Outputs written to `database/prepared/<dataset>/`:

| File | Description |
|------|-------------|
| `synthetic_final.csv` | Best overall synthetic dataset (postprocessed) |
| `synthetic_<model>.csv` | Best per-model synthetic dataset |
| `eval_summary.json` | Loss rankings across all model×loss combos |
| `best` → symlink | Trial directory of the global best run |
| `best_<model>` → symlinks | Trial directory of the best run per model family |

### 6. Cleanup before re-running a test

```bash
rm -rf \
  database/runs_test/<dataset>-* \
  database/prepared/<dataset>/test \
  database/optimization/generative_model/test_<dataset>_*
```

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
apptainer exec --nv biobank.sif python scripts/biobank/01_preprocess_phase1.py
```

Key packages: `ctgan==0.8.0`, `sdv==1.8.0`, `opacus`, `anonymeter==1.0.0`, `hyperopt`, `xgboost==1.7.6`, `torch==1.13.1`, `scipy`, `cuML`, `scikit-posthocs`, `imbalanced-learn`

---

## Usage

```bash
# Step 1: Preprocess raw biobank data
python scripts/biobank/01_preprocess_phase1.py   # merge TSVs, bin dates, pseudonymize
python scripts/biobank/02_preprocess_phase2.py   # label cancer types, aggregate events
# Note: Phase 3 preprocessing (median imputation, missingness indicators, quantile transforms,
# binary/date encoding) is applied automatically by the dataset loader on every run.
# See engine/dataset_helper/preprocessing.py for implementation details.

# Step 2: Train a generative model
# CTGAN — baseline (no DP)
python scripts/optimize/run_tabgen.py --dataset biobank_patient_dead --arch ctgan

# CTGAN + Differential Privacy (opacus, Rényi DP accounting)
python scripts/biobank/03_train.py \
    --dataset biobank_record_vital \
    --private 1 --dp_sigma 1.0 \
    --is_loss_corr 1 --is_loss_dwp 1

# CopulaGAN
python scripts/optimize/run_tabgen.py --dataset biobank_patient_dead --arch copulagan

# TVAE (Tabular VAE)
python scripts/optimize/run_tabgen.py --dataset biobank_patient_dead --arch tvae

# DP-CGANS (alternative DP-GAN)
python scripts/optimize/run_tabgen.py --dataset biobank_patient_dead --arch dpcgans

# CTAB-GAN
python scripts/optimize/run_tabgen.py --dataset biobank_patient_dead --arch ctab

# TabDDPM (single run)
python scripts/optimize/optimize_tabddpm_single.py --dataset biobank_patient_dead

# TabSyn — best overall; runs IORBO hyperparameter optimization
python scripts/optimize/optimize_tabsyn.py --dataset biobank_patient_dead

# Step 3: Hyperparameter optimization (IORBO) for GAN/VAE models
python scripts/optimize/optimize_ctgan.py --dataset biobank_patient_dead --arch ctgan
python scripts/optimize/optimize_tabddpm.py --dataset biobank_patient_dead

# Step 4: Post-process synthetic data
python scripts/biobank/04_postprocess.py

# Step 5: Data-sufficiency analysis
python scripts/analysis/data_sufficiency.py --dataset biobank_record_vital

# Step 6: Statistical significance testing
python scripts/analysis/friedman_nemenyi.py
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
│   ├── tvae.py, copulagan.py, dpcgans.py
│   ├── tab_ddpm/            # TabDDPM (third-party)
│   ├── tabsyn/              # TabSyn (third-party)
│   └── CTAB/                # CTAB-GAN (third-party)
├── scripts/                 # See scripts/README.md for pipeline walkthrough
│   ├── biobank/             # Reference pipeline for PREDICT cohort (numbered 01–04)
│   │   ├── 01_preprocess_phase1.py
│   │   ├── 02_preprocess_phase2.py
│   │   ├── 03_train.py
│   │   └── 04_postprocess.py
│   ├── bianca/              # BIANCA study analysis and distribution comparison
│   ├── optimize/            # IORBO hyperparameter optimisation entry points
│   │   ├── run_tabgen.py                # Train a single generative model
│   │   ├── optimize_ctgan.py            # IORBO for CTGAN/VAE models
│   │   ├── optimize_tabddpm.py          # IORBO for TabDDPM
│   │   ├── optimize_tabddpm_single.py   # Single TabDDPM run
│   │   ├── optimize_tabsyn.py           # IORBO for TabSyn
│   │   └── check_results.py             # Verify optimisation outputs
│   ├── analysis/            # Post-hoc statistical tests and data-sufficiency analysis
│   │   ├── data_sufficiency.py
│   │   ├── data_sufficiency_nemenyi.py
│   │   ├── ml_evaluation.py
│   │   └── friedman_nemenyi.py
│   ├── apptainer/           # Container definitions (biobank.def, biobank_tabddpm.def)
│   └── envs/                # Conda environment setup (setup.sh, setup_py312.sh)
└── database/                # Data directory (gitignored)
    ├── dataset/             # Raw and processed datasets
    ├── gan/                 # GAN training outputs
    └── synthetic/           # Final synthetic data
```

---

## Known Issues & Fixes

### TabDDPM on preprocessed datasets (fixed 2026-05-16)

Two bugs caused all TabDDPM IORBO trials to silently fail with `loss=inf` when running on generic/preprocessed datasets:

**Bug 1 — `assert policy is None` in `models/tab_ddpm/lib/data.py:185`**

The base `database/dataset/config.toml` sets `[train.T] num_nan_policy = "mean"`. TabDDPM's internal pipeline asserts that this policy is `None` when the input data contains no NaN values. Preprocessed datasets have no NaNs, so the assertion always fired.

*Fix:* `engine/dataset_helper/base.py` — `_prep_tabddpm_config_toml_mlp` and `_prep_tabddpm_config_toml_resnet` now set `num_nan_policy = None` (serialised as `"__none__"`) whenever `X_num_train.npy` contains no NaN values.

**Bug 2 — `ModuleNotFoundError: No module named 'models'` in `pipeline.py` subprocess**

`optimize_tabddpm.py` launches `pipeline.py` via `subprocess.run`. The subprocess did not inherit `PYTHONPATH=.`, so `from models.tab_ddpm.tab_ddpm import ...` failed at import time.

*Fix:* `scripts/optimize/optimize_tabddpm.py` — added `env={**os.environ, "PYTHONPATH": "."}` to the `subprocess.run` call.

> **Debugging tip:** Both errors are swallowed by the hyperopt objective's `except Exception`. If all TabDDPM trials return `loss=inf`, inspect the stored reason:
> ```python
> import pickle
> with open("database/optimization/<name>.hyperopt", "rb") as f:
>     trials = pickle.load(f)
> print(trials.trials[0]["result"]["reason"])
> ```

---

## Citation

If you use this work, please cite:

> Vu, M.H., Edler, D., Wibom, C. et al. Anonymization and visualization of health data and biomarkers. *npj Digit. Med.* **9**, 347 (2026). https://doi.org/10.1038/s41746-026-02662-x

```bibtex
@article{vu2026anonymization,
  title     = {Anonymization and visualization of health data and biomarkers},
  author    = {Vu, Minh H and Edler, Daniel and Wibom, Carl and Rosvall, Martin and Melin, Beatrice},
  journal   = {npj Digital Medicine},
  volume    = {9},
  number    = {1},
  pages     = {347},
  year      = {2026},
  publisher = {Nature Publishing Group UK London},
  doi       = {10.1038/s41746-026-02662-x},
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
