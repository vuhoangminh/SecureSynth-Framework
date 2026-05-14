# Scripts

## Folder Layout

```
scripts/
  biobank/     Reference implementation for the PREDICT cohort (GDPR-regulated biobank)
  bianca/      Second reference example: BIANCA clinical dataset analysis
  optimize/    Hyperparameter optimisation (IORBO) for all model architectures
  analysis/    Post-hoc statistical tests and data-sufficiency analysis
  envs/        Environment setup (local: setup.sh / setup_py312.sh)
  apptainer/   HPC container definitions (Apptainer/Singularity)
```

## Pipeline (biobank/ reference implementation)

Run in order:

```bash
# 1. Preprocessing
python scripts/biobank/01_preprocess_phase1.py   # merge TSVs, bin dates, pseudonymize
python scripts/biobank/02_preprocess_phase2.py   # label cancer types, aggregate events

# 2. Train a generative model with differential privacy
python scripts/biobank/03_train.py \
    --dataset biobank_record_vital \
    --private 1 --dp_sigma 1.0 \
    --is_loss_corr 1 --is_loss_dwp 1

# 3. Hyperparameter optimisation (IORBO)
python scripts/optimize/optimize_ctgan.py   --dataset biobank_patient_dead --arch ctgan
python scripts/optimize/optimize_tabsyn.py  --dataset biobank_patient_dead
python scripts/optimize/optimize_tabddpm.py --dataset biobank_patient_dead

# 4. Post-process synthetic data (rejection sampling, plausibility checks)
python scripts/biobank/04_postprocess.py

# 5. Evaluation and statistical tests
python scripts/analysis/data_sufficiency.py     --dataset biobank_record_vital
python scripts/analysis/friedman_nemenyi.py
```

## Using Your Own Dataset

`scripts/biobank/` is a reference implementation for the PREDICT cohort. To adapt the
framework to your own sensitive dataset, create `scripts/<your_dataset>/` and follow
the same numbered pattern (`01_preprocess.py`, `02_train.py`, …). The shared logic
lives in `engine/` and `models/` — no changes needed there.
