# Dataset Configuration Reference

Each dataset is configured via a single TOML file placed in this directory (e.g. `configs/clinical.toml`). The file is loaded by `engine/config_loader.py:load_config()` into a `PipelineConfig` dataclass and passed through the entire pipeline — preprocessing, training, postprocessing, and output.

Only `[data]` and `[columns]` are required. All other sections are optional and fall back to the defaults shown below.

---

## Generating a config with Claude

Not sure how to fill in all the fields? Use one of the prompts below.

### Option A — Quick checklist

Fill in the brackets, then paste the entire block into Claude. Claude will return a ready-to-use `configs/{name}.toml`.

```
I need a SecureSynth-Framework config.toml for my dataset.

Dataset info:
- Name (used as file stem, no spaces): [clinical]
- CSV path (relative to repo root): [data/my_dataset.csv]
- CSV separator: [comma / tab]
- Columns to drop before processing (IDs, surrogate keys): [patient_id, row_num]

Column types:
- Continuous (numeric): [age, bmi, glucose, systolic_bp, diastolic_bp]
- Discrete / categorical: [sex, diagnosis, smoking_status]

Prediction target:
- Target column: [mortality]
- Task: [classification / regression]

Privacy (leave blank if unsure):
- Quasi-identifier columns (could re-identify individuals when combined): [age, sex]
- Sensitive columns (disclosure = privacy breach): [diagnosis, mortality]

Constraints — pandas df.query() syntax, one per line (leave blank if none):
- [diastolic_bp < systolic_bp]
- [age >= 18]

Training:
- Models to train (blank = CTGAN only): [CTGAN, TabSyn]
- Loss variants (blank = vanilla only): [vanilla, cd]
- Max epochs: [10000]

Differential privacy:
- Enable DP?: [no / yes]
- If yes — target epsilon: [3.0], delta: [1e-5]

Output:
- Number of synthetic rows (blank = same as input): []
- Generate SDMetrics HTML report?: [yes / no]

Generate a complete, valid configs/[name].toml following the SecureSynth-Framework TOML schema.
Validate that the target column appears in continuous or discrete, and explain any assumptions made.
```

### Option B — Guided interview

Paste this single prompt into Claude and answer its questions one by one. Better if you are unsure about some fields.

```
I need to create a SecureSynth-Framework config.toml but I am not sure about all the details.

Please interview me step by step. Ask about one topic at a time:
1. Dataset name and CSV file path
2. Which columns are continuous (numeric) vs discrete (categorical)
3. The prediction target column and whether it is classification or regression
4. Columns to drop (IDs, surrogate keys)
5. Privacy-relevant columns — quasi-identifiers and sensitive attributes
6. Logical constraints the synthetic data must satisfy (clinical rules, domain knowledge)
7. Which generative models to train, or use the default (CTGAN only)
8. Whether differential privacy is needed and what epsilon target
9. How many synthetic rows to generate

When you have all the information, output a complete configs/{name}.toml and briefly
explain any assumptions or warnings (e.g. delta too large for the dataset size).
```

---

---

## `[data]` — Input data source *(required)*

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | string | — | Path to the input CSV (or other format) file, relative to the repo root. |
| `format` | string | `"csv"` | File format. Currently only `"csv"` is supported. |
| `separator` | string | `","` | Column delimiter used in the CSV file. Use `"\t"` for TSV. |
| `drop_columns` | list[string] | `[]` | Columns to discard before any processing (e.g. row IDs, surrogate keys). These are never seen by the model or evaluator. |

**Example**

```toml
[data]
path        = "data/clinical.csv"
format      = "csv"
drop_columns = ["patient_id"]
```

---

## `[columns]` — Column roles *(required)*

| Field | Type | Default | Description |
|---|---|---|---|
| `target` | string | — | Name of the prediction target column. Used by ML-utility evaluation (TSTR, AUROC). |
| `task` | string | — | Learning task for the target column. One of `"classification"` or `"regression"`. |
| `continuous` | list[string] | `[]` | Columns treated as continuous numerical features. Passed through quantile transformation and optional PCA. |
| `discrete` | list[string] | `[]` | Columns treated as categorical/ordinal features. Ordinally encoded before model input; inverse-transformed after sampling. |

> **Note:** every column in the dataset should appear in exactly one of `continuous`, `discrete`, or `drop_columns`. The `target` column must also appear in one of the two lists.

**Example**

```toml
[columns]
continuous = ["age", "bmi", "systolic_bp", "diastolic_bp", "glucose"]
discrete   = ["sex", "diagnosis"]
target     = "mortality"
task       = "classification"
```

---

## `[attributes]` — Privacy-relevant column roles *(optional)*

Used by the anonymeter privacy evaluator to compute singling-out, linkability, and inference risks.

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | list[string] | `[]` | Quasi-identifier columns — attributes that could be combined with external data to re-identify an individual (e.g. age, sex, postcode). |
| `sensitive` | list[string] | `[]` | Sensitive attributes whose disclosure constitutes a privacy breach (e.g. diagnosis, disease status). |

**Example**

```toml
[attributes]
key       = ["age", "sex"]
sensitive = ["diagnosis", "diabetes", "hypertension", "mortality"]
```

---

## `[preprocessing]` — Phase 3 preprocessing options *(optional)*

Phase 3 preprocessing runs automatically on every pipeline execution (see `engine/dataset_helper/preprocessing.py`). These fields control its behaviour.

| Field | Type | Default | Description |
|---|---|---|---|
| `date_columns` | list[string] | `[]` | Columns containing date values. `DateEncoder` decomposes each into numeric year/month/day sub-columns and drops the original. |
| `pca` | bool | `false` | Whether to apply PCA dimensionality reduction to the continuous columns after quantile transformation. Useful for high-dimensional metabolomics data. |
| `pca_variance` | float | `0.9` | Fraction of variance to retain when `pca = true`. Controls the number of principal components kept. |
| `pca_exclude` | list[string] | `[]` | Continuous columns to exclude from PCA (e.g. the target column, protected attributes). These pass through untransformed. |
| `missing_noise_std` | float | `0.05` | Standard deviation of Gaussian noise injected into the imputed values of missing continuous entries. Small noise prevents the model from learning a spurious point mass at the imputed median. Set to `0.0` to disable. |

**Example**

```toml
[preprocessing]
date_columns      = ["visit_date"]
pca               = false
pca_variance      = 0.9
pca_exclude       = []
missing_noise_std = 0.05
```

---

## `[training]` — Generative model training *(optional)*

| Field | Type | Default | Description |
|---|---|---|---|
| `gms` | list[string] | `["CTGAN"]` | Generative models to train. Supported values: `"CTGAN"`, `"TVAE"`, `"CopulaGAN"`, `"DPCGANS"`, `"CTAB"`, `"TabDDPM"`, `"TabSyn"`. Multiple models are trained sequentially and evaluated together. |
| `losses` | list[string] | `["vanilla"]` | Loss function variants to apply. `"vanilla"` uses the model's default objective; `"cd"` adds the CorrDst correlation + distribution loss. Both can be listed to train each model under each loss. |
| `epochs` | int | `10000` | Maximum training epochs. Models with early stopping (TabSyn, TabDDPM) may stop earlier. |
| `batch_size` | int | `500` | Mini-batch size for GAN/VAE training. Larger batches improve gradient stability but require more VRAM. TabDDPM and TabSyn have their own batch size controlled via IORBO. |

**Example**

```toml
[training]
gms        = ["CTGAN", "TVAE", "TabSyn"]
losses     = ["vanilla", "cd"]
epochs     = 10000
batch_size = 500
```

---

## `[differential_privacy]` — Formal DP training *(optional)*

Enables Rényi DP training via opacus. Only applies to models with DP support (CTGAN, DP-CGANS, CopulaGAN). The privacy budget (ε, δ) is tracked by `engine/rdp_accountant.py` and logged after every training run.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch. When `false` all other fields in this section are ignored. |
| `epsilon` | float | `1.0` | Target privacy budget ε (lower = stronger privacy). Training stops when the accumulated ε reaches this value. Typical values: `1.0` (strong), `3.0` (moderate), `10.0` (weak). |
| `delta` | float | `1e-5` | Failure probability δ in the (ε, δ)-DP guarantee. Should be much smaller than 1/n where n is the dataset size. |
| `dp_sigma` | float | `1.0` | Noise multiplier σ added to gradients. Higher σ = more noise = smaller ε consumed per step, but degrades data quality. Tuned per dataset (see CLAUDE.md §Bottlenecks). |
| `weight_clip` | float | `0.1` | Per-sample gradient clipping threshold (L2 norm). Must be set to bound the sensitivity of each gradient update. |

**Example**

```toml
[differential_privacy]
enabled     = true
epsilon     = 3.0
delta       = 1e-5
dp_sigma    = 1.2
weight_clip = 0.1
```

---

## `[postprocessing]` — Rejection sampling constraints *(optional)*

After sampling, synthetic rows that violate any constraint are discarded and resampled until the full requested row count is met (up to a maximum retry limit).

| Field | Type | Default | Description |
|---|---|---|---|
| `constraints` | list[string] | `[]` | List of constraint expressions in `pandas.DataFrame.query()` syntax. Each expression must evaluate to `True` for a row to be kept. Columns referenced must exist in the synthetic output. Use this to enforce clinical plausibility rules (e.g. diastolic < systolic BP, age ≥ 18, no male patients with ovarian cancer diagnosis). |

**Example**

```toml
[postprocessing]
constraints = [
    "diastolic_bp < systolic_bp",
    "age >= 18",
    "not (sex == 'male' and diagnosis == 'ovarian_cancer')",
]
```

> **Tip:** test constraint expressions with `df.query(expr)` on your real data before adding them — a typo silently drops all rows.

---

## `[output]` — Synthetic data output *(optional)*

| Field | Type | Default | Description |
|---|---|---|---|
| `n_samples` | int or null | `null` | Number of synthetic rows to generate. When `null` (omitted), matches the row count of the input dataset. |
| `output_dir` | string | `"output/{dataset_name}/synthetic/"` | Directory where synthetic CSV files are written. The placeholder `{dataset_name}` is substituted at runtime. |
| `report` | bool | `false` | Whether to generate an HTML evaluation report (SDMetrics) alongside the synthetic CSV. |

**Example**

```toml
[output]
n_samples  = 5000
output_dir = "output/clinical/synthetic/"
report     = true
```

---

## Full example — `configs/clinical.toml`

```toml
[data]
path         = "data/clinical.csv"
format       = "csv"
drop_columns = ["patient_id"]

[columns]
continuous = ["age", "bmi", "systolic_bp", "diastolic_bp", "glucose", "cholesterol", "creatinine"]
discrete   = ["sex", "diagnosis"]
target     = "mortality"
task       = "classification"

[attributes]
key       = ["age", "sex"]
sensitive = ["diagnosis", "diabetes", "hypertension", "readmission_30d", "mortality"]

[preprocessing]
date_columns      = []
pca               = false
pca_variance      = 0.9
pca_exclude       = []
missing_noise_std = 0.05

[training]
gms        = ["CTGAN", "TVAE", "TabSyn"]
losses     = ["vanilla", "cd"]
epochs     = 10000
batch_size = 500

[differential_privacy]
enabled     = false
epsilon     = 1.0
delta       = 1e-5
dp_sigma    = 1.0
weight_clip = 0.1

[postprocessing]
# pandas df.query() syntax
constraints = [
    "diastolic_bp < systolic_bp",
    "age >= 18",
]

[output]
output_dir = "output/clinical/synthetic/"
report     = true
```
