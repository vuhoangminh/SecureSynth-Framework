# Database Layout Migration

All runs will be re-executed from scratch. Old folders can be deleted safely.

## Path Changes

| Old path | New path | Contents |
|---|---|---|
| `database/dataset/` | `database/prepared/` | Preprocessed CSVs per dataset |
| `database/gan_optimize/` | `database/runs/` | Hyperopt trial outputs |
| `database/optimization/` | `database/optimization/generative_model/` | `.hyperopt` pickles for GAN optimisation |
| `database/optimization_ml_method/` | `database/optimization/ml_method/` | `.hyperopt` pickles for ML evaluation method |
| `data/` | `database/raw/` | Raw input files |

## Notes

- `database/prepared/{dataset}/tvae/` — TVAE checkpoints (previously in `database/tabsyn_tvae/`)
- `database/runs/` replaces `database/gan_optimize/`; trial result JSONs live here
- Old folders left on HPC until re-runs confirm outputs are clean
