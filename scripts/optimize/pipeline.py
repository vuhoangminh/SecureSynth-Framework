"""Subprocess entry point for one model×loss IORBO optimisation.

Called by engine.orchestrator._dispatch_combo with:
    --dataset DATASET --model MODEL --loss LOSS --config CONFIG
    [--is_test 0|1] [--max_trials N]

Prints a JSON object as its final stdout line:
    {"best_trial": <int>, "loss": <float>}
"""
import argparse
import json
import os
import subprocess
import sys

LOSS_TO_VERSION = {"vanilla": 0, "cd": 2}

# model name (lowercase) → (optimizer script, arch flag, is_condvec)
# CTGAN  = condvec=1 (conditional vector sampling enabled, default)
# CTGAN0 = condvec=0 (unconditional sampling)
MODEL_TO_OPTIMIZER = {
    "ctgan":      ("scripts/optimize/optimize_ctgan.py",   "ctgan",     1),
    "ctgan0":     ("scripts/optimize/optimize_ctgan.py",   "ctgan",     0),
    "tvae":       ("scripts/optimize/optimize_ctgan.py",   "tvae",      1),
    "tvae0":      ("scripts/optimize/optimize_ctgan.py",   "tvae",      0),
    "copulagan":  ("scripts/optimize/optimize_ctgan.py",   "copulagan", 1),
    "copulagan0": ("scripts/optimize/optimize_ctgan.py",   "copulagan", 0),
    "tabsyn":     ("scripts/optimize/optimize_tabsyn.py",  "tabsyn",    1),
    "tabddpm":    ("scripts/optimize/optimize_tabddpm.py", "tabddpm",   1),
}


def _best_from_hyperopt(dataset: str, arch: str, loss_version: int, is_test: bool, condvec: int = 1, module: str = "gmdp"):
    """Return (best_trial_index, best_loss) by reading the hyperopt pickle."""
    from engine.utils import hyperopt_utils, path_utils
    import os

    base = f"{dataset}_{arch}_loss_version-{loss_version}-{condvec}"
    if is_test:
        base = "test_" + base
    # Try with module suffix first (current format), then without (legacy)
    for suffix in [f"_module-{module}", ""]:
        hp_path = path_utils.get_hyperopt_path(base + suffix, folder="optimization/generative_model")
        if os.path.exists(hp_path):
            break

    trials = hyperopt_utils.load_project(hp_path, is_print=False)
    best_trial, best_loss = 0, float("inf")
    for i, result in enumerate(trials.results):
        if isinstance(result, dict):
            loss = result.get("loss", float("inf"))
            if loss < best_loss:
                best_trial = i
                best_loss = loss
    return best_trial, best_loss


def main():
    parser = argparse.ArgumentParser(description="Run one model×loss IORBO optimisation")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--loss", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--is_test", type=int, default=0)
    parser.add_argument("--max_trials", type=int, default=30)
    args = parser.parse_args()

    model_key = args.model.lower()
    loss_lower = args.loss.lower()

    if model_key not in MODEL_TO_OPTIMIZER:
        print(f"Unknown model: {args.model!r}. Known: {sorted(MODEL_TO_OPTIMIZER)}", file=sys.stderr)
        sys.exit(1)
    if loss_lower not in LOSS_TO_VERSION:
        print(f"Unknown loss: {args.loss!r}. Known: {sorted(LOSS_TO_VERSION)}", file=sys.stderr)
        sys.exit(1)

    optimizer_script, arch, condvec = MODEL_TO_OPTIMIZER[model_key]
    loss_version = LOSS_TO_VERSION[loss_lower]

    cmd = [
        sys.executable, "-W", "ignore", optimizer_script,
        "--dataset", args.dataset,
        "--arch", arch,
        "--loss_version", str(loss_version),
        "--is_condvec", str(condvec),
        "--is_test", str(args.is_test),
        "--max_trials", str(args.max_trials),
        "--module", "gmdp",
    ]

    env = {**os.environ, "PYTHONPATH": "."}
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    try:
        best_trial, best_loss = _best_from_hyperopt(
            args.dataset, arch, loss_version, bool(args.is_test), condvec
        )
    except Exception as exc:
        print(f"WARNING: could not read hyperopt result: {exc}", file=sys.stderr)
        best_trial, best_loss = 0, float("inf")

    loss_out = best_loss if not (best_loss != best_loss or best_loss == float("inf")) else None
    print(json.dumps({"best_trial": best_trial, "loss": loss_out}))


if __name__ == "__main__":
    main()
