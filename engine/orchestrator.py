from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import engine.progress as progress
from engine.utils.path_utils import get_run_dir

# Translate config-level names to the directory naming used by the optimizers.
_LOSS_TO_LV: dict[str, str] = {"vanilla": "lv0", "cd": "lv2"}

_FAMILY_ALIASES: dict[str, str] = {"ctgan0": "ctgan"}


def _norm_model(model: str) -> str:
    return model.lower()


def _model_family(model_key: str) -> str:
    """Return the canonical family name for a 'MODEL-loss' key."""
    model = model_key.split("-", 1)[0].lower()
    return _FAMILY_ALIASES.get(model, model)


@dataclass
class StepResult:
    ok: bool
    error: Exception | None = None
    data: dict | None = None


def step_preprocess(dataset: str, config_path: str, _dataset_cls=None) -> StepResult:
    """Preprocess *config_path* via GenericDataset and record progress.

    *_dataset_cls* is injectable for unit tests so callers can pass a mock
    without pulling in the full model/GPU import chain.
    """
    if _dataset_cls is None:
        from engine.dataset_helper.generic import GenericDataset  # noqa: lazy
        _dataset_cls = GenericDataset
    try:
        _dataset_cls(config_path)
        progress.mark(dataset, "preprocess", "done")
        return StepResult(ok=True)
    except Exception as exc:
        progress.mark(dataset, "preprocess", "failed")
        return StepResult(ok=False, error=exc)


def _dispatch_combo(
    dataset: str, model: str, loss: str, config_path: str,
    is_test: int = 0, max_trials: int = 30,
) -> dict[str, Any] | None:
    """Run one model×loss optimisation in a subprocess.

    The subprocess must print a JSON object as its final stdout line with keys
    ``best_trial`` (int) and ``loss`` (float).  Returns None on non-zero exit
    or unparseable output.
    """
    proc = subprocess.run(
        [
            sys.executable, "scripts/optimize/pipeline.py",
            "--dataset", dataset,
            "--model", model,
            "--loss", loss,
            "--config", config_path,
            "--is_test", str(is_test),
            "--max_trials", str(max_trials),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
    )
    if proc.returncode != 0:
        return None
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except (json.JSONDecodeError, ValueError):
        return None


def _write_family_symlink(dataset: str, best_trial: int, model_key: str) -> None:
    family = _model_family(model_key)
    model, loss = model_key.split("-", 1)
    arch = _norm_model(model)
    lv = _LOSS_TO_LV.get(loss, loss)
    link = Path("database/prepared") / dataset / f"best_{family}"
    target = Path("../../runs") / f"{dataset}-{arch}-{lv}" / f"trial_{best_trial:04d}"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def _write_best_symlink(dataset: str, best_trial: int, model: str, loss: str) -> None:
    link = Path("database/prepared") / dataset / "best"
    arch = _norm_model(model)
    lv = _LOSS_TO_LV.get(loss, loss)
    target = Path("../../runs") / f"{dataset}-{arch}-{lv}" / f"trial_{best_trial:04d}"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def step_train(
    dataset: str,
    config_path: str,
    gms: list[str],
    losses: list[str],
    is_test: int = 0,
    max_trials: int = 30,
) -> StepResult:
    """Iterate every model×loss combo, dispatch each as a subprocess, handle failures
    independently, then write a ``best/`` symlink to the combo with the lowest loss.
    """
    successes: list[dict[str, Any]] = []

    for gm in gms:
        for loss in losses:
            model_key = f"{gm}-{loss}"
            try:
                result = _dispatch_combo(dataset, gm, loss, config_path, is_test, max_trials)
            except Exception:
                result = None

            if result is not None:
                best_trial = result.get("best_trial", 0)
                best_loss = result.get("loss", float("inf"))
                progress.mark(
                    dataset, "train", "done",
                    model=model_key,
                    best_trial=best_trial,
                    loss=best_loss,
                )
                successes.append({
                    "model": gm,
                    "loss_name": loss,
                    "key": model_key,
                    "best_trial": best_trial,
                    "best_loss": best_loss,
                })
            else:
                progress.mark(dataset, "train", "failed", model=model_key)

    if successes:
        best = min(successes, key=lambda c: c["best_loss"])
        _write_best_symlink(dataset, best["best_trial"], best["model"], best["loss_name"])

    return StepResult(ok=bool(successes))


def step_evaluate(dataset: str, config_path: str) -> StepResult:
    """Rank all trained combos by loss; write eval_summary.json + best_{family}/ symlinks."""
    models_status = (
        progress.load(dataset)
        .get("steps", {}).get("train", {}).get("models", {})
    )
    done = {
        k: v for k, v in models_status.items()
        if v.get("status") == "done"
        and v.get("loss") is not None
        and v.get("loss") != float("inf")
    }

    if not done:
        progress.mark(dataset, "evaluate", "done")
        return StepResult(ok=True, data={})

    family_best: dict[str, dict] = {}
    for model_key, entry in done.items():
        family = _model_family(model_key)
        loss = entry.get("loss", float("inf"))
        if family not in family_best or loss < family_best[family]["loss"]:
            family_best[family] = {
                "model_key": model_key,
                "best_trial": entry.get("best_trial", 0),
                "loss": loss,
            }

    global_best = min(family_best.values(), key=lambda x: x["loss"])

    for info in family_best.values():
        _write_family_symlink(dataset, info["best_trial"], info["model_key"])

    summary = {
        "family_best": family_best,
        "global_best": global_best,
        "all_models": {
            k: {"best_trial": v.get("best_trial", 0), "loss": v.get("loss")}
            for k, v in done.items()
        },
    }
    summary_path = Path("database/prepared") / dataset / "eval_summary.json"
    def _sanitize(obj):
        if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize(summary), fh, indent=2)

    progress.mark(dataset, "evaluate", "done")
    return StepResult(ok=True, data=summary)


def step_postprocess(
    dataset: str,
    config_path: str,
    _dataset_cls=None,
) -> StepResult:
    """Postprocess family-best synthetic CSVs and write synthetic_{family}.csv + synthetic_final.csv.

    Reads eval_summary.json written by step_evaluate.  If the file is absent or
    a synthetic_full.csv is missing, the corresponding output is skipped (no error).
    *_dataset_cls* is injectable for unit tests.
    """
    import pandas as pd

    if _dataset_cls is None:
        from engine.dataset_helper.generic import GenericDataset  # noqa: lazy
        _dataset_cls = GenericDataset

    models_status = (
        progress.load(dataset)
        .get("steps", {})
        .get("train", {})
        .get("models", {})
    )

    done_models = [k for k, v in models_status.items() if v.get("status") == "done"]
    if not done_models:
        progress.mark(dataset, "postprocess", "failed")
        return StepResult(ok=False, error=RuntimeError("No successfully trained models found"))

    prepared_dir = Path("database/prepared") / dataset
    summary_path = prepared_dir / "eval_summary.json"

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as fh:
            eval_summary = json.load(fh)
        family_best = eval_summary.get("family_best", {})
        global_best = eval_summary.get("global_best", {})
        prepared_dir.mkdir(parents=True, exist_ok=True)

        for family, info in family_best.items():
            model_key = info["model_key"]
            model, loss = model_key.split("-", 1)
            synth_path = get_run_dir(
                dataset, _norm_model(model), _LOSS_TO_LV.get(loss, loss), info["best_trial"]
            ) / "synthetic_full.csv"
            if not synth_path.exists():
                continue
            ds = _dataset_cls(config_path)
            df_post = ds.postprocess_synthetic(pd.read_csv(synth_path))
            df_post.to_csv(prepared_dir / f"synthetic_{family}.csv", index=False)

        if global_best:
            model_key = global_best["model_key"]
            model, loss = model_key.split("-", 1)
            synth_path = get_run_dir(
                dataset, _norm_model(model), _LOSS_TO_LV.get(loss, loss), global_best["best_trial"]
            ) / "synthetic_full.csv"
            if synth_path.exists():
                ds = _dataset_cls(config_path)
                df_post = ds.postprocess_synthetic(pd.read_csv(synth_path))
                df_post.to_csv(prepared_dir / "synthetic_final.csv", index=False)

    progress.mark(dataset, "postprocess", "done")
    return StepResult(ok=True)


def run(
    config_path: str,
    _dataset_cls=None,
    is_test: int = 0,
    max_trials: int = 30,
) -> dict[str, Any]:
    """Execute the full pipeline from *config_path* and return the final status dict."""
    from engine.config_loader import load_config  # noqa: lazy

    cfg = load_config(config_path)
    dataset = Path(config_path).stem
    model_keys = [f"{gm}-{loss}" for gm in cfg.training.gms for loss in cfg.training.losses]
    progress.init(dataset, config_path, model_keys)

    step_preprocess(dataset, config_path, _dataset_cls=_dataset_cls)
    step_train(dataset, config_path, cfg.training.gms, cfg.training.losses, is_test, max_trials)
    step_evaluate(dataset, config_path)
    step_postprocess(dataset, config_path, _dataset_cls=_dataset_cls)

    return progress.load(dataset)
