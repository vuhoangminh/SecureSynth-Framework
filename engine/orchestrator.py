from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import engine.progress as progress
from engine.utils.path_utils import get_run_dir


@dataclass
class StepResult:
    ok: bool
    error: Exception | None = None


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
    dataset: str, model: str, loss: str, config_path: str
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
        ],
        capture_output=True,
        text=True,
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


def _write_best_symlink(dataset: str, best_trial: int, model: str, loss: str) -> None:
    link = Path("database/prepared") / dataset / "best"
    target = Path("../../runs") / f"{dataset}-{model}-{loss}" / f"trial_{best_trial:04d}"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def step_train(
    dataset: str,
    config_path: str,
    gms: list[str],
    losses: list[str],
) -> StepResult:
    """Iterate every model×loss combo, dispatch each as a subprocess, handle failures
    independently, then write a ``best/`` symlink to the combo with the lowest loss.
    """
    successes: list[dict[str, Any]] = []

    for gm in gms:
        for loss in losses:
            model_key = f"{gm}-{loss}"
            try:
                result = _dispatch_combo(dataset, gm, loss, config_path)
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
    """Run the data-sufficiency evaluation script for *dataset*."""
    proc = subprocess.run(
        [
            sys.executable, "scripts/analysis/data_sufficiency.py",
            "--dataset", dataset,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        progress.mark(dataset, "evaluate", "done")
        return StepResult(ok=True)
    progress.mark(dataset, "evaluate", "failed")
    return StepResult(ok=False)


def step_postprocess(
    dataset: str,
    config_path: str,
    _dataset_cls=None,
) -> StepResult:
    """Apply postprocess_synthetic() to every done model's best-trial synthetic CSV.

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

    processed = 0
    for model_key, entry in models_status.items():
        if entry.get("status") != "done":
            continue
        model, loss = model_key.split("-", 1)
        best_trial = entry.get("best_trial", 0)
        synth_path = get_run_dir(dataset, model, loss, best_trial) / "synthetic_full.csv"
        if not synth_path.exists():
            continue
        ds = _dataset_cls(config_path)
        df = pd.read_csv(synth_path)
        df_post = ds.postprocess_synthetic(df)
        df_post.to_csv(synth_path, index=False)
        processed += 1

    if processed > 0:
        progress.mark(dataset, "postprocess", "done")
        return StepResult(ok=True)
    progress.mark(dataset, "postprocess", "failed")
    return StepResult(ok=False)


def run(config_path: str, _dataset_cls=None) -> dict[str, Any]:
    """Execute the full pipeline from *config_path* and return the final status dict."""
    from engine.config_loader import load_config  # noqa: lazy

    cfg = load_config(config_path)
    dataset = Path(config_path).stem
    model_keys = [f"{gm}-{loss}" for gm in cfg.training.gms for loss in cfg.training.losses]
    progress.init(dataset, config_path, model_keys)

    step_preprocess(dataset, config_path, _dataset_cls=_dataset_cls)
    step_train(dataset, config_path, cfg.training.gms, cfg.training.losses)
    step_evaluate(dataset, config_path)
    step_postprocess(dataset, config_path, _dataset_cls=_dataset_cls)

    return progress.load(dataset)
