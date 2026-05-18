import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STATUS_FILENAME = "pipeline_status.json"

_STEP_KEYS = ("preprocess", "train", "evaluate", "postprocess")
_VALID_STATUSES = {"pending", "running", "done", "failed"}


def _status_path(dataset: str, is_test: bool = False) -> Path:
    base = Path("database/prepared") / dataset
    return (base / "test" / _STATUS_FILENAME) if is_test else (base / _STATUS_FILENAME)


def load(dataset: str, is_test: bool = False) -> dict[str, Any]:
    """Load the status JSON for *dataset*, or return an empty skeleton."""
    p = _status_path(dataset, is_test)
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save(dataset: str, status: dict[str, Any], is_test: bool = False) -> None:
    p = _status_path(dataset, is_test)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)


def init(dataset: str, config_path: str, model_loss_keys: list[str], is_test: bool = False) -> dict[str, Any]:
    """Create (or overwrite) a fresh status record and persist it.

    *model_loss_keys* is a list of strings like ``"CTGAN-vanilla"`` that
    identify every model×loss combination the pipeline will train.
    """
    status = {
        "dataset": dataset,
        "config": config_path,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {
            "preprocess": {"status": "pending"},
            "train": {
                "models": {
                    key: {"status": "pending", "best_trial": None, "loss": None, "reason": None}
                    for key in model_loss_keys
                }
            },
            "evaluate": {"status": "pending"},
            "postprocess": {"status": "pending"},
        },
    }
    save(dataset, status, is_test)
    return status


def is_done(dataset: str, step: str, model: str | None = None, is_test: bool = False) -> bool:
    """Return True if *step* (and optionally *model* within the train step) is completed."""
    status = load(dataset, is_test)
    if not status:
        return False
    steps = status.get("steps", {})
    if step == "train" and model is not None:
        return (
            steps.get("train", {})
            .get("models", {})
            .get(model, {})
            .get("status") == "done"
        )
    return steps.get(step, {}).get("status") == "done"


def mark(dataset: str, step: str, status_value: str, model: str | None = None, is_test: bool = False, **kwargs: Any) -> None:
    """Update the status of *step* (or a specific *model* within train) and persist.

    Extra keyword arguments (e.g. ``best_trial``, ``loss``, ``reason``) are
    merged into the model entry when ``model`` is supplied.
    """
    if status_value not in _VALID_STATUSES:
        raise ValueError(f"status must be one of {_VALID_STATUSES}, got {status_value!r}")

    record = load(dataset, is_test)
    if not record:
        raise FileNotFoundError(
            f"No pipeline_status.json found for dataset {dataset!r}. Call init() first."
        )

    steps = record.setdefault("steps", {})

    if step == "train" and model is not None:
        models = steps.setdefault("train", {}).setdefault("models", {})
        entry = models.setdefault(model, {"status": "pending", "best_trial": None, "loss": None, "reason": None})
        entry["status"] = status_value
        entry.update(kwargs)
    else:
        step_entry = steps.setdefault(step, {})
        step_entry["status"] = status_value
        step_entry.update(kwargs)

    save(dataset, record, is_test)
