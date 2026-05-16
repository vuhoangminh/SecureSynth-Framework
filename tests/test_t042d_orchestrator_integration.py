"""Integration smoke test for the full orchestrator pipeline (t042d).

All heavy I/O is mocked so no GPU or real data is needed.  The test
verifies that run() wires the four steps in order and that
pipeline_status.json is written with the expected structure.
"""
import json
import textwrap
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import engine.progress as progress
from engine.orchestrator import StepResult, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_train_proc(best_trial: int = 0, loss: float = 0.3) -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = f'{{"best_trial": {best_trial}, "loss": {loss}}}\n'
    return m


def _ok_proc() -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = ""
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def progress_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(
        progress, "_status_path",
        lambda ds: tmp_path / ds / progress._STATUS_FILENAME,
    )
    return tmp_path


@pytest.fixture()
def minimal_config(tmp_path):
    """Minimal TOML + in-line CSV, dataset name = 'testds'."""
    csv = tmp_path / "testds.csv"
    csv.write_text("age,sex,y\n30,0,1\n40,1,0\n35,0,1\n")
    toml = tmp_path / "testds.toml"
    toml.write_text(textwrap.dedent(f"""\
        [data]
        path = "{csv}"

        [columns]
        target = "y"
        task = "classification"
        continuous = ["age"]
        discrete = ["sex"]

        [training]
        gms = ["CTGAN"]
        losses = ["vanilla"]
    """))
    return toml


# ---------------------------------------------------------------------------
# t042d tests
# ---------------------------------------------------------------------------

def test_run_writes_pipeline_status_json(minimal_config, progress_tmp, tmp_path, monkeypatch):
    """run() must produce a valid pipeline_status.json with all step entries."""
    monkeypatch.setattr(
        "engine.orchestrator.get_run_dir",
        lambda ds, model, loss, trial, **kw: (
            tmp_path / "runs" / f"{ds}-{model}-{loss}" / f"trial_{trial:04d}"
        ),
    )

    mock_cls = MagicMock()
    mock_cls.return_value.postprocess_synthetic.return_value = pd.DataFrame(
        {"age": [30], "sex": [0], "y": [1]}
    )

    # First subprocess.run call is from _dispatch_combo (train), subsequent from step_evaluate.
    with patch("engine.orchestrator.subprocess.run",
               side_effect=[_ok_train_proc(), _ok_proc()]), \
         patch("engine.orchestrator._write_best_symlink"):
        run(str(minimal_config), _dataset_cls=mock_cls)

    status_file = progress_tmp / "testds" / progress._STATUS_FILENAME
    assert status_file.exists(), "pipeline_status.json was not written"

    data = json.loads(status_file.read_text())
    assert data["dataset"] == "testds"
    assert data["config"] == str(minimal_config)
    assert "started_at" in data

    assert data["steps"]["preprocess"]["status"] == "done"
    assert data["steps"]["train"]["models"]["CTGAN-vanilla"]["status"] == "done"
    assert data["steps"]["train"]["models"]["CTGAN-vanilla"]["best_trial"] == 0
    assert data["steps"]["train"]["models"]["CTGAN-vanilla"]["loss"] == pytest.approx(0.3)
    assert data["steps"]["evaluate"]["status"] == "done"
    # postprocess fails because no synthetic_full.csv exists (expected)
    assert data["steps"]["postprocess"]["status"] in ("done", "failed")


def test_run_all_steps_execute_regardless_of_preprocess_failure(
    minimal_config, progress_tmp, tmp_path, monkeypatch
):
    """Even if preprocess raises, train/evaluate/postprocess still run."""
    monkeypatch.setattr(
        "engine.orchestrator.get_run_dir",
        lambda ds, model, loss, trial, **kw: (
            tmp_path / "runs" / f"{ds}-{model}-{loss}" / f"trial_{trial:04d}"
        ),
    )

    bad_cls = MagicMock(side_effect=RuntimeError("bad data"))

    with patch("engine.orchestrator.subprocess.run",
               side_effect=[_ok_train_proc(), _ok_proc()]), \
         patch("engine.orchestrator._write_best_symlink"):
        run(str(minimal_config), _dataset_cls=bad_cls)

    data = json.loads(
        (progress_tmp / "testds" / progress._STATUS_FILENAME).read_text()
    )
    assert data["steps"]["preprocess"]["status"] == "failed"
    # Train should still have been attempted
    assert "CTGAN-vanilla" in data["steps"]["train"]["models"]
    assert data["steps"]["evaluate"]["status"] in ("done", "failed")


def test_run_returns_status_dict(minimal_config, progress_tmp, tmp_path, monkeypatch):
    """run() must return the same dict that is written to pipeline_status.json."""
    monkeypatch.setattr(
        "engine.orchestrator.get_run_dir",
        lambda ds, model, loss, trial, **kw: (
            tmp_path / "runs" / f"{ds}-{model}-{loss}" / f"trial_{trial:04d}"
        ),
    )

    with patch("engine.orchestrator.subprocess.run",
               side_effect=[_ok_train_proc(), _ok_proc()]), \
         patch("engine.orchestrator._write_best_symlink"):
        result = run(str(minimal_config), _dataset_cls=MagicMock())

    assert isinstance(result, dict)
    assert result["dataset"] == "testds"
    assert "steps" in result
