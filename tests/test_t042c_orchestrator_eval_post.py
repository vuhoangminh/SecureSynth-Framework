"""Tests for t042c: step_evaluate + step_postprocess."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import engine.progress as progress
from engine.orchestrator import StepResult, step_evaluate, step_postprocess


@pytest.fixture()
def progress_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(
        progress, "_status_path",
        lambda ds: tmp_path / ds / progress._STATUS_FILENAME,
    )
    return tmp_path


def _ok_proc():
    m = MagicMock()
    m.returncode = 0
    m.stdout = ""
    return m


def _fail_proc():
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    return m


# ---------------------------------------------------------------------------
# step_evaluate
# ---------------------------------------------------------------------------

class TestStepEvaluate:
    def test_returns_ok_on_success(self, progress_tmp):
        progress.init("ds", "c.toml", [])
        with patch("engine.orchestrator.subprocess.run", return_value=_ok_proc()):
            result = step_evaluate("ds", "c.toml")
        assert isinstance(result, StepResult)
        assert result.ok is True

    def test_marks_evaluate_done_on_success(self, progress_tmp):
        progress.init("ds", "c.toml", [])
        with patch("engine.orchestrator.subprocess.run", return_value=_ok_proc()):
            step_evaluate("ds", "c.toml")
        assert progress.is_done("ds", "evaluate")

    def test_marks_evaluate_failed_on_nonzero_exit(self, progress_tmp):
        progress.init("ds", "c.toml", [])
        with patch("engine.orchestrator.subprocess.run", return_value=_fail_proc()):
            result = step_evaluate("ds", "c.toml")
        assert result.ok is False
        assert progress.load("ds")["steps"]["evaluate"]["status"] == "failed"

    def test_subprocess_receives_dataset_name(self, progress_tmp):
        progress.init("ds", "c.toml", [])
        with patch("engine.orchestrator.subprocess.run", return_value=_ok_proc()) as mock_run:
            step_evaluate("ds", "c.toml")
        args = mock_run.call_args[0][0]
        assert "ds" in args


# ---------------------------------------------------------------------------
# step_postprocess helpers
# ---------------------------------------------------------------------------

def _make_run_dir(tmp_path, dataset, model, loss, best_trial=0):
    run_dir = tmp_path / "runs" / f"{dataset}-{model}-{loss}" / f"trial_{best_trial:04d}"
    run_dir.mkdir(parents=True)
    (run_dir / "synthetic_full.csv").write_text("a,b\n1,2\n3,4\n")
    return run_dir


def _mock_run_dir(tmp_path):
    return lambda ds, model, loss, trial, **kw: (
        tmp_path / "runs" / f"{ds}-{model}-{loss}" / f"trial_{trial:04d}"
    )


# ---------------------------------------------------------------------------
# step_postprocess
# ---------------------------------------------------------------------------

class TestStepPostprocess:
    def test_postprocess_called_per_done_model(self, progress_tmp, tmp_path, monkeypatch):
        progress.init("ds", "c.toml", ["CTGAN-vanilla", "TabSyn-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)
        progress.mark("ds", "train", "done", model="TabSyn-vanilla", best_trial=0, loss=0.4)

        _make_run_dir(tmp_path, "ds", "ctgan", "lv0")
        _make_run_dir(tmp_path, "ds", "tabsyn", "lv0")
        monkeypatch.setattr("engine.orchestrator.get_run_dir", _mock_run_dir(tmp_path))

        mock_cls = MagicMock()
        mock_cls.return_value.postprocess_synthetic.return_value = pd.DataFrame({"a": [1], "b": [2]})

        step_postprocess("ds", "c.toml", _dataset_cls=mock_cls)

        assert mock_cls.return_value.postprocess_synthetic.call_count == 2

    def test_marks_postprocess_done_when_csvs_exist(self, progress_tmp, tmp_path, monkeypatch):
        progress.init("ds", "c.toml", ["CTGAN-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)
        _make_run_dir(tmp_path, "ds", "ctgan", "lv0")
        monkeypatch.setattr("engine.orchestrator.get_run_dir", _mock_run_dir(tmp_path))

        mock_cls = MagicMock()
        mock_cls.return_value.postprocess_synthetic.return_value = pd.DataFrame({"a": [1]})

        result = step_postprocess("ds", "c.toml", _dataset_cls=mock_cls)
        assert result.ok is True
        assert progress.is_done("ds", "postprocess")

    def test_marks_postprocess_failed_when_no_csvs(self, progress_tmp, tmp_path, monkeypatch):
        progress.init("ds", "c.toml", ["CTGAN-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)
        monkeypatch.setattr("engine.orchestrator.get_run_dir", _mock_run_dir(tmp_path))

        result = step_postprocess("ds", "c.toml", _dataset_cls=MagicMock())
        assert result.ok is False
        assert progress.load("ds")["steps"]["postprocess"]["status"] == "failed"

    def test_skips_failed_models(self, progress_tmp, tmp_path, monkeypatch):
        progress.init("ds", "c.toml", ["CTGAN-vanilla", "TabSyn-vanilla"])
        progress.mark("ds", "train", "failed", model="CTGAN-vanilla")
        progress.mark("ds", "train", "done", model="TabSyn-vanilla", best_trial=0, loss=0.4)

        _make_run_dir(tmp_path, "ds", "tabsyn", "lv0")
        monkeypatch.setattr("engine.orchestrator.get_run_dir", _mock_run_dir(tmp_path))

        mock_cls = MagicMock()
        mock_cls.return_value.postprocess_synthetic.return_value = pd.DataFrame({"a": [1]})

        step_postprocess("ds", "c.toml", _dataset_cls=mock_cls)

        assert mock_cls.return_value.postprocess_synthetic.call_count == 1
