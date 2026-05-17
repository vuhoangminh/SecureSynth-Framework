"""Tests for t042c: step_evaluate + step_postprocess."""
import json
from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# step_evaluate
# ---------------------------------------------------------------------------

class TestStepEvaluate:
    def test_returns_ok(self, progress_tmp):
        progress.init("clinical", "c.toml", [])
        result = step_evaluate("clinical", "c.toml")
        assert isinstance(result, StepResult)
        assert result.ok is True

    def test_marks_evaluate_done(self, progress_tmp):
        progress.init("ds", "c.toml", [])
        step_evaluate("ds", "c.toml")
        assert progress.is_done("ds", "evaluate")

    def test_no_done_models_returns_ok_empty_data(self, progress_tmp, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        progress.init("ds", "c.toml", ["CTGAN-vanilla"])
        progress.mark("ds", "train", "failed", model="CTGAN-vanilla")
        result = step_evaluate("ds", "c.toml")
        assert result.ok is True
        assert result.data == {}

    def test_eval_summary_json_written(self, progress_tmp, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        progress.init("ds", "c.toml", ["CTGAN-vanilla", "TabSyn-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)
        progress.mark("ds", "train", "done", model="TabSyn-vanilla", best_trial=1, loss=0.3)

        result = step_evaluate("ds", "c.toml")
        assert result.ok is True

        summary_path = tmp_path / "database" / "prepared" / "ds" / "eval_summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert "family_best" in data
        assert "global_best" in data
        assert "all_models" in data

    def test_family_best_selects_lowest_loss(self, progress_tmp, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        progress.init("ds", "c.toml", ["CTGAN-vanilla", "CTGAN-cd"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.8)
        progress.mark("ds", "train", "done", model="CTGAN-cd", best_trial=2, loss=0.4)

        step_evaluate("ds", "c.toml")

        data = json.loads((tmp_path / "database/prepared/ds/eval_summary.json").read_text())
        # Both models are in the "ctgan" family; cd variant wins
        assert data["family_best"]["ctgan"]["model_key"] == "CTGAN-cd"
        assert data["family_best"]["ctgan"]["loss"] == 0.4
        assert data["global_best"]["model_key"] == "CTGAN-cd"

    def test_family_symlinks_created(self, progress_tmp, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        progress.init("ds", "c.toml", ["CTGAN-vanilla", "TabSyn-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)
        progress.mark("ds", "train", "done", model="TabSyn-vanilla", best_trial=1, loss=0.3)

        step_evaluate("ds", "c.toml")

        assert (tmp_path / "database/prepared/ds/best_ctgan").is_symlink()
        assert (tmp_path / "database/prepared/ds/best_tabsyn").is_symlink()

    def test_result_data_dict(self, progress_tmp, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        progress.init("ds", "c.toml", ["CTGAN-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)

        result = step_evaluate("ds", "c.toml")

        assert isinstance(result.data, dict)
        assert "family_best" in result.data
        assert "global_best" in result.data


# ---------------------------------------------------------------------------
# step_postprocess helpers
# ---------------------------------------------------------------------------

def _make_run_dir(base, dataset, model, loss, best_trial=0, content="a,b\n1,2\n3,4\n"):
    run_dir = base / "database" / "runs" / f"{dataset}-{model}-{loss}" / f"trial_{best_trial:04d}"
    run_dir.mkdir(parents=True)
    (run_dir / "synthetic_full.csv").write_text(content)
    return run_dir


def _mock_run_dir(base):
    return lambda ds, model, loss, trial, **kw: (
        base / "database" / "runs" / f"{ds}-{model}-{loss}" / f"trial_{trial:04d}"
    )


def _write_eval_summary(base, dataset, family_best, global_best):
    p = base / "database" / "prepared" / dataset
    p.mkdir(parents=True, exist_ok=True)
    data = {
        "family_best": family_best,
        "global_best": global_best,
        "all_models": {info["model_key"]: {"best_trial": info["best_trial"], "loss": info["loss"]}
                       for info in family_best.values()},
    }
    (p / "eval_summary.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# step_postprocess
# ---------------------------------------------------------------------------

class TestStepPostprocess:
    def test_postprocess_called_per_family(self, progress_tmp, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("engine.orchestrator.get_run_dir", _mock_run_dir(tmp_path))

        progress.init("ds", "c.toml", ["CTGAN-vanilla", "TabSyn-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)
        progress.mark("ds", "train", "done", model="TabSyn-vanilla", best_trial=0, loss=0.4)

        _make_run_dir(tmp_path, "ds", "ctgan", "lv0")
        _make_run_dir(tmp_path, "ds", "tabsyn", "lv0")
        _write_eval_summary(tmp_path, "ds",
            family_best={
                "ctgan":  {"model_key": "CTGAN-vanilla",  "best_trial": 0, "loss": 0.5},
                "tabsyn": {"model_key": "TabSyn-vanilla", "best_trial": 0, "loss": 0.4},
            },
            global_best={"model_key": "TabSyn-vanilla", "best_trial": 0, "loss": 0.4},
        )

        mock_cls = MagicMock()
        mock_cls.return_value.postprocess_synthetic.return_value = pd.DataFrame({"a": [1], "b": [2]})

        step_postprocess("ds", "c.toml", _dataset_cls=mock_cls)

        assert mock_cls.return_value.postprocess_synthetic.call_count == 3  # ctgan + tabsyn + global
        assert (tmp_path / "database/prepared/ds/synthetic_ctgan.csv").exists()
        assert (tmp_path / "database/prepared/ds/synthetic_tabsyn.csv").exists()

    def test_final_csv_written(self, progress_tmp, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("engine.orchestrator.get_run_dir", _mock_run_dir(tmp_path))

        progress.init("ds", "c.toml", ["CTGAN-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)
        _make_run_dir(tmp_path, "ds", "ctgan", "lv0")
        _write_eval_summary(tmp_path, "ds",
            family_best={"ctgan": {"model_key": "CTGAN-vanilla", "best_trial": 0, "loss": 0.5}},
            global_best={"model_key": "CTGAN-vanilla", "best_trial": 0, "loss": 0.5},
        )

        mock_cls = MagicMock()
        mock_cls.return_value.postprocess_synthetic.return_value = pd.DataFrame({"a": [1]})

        result = step_postprocess("ds", "c.toml", _dataset_cls=mock_cls)
        assert result.ok is True
        assert (tmp_path / "database/prepared/ds/synthetic_final.csv").exists()

    def test_marks_postprocess_done(self, progress_tmp, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("engine.orchestrator.get_run_dir", _mock_run_dir(tmp_path))

        progress.init("ds", "c.toml", ["CTGAN-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)
        _make_run_dir(tmp_path, "ds", "ctgan", "lv0")
        _write_eval_summary(tmp_path, "ds",
            family_best={"ctgan": {"model_key": "CTGAN-vanilla", "best_trial": 0, "loss": 0.5}},
            global_best={"model_key": "CTGAN-vanilla", "best_trial": 0, "loss": 0.5},
        )

        mock_cls = MagicMock()
        mock_cls.return_value.postprocess_synthetic.return_value = pd.DataFrame({"a": [1]})

        result = step_postprocess("ds", "c.toml", _dataset_cls=mock_cls)
        assert result.ok is True
        assert progress.is_done("ds", "postprocess")

    def test_succeeds_when_done_models_but_no_csvs(self, progress_tmp, tmp_path, monkeypatch):
        # done models + no eval_summary.json (test/non-full mode) — should not fail
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("engine.orchestrator.get_run_dir", _mock_run_dir(tmp_path))

        progress.init("ds", "c.toml", ["CTGAN-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)

        result = step_postprocess("ds", "c.toml", _dataset_cls=MagicMock())
        assert result.ok is True
        assert progress.load("ds")["steps"]["postprocess"]["status"] == "done"

    def test_marks_postprocess_failed_when_no_done_models(self, progress_tmp, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("engine.orchestrator.get_run_dir", _mock_run_dir(tmp_path))

        progress.init("ds", "c.toml", ["CTGAN-vanilla"])
        progress.mark("ds", "train", "failed", model="CTGAN-vanilla")

        result = step_postprocess("ds", "c.toml", _dataset_cls=MagicMock())
        assert result.ok is False
        assert progress.load("ds")["steps"]["postprocess"]["status"] == "failed"

    def test_no_csv_graceful(self, progress_tmp, tmp_path, monkeypatch):
        # eval_summary.json present but synthetic_full.csv absent — graceful no-op
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("engine.orchestrator.get_run_dir", _mock_run_dir(tmp_path))

        progress.init("ds", "c.toml", ["CTGAN-vanilla"])
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=0, loss=0.5)
        _write_eval_summary(tmp_path, "ds",
            family_best={"ctgan": {"model_key": "CTGAN-vanilla", "best_trial": 0, "loss": 0.5}},
            global_best={"model_key": "CTGAN-vanilla", "best_trial": 0, "loss": 0.5},
        )
        # No synthetic_full.csv created

        result = step_postprocess("ds", "c.toml", _dataset_cls=MagicMock())
        assert result.ok is True
        assert not (tmp_path / "database/prepared/ds/synthetic_final.csv").exists()
