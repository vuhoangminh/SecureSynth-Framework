"""Tests for t042b: orchestrator.step_train."""
import json
from unittest.mock import MagicMock, call, patch

import pytest

import engine.progress as progress
from engine.orchestrator import StepResult, step_train


@pytest.fixture()
def progress_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(
        progress, "_status_path",
        lambda ds: tmp_path / ds / progress._STATUS_FILENAME,
    )
    return tmp_path


def _make_proc(returncode: int = 0, best_trial: int = 2, loss: float = 0.42) -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stdout = f"Training…\n{json.dumps({'best_trial': best_trial, 'loss': loss})}\n"
    return m


def _fail_proc() -> MagicMock:
    return _make_proc(returncode=1)


class TestStepTrainAllCombos:
    def test_all_combos_dispatched(self, progress_tmp):
        gms = ["CTGAN", "TabSyn"]
        losses = ["vanilla", "cd"]
        keys = [f"{g}-{l}" for g in gms for l in losses]
        progress.init("ds", "c.toml", keys)

        with patch("engine.orchestrator.subprocess.run", return_value=_make_proc()) as mock_run, \
             patch("engine.orchestrator._write_best_symlink"):
            step_train("ds", "c.toml", gms, losses)

        assert mock_run.call_count == len(gms) * len(losses)

    def test_one_failing_combo_does_not_abort_others(self, progress_tmp):
        progress.init("ds", "c.toml", ["CTGAN-vanilla", "TabSyn-vanilla"])

        with patch("engine.orchestrator.subprocess.run",
                   side_effect=[_fail_proc(), _make_proc()]), \
             patch("engine.orchestrator._write_best_symlink"):
            result = step_train("ds", "c.toml", ["CTGAN", "TabSyn"], ["vanilla"])

        assert result.ok is True
        rec = progress.load("ds")
        assert rec["steps"]["train"]["models"]["CTGAN-vanilla"]["status"] == "failed"
        assert rec["steps"]["train"]["models"]["TabSyn-vanilla"]["status"] == "done"

    def test_all_failing_returns_not_ok(self, progress_tmp):
        progress.init("ds", "c.toml", ["CTGAN-vanilla"])

        with patch("engine.orchestrator.subprocess.run", return_value=_fail_proc()):
            result = step_train("ds", "c.toml", ["CTGAN"], ["vanilla"])

        assert result.ok is False


class TestStepTrainBestSymlink:
    def test_best_symlink_written_after_success(self, progress_tmp):
        progress.init("ds", "c.toml", ["CTGAN-vanilla"])

        with patch("engine.orchestrator.subprocess.run",
                   return_value=_make_proc(best_trial=3, loss=0.3)), \
             patch("engine.orchestrator._write_best_symlink") as mock_sym:
            step_train("ds", "c.toml", ["CTGAN"], ["vanilla"])

        mock_sym.assert_called_once_with("ds", 3, "CTGAN", "vanilla")

    def test_best_symlink_points_to_lowest_loss_combo(self, progress_tmp):
        progress.init("ds", "c.toml", ["CTGAN-vanilla", "TabSyn-vanilla"])

        side_effects = [
            _make_proc(best_trial=1, loss=0.8),   # CTGAN-vanilla — worse
            _make_proc(best_trial=4, loss=0.2),   # TabSyn-vanilla — better
        ]
        with patch("engine.orchestrator.subprocess.run", side_effect=side_effects), \
             patch("engine.orchestrator._write_best_symlink") as mock_sym:
            step_train("ds", "c.toml", ["CTGAN", "TabSyn"], ["vanilla"])

        mock_sym.assert_called_once_with("ds", 4, "TabSyn", "vanilla")

    def test_best_symlink_not_written_when_all_fail(self, progress_tmp):
        progress.init("ds", "c.toml", ["CTGAN-vanilla"])

        with patch("engine.orchestrator.subprocess.run", return_value=_fail_proc()), \
             patch("engine.orchestrator._write_best_symlink") as mock_sym:
            step_train("ds", "c.toml", ["CTGAN"], ["vanilla"])

        mock_sym.assert_not_called()


class TestStepTrainProgressMark:
    def test_mark_done_with_model_key_and_metadata(self, progress_tmp):
        progress.init("ds", "c.toml", ["CTGAN-vanilla"])

        with patch("engine.orchestrator.subprocess.run",
                   return_value=_make_proc(best_trial=5, loss=0.55)), \
             patch("engine.orchestrator._write_best_symlink"):
            step_train("ds", "c.toml", ["CTGAN"], ["vanilla"])

        rec = progress.load("ds")
        entry = rec["steps"]["train"]["models"]["CTGAN-vanilla"]
        assert entry["status"] == "done"
        assert entry["best_trial"] == 5
        assert entry["loss"] == pytest.approx(0.55)

    def test_mark_failed_for_failing_combo(self, progress_tmp):
        progress.init("ds", "c.toml", ["CTGAN-vanilla"])

        with patch("engine.orchestrator.subprocess.run", return_value=_fail_proc()):
            step_train("ds", "c.toml", ["CTGAN"], ["vanilla"])

        rec = progress.load("ds")
        assert rec["steps"]["train"]["models"]["CTGAN-vanilla"]["status"] == "failed"
