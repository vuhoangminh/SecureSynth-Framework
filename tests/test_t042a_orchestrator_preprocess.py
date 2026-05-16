"""Tests for t042a: StepResult dataclass + orchestrator.step_preprocess.

GenericDataset is injected via _dataset_cls to avoid pulling in the full
model/GPU import chain (rdt, icecream, copulas, …) in the unit-test env.
"""
from unittest.mock import MagicMock

import pytest

import engine.progress as progress
from engine.orchestrator import StepResult, step_preprocess


@pytest.fixture()
def progress_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(
        progress, "_status_path",
        lambda ds: tmp_path / ds / progress._STATUS_FILENAME,
    )
    progress.init("ds", "configs/test.toml", [])
    return tmp_path


class TestStepResult:
    def test_ok_true(self):
        r = StepResult(ok=True)
        assert r.ok is True
        assert r.error is None

    def test_ok_false_with_error(self):
        exc = ValueError("bad")
        r = StepResult(ok=False, error=exc)
        assert r.ok is False
        assert r.error is exc


class TestStepPreprocess:
    def test_returns_step_result_ok_on_success(self, progress_tmp):
        result = step_preprocess("ds", "configs/test.toml", _dataset_cls=MagicMock())
        assert isinstance(result, StepResult)
        assert result.ok is True
        assert result.error is None

    def test_dataset_cls_called_with_config_path(self, progress_tmp):
        mock_cls = MagicMock()
        step_preprocess("ds", "configs/test.toml", _dataset_cls=mock_cls)
        mock_cls.assert_called_once_with("configs/test.toml")

    def test_marks_preprocess_done_on_success(self, progress_tmp):
        step_preprocess("ds", "configs/test.toml", _dataset_cls=MagicMock())
        assert progress.is_done("ds", "preprocess")

    def test_returns_not_ok_on_exception(self, progress_tmp):
        bad_cls = MagicMock(side_effect=RuntimeError("fail"))
        result = step_preprocess("ds", "configs/test.toml", _dataset_cls=bad_cls)
        assert isinstance(result, StepResult)
        assert result.ok is False
        assert isinstance(result.error, RuntimeError)

    def test_marks_preprocess_failed_on_exception(self, progress_tmp):
        bad_cls = MagicMock(side_effect=ValueError("oops"))
        step_preprocess("ds", "configs/test.toml", _dataset_cls=bad_cls)
        rec = progress.load("ds")
        assert rec["steps"]["preprocess"]["status"] == "failed"
