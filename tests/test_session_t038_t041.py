"""Tests for t038–t041 (session 2026-05-16).

t038 — path_utils.get_run_dir()
t039 — database/dataset → database/prepared in base.py / generic.py
t040 — engine/config_validator.validate_config()
t041 — engine/progress read/write/is_done/mark
"""

import csv
import json
import tempfile
import warnings
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# t038 — get_run_dir
# ---------------------------------------------------------------------------

from engine.utils.path_utils import get_run_dir, get_folder_technical_paper


class TestGetRunDir:
    def test_normal_path(self):
        p = get_run_dir("adult", "CTGAN", "vanilla", 0)
        assert p == Path("database/runs/adult-CTGAN-vanilla/trial_0000")

    def test_trial_zero_padded(self):
        p = get_run_dir("adult", "TabSyn", "cd", 42)
        assert str(p).endswith("trial_0042")

    def test_is_test_flag(self):
        p = get_run_dir("adult", "CTGAN", "vanilla", 1, is_test=True)
        assert "runs_test" in str(p)
        assert "runs/" not in str(p)

    def test_returns_path_object(self):
        assert isinstance(get_run_dir("ds", "m", "l", 0), Path)

    def test_get_folder_technical_paper_deprecated(self):
        """get_folder_technical_paper must emit DeprecationWarning."""
        import argparse
        args = argparse.Namespace(
            loss_version=0, is_test=False, is_drop_id=True,
            dataset="adult", arch="ctgan", private=False,
            batch_size=500, epochs=300,
            embedding_dim=128, discriminator_dim=256, generator_dim=256,
            is_loss_corr=0.0, is_loss_dwp=0.0, is_condvec=1,
            row_number=None, generator_lr=2e-4,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_folder_technical_paper(args)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


# ---------------------------------------------------------------------------
# t039 — database/dataset → database/prepared (source-level check)
# ---------------------------------------------------------------------------

class TestPathRenaming:
    def _grep(self, filepath, pattern):
        text = Path(filepath).read_text()
        return [line.strip() for line in text.splitlines() if pattern in line]

    def test_base_py_no_hardcoded_dataset_path(self):
        hits = self._grep(
            "engine/dataset_helper/base.py", '"database/dataset'
        )
        assert hits == [], f"Remaining hardcoded database/dataset refs: {hits}"

    def test_generic_py_no_hardcoded_dataset_path(self):
        hits = self._grep(
            "engine/dataset_helper/generic.py", '"database/dataset'
        )
        assert hits == [], f"Remaining hardcoded database/dataset refs: {hits}"

    def test_base_py_has_prepared(self):
        text = Path("engine/dataset_helper/base.py").read_text()
        assert "database/prepared" in text

    def test_generic_py_has_prepared(self):
        text = Path("engine/dataset_helper/generic.py").read_text()
        assert "database/prepared" in text


# ---------------------------------------------------------------------------
# t040 — config_validator.validate_config
# ---------------------------------------------------------------------------

from engine.config_loader import (
    DataConfig, ColumnsConfig, AttributesConfig,
    PreprocessingConfig, TrainingConfig, DifferentialPrivacyConfig,
    PostprocessingConfig, OutputConfig, PipelineConfig,
)
from engine.config_validator import validate_config


def _make_cfg(**overrides) -> PipelineConfig:
    """Build a minimal valid PipelineConfig, applying any overrides."""
    defaults = dict(
        data=DataConfig(path="PLACEHOLDER"),
        columns=ColumnsConfig(
            target="y",
            task="classification",
            continuous=["age", "bmi"],
            discrete=["sex"],
        ),
        attributes=AttributesConfig(),
        preprocessing=PreprocessingConfig(),
        training=TrainingConfig(gms=["CTGAN"], losses=["vanilla"]),
        differential_privacy=DifferentialPrivacyConfig(),
        postprocessing=PostprocessingConfig(),
        output=OutputConfig(),
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


@pytest.fixture()
def tmp_csv(tmp_path):
    """Write a tiny CSV and return its path string."""
    p = tmp_path / "data.csv"
    with open(p, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["age", "bmi", "sex", "y"])
        for i in range(20):
            writer.writerow([30 + i, 22.0 + i * 0.1, i % 2, i % 2])
    return str(p)


class TestValidateConfig:
    def test_valid_config_no_issues(self, tmp_csv):
        cfg = _make_cfg(data=DataConfig(path=tmp_csv))
        assert validate_config(cfg) == []

    def test_missing_data_path(self):
        cfg = _make_cfg(data=DataConfig(path="/nonexistent/file.csv"))
        issues = validate_config(cfg)
        assert any("not found" in i for i in issues)

    def test_column_missing_from_file(self, tmp_csv):
        cfg = _make_cfg(
            data=DataConfig(path=tmp_csv),
            columns=ColumnsConfig(
                target="y", task="classification",
                continuous=["age", "bmi", "MISSING_COL"], discrete=["sex"],
            ),
        )
        issues = validate_config(cfg)
        assert any("MISSING_COL" in i for i in issues)

    def test_bad_task(self, tmp_csv):
        cfg = _make_cfg(
            data=DataConfig(path=tmp_csv),
            columns=ColumnsConfig(target="y", task="clustering",
                                  continuous=["age", "bmi"], discrete=["sex"]),
        )
        issues = validate_config(cfg)
        assert any("task" in i for i in issues)

    def test_bad_gm(self, tmp_csv):
        cfg = _make_cfg(
            data=DataConfig(path=tmp_csv),
            training=TrainingConfig(gms=["CTGAN", "FakeGAN"], losses=["vanilla"]),
        )
        issues = validate_config(cfg)
        assert any("FakeGAN" in i for i in issues)

    def test_bad_loss(self, tmp_csv):
        cfg = _make_cfg(
            data=DataConfig(path=tmp_csv),
            training=TrainingConfig(gms=["CTGAN"], losses=["vanilla", "badloss"]),
        )
        issues = validate_config(cfg)
        assert any("badloss" in i for i in issues)

    def test_invalid_query_constraint(self, tmp_csv):
        cfg = _make_cfg(
            data=DataConfig(path=tmp_csv),
            postprocessing=PostprocessingConfig(constraints=["age >>> bmi"]),
        )
        issues = validate_config(cfg)
        assert any("constraint" in i for i in issues)

    def test_valid_query_constraint(self, tmp_csv):
        cfg = _make_cfg(
            data=DataConfig(path=tmp_csv),
            postprocessing=PostprocessingConfig(constraints=["age > 18"]),
        )
        assert validate_config(cfg) == []

    def test_target_in_drop_columns(self, tmp_csv):
        cfg = _make_cfg(
            data=DataConfig(path=tmp_csv, drop_columns=["y"]),
            columns=ColumnsConfig(target="y", task="classification",
                                  continuous=["age", "bmi"], discrete=["sex"]),
        )
        issues = validate_config(cfg)
        assert any("drop_columns" in i for i in issues)

    def test_dp_delta_warning(self, tmp_csv):
        # 20 rows → threshold 0.05; delta=0.1 should warn
        cfg = _make_cfg(
            data=DataConfig(path=tmp_csv),
            differential_privacy=DifferentialPrivacyConfig(
                enabled=True, delta=0.1
            ),
        )
        issues = validate_config(cfg)
        assert any("delta" in i and "WARNING" in i for i in issues)

    def test_dp_delta_ok(self, tmp_csv):
        cfg = _make_cfg(
            data=DataConfig(path=tmp_csv),
            differential_privacy=DifferentialPrivacyConfig(
                enabled=True, delta=1e-5
            ),
        )
        issues = validate_config(cfg)
        assert not any("delta" in i and "WARNING" in i for i in issues)


# ---------------------------------------------------------------------------
# t041 — progress module
# ---------------------------------------------------------------------------

import engine.progress as progress


@pytest.fixture()
def patched_progress(tmp_path, monkeypatch):
    """Redirect _status_path to a temp directory."""
    monkeypatch.setattr(
        progress, "_status_path",
        lambda ds: tmp_path / ds / progress._STATUS_FILENAME,
    )
    return tmp_path


class TestProgress:
    KEYS = ["CTGAN-vanilla", "TabSyn-cd"]

    def test_init_creates_file(self, patched_progress):
        rec = progress.init("ds", "configs/clinical.toml", self.KEYS)
        p = patched_progress / "ds" / progress._STATUS_FILENAME
        assert p.exists()
        assert rec["dataset"] == "ds"

    def test_all_steps_start_pending(self, patched_progress):
        progress.init("ds", "configs/clinical.toml", self.KEYS)
        for step in ("preprocess", "evaluate", "postprocess"):
            assert not progress.is_done("ds", step)
        for key in self.KEYS:
            assert not progress.is_done("ds", "train", model=key)

    def test_mark_step_done(self, patched_progress):
        progress.init("ds", "configs/clinical.toml", self.KEYS)
        progress.mark("ds", "preprocess", "done")
        assert progress.is_done("ds", "preprocess")

    def test_mark_model_done_with_metadata(self, patched_progress):
        progress.init("ds", "configs/clinical.toml", self.KEYS)
        progress.mark("ds", "train", "done", model="CTGAN-vanilla", best_trial=3, loss=0.55)
        assert progress.is_done("ds", "train", model="CTGAN-vanilla")
        assert not progress.is_done("ds", "train", model="TabSyn-cd")
        rec = progress.load("ds")
        entry = rec["steps"]["train"]["models"]["CTGAN-vanilla"]
        assert entry["best_trial"] == 3
        assert entry["loss"] == pytest.approx(0.55)

    def test_load_empty_dataset_returns_empty(self, patched_progress):
        assert progress.load("nonexistent") == {}

    def test_is_done_missing_dataset(self, patched_progress):
        assert not progress.is_done("nonexistent", "preprocess")

    def test_mark_invalid_status_raises(self, patched_progress):
        progress.init("ds", "configs/clinical.toml", self.KEYS)
        with pytest.raises(ValueError):
            progress.mark("ds", "preprocess", "invalid_status")

    def test_mark_before_init_raises(self, patched_progress):
        with pytest.raises(FileNotFoundError):
            progress.mark("ds", "preprocess", "done")

    def test_json_is_human_readable(self, patched_progress):
        progress.init("ds", "configs/clinical.toml", self.KEYS)
        p = patched_progress / "ds" / progress._STATUS_FILENAME
        data = json.loads(p.read_text())
        assert "started_at" in data
        assert "steps" in data
