"""
Unit tests for GenericDataset (config-driven loader).

Covers:
  t015 — _encode_label(): discrete cols are float-encoded; inverse-transform recovers originals
  t016 — _get_type_columns() auto-detect: n_unique > 15 → continuous, else discrete
  t017 — load_config() validation: ValueError for missing [data], target, task
  t018 — dynamic registry: configs/dummy.toml → get_dataset('dummy') with no code changes
"""

import os
import pickle
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.config_loader import load_config
from engine.datasets import get_dataset


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _write_tiny_dataset(tmp_path, name, df, continuous, discrete, target, task):
    """Write a CSV + minimal TOML and the required database dirs under tmp_path."""
    (tmp_path / "data").mkdir(exist_ok=True)
    df.to_csv(tmp_path / "data" / f"{name}.csv", index=False)

    def toml_list(items):
        return "[" + ", ".join(f'"{c}"' for c in items) + "]"

    toml = textwrap.dedent(f"""\
        [data]
        path = "data/{name}.csv"
        format = "csv"

        [columns]
        continuous = {toml_list(continuous)}
        discrete   = {toml_list(discrete)}
        target     = "{target}"
        task       = "{task}"
    """)
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / f"{name}.toml").write_text(toml)

    (tmp_path / "database" / "dataset" / name).mkdir(parents=True)


# ---------------------------------------------------------------------------
# t015 — _encode_label()
# ---------------------------------------------------------------------------

class TestEncodeLabel:
    """Discrete cols (sex, diagnosis) must be float-encoded and inverse-transformable."""

    def _load(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(0)
        n = 60
        df = pd.DataFrame({
            "age":       rng.uniform(20, 80, n),
            "bmi":       rng.uniform(18, 40, n),
            "sex":       rng.choice(["M", "F"], n),
            "diagnosis": rng.choice(["TypeA", "TypeB", "TypeC"], n),
            "mortality": rng.integers(0, 2, n),
        })
        _write_tiny_dataset(tmp_path, "tiny",
                            df=df,
                            continuous=["age", "bmi"],
                            discrete=["sex", "diagnosis"],
                            target="mortality",
                            task="classification")
        monkeypatch.chdir(tmp_path)
        return get_dataset("tiny", is_encode=True)

    def test_discrete_cols_are_float(self, tmp_path, monkeypatch):
        ds = self._load(tmp_path, monkeypatch)
        for col in ("sex", "diagnosis"):
            assert pd.api.types.is_float_dtype(ds.data[col]), \
                f"{col} should be float after label encoding"

    def test_discrete_col_values_are_consecutive_labels(self, tmp_path, monkeypatch):
        ds = self._load(tmp_path, monkeypatch)
        sex_vals = set(ds.data["sex"].dropna().astype(int))
        assert sex_vals == {0, 1}, f"sex encoded to {sex_vals}, expected {{0, 1}}"

        diag_vals = set(ds.data["diagnosis"].dropna().astype(int))
        assert diag_vals == {0, 1, 2}, \
            f"diagnosis encoded to {diag_vals}, expected {{0, 1, 2}}"

    def test_inverse_transform_recovers_original_categories(self, tmp_path, monkeypatch):
        ds = self._load(tmp_path, monkeypatch)
        with open(ds.label_encoder_path, "rb") as f:
            encoder = pickle.load(f)

        # MultiColumnLabelEncoder.inverse_transform iterates over all encoded
        # columns (including the target), so pass the full discrete set.
        recovered = encoder.inverse_transform(ds.data[ds.discrete_columns].copy())

        assert set(recovered["sex"].unique()) <= {"M", "F"}
        assert set(recovered["diagnosis"].unique()) <= {"TypeA", "TypeB", "TypeC"}


# ---------------------------------------------------------------------------
# t016 — _get_type_columns() auto-detect heuristic
# ---------------------------------------------------------------------------

class TestAutoDetect:
    """With empty continuous/discrete lists, n_unique > 15 → continuous, else → discrete."""

    def _load(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(1)
        n = 60
        df = pd.DataFrame({
            # 60 distinct floats → n_unique > 15 → should be "continuous"
            "hi_unique": rng.uniform(0, 1000, n),
            # 5 unique string values → n_unique ≤ 15 → should be "discrete"
            "lo_unique": rng.choice(["A", "B", "C", "D", "E"], n),
            "label":     rng.integers(0, 2, n),
        })
        _write_tiny_dataset(tmp_path, "autodet",
                            df=df,
                            continuous=[],  # empty → auto-detect fires for all cols
                            discrete=[],
                            target="label",
                            task="classification")
        monkeypatch.chdir(tmp_path)
        return get_dataset("autodet", is_encode=True)

    def test_high_unique_col_detected_as_continuous(self, tmp_path, monkeypatch):
        ds = self._load(tmp_path, monkeypatch)
        assert ds.type_columns["hi_unique"] == "continuous"

    def test_low_unique_col_detected_as_discrete(self, tmp_path, monkeypatch):
        ds = self._load(tmp_path, monkeypatch)
        assert ds.type_columns["lo_unique"] == "discrete"


# ---------------------------------------------------------------------------
# t017 — load_config() validation errors
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_missing_data_section_raises(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text('[columns]\ntarget = "y"\ntask = "classification"\n')
        with pytest.raises(ValueError, match=r"\[data\]"):
            load_config(str(p))

    def test_missing_target_raises(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text(
            '[data]\npath = "x.csv"\n\n'
            '[columns]\ntask = "classification"\n'
        )
        with pytest.raises(ValueError, match="target"):
            load_config(str(p))

    def test_missing_task_raises(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text(
            '[data]\npath = "x.csv"\n\n'
            '[columns]\ntarget = "y"\n'
        )
        with pytest.raises(ValueError, match="task"):
            load_config(str(p))


# ---------------------------------------------------------------------------
# t018 — dynamic registry: drop configs/dummy.toml → get_dataset('dummy')
# ---------------------------------------------------------------------------

class TestDynamicRegistry:
    def test_get_dataset_loads_from_toml_without_code_changes(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(2)
        n = 30
        df = pd.DataFrame({
            "feature_a": rng.uniform(0, 100, n),
            "feature_b": rng.choice(["x", "y"], n),
            "outcome":   rng.integers(0, 2, n),
        })
        _write_tiny_dataset(tmp_path, "dummy",
                            df=df,
                            continuous=["feature_a"],
                            discrete=["feature_b"],
                            target="outcome",
                            task="classification")
        monkeypatch.chdir(tmp_path)

        ds = get_dataset("dummy")

        assert ds.data.shape[0] == n
        assert set(ds.columns) == {"feature_a", "feature_b", "outcome"}
        assert ds.target == "outcome"
