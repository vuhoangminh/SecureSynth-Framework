"""
Smoke / integration tests for the full training pipeline.

Run these with: pytest tests/test_pipeline.py -v -s

Skipped automatically when data/clinical.csv is absent (e.g., bare CI).

t019 — run_tabgen.py: CTGAN trains for 5 epochs and writes synthetic_full.csv
t020 — optimize_ctgan.py: IORBO runs 1 hyperopt trial without error
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Repo root must be on PYTHONPATH so subprocess scripts can import engine/models
REPO_ROOT = str(Path(__file__).parent.parent)
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": REPO_ROOT}

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

CLINICAL_CSV = Path("data/clinical.csv")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: full-pipeline smoke tests (require data on disk)"
    )


@pytest.fixture(autouse=True, scope="module")
def require_clinical_data():
    if not CLINICAL_CSV.exists():
        pytest.skip("data/clinical.csv not found — skipping pipeline smoke tests")


# ---------------------------------------------------------------------------
# t019 — run_tabgen.py generates synthetic_full.csv
# ---------------------------------------------------------------------------

def test_run_tabgen_generates_synthetic_csv(tmp_path):
    """5-epoch dry run: confirm synthetic_full.csv is written."""
    dir_logs = tmp_path / "gan"

    result = subprocess.run(
        [
            sys.executable, "-W", "ignore",
            "scripts/optimize/run_tabgen.py",
            "--dataset", "clinical",
            "--arch", "ctgan",
            "--epochs", "5",
            "--n_run", "1",
            "--is_test", "1",
            "--checkpoint_freq", "100",
            "--dir_logs", str(dir_logs),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=_SUBPROCESS_ENV,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, (
        f"run_tabgen.py exited {result.returncode}\n--- stderr ---\n{result.stderr[-2000:]}"
    )

    # synthetic_full.csv is only written when --row_number is used;
    # the standard output is fake_{epoch:05}.csv written at the final epoch.
    fake_files = list(dir_logs.rglob("fake_*.csv"))
    assert fake_files, (
        f"No fake_*.csv found under {dir_logs}.\n"
        f"stdout tail:\n{result.stdout[-1000:]}"
    )
    assert fake_files[0].stat().st_size > 0, "fake CSV is empty"


# ---------------------------------------------------------------------------
# t020 — optimize_ctgan.py completes 1 hyperopt trial
# ---------------------------------------------------------------------------

def test_optimize_ctgan_one_trial():
    """1 IORBO trial (is_test=1 → 100 epochs): confirm clean exit."""
    # evaluate_technical_paper reads from database/gan_optimize/ unconditionally,
    # so dir_logs cannot be redirected to tmp — we accept writes to database/.
    result = subprocess.run(
        [
            sys.executable, "-W", "ignore",
            "scripts/optimize/optimize_ctgan.py",
            "--dataset", "clinical",
            "--arch", "ctgan",
            "--max_trials", "1",
            "--is_test", "1",
            "--module", "public",
            "--loss_version", "0",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_SUBPROCESS_ENV,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, (
        f"optimize_ctgan.py exited {result.returncode}\n--- stderr ---\n{result.stderr[-2000:]}"
    )
