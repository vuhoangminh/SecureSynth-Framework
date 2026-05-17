"""Tests for t045b: run.py --dry-run output.

Coverage:
- --dry-run exits 0
- output contains validation checklist section
- output contains model×loss combo list
- output contains disk/resource estimate
- --dry-run writes zero new files under database/
"""
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_run():
    spec = importlib.util.spec_from_file_location("_run", _REPO / "run.py")
    mod = importlib.util.module_from_spec(spec)
    orig_argv = sys.argv[:]
    sys.argv = ["run.py"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = orig_argv
    return mod


def _minimal_config(tmp_path, data_csv: Path | None = None, extra: str = "") -> Path:
    """Write a minimal but valid TOML config; use a real CSV path if provided."""
    if data_csv is None:
        # create a tiny CSV so the validator can pass all checks
        data_csv = tmp_path / "data.csv"
        data_csv.write_text("age,sex,target\n30,M,0\n40,F,1\n50,M,0\n")
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        f'[data]\npath = "{data_csv}"\nformat = "csv"\n'
        '[columns]\ncontinuous = ["age"]\ndiscrete = ["sex"]\n'
        'target = "target"\ntask = "classification"\n'
        '[training]\ngms = ["ctgan", "tvae"]\nlosses = ["vanilla", "cd"]\n'
        + extra
    )
    return cfg


# ---------------------------------------------------------------------------
# exit code
# ---------------------------------------------------------------------------

def test_dry_run_exits_0(tmp_path):
    cfg = _minimal_config(tmp_path)
    mod = _load_run()
    code = mod.main(["--config", str(cfg), "--dry-run"])
    assert code == 0


# ---------------------------------------------------------------------------
# checklist section in output
# ---------------------------------------------------------------------------

def test_dry_run_contains_checklist_header(tmp_path, capsys):
    cfg = _minimal_config(tmp_path)
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    assert "Validation checklist" in out


def test_dry_run_checklist_shows_pass_marks(tmp_path, capsys):
    cfg = _minimal_config(tmp_path)
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    assert "✓" in out


def test_dry_run_checklist_shows_fail_marks_on_bad_config(tmp_path, capsys):
    # data.path deliberately missing → ✗ in checklist; dry-run always prints it
    cfg = tmp_path / "bad.toml"
    cfg.write_text(
        '[data]\npath = "/nonexistent_xy.csv"\nformat = "csv"\n'
        '[columns]\ntarget = "y"\ntask = "classification"\n'
        '[training]\ngms = ["ctgan"]\nlosses = ["vanilla"]\n'
    )
    mod = _load_run()
    code = mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    assert "✗" in out          # checklist still shown even with errors
    assert code == 1           # but exit code is 1 so caller knows it failed


# ---------------------------------------------------------------------------
# model×loss combo list in output
# ---------------------------------------------------------------------------

def test_dry_run_contains_combo_list(tmp_path, capsys):
    cfg = _minimal_config(tmp_path)
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    # ctgan and tvae × vanilla and cd = 4 combos
    assert "ctgan×vanilla" in out.lower() or "ctgan" in out.lower()
    assert "×" in out


def test_dry_run_combo_count_matches_config(tmp_path, capsys):
    cfg = _minimal_config(tmp_path)  # 2 gms × 2 losses = 4 combos
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    # Each combo is on its own bullet line
    combo_lines = [ln for ln in out.splitlines() if "×" in ln and "•" in ln]
    assert len(combo_lines) == 4


def test_dry_run_single_combo(tmp_path, capsys):
    data_csv = tmp_path / "data.csv"
    data_csv.write_text("age,target\n30,0\n40,1\n")
    cfg = tmp_path / "single.toml"
    cfg.write_text(
        f'[data]\npath = "{data_csv}"\nformat = "csv"\n'
        '[columns]\ncontinuous = ["age"]\ntarget = "target"\ntask = "classification"\n'
        '[training]\ngms = ["ctgan"]\nlosses = ["vanilla"]\n'
    )
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    combo_lines = [ln for ln in out.splitlines() if "×" in ln and "•" in ln]
    assert len(combo_lines) == 1


# ---------------------------------------------------------------------------
# resource / disk estimate in output
# ---------------------------------------------------------------------------

def test_dry_run_contains_disk_estimate(tmp_path, capsys):
    cfg = _minimal_config(tmp_path)
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    assert "disk" in out.lower() or "GB" in out or "MB" in out


def test_dry_run_contains_row_count(tmp_path, capsys):
    cfg = _minimal_config(tmp_path)  # tiny CSV has 3 data rows
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    assert "Rows" in out or "rows" in out


def test_dry_run_contains_gpu_flag(tmp_path, capsys):
    # tabsyn requires GPU → should appear in output
    data_csv = tmp_path / "d.csv"
    data_csv.write_text("age,target\n30,0\n40,1\n")
    cfg = tmp_path / "gpu.toml"
    cfg.write_text(
        f'[data]\npath = "{data_csv}"\nformat = "csv"\n'
        '[columns]\ncontinuous = ["age"]\ntarget = "target"\ntask = "classification"\n'
        '[training]\ngms = ["tabsyn"]\nlosses = ["vanilla"]\n'
    )
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    assert "GPU" in out or "gpu" in out.lower()


def test_dry_run_gpu_not_required_for_ctgan_only(tmp_path, capsys):
    data_csv = tmp_path / "d.csv"
    data_csv.write_text("age,target\n30,0\n40,1\n")
    cfg = tmp_path / "nogpu.toml"
    cfg.write_text(
        f'[data]\npath = "{data_csv}"\nformat = "csv"\n'
        '[columns]\ncontinuous = ["age"]\ntarget = "target"\ntask = "classification"\n'
        '[training]\ngms = ["ctgan"]\nlosses = ["vanilla"]\n'
    )
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    # GPU required line should say "no"
    gpu_line = next((ln for ln in out.splitlines() if "GPU" in ln), "")
    assert "no" in gpu_line.lower()


# ---------------------------------------------------------------------------
# no files written to database/
# ---------------------------------------------------------------------------

def test_dry_run_writes_no_files(tmp_path):
    cfg = _minimal_config(tmp_path)
    db = _REPO / "database"

    def _snapshot(directory: Path) -> set[Path]:
        if not directory.exists():
            return set()
        return {p for p in directory.rglob("*") if p.is_file()}

    before = _snapshot(db)
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    after = _snapshot(db)

    new_files = after - before
    assert new_files == set(), f"--dry-run wrote unexpected files: {new_files}"


# ---------------------------------------------------------------------------
# execution plan section in output
# ---------------------------------------------------------------------------

def test_dry_run_contains_execution_plan(tmp_path, capsys):
    cfg = _minimal_config(tmp_path)
    mod = _load_run()
    mod.main(["--config", str(cfg), "--dry-run"])
    out = capsys.readouterr().out
    assert "Execution plan" in out
    assert "preprocess" in out
    assert "train" in out
