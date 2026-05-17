"""Tests for t045a: run.py argparse + config load/validate entry point.

Coverage:
- --config is required (missing → exit 2)
- --dry-run flag is accepted
- nonexistent config path → exit 1
- config with validation errors → exit 1, errors printed to stderr
- valid config + mocked orchestrator → exit 0
- --dry-run with valid config → exit 0, no orchestrator called
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_run(extra_mocks: dict | None = None):
    """Import run.py with heavy engine deps mocked out."""
    mocks = {
        "engine.orchestrator": MagicMock(),
        "engine.progress": MagicMock(),
    }
    if extra_mocks:
        mocks.update(extra_mocks)

    originals = {}
    for name, mock in mocks.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = mock

    spec = importlib.util.spec_from_file_location("_run", _REPO / "run.py")
    mod = importlib.util.module_from_spec(spec)
    orig_argv = sys.argv[:]
    sys.argv = ["run.py"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = orig_argv
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return mod


# ---------------------------------------------------------------------------
# argparse contract
# ---------------------------------------------------------------------------

def test_config_required():
    mod = _load_run()
    with pytest.raises(SystemExit) as exc:
        mod.main([])
    assert exc.value.code == 2  # argparse exits 2 for missing required arg


def test_dry_run_flag_accepted(tmp_path):
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        '[data]\npath = "/nonexistent.csv"\nformat = "csv"\n'
        '[columns]\ntarget = "y"\ntask = "classification"\n'
    )
    mod = _load_run()
    with patch.object(mod, "_load_and_validate") as mock_lav:
        mock_lav.return_value = (MagicMock(), [])
        with patch.object(mod, "_print_dry_run"):
            code = mod.main(["--config", str(cfg_file), "--dry-run"])
    assert code == 0


# ---------------------------------------------------------------------------
# file-not-found exit
# ---------------------------------------------------------------------------

def test_nonexistent_config_exits_1(tmp_path):
    mod = _load_run()
    with pytest.raises(SystemExit) as exc:
        mod.main(["--config", str(tmp_path / "no_such.toml")])
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# validation-error exit
# ---------------------------------------------------------------------------

def test_validation_errors_exit_1(tmp_path):
    cfg_file = tmp_path / "bad.toml"
    # valid parse but references a missing data path → validator returns errors
    cfg_file.write_text(
        '[data]\npath = "/nonexistent_data.csv"\nformat = "csv"\n'
        '[columns]\ntarget = "y"\ntask = "classification"\n'
    )
    mod = _load_run()
    code = mod.main(["--config", str(cfg_file)])
    assert code == 1


def test_validation_errors_printed_to_stderr(tmp_path, capsys):
    cfg_file = tmp_path / "bad.toml"
    cfg_file.write_text(
        '[data]\npath = "/nonexistent_data.csv"\nformat = "csv"\n'
        '[columns]\ntarget = "y"\ntask = "classification"\n'
    )
    mod = _load_run()
    mod.main(["--config", str(cfg_file)])
    captured = capsys.readouterr()
    assert "not found" in captured.err or "ERROR" in captured.err


# ---------------------------------------------------------------------------
# valid config + mocked orchestrator
# ---------------------------------------------------------------------------

def test_valid_config_calls_pipeline_exits_0(tmp_path):
    cfg_file = tmp_path / "ok.toml"
    # Points at a real data file (the clinical CSV if present, else skip)
    data_csv = _REPO / "database" / "raw" / "clinical.csv"
    if not data_csv.exists():
        pytest.skip("clinical.csv not present — skipping integration path")

    cfg_file.write_text(
        f'[data]\npath = "{data_csv}"\nformat = "csv"\n'
        '[columns]\ntarget = "mortality"\ntask = "classification"\n'
        '[training]\ngms = ["ctgan"]\nlosses = ["vanilla"]\n'
    )
    mod = _load_run()
    pipeline_mock = MagicMock(return_value={"preprocess": {"status": "done"}})

    with patch.object(mod, "run_pipeline", pipeline_mock):
        code = mod.main(["--config", str(cfg_file)])

    assert code == 0
    pipeline_mock.assert_called_once()
    call_args = pipeline_mock.call_args
    assert call_args[0][0] == str(cfg_file)  # first positional arg is config_path


def test_valid_config_mocked_pipeline_no_data(tmp_path):
    """run_pipeline is called even when data file doesn't exist on disk
    — validation is bypassed via patching validate_config."""
    cfg_file = tmp_path / "ok.toml"
    cfg_file.write_text(
        '[data]\npath = "/fake/data.csv"\nformat = "csv"\n'
        '[columns]\ntarget = "y"\ntask = "classification"\n'
        '[training]\ngms = ["ctgan"]\nlosses = ["vanilla"]\n'
    )
    mod = _load_run()
    pipeline_mock = MagicMock(return_value={"preprocess": {"status": "done"}})

    with patch("engine.config_validator.validate_config", return_value=[]), \
         patch.object(mod, "run_pipeline", pipeline_mock):
        code = mod.main(["--config", str(cfg_file)])

    assert code == 0
    pipeline_mock.assert_called_once()


# ---------------------------------------------------------------------------
# --dry-run does NOT call orchestrator
# ---------------------------------------------------------------------------

def test_dry_run_does_not_call_pipeline(tmp_path):
    cfg_file = tmp_path / "ok.toml"
    cfg_file.write_text(
        '[data]\npath = "/fake/data.csv"\nformat = "csv"\n'
        '[columns]\ntarget = "y"\ntask = "classification"\n'
        '[training]\ngms = ["ctgan"]\nlosses = ["vanilla"]\n'
    )
    mod = _load_run()
    pipeline_mock = MagicMock(return_value={})

    with patch("engine.config_validator.validate_config", return_value=[]), \
         patch.object(mod, "run_pipeline", pipeline_mock):
        code = mod.main(["--config", str(cfg_file), "--dry-run"])

    assert code == 0
    pipeline_mock.assert_not_called()
