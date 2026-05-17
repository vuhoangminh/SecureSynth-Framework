"""Tests for t045c: run.py step dispatch + dual logging + final summary table.

Coverage:
- log file created at expected path after run_pipeline()
- log file contains START/END entries for each step
- summary dict contains one key per model×loss combo + preprocess/evaluate/postprocess
- failed steps are marked "failed" in summary
- re-run with all steps already done skips all (via mocked is_done)
- re-run with only some steps done skips those, runs the rest
- summary table (rich output) contains combo names
- summary table marks failed steps with failure indicator
"""
import importlib.util
import io
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, call

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


def _fake_result(ok: bool = True):
    r = MagicMock()
    r.ok = ok
    return r


def _make_steps(ok: bool = True, is_done_val: bool = False):
    """Return a minimal _steps dict with all fns returning StepResult(ok=ok)."""
    return {
        "preprocess": MagicMock(return_value=_fake_result(ok)),
        "train": MagicMock(return_value=_fake_result(ok)),
        "evaluate": MagicMock(return_value=_fake_result(ok)),
        "postprocess": MagicMock(return_value=_fake_result(ok)),
        "is_done": MagicMock(return_value=is_done_val),
    }


def _silent_console():
    """A rich Console that discards output (no terminal escape codes in tests)."""
    from rich.console import Console
    return Console(file=io.StringIO(), highlight=False)


def _config(tmp_path) -> tuple:
    """Write a minimal TOML config with a tiny CSV. Returns (cfg_path_str, cfg)."""
    from engine.config_loader import load_config

    data_csv = tmp_path / "data.csv"
    data_csv.write_text("age,sex,target\n30,M,0\n40,F,1\n50,M,0\n")
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        f'[data]\npath = "{data_csv}"\nformat = "csv"\n'
        '[columns]\ncontinuous = ["age"]\ndiscrete = ["sex"]\n'
        'target = "target"\ntask = "classification"\n'
        '[training]\ngms = ["ctgan", "tvae"]\nlosses = ["vanilla", "cd"]\n'
    )
    cfg = load_config(str(cfg_path))
    return str(cfg_path), cfg


# ---------------------------------------------------------------------------
# log file
# ---------------------------------------------------------------------------

def test_log_file_created(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps()

    mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    log_path = _REPO / "database" / "prepared" / "cfg" / "pipeline.log"
    assert log_path.exists(), f"Expected log file at {log_path}"


def test_log_file_path_uses_dataset_name(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    # dataset name == stem of cfg file = "cfg"
    steps = _make_steps()
    mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    log_path = _REPO / "database" / "prepared" / "cfg" / "pipeline.log"
    assert log_path.exists()


def test_log_file_contains_start_end_entries(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps()
    mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    log_path = _REPO / "database" / "prepared" / "cfg" / "pipeline.log"
    content = log_path.read_text()
    assert "START preprocess" in content
    assert "END   preprocess" in content


def test_log_file_contains_done_entry(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps()
    mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    log_path = _REPO / "database" / "prepared" / "cfg" / "pipeline.log"
    assert "DONE  pipeline finished" in log_path.read_text()


# ---------------------------------------------------------------------------
# summary dict keys / combo coverage
# ---------------------------------------------------------------------------

def test_summary_contains_all_combos(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)   # 2 gms × 2 losses = 4 combos
    steps = _make_steps()

    summary = mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    expected_combos = {"ctgan-vanilla", "ctgan-cd", "tvae-vanilla", "tvae-cd"}
    assert expected_combos <= summary.keys()


def test_summary_contains_pipeline_steps(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps()

    summary = mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    for step in ("preprocess", "evaluate", "postprocess"):
        assert step in summary, f"Missing step {step!r} in summary"


def test_summary_one_row_per_combo(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps()

    summary = mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    combo_keys = [k for k in summary if k not in ("preprocess", "evaluate", "postprocess")]
    assert len(combo_keys) == 4  # 2 gms × 2 losses


# ---------------------------------------------------------------------------
# failed step marked in summary
# ---------------------------------------------------------------------------

def test_failed_preprocess_marked_in_summary(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps()
    steps["preprocess"] = MagicMock(return_value=_fake_result(ok=False))

    summary = mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    assert summary["preprocess"]["status"] == "failed"


def test_failed_train_combo_marked_in_summary(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps()
    # first call to train fails, rest succeed
    call_count = [0]
    def _train(gm, loss):
        call_count[0] += 1
        return _fake_result(ok=(call_count[0] > 1))
    steps["train"] = _train

    summary = mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    failed_combos = [k for k, v in summary.items() if v["status"] == "failed"]
    assert len(failed_combos) == 1


def test_failed_step_logged_to_file(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps()
    steps["evaluate"] = MagicMock(side_effect=RuntimeError("eval exploded"))

    mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    log_path = _REPO / "database" / "prepared" / "cfg" / "pipeline.log"
    content = log_path.read_text()
    assert "ERROR" in content
    assert "evaluate" in content


# ---------------------------------------------------------------------------
# skip completed steps via mocked is_done
# ---------------------------------------------------------------------------

def test_skip_all_when_is_done_true(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps(is_done_val=True)

    summary = mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    # no step function should have been called
    steps["preprocess"].assert_not_called()
    steps["evaluate"].assert_not_called()
    steps["postprocess"].assert_not_called()
    steps["train"].assert_not_called()

    # all entries should be "skipped"
    assert all(v["status"] == "skipped" for v in summary.values())


def test_skip_preprocess_only(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)

    def _is_done(dataset, step, model=None):
        return step == "preprocess"

    steps = _make_steps()
    steps["is_done"] = _is_done

    summary = mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    assert summary["preprocess"]["status"] == "skipped"
    steps["preprocess"].assert_not_called()
    # other steps should have run
    assert summary["evaluate"]["status"] == "done"
    steps["evaluate"].assert_called_once()


def test_skip_records_duration_zero(tmp_path):
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps(is_done_val=True)

    summary = mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=_silent_console())

    for v in summary.values():
        assert v["duration"] == 0.0


# ---------------------------------------------------------------------------
# summary table output (rich rendered to a StringIO buffer)
# ---------------------------------------------------------------------------

def test_summary_table_contains_combo_names(tmp_path):
    from rich.console import Console
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps()

    buf = io.StringIO()
    console = Console(file=buf, highlight=False, width=120)
    mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=console)

    output = buf.getvalue()
    assert "ctgan-vanilla" in output
    assert "tvae-cd" in output


def test_summary_table_marks_failed_step(tmp_path):
    from rich.console import Console
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps()
    steps["evaluate"] = MagicMock(return_value=_fake_result(ok=False))

    buf = io.StringIO()
    console = Console(file=buf, highlight=False, width=120)
    mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=console)

    output = buf.getvalue()
    assert "failed" in output.lower() or "✗" in output


def test_summary_table_marks_skipped_step(tmp_path):
    from rich.console import Console
    mod = _load_run()
    cfg_path, cfg = _config(tmp_path)
    steps = _make_steps(is_done_val=True)

    buf = io.StringIO()
    console = Console(file=buf, highlight=False, width=120)
    mod.run_pipeline(cfg_path, cfg, _steps=steps, _console=console)

    output = buf.getvalue()
    assert "skipped" in output.lower() or "↷" in output
