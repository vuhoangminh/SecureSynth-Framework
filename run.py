"""SecureSynth-Framework — one-click pipeline entry point.

Usage:
    python run.py --config configs/clinical.toml
    python run.py --config configs/clinical.toml --dry-run
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="SecureSynth end-to-end synthetic data pipeline",
    )
    p.add_argument(
        "--config",
        required=True,
        metavar="TOML",
        help="Path to the pipeline config file (TOML format)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print execution plan; do not write any files",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Smoke-test mode: 20 epochs, 2 IORBO trials per combo",
    )
    return p


def _load_and_validate(config_path: str) -> tuple:
    """Load config and run validation.  Returns (cfg, issues).

    Exits with code 1 on load errors (missing file, parse error, missing
    required fields).  Returns the issues list for the caller to handle
    (so --dry-run can display them without redundantly exiting).
    """
    path = Path(config_path)
    if not path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from engine.config_loader import load_config
        cfg = load_config(config_path)
    except Exception as exc:
        print(f"ERROR: failed to parse config {config_path!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    from engine.config_validator import validate_config
    issues = validate_config(cfg)
    return cfg, issues


def _print_validation_issues(issues: list[str]) -> None:
    errors = [i for i in issues if not i.startswith("WARNING")]
    warnings = [i for i in issues if i.startswith("WARNING")]
    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}", file=sys.stderr)


def _has_errors(issues: list[str]) -> bool:
    return any(not i.startswith("WARNING") for i in issues)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg, issues = _load_and_validate(args.config)

    if args.dry_run:
        # Always print the full diagnostic, even when there are errors — that is
        # the point of --dry-run: show exactly what is wrong before committing.
        _print_dry_run(cfg, args.config)
        if issues:
            _print_validation_issues(issues)
        if _has_errors(issues):
            print("Fix the errors above before running the pipeline.", file=sys.stderr)
            return 1
        return 0

    if issues:
        _print_validation_issues(issues)

    if _has_errors(issues):
        print("Aborting: fix the errors above before running the pipeline.", file=sys.stderr)
        return 1

    is_test = 1 if args.test else 0
    max_trials = 2 if args.test else 30
    summary = run_pipeline(args.config, cfg, is_test=is_test, max_trials=max_trials)
    failed = [k for k, v in summary.items() if v["status"] == "failed"]
    return 1 if failed else 0


def _fmt_bytes(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024**3:.1f} GB"
    return f"{b / 1024**2:.0f} MB"


def _print_dry_run(cfg, config_path: str) -> None:
    from pathlib import Path as _P
    from engine.config_validator import run_checklist, estimate_resources

    W = 62
    print(f"\n{'='*W}")
    print("  DRY RUN — no files will be written")
    print(f"{'='*W}")
    print(f"  Config  : {config_path}")
    print(f"  Dataset : {_P(config_path).stem}")
    print()

    # --- Validation checklist ---
    print("  Validation checklist")
    print(f"  {'-'*40}")
    checks = run_checklist(cfg)
    for c in checks:
        mark = "✓" if c.passed else "✗"
        detail = f"  ({c.detail})" if c.detail else ""
        print(f"    {mark}  {c.label}{detail}")
    print()

    # --- Resource estimates ---
    res = estimate_resources(cfg)
    print("  Resource estimates")
    print(f"  {'-'*40}")
    rows_str = f"{res.n_rows:,}" if res.n_rows is not None else "unknown"
    print(f"    Rows in dataset  : {rows_str}")
    print(f"    Raw data size    : {_fmt_bytes(res.raw_bytes)}")
    print(f"    Est. disk usage  : {_fmt_bytes(res.estimated_bytes)}"
          f"  ({res.n_combos} combo{'s' if res.n_combos != 1 else ''} × {_fmt_bytes(res.raw_bytes)} × 3)")
    print(f"    GPU required     : {'yes' if res.needs_gpu else 'no'}")
    dp = cfg.differential_privacy
    print(f"    Differential DP  : {'enabled  ε=' + str(dp.epsilon) + '  δ=' + str(dp.delta) if dp.enabled else 'disabled'}")
    print()

    # --- Model × loss combinations ---
    print("  Model × loss combinations")
    print(f"  {'-'*40}")
    for combo in res.combos:
        print(f"    • {combo}")
    print()

    # --- Execution plan ---
    print("  Execution plan")
    print(f"  {'-'*40}")
    print("    1. preprocess   — prepare artefacts in database/prepared/")
    print("    2. train        — IORBO hyperparameter search per combo")
    print("    3. evaluate     — data-sufficiency analysis")
    print("    4. postprocess  — apply constraints to best synthetic output")
    print(f"{'='*W}\n")


def _setup_file_logger(dataset: str, is_test: bool = False) -> tuple[Path, logging.Logger]:
    """Create (or reuse) a FileHandler logger for *dataset*. Returns (log_path, logger)."""
    out_dir = Path("database/prepared") / dataset / ("test" if is_test else "")
    log_path = out_dir / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"securesynth.{dataset}")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
        logger.addHandler(fh)
    return log_path, logger


def _print_summary_table(summary: dict, all_labels: list[str], console) -> None:
    """Render the final run summary as a rich Table to *console*."""
    from rich.table import Table

    tbl = Table(title="Pipeline summary", show_header=True, header_style="bold")
    tbl.add_column("Step / Combo", style="cyan", min_width=22)
    tbl.add_column("Status", justify="center", min_width=10)
    tbl.add_column("Duration", justify="right", min_width=9)
    tbl.add_column("Best loss", justify="right", min_width=10)

    _STATUS_STYLE = {"done": "green", "failed": "red", "skipped": "yellow"}
    _STATUS_MARK = {"done": "✓ done", "failed": "✗ failed", "skipped": "↷ skipped"}

    for label in all_labels:
        if label == "---":
            tbl.add_section()
            continue
        entry = summary.get(label, {"status": "pending", "duration": None, "loss": None})
        status = entry["status"]
        mark = _STATUS_MARK.get(status, status)
        style = _STATUS_STYLE.get(status, "")
        dur = entry.get("duration")
        dur_str = f"{dur:.1f}s" if dur else "—"
        loss = entry.get("loss")
        loss_str = f"{loss:.4f}" if loss is not None else "—"
        display = entry.get("display")
        status_cell = display if display else (f"[{style}]{mark}[/{style}]" if style else mark)
        tbl.add_row(label, status_cell, dur_str, loss_str)

    console.print(tbl)


def _print_output_panel(dataset: str, console, is_test: bool = False) -> None:
    from rich.panel import Panel
    prepared_dir = Path("database/prepared") / dataset
    out_dir = prepared_dir / "test" if is_test else prepared_dir
    if not (out_dir / "synthetic_final.csv").exists():
        return
    paths = sorted(out_dir.glob("synthetic_*.csv"))
    console.print(Panel("\n".join(str(p) for p in paths),
                        title="Output files", border_style="green"))


def run_pipeline(
    config_path: str,
    cfg,
    _steps: dict | None = None,
    _console=None,
    is_test: int = 0,
    max_trials: int = 30,
) -> dict:
    """Execute the full pipeline with dual logging (rich Live + file handler).

    Returns a summary dict mapping each step label to
    ``{"status": str, "duration": float|None, "loss": float|None}``.

    *_steps* injection keys (all optional; omit to use the real orchestrator):
        "preprocess"  -> callable() -> StepResult
        "train"       -> callable(gm: str, loss: str) -> StepResult
        "evaluate"    -> callable() -> StepResult
        "postprocess" -> callable() -> StepResult
        "is_done"     -> callable(dataset, step, model=None) -> bool
    """
    import threading
    import engine.progress as progress
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text

    dataset = Path(config_path).stem
    log_path, logger = _setup_file_logger(dataset, is_test=bool(is_test))

    live_console = _console if _console is not None else Console(stderr=True)
    summary_console = _console if _console is not None else Console()

    combos = [
        f"{gm}-{loss}"
        for gm in cfg.training.gms
        for loss in cfg.training.losses
    ]
    all_labels = ["preprocess", *combos, "evaluate", "postprocess"]

    try:
        progress.init(dataset, config_path, combos, is_test=bool(is_test))
    except Exception:
        pass

    steps = _steps or {}
    _progress_is_done = lambda d, s, m=None: progress.is_done(d, s, m, is_test=bool(is_test))
    is_done_fn = steps.get("is_done", _progress_is_done)
    summary: dict[str, dict] = {}

    _SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _STATUS_STYLE = {"done": "green", "failed": "red", "skipped": "yellow"}
    _STATUS_MARK  = {"done": "✓ done", "failed": "✗ failed", "skipped": "↷ skipped"}
    _cur: dict = {"label": None, "t0": None, "tick": 0}

    def _build_renderable():
        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("Step / Combo", style="cyan", min_width=24)
        tbl.add_column("Status", justify="center", min_width=14)
        tbl.add_column("Duration", justify="right", min_width=9)
        tbl.add_column("Best loss", justify="right", min_width=10)

        now = time.monotonic()
        spin = _SPINNER[_cur["tick"] % len(_SPINNER)]

        for label in all_labels:
            if label == "---":
                tbl.add_section()
                continue
            if label == _cur["label"]:
                elapsed = now - _cur["t0"]
                status_cell = f"[yellow]{spin} {elapsed:.1f}s...[/yellow]"
                tbl.add_row(label, status_cell, "—", "—")
            elif label in summary:
                entry = summary[label]
                st = entry["status"]
                mark = _STATUS_MARK.get(st, st)
                style = _STATUS_STYLE.get(st, "")
                display = entry.get("display")
                status_cell = display if display else (f"[{style}]{mark}[/{style}]" if style else mark)
                dur = entry.get("duration")
                loss = entry.get("loss")
                tbl.add_row(
                    label, status_cell,
                    f"{dur:.1f}s" if dur else "—",
                    f"{loss:.4f}" if loss is not None else "—",
                )
            else:
                tbl.add_row(label, "[dim]pending[/dim]", "—", "—")

        if _cur["label"]:
            elapsed = now - _cur["t0"]
            spin2 = _SPINNER[(_cur["tick"] + 2) % len(_SPINNER)]
            bar = Text()
            bar.append(f"\n  {spin2}  ", style="yellow")
            bar.append(_cur["label"], style="cyan bold")
            bar.append("  running...  ", style="dim")
            bar.append(f"{elapsed:.1f}s", style="bold white")
            return Group(tbl, bar)
        return tbl

    def _default(step_name, *a):
        from engine import orchestrator as _o
        fn = getattr(_o, f"step_{step_name}")
        return fn(dataset, config_path, *a)

    preprocess_fn = steps.get("preprocess", lambda: _default("preprocess", is_test))
    train_fn      = steps.get("train")
    evaluate_fn   = steps.get("evaluate",   lambda: _default("evaluate",   is_test))
    postprocess_fn= steps.get("postprocess",lambda: _default("postprocess", is_test))

    with Live(console=live_console, refresh_per_second=4) as live:

        def _run_step(label: str, fn, step_key: str | None = None) -> bool:
            if step_key and is_done_fn(dataset, step_key):
                summary[label] = {"status": "skipped", "duration": 0.0, "loss": None}
                logger.info(f"SKIP  {label}")
                live.update(_build_renderable())
                return True

            logger.info(f"START {label}")
            t0 = time.monotonic()
            _cur["label"] = label
            _cur["t0"]    = t0

            result_box: list = [None]
            exc_box:    list = [None]

            def _worker():
                try:
                    result_box[0] = fn()
                except Exception as e:
                    exc_box[0] = e

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            while thread.is_alive():
                _cur["tick"] += 1
                live.update(_build_renderable())
                time.sleep(0.25)
            thread.join()

            _cur["label"] = None
            duration = time.monotonic() - t0

            if exc_box[0]:
                logger.error(f"ERROR {label}: {exc_box[0]}")
                ok = False
            else:
                result = result_box[0]
                ok = result.ok if hasattr(result, "ok") else bool(result)
                if not ok and hasattr(result, "error") and result.error:
                    logger.error(f"ERROR {label}: {result.error}")

            status = "done" if ok else "failed"
            summary[label] = {"status": status, "duration": duration, "loss": None}
            logger.info(f"END   {label}  status={status}  duration={duration:.1f}s")
            live.update(_build_renderable())
            return ok

        _run_step("preprocess", preprocess_fn, step_key="preprocess")

        for combo in combos:
            gm, loss = combo.split("-", 1)
            if is_done_fn(dataset, "train", combo):
                combo_loss = (
                    progress.load(dataset, is_test=bool(is_test))
                    .get("steps", {}).get("train", {}).get("models", {})
                    .get(combo, {}).get("loss")
                )
                summary[combo] = {"status": "skipped", "duration": 0.0, "loss": combo_loss}
                logger.info(f"SKIP  {combo}")
                live.update(_build_renderable())
                continue
            if train_fn is not None:
                fn = lambda g=gm, l=loss: train_fn(g, l)
            else:
                fn = lambda g=gm, l=loss: _default("train", [g], [l], is_test, max_trials)
            _run_step(combo, fn)
            combo_loss = (
                progress.load(dataset, is_test=bool(is_test))
                .get("steps", {}).get("train", {}).get("models", {})
                .get(combo, {}).get("loss")
            )
            if combo_loss is not None:
                summary[combo]["loss"] = combo_loss
            live.update(_build_renderable())

        _run_step("evaluate", evaluate_fn, step_key="evaluate")

        eval_summary_path = Path("database/prepared") / dataset / ("test" if is_test else "") / "eval_summary.json"
        if eval_summary_path.exists():
            try:
                with open(eval_summary_path) as fh:
                    eval_data = json.load(fh)
                family_best = eval_data.get("family_best", {})
                global_best = eval_data.get("global_best", {})
                if family_best:
                    all_labels.append("---")
                    for family, info in sorted(family_best.items(),
                                               key=lambda x: x[1]["loss"] if x[1]["loss"] is not None else float("inf")):
                        lbl = f"↳ {family}"
                        all_labels.append(lbl)
                        summary[lbl] = {
                            "status": "done", "duration": None,
                            "loss": info["loss"], "display": info["model_key"],
                        }
                    if global_best:
                        lbl = "★ best overall"
                        all_labels.append(lbl)
                        summary[lbl] = {
                            "status": "done", "duration": None,
                            "loss": global_best["loss"], "display": global_best["model_key"],
                        }
                    live.update(_build_renderable())
            except Exception:
                pass

        _run_step("postprocess", postprocess_fn, step_key="postprocess")

    logger.info(f"DONE  pipeline finished. Log: {log_path}")
    _print_summary_table(summary, all_labels, summary_console)
    _print_output_panel(dataset, summary_console, is_test=bool(is_test))
    return summary


if __name__ == "__main__":
    sys.exit(main())
