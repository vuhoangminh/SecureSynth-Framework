"""SecureSynth-Framework — one-click pipeline entry point.

Usage:
    python run.py --config configs/clinical.toml
    python run.py --config configs/clinical.toml --dry-run
"""
import argparse
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

    summary = run_pipeline(args.config, cfg)
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


def _setup_file_logger(dataset: str) -> tuple[Path, logging.Logger]:
    """Create (or reuse) a FileHandler logger for *dataset*. Returns (log_path, logger)."""
    log_path = Path("database/prepared") / dataset / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"securesynth.{dataset}")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
        logger.addHandler(fh)
    return log_path, logger


def _print_summary_table(summary: dict, combos: list[str], console) -> None:
    """Render the final run summary as a rich Table to *console*."""
    from rich.table import Table

    tbl = Table(title="Pipeline summary", show_header=True, header_style="bold")
    tbl.add_column("Step / Combo", style="cyan", min_width=22)
    tbl.add_column("Status", justify="center", min_width=10)
    tbl.add_column("Duration", justify="right", min_width=9)
    tbl.add_column("Best loss", justify="right", min_width=10)

    _STATUS_STYLE = {"done": "green", "failed": "red", "skipped": "yellow"}
    _STATUS_MARK = {"done": "✓ done", "failed": "✗ failed", "skipped": "↷ skipped"}

    for label in ("preprocess", *combos, "evaluate", "postprocess"):
        entry = summary.get(label, {"status": "pending", "duration": None, "loss": None})
        status = entry["status"]
        mark = _STATUS_MARK.get(status, status)
        style = _STATUS_STYLE.get(status, "")
        dur = entry.get("duration")
        dur_str = f"{dur:.1f}s" if dur else "—"
        loss = entry.get("loss")
        loss_str = f"{loss:.4f}" if loss is not None else "—"
        tbl.add_row(label, f"[{style}]{mark}[/{style}]", dur_str, loss_str)

    console.print(tbl)


def run_pipeline(
    config_path: str,
    cfg,
    _steps: dict | None = None,
    _console=None,
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
    import engine.progress as progress
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    dataset = Path(config_path).stem
    log_path, logger = _setup_file_logger(dataset)

    # Console for the Live display (stderr keeps it off capsys.out in tests)
    live_console = _console if _console is not None else Console(stderr=True)
    # Summary table always goes to stdout so it's visible in the terminal
    summary_console = _console if _console is not None else Console()

    combos = [
        f"{gm}-{loss}"
        for gm in cfg.training.gms
        for loss in cfg.training.losses
    ]

    try:
        progress.init(dataset, config_path, combos)
    except Exception:
        pass  # not fatal — directory may not exist in test environments

    steps = _steps or {}
    is_done_fn = steps.get("is_done", progress.is_done)

    summary: dict[str, dict] = {}

    # --- live status table (updates in-place while running) ---
    live_tbl = Table(show_header=True, box=None)
    live_tbl.add_column("Step", style="cyan", min_width=22)
    live_tbl.add_column("Status")

    def _run_step(label: str, fn, step_key: str | None = None) -> bool:
        if step_key and is_done_fn(dataset, step_key):
            summary[label] = {"status": "skipped", "duration": 0.0, "loss": None}
            logger.info(f"SKIP  {label}")
            return True
        logger.info(f"START {label}")
        t0 = time.monotonic()
        try:
            result = fn()
            ok = result.ok if hasattr(result, "ok") else bool(result)
        except Exception as exc:
            logger.error(f"ERROR {label}: {exc}")
            ok = False
        duration = time.monotonic() - t0
        status = "done" if ok else "failed"
        summary[label] = {"status": status, "duration": duration, "loss": None}
        logger.info(f"END   {label}  status={status}  duration={duration:.1f}s")
        return ok

    def _default(step_name, *a):
        from engine import orchestrator as _o
        fn = getattr(_o, f"step_{step_name}")
        return fn(dataset, config_path, *a)

    preprocess_fn = steps.get("preprocess", lambda: _default("preprocess"))
    train_fn = steps.get("train")  # callable(gm, loss) or None → use orchestrator
    evaluate_fn = steps.get("evaluate", lambda: _default("evaluate"))
    postprocess_fn = steps.get("postprocess", lambda: _default("postprocess"))

    with Live(live_tbl, console=live_console, refresh_per_second=4):
        _run_step("preprocess", preprocess_fn, step_key="preprocess")

        for combo in combos:
            gm, loss = combo.split("-", 1)
            # Train combos use model= keyword for the is_done check
            if is_done_fn(dataset, "train", combo):
                summary[combo] = {"status": "skipped", "duration": 0.0, "loss": None}
                logger.info(f"SKIP  {combo}")
                continue
            if train_fn is not None:
                fn = lambda g=gm, l=loss: train_fn(g, l)
            else:
                fn = lambda g=gm, l=loss: _default("train", [g], [l])
            _run_step(combo, fn)

        _run_step("evaluate", evaluate_fn, step_key="evaluate")
        _run_step("postprocess", postprocess_fn, step_key="postprocess")

    logger.info(f"DONE  pipeline finished. Log: {log_path}")
    _print_summary_table(summary, combos, summary_console)
    return summary


if __name__ == "__main__":
    sys.exit(main())
