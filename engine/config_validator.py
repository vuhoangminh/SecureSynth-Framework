import os
import csv
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from engine.config_loader import PipelineConfig

_KNOWN_GMS = {
    "ctgan", "ctgan0",
    "copulagan", "copulagan0",
    "tvae", "tvae0",
    "dpcgans", "ctab",
    "tabddpm", "tabsyn",
}
_KNOWN_LOSSES = {"vanilla", "cd"}
_KNOWN_TASKS = {"classification", "regression"}
_GPU_MODELS = {"tabddpm", "tabsyn"}

_DISK_WARN_BYTES = 10 * 1024 ** 3  # 10 GB
_DISK_FACTOR = 3  # raw + synthetic + artefacts


@dataclass
class CheckResult:
    label: str
    passed: bool
    detail: str = ""


@dataclass
class ResourceEstimate:
    n_rows: int | None
    raw_bytes: int
    estimated_bytes: int
    n_combos: int
    needs_gpu: bool
    combos: List[str] = field(default_factory=list)


def validate_config(cfg: PipelineConfig) -> list[str]:
    """Return a list of error/warning strings; empty list means valid."""
    issues: list[str] = []

    # 1. data.path exists
    if not os.path.exists(cfg.data.path):
        issues.append(f"data.path not found: {cfg.data.path!r}")
        return issues  # can't do column checks without the file

    # 2. Columns present in CSV header
    header = _read_header(cfg.data.path, cfg.data.format, cfg.data.separator)
    if header is not None:
        declared = set(cfg.columns.continuous) | set(cfg.columns.discrete) | set(cfg.data.drop_columns)
        missing = declared - header
        if missing:
            issues.append(f"columns declared in config but missing from file: {sorted(missing)}")

        # 3. target not in drop_columns (auto-detection handles unlisted columns)
        if cfg.columns.target in set(cfg.data.drop_columns):
            issues.append(
                f"columns.target={cfg.columns.target!r} is listed in data.drop_columns and will be removed"
            )

    # 4. task
    if cfg.columns.task not in _KNOWN_TASKS:
        issues.append(
            f"columns.task={cfg.columns.task!r} unrecognised; expected one of {sorted(_KNOWN_TASKS)}"
        )

    # 5. postprocessing.constraints — valid pandas query syntax
    if cfg.postprocessing.constraints:
        all_cols = cfg.columns.continuous + cfg.columns.discrete
        dummy = pd.DataFrame(columns=all_cols)
        for expr in cfg.postprocessing.constraints:
            try:
                dummy.query(expr)
            except Exception as exc:
                issues.append(f"postprocessing constraint invalid query {expr!r}: {exc}")

    # 6. training.gms recognised
    for gm in cfg.training.gms:
        if gm.lower() not in _KNOWN_GMS:
            issues.append(
                f"training.gms: {gm!r} unrecognised; known: {sorted(_KNOWN_GMS)}"
            )

    # 7. training.losses
    for loss in cfg.training.losses:
        if loss.lower() not in _KNOWN_LOSSES:
            issues.append(
                f"training.losses: {loss!r} unrecognised; expected one of {sorted(_KNOWN_LOSSES)}"
            )

    # 8. DP delta warning
    if cfg.differential_privacy.enabled:
        n = _estimate_row_count(cfg.data.path, cfg.data.format, cfg.data.separator)
        if n is not None and n > 0:
            threshold = 1.0 / n
            if cfg.differential_privacy.delta >= threshold:
                issues.append(
                    f"WARNING: differential_privacy.delta={cfg.differential_privacy.delta} "
                    f">= 1/n={threshold:.2e} (n={n}); DP guarantee may be meaningless"
                )

    # 9. Disk usage estimate
    n_combos = len(cfg.training.gms) * len(cfg.training.losses)
    raw_bytes = os.path.getsize(cfg.data.path)
    estimated = raw_bytes * n_combos * 3  # raw + synthetic + artefacts, rough factor
    if estimated > _DISK_WARN_BYTES:
        issues.append(
            f"WARNING: estimated disk usage ~{estimated / 1024**3:.1f} GB "
            f"({n_combos} model×loss combos × {raw_bytes / 1024**2:.0f} MB input)"
        )

    return issues


def run_checklist(cfg: PipelineConfig) -> list[CheckResult]:
    """Return a structured checklist of validation checks with pass/fail status."""
    results: list[CheckResult] = []

    # 1. data.path
    path_ok = os.path.exists(cfg.data.path)
    results.append(CheckResult(
        "data.path exists",
        path_ok,
        cfg.data.path if path_ok else f"not found: {cfg.data.path!r}",
    ))

    # 2. columns in file (only if path accessible)
    if path_ok:
        header = _read_header(cfg.data.path, cfg.data.format, cfg.data.separator)
        if header is not None:
            declared = set(cfg.columns.continuous) | set(cfg.columns.discrete) | set(cfg.data.drop_columns)
            missing = declared - header
            if missing:
                results.append(CheckResult("columns in file", False, f"missing: {sorted(missing)}"))
            else:
                results.append(CheckResult("columns in file", True, f"{len(declared)} declared columns present"))

            # 3. target not dropped
            target_ok = cfg.columns.target not in set(cfg.data.drop_columns)
            results.append(CheckResult(
                "target not in drop_columns",
                target_ok,
                "" if target_ok else f"{cfg.columns.target!r} will be removed",
            ))

    # 4. task
    task_ok = cfg.columns.task in _KNOWN_TASKS
    results.append(CheckResult(
        "task recognised",
        task_ok,
        cfg.columns.task if task_ok else f"{cfg.columns.task!r} not in {sorted(_KNOWN_TASKS)}",
    ))

    # 5. models
    bad_gms = [g for g in cfg.training.gms if g.lower() not in _KNOWN_GMS]
    results.append(CheckResult(
        "models recognised",
        not bad_gms,
        ", ".join(cfg.training.gms) if not bad_gms else f"unknown: {bad_gms}",
    ))

    # 6. losses
    bad_losses = [l for l in cfg.training.losses if l.lower() not in _KNOWN_LOSSES]
    results.append(CheckResult(
        "losses recognised",
        not bad_losses,
        ", ".join(cfg.training.losses) if not bad_losses else f"unknown: {bad_losses}",
    ))

    # 7. postprocessing constraints
    if cfg.postprocessing.constraints:
        all_cols = cfg.columns.continuous + cfg.columns.discrete
        dummy = pd.DataFrame(columns=all_cols)
        bad_exprs = []
        for expr in cfg.postprocessing.constraints:
            try:
                dummy.query(expr)
            except Exception:
                bad_exprs.append(expr)
        results.append(CheckResult(
            "postprocessing constraints",
            not bad_exprs,
            f"{len(cfg.postprocessing.constraints)} constraint(s) valid" if not bad_exprs
            else f"invalid: {bad_exprs}",
        ))

    # 8. DP delta (only if enabled and file readable)
    if cfg.differential_privacy.enabled and path_ok:
        n = _estimate_row_count(cfg.data.path, cfg.data.format, cfg.data.separator)
        if n is not None and n > 0:
            threshold = 1.0 / n
            dp_ok = cfg.differential_privacy.delta < threshold
            results.append(CheckResult(
                "DP delta < 1/n",
                dp_ok,
                f"δ={cfg.differential_privacy.delta:.2e}, 1/n={threshold:.2e} (n={n})"
                + ("" if dp_ok else " — DP guarantee may be meaningless"),
            ))

    return results


def estimate_resources(cfg: PipelineConfig) -> ResourceEstimate:
    """Return resource estimates without side effects."""
    n_rows = None
    raw_bytes = 0

    if os.path.exists(cfg.data.path):
        raw_bytes = os.path.getsize(cfg.data.path)
        n_rows = _estimate_row_count(cfg.data.path, cfg.data.format, cfg.data.separator)

    combos = [
        f"{gm}×{loss}"
        for gm in cfg.training.gms
        for loss in cfg.training.losses
    ]
    n_combos = len(combos)
    estimated_bytes = raw_bytes * n_combos * _DISK_FACTOR
    needs_gpu = any(g.lower() in _GPU_MODELS for g in cfg.training.gms)

    return ResourceEstimate(
        n_rows=n_rows,
        raw_bytes=raw_bytes,
        estimated_bytes=estimated_bytes,
        n_combos=n_combos,
        needs_gpu=needs_gpu,
        combos=combos,
    )


def _read_header(path: str, fmt: str, sep: str) -> set | None:
    try:
        if fmt == "parquet":
            import pandas as pd
            return set(pd.read_parquet(path, columns=[]).columns) | set(pd.read_parquet(path).columns[:0])
        sep = "\t" if fmt == "tsv" else sep
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=sep)
            return set(next(reader))
    except Exception:
        return None


def _estimate_row_count(path: str, fmt: str, sep: str) -> int | None:
    try:
        if fmt == "parquet":
            import pandas as pd
            return len(pd.read_parquet(path))
        sep = "\t" if fmt == "tsv" else sep
        with open(path, encoding="utf-8") as fh:
            return sum(1 for _ in fh) - 1  # subtract header
    except Exception:
        return None
