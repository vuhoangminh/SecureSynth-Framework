import os
import csv

import pandas as pd

from engine.config_loader import PipelineConfig

_KNOWN_GMS = {
    "ctgan", "copulagan", "dpcgans", "ctab", "tvae",
    "tabddpm", "tabsyn",
}
_KNOWN_LOSSES = {"vanilla", "cd"}
_KNOWN_TASKS = {"classification", "regression"}

_DISK_WARN_BYTES = 10 * 1024 ** 3  # 10 GB


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
