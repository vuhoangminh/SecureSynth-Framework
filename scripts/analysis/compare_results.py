"""
compare_results.py — rich terminal table + matplotlib heatmap of model rankings.

Usage (biobank):
  python scripts/analysis/compare_results.py \
    --path_df_score database/results/biobank_compare_gmdp.csv

Usage (clinical / generic):
  python scripts/analysis/compare_results.py \
    --path_df_score database/results/clinical_compare_gmdp.csv \
    --datasets test_clinical \
    --models ctgan copulagan tvae tabddpm tabsyn \
    --condvecs 1
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parents[2]))
from engine.config import config
from scripts.analysis.friedman_nemenyi import get_rankings

BIOBANK_DATASETS = [
    "biobank_record_vital",
    "biobank_record_dead",
    "biobank_record_icd7",
    "biobank_record_icd9",
    "biobank_record_icdo2",
    "biobank_record_icdo3",
    "biobank_patient_dead",
    "biobank_sen_meta",
    "biobank_sen_prostate",
    "biobank_sen_breast",
    "biobank_sen_colorectal",
    "biobank_sen_uroandkid",
    "biobank_sen_lung",
    "biobank_sen_pancreatic",
    "biobank_sen_haematological",
    "biobank_phase3_cancer_all",
    "biobank_phase3_cancer_bio_all",
    "biobank_phase3_cancer_185",
    "biobank_phase3_cancer_174",
    "biobank_phase3_cancer_153",
    "biobank_phase3_cancer_162",
    "biobank_phase3_cancer_188",
    "biobank_phase3_cancer_172",
    "biobank_phase3_cancer_154",
    "biobank_phase3_cancer_173",
    "biobank_phase3_cancer_200",
]


def _display_name(key):
    """Map internal model key (e.g. 'ctgan_1_0') to a human-readable label."""
    try:
        parts = key.rsplit("_", 1)
        base, lv = parts[0], parts[1]
        if "tabddpm" in base:
            display = config.DICT_MAPPING_MODEL.get(base, base)
        else:
            bparts = base.rsplit("_", 1)
            model, condvec = bparts[0], bparts[1]
            display = config.DICT_MAPPING_MODEL.get(model, model)
            if condvec == "0":
                display += "*"
        if lv != "0":
            display += f" lv{lv}"
        return display
    except Exception:
        return key


def build_rank_df(R):
    """Convert rankings dict {dataset: {model_key: rank}} to a DataFrame."""
    return pd.DataFrame(R).T  # rows=datasets, cols=model_keys


def show_rich_table(df, title="Model Rankings (1 = best)"):
    console = Console()
    col_keys = list(df.columns)
    avg = df.mean(axis=1)
    df_sorted = df.loc[avg.sort_values().index]

    table = Table(title=title, show_lines=True)
    table.add_column("Dataset", style="bold cyan", no_wrap=True)
    for key in col_keys:
        table.add_column(_display_name(key), justify="center")
    table.add_column("Avg", justify="center", style="bold")

    for dataset, row in df_sorted.iterrows():
        row_vals = row[col_keys]
        min_rank = row_vals.min()
        max_rank = row_vals.max()
        cells = []
        for key in col_keys:
            v = row[key]
            if pd.isna(v):
                cells.append("-")
            elif v == min_rank:
                cells.append(f"[bold green]{int(v)}[/bold green]")
            elif v == max_rank:
                cells.append(f"[red]{int(v)}[/red]")
            else:
                cells.append(str(int(v)))
        cells.append(f"{row_vals.mean():.1f}")
        table.add_row(dataset, *cells)

    avg_per_model = df_sorted[col_keys].mean(axis=0)
    avg_cells = [f"{avg_per_model[k]:.1f}" for k in col_keys]
    avg_cells.append(f"{avg_per_model.mean():.1f}")
    table.add_row("[bold]Average[/bold]", *avg_cells, style="bold yellow")

    console.print(table)
    return df_sorted


def save_heatmap(df, output_path, title="Model Rankings"):
    col_keys = [c for c in df.columns if c != "avg"]
    data = df[col_keys].values.astype(float)
    n_models = len(col_keys)

    fig_w = max(8, n_models * 1.6)
    fig_h = max(4, len(df) * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        data,
        cmap="RdYlGn_r",
        aspect="auto",
        vmin=1,
        vmax=n_models,
        interpolation="nearest",
    )

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(
        [_display_name(k) for k in col_keys],
        rotation=40,
        ha="right",
        fontsize=9,
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=8)

    for i in range(len(df)):
        for j in range(n_models):
            v = data[i, j]
            if not np.isnan(v):
                text_color = "white" if v <= n_models / 2 else "black"
                ax.text(j, i, str(int(v)), ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Rank (1 = best)", fontsize=9)

    avg_rank = df[col_keys].mean(axis=0)
    order_labels = [f"{_display_name(k)}\n(avg {avg_rank[k]:.1f})" for k in col_keys]
    ax.set_xticklabels(order_labels, rotation=40, ha="right", fontsize=8)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved → {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path_df_score", default="database/results/biobank_compare_gmdp.csv",
                        help="Path to the pre-computed scores CSV")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to include (default: all 26 biobank datasets)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to include (default: all 7)")
    parser.add_argument("--condvecs", nargs="+", type=int, default=None,
                        help="Condvec values to include (default: [0, 1])")
    parser.add_argument("--output", default=None,
                        help="Output path for heatmap PNG (default: same dir as path_df_score)")
    parser.add_argument("--title", default="Model Rankings", help="Figure title")
    args = parser.parse_args()

    datasets = args.datasets or BIOBANK_DATASETS

    R = get_rankings(
        datasets=datasets,
        models=args.models,
        condvecs=args.condvecs,
        path_df_score=args.path_df_score,
    )

    if not R:
        print("No rankings found — check that the scores CSV exists and contains data.")
        sys.exit(1)

    df = build_rank_df(R)
    df_sorted = show_rich_table(df, title=args.title)

    output = args.output or str(Path(args.path_df_score).with_suffix(".png"))
    save_heatmap(df_sorted, output, title=args.title)


if __name__ == "__main__":
    main()
