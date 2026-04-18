import os
import glob
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
from itertools import combinations

from engine.datasets import get_dataset
import engine.utils.data_utils as data_utils

from termcolor import colored


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def categorize_columns(df, target="icd9_code"):
    def _categorize_columns_from_input(df):
        cont_cols, date_cols, str_cols, mixed_cols = [], [], [], []

        for col in df.columns:
            s = df[col]
            nonnull = s.dropna()

            # 1) If entirely NaN → mixed
            if nonnull.empty:
                mixed_cols.append(col)
                continue

            # 2) Already datetime dtype?
            if pd.api.types.is_datetime64_any_dtype(s):
                date_cols.append(col)
                continue

            # 3) Boolean columns count as numeric
            if pd.api.types.is_bool_dtype(s):
                cont_cols.append(col)
                continue

            # 4) All non-null are strings?
            if nonnull.map(lambda x: isinstance(x, str)).all():
                # Try parse as date
                try:
                    pd.to_datetime(nonnull, errors="raise", infer_datetime_format=True)
                    date_cols.append(col)
                except Exception:
                    str_cols.append(col)
                continue

            # 5) Pure numeric?
            try:
                pd.to_numeric(nonnull, errors="raise")
                cont_cols.append(col)
                continue
            except Exception:
                pass

            # 6) Anything else → mixed
            mixed_cols.append(col)
        return cont_cols, date_cols, str_cols, mixed_cols

    cont_cols_input, date_cols_input, str_cols_input, mixed_cols_input = (
        _categorize_columns_from_input(df)
    )

    for col in df.columns:
        n_unique = df[col].dropna().nunique()

        is_continuous = (
            n_unique > 6
            and col in cont_cols_input
            and not (col in cont_cols_input and target == col)
        )

        if target == col:
            a = 2

        if is_continuous:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("object")

    list_discrete = df.select_dtypes(include=["object"]).columns.tolist()
    list_continuous = [col for col in df.columns if col not in list_discrete]

    return df, list_discrete, list_continuous


def find_continuous_pairs(df, cont_cols, threshold=0.8):

    results = []
    for x, y in combinations(cont_cols, 2):
        r = df[x].corr(df[y])
        if pd.notna(r) and abs(r) >= threshold:
            results.append((x, y, r))
    return pd.DataFrame(results, columns=["var1", "var2", "corr"]).sort_values(
        "corr", ascending=False
    )


def find_dd_high_assoc(df, discrete_cols, threshold=0.9):
    """
    For each pair of discrete columns C1, C2 and each category v1 in C1:
      compute P(C2 == v2 | C1 == v1) for the most frequent non-NaN v2.
      If that P ≥ threshold, report (C1, v1, C2, v2, P).
    """
    results = []
    total = sum(len(df[c1].dropna().unique()) for c1 in discrete_cols) * (
        len(discrete_cols) - 1
    )

    with tqdm(total=total, desc="Analyzing associations") as pbar:
        for c1 in discrete_cols:
            for c2 in discrete_cols:
                if c1 == c2:
                    continue
                for v1, sub in df.groupby(c1):
                    if len(sub) < 2:
                        pbar.update(1)
                        continue
                    vc = sub[c2].value_counts(normalize=True, dropna=False)
                    # Remove NaN index if it's the top value
                    vc = vc[vc.index.notna()]
                    if vc.empty:
                        pbar.update(1)
                        continue
                    top_v2, p = vc.index[0], vc.iloc[0]
                    if p >= threshold:
                        results.append((c1, v1, c2, top_v2, p))
                    pbar.update(1)

    return pd.DataFrame(
        results, columns=["col1", "val1", "col2", "val2", "p_cond"]
    ).sort_values("p_cond", ascending=False)


def find_continuous_discrete(
    df: pd.DataFrame,
    discrete_cols: list,
    cont_cols: list,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    One-hot encode discrete_cols → dummy variables.
    For each dummy and each cont_col, compute Pearson r.
    Return all pairs with |r| >= threshold, sorted descending.
    """
    # 1) One-hot encode all discrete columns
    df_dummies = pd.get_dummies(df[discrete_cols], drop_first=False)

    results = []

    total = len(df_dummies.columns) * len(cont_cols)
    with tqdm(total=total, desc="Analyzing dummy-cont correlations") as pbar:
        # 2) For each dummy and each continuous column, compute correlation
        for dummy_col in df_dummies.columns:
            series_dummy = df_dummies[dummy_col]
            for cont_col in cont_cols:
                # make sure cont_col is numeric
                series_cont = pd.to_numeric(df[cont_col], errors="coerce")
                if series_cont.isna().all():
                    pbar.update(1)
                    continue
                r = series_dummy.corr(series_cont)
                if pd.notna(r) and abs(r) >= threshold:
                    results.append((dummy_col, cont_col, r))
                pbar.update(1)

    # 3) Build result DataFrame
    if not results:
        return pd.DataFrame(columns=["dummy", "cont", "corr"])
    df_out = pd.DataFrame(results, columns=["dummy", "cont", "corr"])
    return df_out.sort_values("corr", ascending=False).reset_index(drop=True)


def extract_columns_by_substring(df, list_included):
    """
    Extracts columns from a Pandas DataFrame whose names contain any of the
    substrings present in the provided list.

    Args:
      df (pd.DataFrame): The input DataFrame.
      list_included (list): A list of substrings to search for in the column names.

    Returns:
      pd.DataFrame: A new DataFrame containing only the columns whose names
                    include at least one of the substrings from list_included.
    """
    included_columns = []
    for col in df.columns:
        for item in list_included:
            if item in col:
                included_columns.append(col)
                break  # Move to the next column once a match is found
    return df[included_columns]


parser = argparse.ArgumentParser(add_help=False)
# generate subsets of a pandas DataFrame by subsampling rows and shuffling columns before sampling
parser.add_argument(
    "--dataset",
    default="dummy.csv",
    type=str,
)
parser.add_argument(
    "--folder",
    # default="_database/gan_optimize",
    default="database/dataset/daniel",
    type=str,
)


args = parser.parse_args()

csv_file = os.path.join(args.folder, args.dataset)

# df = pd.read_csv(csv_file, sep="\t", header=0, index_col=0)
df = pd.read_csv(csv_file, sep="\t", header=0)

print(df)

list_included = [
    "sex",
    "age",
    "marital_status",
    "education",
    "cohabitant",
    "sm_",
    "sn_",
    "health_status_year",
    "mi_stroke_family",
    "diabetes",
    "distance_work",
    "cholesterol",
    "height",
    "weight",
    "bmi",
    "waist",
    "cholesterol",
    "bloodsugar",
    "bp_",
    "pa_index",
]


print()
print()
print()
print("After extracting:")
# df = extract_columns_by_substring(df, list_included)
columns_to_drop = [col for col in df.columns if "date" in col.lower()]
df = df.drop(columns=columns_to_drop)

print(df)
print()


# Define which columns are discrete vs continuous
print()
print()
print()
print()
print()
print(">> categorize_columns")
df, list_discrete, list_continuous = categorize_columns(df)
print()
print()
print()
print()
print()
# print(">> find_high_global_correlations")
print()
print()
print()
print()
print()
df_global_cont = find_continuous_pairs(
    df,
    list_continuous,
    threshold=0.3,
)
df_global_cont.to_csv(
    "database/analysis/df_global_cont.csv", sep="\t", encoding="utf-8"
)
print("=== Continuous-Continuous ===\n", df_global_cont)
print()
print()

df_global_disc = find_dd_high_assoc(
    df,
    list_discrete,
    threshold=0.3,
)
df_global_disc.to_csv(
    "database/analysis/df_global_disc.csv", sep="\t", encoding="utf-8"
)
print("=== Discrete-Discrete ===\n", df_global_disc)
print()
print()

df_cont_disc = find_continuous_discrete(
    df,
    list_discrete,
    list_continuous,
    threshold=0.3,
)
df_cont_disc.to_csv("database/analysis/df_cont_disc.csv", sep="\t", encoding="utf-8")
print("=== Continuous-Discrete ===\n", df_cont_disc)
print()
print()
