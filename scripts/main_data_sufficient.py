import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from engine.datasets import *
from engine.config import config
from scripts.perform_friedman_nemenyi_test import nemenyi_test
from scipy.interpolate import UnivariateSpline
from kneed import KneeLocator


# --- Evaluation mapping definition ---
# Use tuples as keys because lists are not hashable and cannot be dict keys
EVALUATION_MAP = {
    tuple(["statistics", "ml", "ml_augment", "dp"]): "all",
    tuple(["statistics"]): "statistics",
    tuple(["ml", "ml_augment"]): "ml",
    tuple(["dp"]): "dp",
}


LIST_DATASETS = [
    "biobank_patient_dead",
    "biobank_record_dead",
    "biobank_record_vital",
    "biobank_record_icd7",
    "biobank_record_icd9",
    "biobank_record_icdo2",
    "biobank_record_icdo3",
    # phase 2
    "biobank_sen_meta",
    "biobank_sen_prostate",
    "biobank_sen_breast",
    "biobank_sen_colorectal",
    "biobank_sen_uroandkid",
    "biobank_sen_lung",
    "biobank_sen_pancreatic",
    "biobank_sen_haematological",
    # phase 3
    "biobank_phase3_cancer_all",
    # "biobank_phase3_cancer_bio_all",
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


# -----------------------------------------------------------------------------
# 1. ARGPARSE
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument(
    "--dataset",
    default="biobank_patient_dead",
    type=str,
)
parser.add_argument(
    "--dir_logs",
    type=str,
    default="database/gan/",
    help="dir logs",
)
parser.add_argument(
    "-a",
    "--arch",
    default="ctgan",
    choices=[
        "ctgan",
        "tvae",
        "copulagan",
        "dpcgans",
        "ctab",
        "tabddpm",
        "tabsyn",
    ],
)
parser.add_argument(
    "--is_condvec",
    default=1,
    type=int,
)
# conventional/proposed method
parser.add_argument(
    "--loss_version",
    default=0,
    choices=[
        0,  # conventional
        1,  # version 1: submitted to NIPS24
        2,  # version 2: generalize mean loss to distribution loss and correct correlation loss
        3,  # version 3: generalize mean loss to normalized distribution loss and correct correlation loss
        4,  # version 4: only correlation loss
        5,  # version 5: only distribution loss
    ],
    type=int,
)
# generate subsets of a pandas DataFrame by subsampling rows and shuffling columns before sampling
parser.add_argument(
    "--row_number",
    default=None,
    type=int,
)

# Parse the arguments
args = parser.parse_args()


# -----------------------------------------------------------------------------
# 2. UTILITIES FOR FOLDER DISCOVERY
# -----------------------------------------------------------------------------
def get_paths(paths, dataset=None):
    if dataset is not None:
        p = []
        for path in paths:
            if dataset in path:
                p.append(path)
        paths = p
    return paths


# Function to extract rownum (xxx)
def extract_rownum(path):
    match = re.search(r"rownum_(\d+)", path)
    return int(match.group(1)) if match else float("inf")  # Handle missing cases


def get_folders(directory):
    """
    Gets a list of all folders (subdirectories) within a given directory.

    Args:
      directory: The path to the directory to search.

    Returns:
      A list of strings, where each string is the name of a folder within the directory.
    """
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders


def find_folders_containing_all_substrings(string_list, *substrings):
    """
    Returns a list of strings from the input list that contain all the provided substrings.

    Args:
      string_list: A list of strings to search through.
      *substrings: A variable number of substring arguments.

    Returns:
      A new list containing only the strings that contain all the specified substrings.
    """
    result = []
    for s in string_list:
        contains_all = True
        for sub in substrings:
            if sub not in s:
                contains_all = False
                break  # No need to check other substrings for this string
        if contains_all:
            result.append(s)
    return result


def plot(d, dfs):
    for folder, rownum in d.items():
        filepath = os.path.join(dir, folder, "df_score_data_sufficient.csv")
        try:
            df_score_data_sufficient = pd.read_csv(
                filepath,
                sep="\t",
                header=0,
                index_col=0,
            )
            dfs[rownum] = df_score_data_sufficient
            print(f"Read DataFrame from: {filepath}")
            print(df_score_data_sufficient)
        except FileNotFoundError:
            print(f"File not found: {filepath}")

    all_metrics = list(list(dfs.values())[0]["metric"])

    # Iterate through each unique metric
    for metric in list(all_metrics):
        metric_values = {}
        valid_values = []  # List to store only valid (non-NaN, non-Inf) values

        # Collect the 'value' for the current metric from each DataFrame
        for rownum, df in dfs.items():
            if "metric" in df.columns and "value" in df.columns:
                metric_row = df[df["metric"] == metric]
                if not metric_row.empty:
                    value = metric_row["value"].iloc[0]
                    metric_values[rownum] = value
                    if not (pd.isna(value) or np.isinf(value)):
                        valid_values.append(value)
                else:
                    metric_values[rownum] = (
                        np.nan
                    )  # Or some other indicator for missing metric

        # Check if there are at least two different valid values
        unique_valid_values = set(valid_values)
        if len(unique_valid_values) > 1:
            # Prepare data for plotting
            rownums = sorted(metric_values.keys())
            values = [metric_values[r] for r in rownums]

            # Create sequential x-axis values
            sequential_x = range(1, len(rownums) + 1)

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(sequential_x, values, marker="o")
            plt.xlabel("Rownum")
            plt.ylabel("Value")
            plt.title(f"Metric: {metric}")
            plt.xticks(
                sequential_x, rownums
            )  # Set x-axis ticks to sequential values, labels to original rownums
            plt.grid(True)
            # plt.show()
        else:
            if not unique_valid_values:
                print(
                    f"Skipping metric '{metric}' as all values are NaN or Inf across all DataFrames."
                )
            else:
                print(
                    f"Skipping metric '{metric}' as all valid values are the same ({unique_valid_values.pop()})."
                )


# -----------------------------------------------------------------------------
# 3. METRIC PREPROCESSING & SCORING (from earlier)
# -----------------------------------------------------------------------------
def compute_composite_scores(df: pd.DataFrame, smooth_window: int = 3) -> pd.DataFrame:
    g = df.groupby("row_number")["score"].mean().reset_index()
    # g = df.groupby("row_number")["score"].median().reset_index()
    g.rename(columns={"score": "composite_score"}, inplace=True)
    # Add a smoothed version of the composite score
    # g["composite_smooth"] = (
    #     g["composite_score"].rolling(window=smooth_window, center=True).mean()
    # )

    return g


def find_elbow(row_nums, scores):
    """
    Identify the elbow point in a curve of score vs. number of rows using first-order difference.

    Args:
        row_nums (list or np.ndarray): List of row counts (e.g., [50, 100, 500, ...]), sorted ascending.
        scores (list or np.ndarray): Corresponding composite scores (e.g., utility, accuracy, etc.)

    Returns:
        int: The row_number (from row_nums) where the largest drop in score occurs,
             indicating the elbow point. If there is no drop (constant or increasing), return the last row_number.
    """
    diffs = np.diff(scores)  # difference between consecutive scores
    if len(diffs) == 0:
        return row_nums[-1]  # fallback if scores are too short
    return int(row_nums[np.argmin(diffs)])  # point with biggest drop


def update_row_metric(value, is_higher_is_better, fill_inf_nan=10_000):
    """Handle NaN/inf and apply sign flip based on whether higher is better."""
    if pd.isna(value) or np.isinf(value):
        return fill_inf_nan if is_higher_is_better else -fill_inf_nan
    else:
        return value if is_higher_is_better else -value


def update_df_all_metrics(df_all, dict_mapping_metrics, fill_inf_nan=10_000):
    """Update the 'value' column based on evaluation and metric type."""
    df_all = df_all.copy()

    for idx, row in df_all.iterrows():
        evaluation = row["evaluation"]
        metric = row["metric"]
        value = row["value"]

        if evaluation == "statistics":
            is_higher_is_better = dict_mapping_metrics.get(metric) != "lower"
        elif evaluation == "ml":
            is_higher_is_better = False
        elif evaluation == "ml_augment":
            is_higher_is_better = not ("mae" in metric or "mse" in metric)
        elif evaluation == "dp":
            is_higher_is_better = dict_mapping_metrics.get(metric) != "lower"
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation}")

        df_all.at[idx, "value"] = update_row_metric(
            value, is_higher_is_better, fill_inf_nan
        )

    return df_all


def generate_metric_thresholds(metrics, evaluation, fill_inf_nan=10_000):
    """
    Generate good threshold ranges (low_good, high_good) for each metric.
    All metrics are assumed to be higher-is-better (already transformed).

    Args:
        metrics: list of metric names in your dataframe
        evaluation: one of ['statistics', 'ml', 'ml_augment', 'dp']
        fill_inf_nan: upper bound for good values (used as high_good)

    Returns:
        dict of metric → {"low_good": x, "high_good": y}
    """
    thresholds = {}

    for m in metrics:
        if evaluation == "statistics":
            if "kl_divergence" in m or "kolmogorov_smirnov" in m or "dwp" in m:
                thresholds[m] = {"low_good": -1, "high_good": 0}
            elif "chisquare" in m:
                thresholds[m] = {"low_good": 0.05, "high_good": 1}
            elif "cramer" in m:
                thresholds[m] = {"low_good": 0, "high_good": 1}
            elif "pearson" in m:
                thresholds[m] = {"low_good": 0, "high_good": 1}
            else:
                thresholds[m] = {"low_good": 0, "high_good": 1}  # fallback

        elif evaluation == "dp":
            if "k_anonymization_safe" in m or "l_diversity_safe" in m:
                thresholds[m] = {"low_good": 1.0, "high_good": fill_inf_nan}
            elif "k_anonymization_synthetic" in m:
                thresholds[m] = {"low_good": 0, "high_good": 2000}
            elif "l_diversity_synthetic" in m:
                thresholds[m] = {"low_good": 2, "high_good": 10}
            elif "k_map" in m:
                thresholds[m] = {"low_good": 5, "high_good": 10}
            elif "delta_presence" in m:
                thresholds[m] = {"low_good": -100, "high_good": 0}
            elif "re_identification_score" in m:
                thresholds[m] = {"low_good": -1, "high_good": -0.15}
            elif "mia_accuracy" in m or "mia_auc" in m:
                thresholds[m] = {"low_good": -1, "high_good": -0.5}
            elif "categorical_zero_cap" in m or "categorical_generalized_cap" in m:
                thresholds[m] = {"low_good": 0.8, "high_good": 1}
            elif "nndr" in m:
                thresholds[m] = {"low_good": 0.8, "high_good": 1}
            elif "dcr" in m:
                thresholds[m] = {"low_good": 0, "high_good": 100}
            else:
                thresholds[m] = {"low_good": 0.6, "high_good": 10}  # fallback

        elif evaluation == "ml":
            thresholds[m] = {"low_good": -1, "high_good": 0}

        elif evaluation == "ml_augment":
            if "mae" in m or "mse" in m:
                thresholds[m] = {"low_good": -10, "high_good": -0.01}
            else:
                thresholds[m] = {"low_good": 0, "high_good": 1}

        else:
            raise ValueError(f"Unknown evaluation type: {evaluation}")

    return thresholds


def preprocess_metrics(df: pd.DataFrame, metric_thresholds: dict) -> pd.DataFrame:
    """
    Assigns a score between 0 and 1 to each metric value.
    Assumes all metrics are higher-is-better.
    """
    df = df.copy()
    df["score"] = np.nan

    for idx, row in df.iterrows():
        m = row["metric"]
        v = row["value"]

        if m in metric_thresholds:
            lo, hi = metric_thresholds[m]["low_good"], metric_thresholds[m]["high_good"]

            if v <= lo:
                s = 0.0
            elif v >= hi:
                s = 1.0
            else:
                s = (v - lo) / (hi - lo)
        else:
            s = 0.5  # fallback if metric not in thresholds

        df.at[idx, "score"] = s

    return df


def filter_df_all_by_metric_consistency(df_all, value_col="value"):
    """
    Removes any (evaluation, metric) group from df_all if, within that group:
      1) Any expected row_number is missing, or
      2) All values of `value_col` are identical.

    Assumes df_all has columns: 'evaluation', 'metric', 'row_number', and `value_col`.

    Parameters:
        df_all (pd.DataFrame): Input DataFrame.
        value_col (str): Name of the column containing metric values.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Determine the full set of row_numbers we expect
    all_rownums = set(df_all["row_number"].unique())

    # Collect groups to drop
    bad_groups = set()

    # Group by evaluation & metric
    for (eval_name, metric_name), group in df_all.groupby(["evaluation", "metric"]):
        present_rownums = set(group["row_number"].unique())

        # Condition 1: missing any row_number
        if present_rownums != all_rownums:
            bad_groups.add((eval_name, metric_name))
            continue

        # Extract the values, dropping NaNs
        vals = group[value_col].values
        # If any NaN in the values → drop
        if np.isnan(vals).any():
            bad_groups.add((eval_name, metric_name))
            continue

        # Condition 2: all values identical
        if len(np.unique(vals)) <= 1:
            bad_groups.add((eval_name, metric_name))

    # Now filter out any rows belonging to those bad (evaluation, metric) pairs
    mask = df_all.set_index(["evaluation", "metric"]).index.isin(bad_groups)
    df_cleaned = df_all.loc[~mask].reset_index(drop=True)
    return df_cleaned


def smooth_scores(x, y):
    # 1) Fit a smoothing spline
    #    - s controls the smoothness: larger s => smoother curve
    spl = UnivariateSpline(x, y, s=1 * len(x))
    y_smooth = -spl(x)

    # 2) Use KneeLocator to find the elbow/knee
    knee = KneeLocator(
        x,
        y_smooth,
        curve="convex",  # your composite should be non‑decreasing/convex
        direction="increasing",
    ).knee

    return x, y_smooth, knee


def prepare_data(dataset, arch, loss_version, is_condvec):
    dir = "database/20250815_gan_optimize_data_sufficient/"
    folders = get_folders(dir)
    folders.sort()

    folders = find_folders_containing_all_substrings(
        folders,
        dataset,
        arch,
        f"lv_{loss_version}",
        f"condvec_{is_condvec}",
    )

    d = {}
    for folder in folders:
        rownum = extract_rownum(folder)
        d[folder] = rownum

    d = dict(sorted(d.items(), key=lambda item: item[1]))

    # Dictionary to store the DataFrames
    dfs = {}

    for folder, rownum in d.items():
        filepath = os.path.join(dir, folder, "df_score_data_sufficient.csv")
        try:
            df_score_data_sufficient = pd.read_csv(
                filepath,
                sep="\t",
                header=0,
                index_col=0,
            )

            # fix writing bug
            df_score_data_sufficient = df_score_data_sufficient[
                df_score_data_sufficient["dataset"] == dataset
            ].copy()

            dfs[rownum] = df_score_data_sufficient
            # print(f"Read DataFrame from: {filepath}")
            # print(df_score_data_sufficient)
        except FileNotFoundError:
            print(f"File not found: {filepath}")

    # 4.3 Concatenate into one DataFrame
    df_all = pd.concat(dfs.values(), ignore_index=True)
    return df_all


def preprocess(df_all, evaluations=["statistics", "ml", "ml_augment", "dp"]):
    df_all = df_all[df_all["evaluation"].isin(evaluations)].copy()
    df_all = filter_df_all_by_metric_consistency(df_all)
    df_all = update_df_all_metrics(
        df_all,
        config.DICT_MAPPING_METRICS,
    )
    return df_all


def process_dataset_model_ranking(df_all, evaluations, is_plot=False):
    df_pivoted = df_all.pivot_table(
        index=["evaluation", "metric"], columns="row_number", values="value"
    )
    df_pivoted = df_pivoted.reset_index(drop=True)

    sign, ranks, CD, sign_bool, nemenyi_p_values = nemenyi_test(X=df_pivoted.values)

    rownums, scores_smooth, best_row = smooth_scores(
        list(df_pivoted.columns),
        ranks.mean(axis=0),
    )
    scores = ranks.mean(axis=0)

    # # 4.6 Find threshold by elbow
    # best_row = find_elbow(rownums, scores_smooth)
    # print(f"Suggested minimum sample size: {best_row} rows")

    if is_plot:
        # 4.7 Plot composite vs. row_number
        plt.figure(figsize=(10, 6))
        plt.plot(
            rownums,
            scores,
            rownums,
            scores_smooth,
            # marker="o",
            # linestyle="-",
            linewidth=2,
            label="Composite Score",
        )
        plt.axvline(best_row, color="red", linestyle="--", label=f"Elbow ~ {best_row}")
        plt.xlabel("Number of Training Rows", fontsize=14)
        plt.ylabel("Composite Score (0-1)", fontsize=14)
        plt.title(
            f"Data Sufficiency for {args.arch} | {args.dataset} | lv={args.loss_version} | condvec={args.is_condvec} | {evaluations}",
            fontsize=16,
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # path_utils.make_dir("database/temp")
        # out_png = os.path.join("database/temp", "data_sufficiency_threshold.png")
        # plt.savefig(out_png, dpi=300)
        # print(f"Saved plot to {out_png}")

        # plt.show()

    return best_row


def process_dataset_model_composite_score(df_all, evaluations, is_plot=False):

    # 4.4 Define your metric thresholds (example placeholder—adapt as needed)
    # Example: generate thresholds for all metrics in your df
    metric_thresholds = {}
    for eval_type in ["statistics", "ml", "ml_augment", "dp"]:
        metric_list = df_all[df_all["evaluation"] == eval_type]["metric"].unique()
        d = generate_metric_thresholds(metric_list, eval_type)
        metric_thresholds.update(d)

    # 4.5 Preprocess & compute composite
    df_scored = preprocess_metrics(df_all, metric_thresholds)
    df_composite = compute_composite_scores(df_scored)
    df_composite.sort_values("row_number", inplace=True)

    # 4.6 Find threshold by elbow
    rownums, scores_smooth, best_row = smooth_scores(
        df_composite["row_number"].values,
        df_composite["composite_score"].values,
    )
    scores = df_composite["composite_score"].values

    # best_row = find_elbow(
    #     df_composite["row_number"].values, df_composite["composite_score"].values
    # )
    # print(f"Suggested minimum sample size: {best_row} rows")

    if is_plot:
        # 4.7 Plot composite vs. row_number
        plt.figure(figsize=(10, 6))
        plt.plot(
            rownums,
            scores,
            rownums,
            scores_smooth,
            # marker="o",
            # linestyle="-",
            linewidth=2,
            label="Composite Score",
        )
        plt.axvline(best_row, color="red", linestyle="--", label=f"Elbow ~ {best_row}")
        plt.xlabel("Number of Training Rows", fontsize=14)
        plt.ylabel("Composite Score (0-1)", fontsize=14)
        plt.title(
            f"Data Sufficiency for {args.arch} | {args.dataset} | lv={args.loss_version} | condvec={args.is_condvec} | {evaluations}",
            fontsize=16,
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # path_utils.make_dir("database/temp")
        # out_png = os.path.join("database/temp", "data_sufficiency_threshold.png")
        # plt.savefig(out_png, dpi=300)
        # print(f"Saved plot to {out_png}")

        # plt.show()

    a = 2

    return best_row


def update_score_dataframe(
    df_score, dataset, arch, loss_version, is_condvec, evaluations, best_row
):
    """
    Creates df_score if None, otherwise appends a new row with run details.

    Args:
        df_score (pd.DataFrame or None): The existing score DataFrame or None.
        args (object): An object with dataset, arch, is_condvec, loss_version attributes.
        evaluations (list): The list of evaluations used.
        best_row: The result from process_dataset_model_ranking.

    Returns:
        pd.DataFrame: The updated df_score DataFrame.
    """
    # Map evaluations list to a string
    # Use .get() with a default in case the evaluation list doesn't match a known pattern
    mapped_evaluations = EVALUATION_MAP.get(
        tuple(evaluations), str(evaluations)
    )  # Default to string representation

    # Create a dictionary for the new row
    new_row_dict = {
        "dataset": dataset,
        "arch": arch,
        "loss_version": loss_version,
        "is_condvec": is_condvec,
        "evaluations": mapped_evaluations,
        "best_row": best_row,
    }

    # Convert dictionary to a DataFrame row
    new_row_df = pd.DataFrame([new_row_dict])

    # Check if df_score needs to be created or appended to
    if df_score is None:
        df_score = new_row_df
        # print("Created new df_score DataFrame.")
    else:
        df_score = pd.concat([df_score, new_row_df], ignore_index=True)
        # print("Appended new row to df_score.")

    return df_score


# -----------------------------------------------------------------------------
# 4. MAIN PIPELINE
# -----------------------------------------------------------------------------
def main_ranking():
    df_score = None
    for dataset in [
        # phase 1
        "biobank_patient_dead",
        "biobank_record_dead",
        "biobank_record_vital",
        "biobank_record_icd7",
        "biobank_record_icd9",
        "biobank_record_icdo2",
        "biobank_record_icdo3",
        # phase 2
        "biobank_sen_meta",
        "biobank_sen_prostate",
        "biobank_sen_breast",
        "biobank_sen_colorectal",
        "biobank_sen_uroandkid",
        "biobank_sen_lung",
        "biobank_sen_pancreatic",
        "biobank_sen_haematological",
        # phase 3
        "biobank_phase3_cancer_all",
        "biobank_phase3_cancer_185",
        "biobank_phase3_cancer_174",
        "biobank_phase3_cancer_153",
        "biobank_phase3_cancer_162",
        "biobank_phase3_cancer_188",
        "biobank_phase3_cancer_172",
        "biobank_phase3_cancer_154",
        "biobank_phase3_cancer_173",
        "biobank_phase3_cancer_200",
        # "biobank_phase3_cancer_bio_all",
    ]:
        for arch in [
            "ctgan",
            "tvae",
            "tabsyn",
        ]:
            for loss_version in [0]:
                for is_condvec in [1]:

                    print(">> processing", dataset, arch, loss_version, is_condvec)
                    # try:
                    #     df = prepare_data(dataset, arch, loss_version, is_condvec)
                    # except:
                    #     continue
                    df = prepare_data(dataset, arch, loss_version, is_condvec)

                    df["row_number"] = df["row_number"].astype(int)
                    for evaluations in [
                        [
                            "statistics",
                            "ml",
                            "ml_augment",
                            "dp",
                        ],
                        ["statistics"],
                        ["ml", "ml_augment"],
                        ["dp"],
                    ]:
                        df_all = preprocess(df.copy(), evaluations)
                        best_row = process_dataset_model_ranking(
                            df_all, evaluations, is_plot=False
                        )

                        df_score = update_score_dataframe(
                            df_score,
                            dataset,
                            arch,
                            loss_version,
                            is_condvec,
                            evaluations,
                            best_row,
                        )

    df_score.to_csv(
        "database/results/biobank_data_sufficient.csv",
        sep="\t",
        encoding="utf-8",
    )


def main_composite_score():
    df_score = None
    for dataset in [
        # phase 1
        "biobank_patient_dead",
        "biobank_record_dead",
        "biobank_record_vital",
        "biobank_record_icd7",
        "biobank_record_icd9",
        "biobank_record_icdo2",
        "biobank_record_icdo3",
        # phase 2
        "biobank_sen_meta",
        "biobank_sen_prostate",
        "biobank_sen_breast",
        "biobank_sen_colorectal",
        "biobank_sen_uroandkid",
        "biobank_sen_lung",
        "biobank_sen_pancreatic",
        "biobank_sen_haematological",
        # phase 3
        "biobank_phase3_cancer_all",
        "biobank_phase3_cancer_185",
        "biobank_phase3_cancer_174",
        "biobank_phase3_cancer_153",
        "biobank_phase3_cancer_162",
        "biobank_phase3_cancer_188",
        "biobank_phase3_cancer_172",
        "biobank_phase3_cancer_154",
        "biobank_phase3_cancer_173",
        "biobank_phase3_cancer_200",
        # "biobank_phase3_cancer_bio_all",
    ]:
        for arch in [
            "ctgan",
            "tvae",
            "tabsyn",
        ]:
            for loss_version in [0]:
                for is_condvec in [1]:

                    print(">> processing", dataset, arch, loss_version, is_condvec)
                    try:
                        df = prepare_data(dataset, arch, loss_version, is_condvec)
                    except:
                        continue

                    df["row_number"] = df["row_number"].astype(int)
                    for evaluations in [
                        [
                            "statistics",
                            "ml",
                            "ml_augment",
                            "dp",
                        ],
                        ["statistics"],
                        ["ml", "ml_augment"],
                        ["dp"],
                    ]:
                        df_all = preprocess(df.copy(), evaluations)
                        best_row = process_dataset_model_composite_score(
                            df_all,
                            evaluations,
                            # is_plot=True,
                        )

                        df_score = update_score_dataframe(
                            df_score,
                            dataset,
                            arch,
                            loss_version,
                            is_condvec,
                            evaluations,
                            best_row,
                        )

    df_score.to_csv(
        "database/results/biobank_data_sufficient_composite.csv",
        sep="\t",
        encoding="utf-8",
    )


if __name__ == "__main__":
    # main_ranking()
    main_composite_score()

    # plt.show()


"""
should aggregate multiple datasets
"""
