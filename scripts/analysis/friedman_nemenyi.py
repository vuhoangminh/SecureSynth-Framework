import os
import scipy.stats as stat
import numpy as np
import pandas as pd
import re
import scikit_posthocs as sp
import engine.utils.hyperopt_utils as hyperopt_utils
from engine.config import config


__all__ = [
    "multivariate_normal",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "F_score",
    "fleiss_kappa",
    "r2_score",
    "compute_ranks",
    "nemenyi_test",
]


def r2_score(y_true, y_pred):
    """R squared (coefficient of determination) regression score function.
    Best possible score is 1.0, lower values are worse.
    Parameters
    ----------
    y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.
    y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.
    Returns
    -------
    z : float
        The R^2 score.
    Notes
    -----
    This is not a symmetric function.
    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).
    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <http://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    Examples
    --------
    >>> from parsimony.utils.stats import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.948...
    """
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        else:
            return 0.0

    return 1 - numerator / denominator


def _critical_nemenyi_value(p_value, num_models):
    """Critical values for the Nemenyi test.
    Table obtained from: https://gist.github.com/garydoranjr/5016455
    """
    values = [  # p   0.01   0.05   0.10    Models
        [2.576, 1.960, 1.645],  # 2
        [2.913, 2.344, 2.052],  # 3
        [3.113, 2.569, 2.291],  # 4
        [3.255, 2.728, 2.460],  # 5
        [3.364, 2.850, 2.589],  # 6
        [3.452, 2.948, 2.693],  # 7
        [3.526, 3.031, 2.780],  # 8
        [3.590, 3.102, 2.855],  # 9
        [3.646, 3.164, 2.920],  # 10
        [3.696, 3.219, 2.978],  # 11
        [3.741, 3.268, 3.030],  # 12
        [3.781, 3.313, 3.077],  # 13
        [3.818, 3.354, 3.120],  # 14
        [3.853, 3.391, 3.159],  # 15
        [3.884, 3.426, 3.196],  # 16
        [3.914, 3.458, 3.230],  # 17
        [3.941, 3.489, 3.261],  # 18
        [3.967, 3.517, 3.291],  # 19
        [3.992, 3.544, 3.319],  # 20
        [4.015, 3.569, 3.346],  # 21
        [4.037, 3.593, 3.371],  # 22
        [4.057, 3.616, 3.394],  # 23
        [4.077, 3.637, 3.417],  # 24
        [4.096, 3.658, 3.439],  # 25
        [4.114, 3.678, 3.459],  # 26
        [4.132, 3.696, 3.479],  # 27
        [4.148, 3.714, 3.498],  # 28
        [4.164, 3.732, 3.516],  # 29
        [4.179, 3.749, 3.533],  # 30
        [4.194, 3.765, 3.550],  # 31
        [4.208, 3.780, 3.567],  # 32
        [4.222, 3.795, 3.582],  # 33
        [4.236, 3.810, 3.597],  # 34
        [4.249, 3.824, 3.612],  # 35
        [4.261, 3.837, 3.626],  # 36
        [4.273, 3.850, 3.640],  # 37
        [4.285, 3.863, 3.653],  # 38
        [4.296, 3.876, 3.666],  # 39
        [4.307, 3.888, 3.679],  # 40
        [4.318, 3.899, 3.691],  # 41
        [4.329, 3.911, 3.703],  # 42
        [4.339, 3.922, 3.714],  # 43
        [4.349, 3.933, 3.726],  # 44
        [4.359, 3.943, 3.737],  # 45
        [4.368, 3.954, 3.747],  # 46
        [4.378, 3.964, 3.758],  # 47
        [4.387, 3.973, 3.768],  # 48
        [4.395, 3.983, 3.778],  # 49
        [4.404, 3.992, 3.788],  # 50
    ]

    if num_models < 2 or num_models > 50:
        raise ValueError("num_models must be in [2, 50].")

    if p_value == 0.01:
        return values[num_models - 2][0]
    elif p_value == 0.05:
        return values[num_models - 2][1]
    elif p_value == 0.10:
        return values[num_models - 2][2]
    else:
        raise ValueError("p_value must be in {0.01, 0.05, 0.10}")


def compute_ranks(X, method="average"):
    """Assign ranks to data, dealing with ties appropriately.
    Uses scipy.stats.rankdata to compute the ranks of each row of the matrix X.
    Parameters
    ----------
    X : numpy array
        Computes the ranks of the rows of X.
    method : str
        The method used to assign ranks to tied elements. Must be one of
        "average", "min", "max", "dense" and "ordinal".
    Returns
    -------
    R : numpy array
        A matrix with the ranks computed from X. Has the same shape as X. Ranks
        begin at 1.
    """
    if method not in ["average", "min", "max", "dense", "ordinal"]:
        raise ValueError(
            'Method must be one of "average", "min", "max", ' '"dense" and "ordinal".'
        )

    n = X.shape[0]
    R = np.zeros(X.shape)
    for i in range(n):
        r = stat.rankdata(X[i, :], method=method)
        R[i, :] = r

    return R


def nemenyi_test(
    X, p_value=0.05, return_ranks=True, return_critval=True, return_p_value=True
):
    """Performs the Nemenyi test for comparing a set of classifiers to each
    other.
    Parameters
    ----------
    X : numpy array of shape (num_datasets, num_models)
        The scores of the num_datasets datasets for each of the num_models
        models. X must have at least one row and between 2 and 50 columns.
    p_value : float
        The p-value of the test. Must be one of 0.01, 0.05 or 0.1. Default is
        p_value=0.05.
    return_ranks : bool
        Whether or not to return the computed ranks. Default is False, do not
        return the ranks.
    return_critval : bool
        Whether or not to return the computed critical value. Default is False,
        do not return the critical value.
    """
    num_datasets, num_models = X.shape
    R = compute_ranks(X)
    crit_val = _critical_nemenyi_value(p_value, num_models)
    CD = crit_val * np.sqrt(num_models * (num_models + 1) / (6.0 * num_datasets))

    sign = np.zeros((num_models, num_models))

    sign_bool = np.zeros((num_models, num_models))
    for j1 in range(num_models):
        for j2 in range(num_models):
            if np.abs(np.mean(R[:, j1] - R[:, j2])) > CD:
                sign[j1, j2] = 1
            else:
                sign[j1, j2] = 0
            if np.mean(R[:, j1] - R[:, j2]) > 0:
                sign_bool[j1, j2] = 1
            else:
                sign_bool[j1, j2] = -1

    # Perform Nemenyi post-hoc test
    data = X
    nemenyi_p_values = sp.posthoc_nemenyi_friedman(data)

    return sign, R, CD, sign_bool, nemenyi_p_values


# ==========================================================================================
# Added from here
# ==========================================================================================

LIST_DATASETS = [
    # small
    "abalone",  # done
    "wilt",  # done
    "churn2",  # done
    "diabetes_openml",  # done
    "buddy",  # done
    "gesture",  # done
    "insurance",  # done
    "king",  # done
    "cardio",  # done
    "california",  # done
    "house_16h",  # done
    "adult",  # done
    "house",  # done
    "news",  # done
    "diabetesbalanced",  # 4
    # large
    "diabetes",  # 9
    "mnist12",
    # super large
    "higgs-small",  # fixed name
    "miniboone",
    "credit",
]


LIST_METHOD_CONDVEC = ["ctgan", "copulagan", "dpcgans"]

DICT_MAPPING_METRICS = {  # higher/lower is better
    "kl_divergence_discrete": "lower",
    "kl_divergence_continuous": "lower",
    "chisquare_discrete": "lower",
    "kolmogorov_smirnov_continuous": "higher",
    "cramer_discrete": "higher",
    "pearson_continuous": "higher",
    "dwp_discrete": "lower",
    "dwp_continuous": "lower",
}


def perform_test(df, is_write=True, verbose=1, return_p_value=False, return_rank=False):
    df_data = df.iloc[:, :]
    X = df_data.values

    models = list(df.columns.values)
    models.append("total")

    p_value = 0.05  # 95%
    # p_value = 0.01  # 99%
    sign, R, CD, sign_bool, nemenyi_p_values = nemenyi_test(X, p_value=p_value)

    if return_p_value:
        x = sign_bool
        column_sums = x.sum(axis=1) + 1
    else:
        x = np.multiply(sign, sign_bool)
        column_sums = x.sum(axis=1)

    # Create a new column for the sum values
    arr_with_sum_column = np.column_stack((x, column_sums))

    df_result = pd.DataFrame(arr_with_sum_column, columns=models)

    index_list = list(df.columns.values)
    index_series = pd.Series(index_list)

    # Set the index using the Series
    df_result = df_result.set_index(index_series)
    df_result_new = df_result.copy()

    # Iterate over rows and columns
    i_row = 0
    for index, row in df_result.iterrows():
        i_col = 0
        for col in df_result.columns:
            if index == col:
                df_result.loc[index, col] = ""
                df_result_new.loc[index, col] = ""
            else:
                if col != "total":
                    if df_result.loc[index, col] == 0:
                        df_result.loc[index, col] = "0"
                        df_result_new.loc[index, col] = "0"
                    elif df_result.loc[index, col] > 0:
                        df_result.loc[index, col] = "+"
                        df_result_new.loc[index, col] = "+"
                    else:
                        df_result.loc[index, col] = "-"
                        df_result_new.loc[index, col] = "-"
                    if return_p_value:
                        df_result.loc[index, col] = (
                            df_result.loc[index, col]
                            + f"(p={float(nemenyi_p_values.values[i_row, i_col]):.3f})"
                        )
                        if nemenyi_p_values.values[i_row, i_col] <= 0.01:
                            df_result_new.loc[index, col] = (
                                df_result_new.loc[index, col] * 3
                            )
                        elif nemenyi_p_values.values[i_row, i_col] <= 0.05:
                            df_result_new.loc[index, col] = (
                                df_result_new.loc[index, col] * 2
                            )
            i_col += 1
        i_row += 1

    if return_rank:
        if return_p_value:
            return (
                df_result.drop(columns=df_result.columns[-1]),
                df_result_new.drop(columns=df_result.columns[-1]),
                R,
            )
        else:
            return df_result, R.mean(axis=0)
    else:
        if return_p_value:
            return (
                df_result.drop(columns=df_result.columns[-1]),
                df_result_new.drop(columns=df_result.columns[-1]),
            )
        else:
            return df_result


def get_group_datasets(group):
    if group == "regression":
        return ["house", "news"]
    elif group == "classification":
        return [
            "adult",
            "covertype",
            "credit",
            "diabetes",
            "diabetesbalanced",
            "intrusion",
            "mnist12",
            "mnist28",
        ]
    elif group == "all":
        return LIST_DATASETS
    elif group == "mnist":
        return ["mnist12", "mnist28"]
    elif group == "only_discrete":
        return ["diabetes", "diabetesbalanced", "mnist12", "mnist28"]
    elif group == "only_continuous":
        return ["credit"]
    elif group == "small":  # < 100 000 samples
        return [
            "adult",
            "diabetes",
            "mnist12",
            "mnist28",
            "house",
            "news",
        ]
    elif group == "big":
        return [
            "covertype",
            "credit",
            "diabetesbalanced",
            "intrusion",
        ]
    else:
        raise ValueError(f"{group} is not defined...")


def get_group_evaluation(evaluation):
    if evaluation == "statistics":
        return ""
    elif evaluation == "ml":
        return "_ml"
    elif evaluation == "ml_augment":
        return "_ml_augment"
    else:
        raise ValueError(f"{evaluation} is not defined...")


def get_group_method():
    return [
        "ctgan",
        "tvae",
        "copulagan",
        "dpcgans",
        "ctab",
    ]


def get_df(dataset, evaluation, mode, notebook_path=None):
    path = f"database/evaluation/{mode}/{dataset}{get_group_evaluation(evaluation)}.csv"

    if notebook_path is not None:
        path = os.path.join(notebook_path, path)

    df = pd.read_csv(path, index_col=0, header=0, sep="\t")

    if evaluation == "statistics":
        df = df.iloc[:, :-2]
    return df


def get_row_score(df, model, losscorr, lossdwp, condvec, moment=None):
    if moment is None:
        row = df.loc[
            (df["model"] == model)
            & (df["losscorr"] == losscorr)
            & (df["lossdwp"] == lossdwp)
            & (df["condvec"] == condvec)
        ]
    else:
        row = df.loc[
            (df["model"] == model)
            & (df["losscorr"] == losscorr)
            & (df["lossdwp"] == lossdwp)
            & (df["condvec"] == condvec)
            & (df["moment"] == moment)
        ]

    return row


def preprocess_metrics(row, evaluation):
    row = row.replace([np.inf], 10_000_000_000)
    row = row.replace([np.nan], 10_000_000_000)

    for metric in row.columns:
        if evaluation == "statistics":
            if DICT_MAPPING_METRICS[metric] == "lower":
                row[metric] = -row[metric]
        elif evaluation == "ml":  # lower is better
            row[metric] = -row[metric]
        elif (
            evaluation == "ml_augment"
        ):  # higher is better except for regression task (mae + mse)
            if "mae" in metric or "mse" in metric:
                row[metric] = -row[metric]  # regression task
    return row


def extract_score_list(row, evaluation):
    row = row.drop(
        columns=[
            "folder",
            "filename",
            "model",
            "losscorr",
            "lossdwp",
            "condvec",
        ]
    )

    row = preprocess_metrics(row, evaluation)

    return row.values.tolist()[0]


def print_latex(df_score):
    def extract_numbers(text):
        numbers = re.findall(r"\d+", text)
        return "".join(str(num) for num in numbers)

    # Iterate over rows and columns
    i_row = 0
    text = ""
    for index, row in df_score.iterrows():
        text += "{:<5s}".format(f"${extract_numbers(index)}$")
        for col in df_score.columns:
            if index == col:
                text += " & {:<5s}".format(f"~~~")
            else:
                text += " & {:<5s}".format(f"${df_score.loc[index, col]}$")
        text += " \\\\ \n"
    return print(text)


def get_bold_style(d, w):
    if d > 0.5:
        return f"\\hl{{{w}}}"
    elif d == 0.5:
        return f"\\zr{{{w}}}"
    else:
        return f"\\nt{{{w}}}"


def print_latex_methods(d):
    # Iterate over rows and columns
    i_row = 0
    text = ""
    for key in [
        "method",
        "statistics",
        "ml",
        "ml_augment",
        "comp",
        "empty",
        "statistics_wr",
        "ml_wr",
        "ml_augment_wr",
        "comp_wr",
    ]:
        if key == "method":
            text += "{:<20s}".format(f"\\h{{{config.DICT_MAPPING_MODEL[d[key]]}}}")
            text += "&"
        elif key == "dataset":
            text += "{:<20s}".format(f"{d[key]}")
            text += "&"
        elif key == "empty":
            text += " &".format(f"${d[key]}$")
        elif "_wr" in key:
            key_ = key + "se"
            s = get_bold_style(d[key], d[key_])
            text += " & {:<20s}".format(f"{s}")
        elif "_wrse" in key:
            pass
        else:
            if d[key] == "--":
                k = "-"
            elif d[key] == "++":
                k = "+"
            elif d[key] == "---":
                k = "--"
            elif d[key] == "+++":
                k = "++"
            else:
                k = "0"
            text += " & {:<5s}".format(f"${k}$")
    text += " \\\\"
    return print(text)


def print_latex_datasets(d):
    # Iterate over rows and columns
    i_row = 0
    text = ""
    for key in [
        "dataset",
        "statistics",
        "ml",
        "ml_augment",
        "comp",
        "empty",
        "statistics_wr",
        "ml_wr",
        "ml_augment_wr",
        "comp_wr",
    ]:
        if key == "dataset":
            s = config.DICT_STRING_DATASETS[d[key]]
            s = f"\\h{{{s}}}"
            text += "{:<20s}".format(s)
            text += "&"
        elif key == "empty":
            text += " &".format(f"${d[key]}$")
        elif "_wr" in key:
            key_ = key + "se"
            s = get_bold_style(d[key], d[key_])
            text += " & {:<20s}".format(f"{s}")
        elif "_wrse" in key:
            pass
        else:
            if d[key] == "--":
                k = "-"
            elif d[key] == "++":
                k = "+"
            elif d[key] == "---":
                k = "--"
            elif d[key] == "+++":
                k = "++"
            else:
                k = "0"
            text += " & {:<5s}".format(f"${k}$")
    text += " \\\\"
    return print(text)


def print_latex_losses(df_score):
    def extract_numbers(text):
        numbers = re.findall(r"\d+", text)
        return "".join(str(num) for num in numbers)

    # Iterate over rows and columns
    i_row = 0
    text = ""
    for index, row in df_score.iterrows():
        text += "{:<5s}".format(f"${extract_numbers(index)}$")
        for col in df_score.columns:
            if index in col:
                text += " & {:<5s}".format(f"~~~")
            else:
                text += " & {:<5s}".format(f"${df_score.loc[index, col]}$")
        text += " \\\\ \n"
    return print(text)


def bootstrap_standard_error(data, num_bootstrap_samples=1000):
    """
    Calculate the bootstrap estimate of standard error for a numpy array of 0s and 1s.

    Parameters:
    data (numpy array): Array containing 0s and 1s.
    num_bootstrap_samples (int): Number of bootstrap samples to generate. Default is 1000.

    Returns:
    float: The bootstrap estimate of the standard error.
    """
    # Store bootstrap sample means
    bootstrap_means = []

    # Perform bootstrap sampling
    for _ in range(num_bootstrap_samples):
        # Generate a bootstrap sample by randomly sampling with replacement from the data
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Calculate the mean of the bootstrap sample and store it
        bootstrap_means.append(np.mean(bootstrap_sample))

    # Calculate the standard error as the standard deviation of the bootstrap sample means
    standard_error = np.std(bootstrap_means)

    return standard_error


def write_hyperopt_phase3(
    path_loss,
    _loss_version,
    dataset,
    model,
    condvec=1,
    model_type="",
    row_number="",
):
    def find_folder_with_strings(directory_path, string1, string2):
        """
        Finds a folder in a directory whose name contains both string1 and string2.

        Args:
            directory_path (str): The path to the directory to search in.
            string1 (str): The first string to look for in the folder name.
            string2 (str): The second string to look for in the folder name.

        Returns:
            str: The path to the first folder found that matches the criteria,
                or None if no such folder is found.
        """
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found at {directory_path}")
            return None

        for entry in os.listdir(directory_path):
            full_path = os.path.join(directory_path, entry)
            if os.path.isdir(full_path):
                if string1 in entry and string2 in entry:
                    return full_path

        return None

    def create_metric_value_dicts(df):
        """
        Transforms a DataFrame into separate dictionaries based on 'evaluation' types,
        mapping 'metric' to 'value' for each.

        Args:
            df (pd.DataFrame): The input DataFrame containing 'evaluation', 'metric', and 'value' columns.

        Returns:
            dict: A dictionary where keys are 'scores_evaluation_type' (e.g., 'scores_statistics')
                and values are the corresponding dictionaries of {'metric': 'value'}.
        """
        # Ensure only the relevant columns are considered for processing
        # This also helps if the input df has other columns not needed for this task.
        relevant_df = df[["evaluation", "metric", "value"]].copy()

        # Get unique evaluation types to iterate over
        evaluation_types = relevant_df["evaluation"].unique()

        # Dictionary to hold all the resulting metric-value dictionaries
        output_dicts = {}

        for eval_type in evaluation_types:
            # Filter the DataFrame for the current evaluation type
            subset_df = relevant_df[relevant_df["evaluation"] == eval_type]

            # Create a dictionary from 'metric' and 'value' columns for the current subset
            # Using a dictionary comprehension for conciseness
            metric_value_dict = {
                row["metric"]: row["value"] for index, row in subset_df.iterrows()
            }

            # Construct the desired dictionary name (e.g., 'scores_statistics')
            dict_name = f"scores_{eval_type}"

            # Store the created dictionary in the output_dicts
            output_dicts[dict_name] = metric_value_dict

        return output_dicts

    if row_number != "":
        base_name = (
            f"{dataset}-rownum_{row_number}-{model}-lv_{_loss_version}_model-{model_type}"
            if model_type != ""
            else f"{dataset}-{model}-lv_{_loss_version}"
        )
    else:
        base_name = (
            f"{dataset}-{model}-lv_{_loss_version}_model-{model_type}"
            if model_type != ""
            else f"{dataset}-{model}-lv_{_loss_version}"
        )

    folder = find_folder_with_strings(
        "database/20250824_gan_optimize_phase3_only_score",
        base_name,
        f"condvec_{condvec}",
    )
    path_df_score = f"{folder}/df_score_data_sufficient.csv"
    if os.path.exists(path_df_score):
        df_score = pd.read_csv(path_df_score, sep="\t", header=0, index_col=0)
        result_dicts = create_metric_value_dicts(df_score)

        data_dict = {
            "tid": 0,
            "spec": None,
            "result": {
                "loss": 1,
                "status": "ok",
                "reason": "success",
                "params": {},
                "scores_statistics": result_dicts["scores_statistics"],
                "scores_ml": result_dicts["scores_ml"],
                "scores_ml_augment": result_dicts["scores_ml_augment"],
                "scores_dp": result_dicts["scores_dp"],
            },
            "misc": {},
            "exp_key": None,
            "owner": None,
            "version": 0,
            "book_time": None,
            "refresh_time": None,
        }

        from hyperopt import Trials
        import pickle

        trials = Trials()
        trials.trials.append(data_dict)
        with open(path_loss, "wb") as f:
            pickle.dump(trials, f)


def write_df_ranking(
    datasets,
    evaluations=["statistics", "ml", "ml_augment", "dp"],
    models=None,
    condvecs=None,
    dir="database/20241003_optimization_biobank",
    path_df_score=f"database/results/biobank_compare_gmdp.csv",
    verbose=True,
):
    def udpate_df(
        _df_score,
        _path_loss,
        _loss_version,
        dataset,
        model,
        condvec=1,
        model_type="",
    ):
        if os.path.exists(_path_loss) and os.path.exists(_path_loss):
            I0 = hyperopt_utils.IncrementalObjectiveOptimizationGenerativeModel(
                _path_loss, is_print=False
            )
            I0.update_trials_losses(evaluations=evaluations)

            for evaluation in evaluations:
                if model_type == "":
                    v = f"{model}_{condvec}_{loss_version}"
                else:
                    v = f"{model}_{model_type}_{loss_version}"

                r0 = I0.get_row_fmin_evaluation("value", evaluation)
                df_temp = r0.copy()
                df_temp.insert(0, "model_name", v)
                df_temp.insert(0, "loss_version", _loss_version)
                df_temp.insert(0, "evaluation", evaluation)
                df_temp.insert(0, "model_type", model_type)
                df_temp.insert(0, "condvec", condvec)
                df_temp.insert(0, "model", model)
                df_temp.insert(0, "dataset", dataset)

                if _df_score is None:
                    _df_score = (
                        df_temp.copy()
                    )  # Use copy to avoid referencing the same object
                else:
                    _df_score = pd.concat(
                        [_df_score, df_temp], axis=0, ignore_index=True
                    )

        return _df_score

    if os.path.exists(path_df_score):
        df_score = pd.read_csv(path_df_score, sep="\t", header=0, index_col=0)
        return df_score
    else:
        df_score = None

    for dataset in datasets:
        print(f"    >> dataset: {dataset}") if verbose else None
        for condvec in condvecs:
            print(f"        >> condvec: {condvec}") if verbose else None
            for model in models:
                print(f"            >> model: {model}") if verbose else None
                for loss_version in [0, 2]:
                    (
                        print(f"                >> loss version: {loss_version}")
                        if verbose
                        else None
                    )

                    if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                        (
                            print(
                                f"            >> ignore model: {model} and condvec: {condvec}"
                            )
                            if verbose
                            else None
                        )
                        continue

                    if model == "tabddpm":
                        for model_type in ["mlp", "resnet"]:
                            # for model_type in ["mlp"]:
                            (
                                print(f"                >> model_type: {model_type}")
                                if verbose
                                else None
                            )

                            path_loss = f"{dir}/{dataset}_{model}_loss_version-{loss_version}_model-{model_type}_module-gmdp.hyperopt"

                            if "biobank_phase3" in dataset:
                                write_hyperopt_phase3(
                                    path_loss,
                                    loss_version,
                                    dataset,
                                    model,
                                    condvec,
                                    model_type="",
                                )

                            (
                                print(f"                    >> {path_loss}")
                                if verbose
                                else None
                            )

                            df_score = udpate_df(
                                df_score,
                                path_loss,
                                loss_version,
                                dataset,
                                model,
                                condvec,
                                model_type=model_type,
                            )

                    elif model == "tabsyn":
                        path_loss = f"{dir}/{dataset}_{model}_loss_version-{loss_version}-{condvec}_module-gmdp.hyperopt"
                        (
                            print(f"                    >> {path_loss}")
                            if verbose
                            else None
                        )

                        if "biobank_phase3" in dataset:
                            write_hyperopt_phase3(
                                path_loss,
                                loss_version,
                                dataset,
                                model,
                                condvec,
                                model_type="",
                            )

                        df_score = udpate_df(
                            df_score,
                            path_loss,
                            loss_version,
                            dataset,
                            model,
                            condvec,
                            model_type="",
                        )

                    else:
                        if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                            (
                                print(
                                    f"            >> ignore model: {model} and condvec: {condvec}"
                                )
                                if verbose
                                else None
                            )
                            continue

                        path_loss = f"{dir}/{dataset}_{model}_loss_version-{loss_version}-{condvec}_module-gmdp.hyperopt"

                        if "biobank_phase3" in dataset:
                            write_hyperopt_phase3(
                                path_loss,
                                loss_version,
                                dataset,
                                model,
                                condvec,
                                model_type="",
                            )

                        (print(f"                >> {path_loss}") if verbose else None)

                        # df_score = udpate_df(
                        #     df_score,
                        #     path_loss,
                        #     loss_version,
                        #     dataset,
                        #     model,
                        #     condvec,
                        #     model_type="",
                        # )

                        # a = 2

                        try:
                            df_score = udpate_df(
                                df_score,
                                path_loss,
                                loss_version,
                                dataset,
                                model,
                                condvec,
                                model_type="",
                            )
                        except:
                            pass

    df_score.to_csv(path_df_score, sep="\t", encoding="utf-8")
    return df_score


def get_df_score_comparison(
    datasets,
    evaluations,
    models=None,
    condvecs=None,
    model_types=None,
    # dir="database/optimization",
    path_df_score="database/results/biobank_compare_gmdp.csv",
    verbose=False,
    is_read_existing=True,
    is_read_error=False,
):

    def get_row(
        _df_score_full,
        dataset,
        model,
        condvec,
        evaluation,
        model_type=None,
        loss_version=0,
    ):
        """Checks if a single row in the DataFrame has the specified values in columns 'dataset', 'model', 'condvec', and if the 'metric' column contains a substring of the specified evaluation.

        Args:
            _df_score_full: The pandas DataFrame.
            dataset: The value for the 'dataset' column.
            model: The value for the 'model' column.
            condvec: The value for the 'condvec' column.
            evaluation: The value for the 'evaluation' column.
            model_type: The value for the 'model_type' column (optional).

        Returns:
            The matching row in the DataFrame, or None if no match is found.
        """

        if model_type is None:
            return _df_score_full[
                (_df_score_full["dataset"] == dataset)
                & (_df_score_full["model"] == model)
                & (_df_score_full["condvec"] == condvec)
                & (_df_score_full["evaluation"] == evaluation)
                & (_df_score_full["loss_version"] == loss_version)
            ]
        else:
            return _df_score_full[
                (_df_score_full["dataset"] == dataset)
                & (_df_score_full["model"] == model)
                & (_df_score_full["condvec"] == condvec)
                & (_df_score_full["model_type"] == model_type)
                & (_df_score_full["evaluation"] == evaluation)
                & (_df_score_full["loss_version"] == loss_version)
            ]

    if os.path.exists(path_df_score):
        df_score_full = pd.read_csv(path_df_score, sep="\t", header=0, index_col=0)
    else:
        raise FileExistsError

    df_score = None
    for evaluation in evaluations:
        df_score_evaluation = None
        for dataset in datasets:
            print(f"    >> dataset: {dataset}") if verbose else None
            for condvec in condvecs:
                print(f"        >> condvec: {condvec}") if verbose else None
                for model in models:
                    print(f"            >> model: {model}") if verbose else None
                    for loss_version in [0, 2]:
                        if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                            (
                                print(
                                    f"            >> ignore model: {model} and condvec: {condvec}"
                                )
                                if verbose
                                else None
                            )
                            continue

                        if model == "tabsyn":
                            a = 2

                        if model == "tabddpm":
                            for model_type in model_types:
                                (
                                    print(
                                        f"                >> model_type: {model_type}"
                                    )
                                    if verbose
                                    else None
                                )

                                df_temp = get_row(
                                    df_score_full,
                                    dataset,
                                    model,
                                    condvec,
                                    evaluation,
                                    model_type,
                                    loss_version,
                                )

                                if not df_temp.empty:
                                    model_name = df_temp["model_name"].iloc[0]
                                    df_temp = df_temp.rename(
                                        columns={
                                            "value": model_name,
                                        }
                                    )

                                    if is_read_error:
                                        error = df_temp["error"].values[0]
                                    df_temp = df_temp[["dataset", "metric", model_name]]
                                    df_temp = df_temp.reset_index(drop=True)

                                    if is_read_error and not pd.isna(error):
                                        df_temp.loc[0, model_name] = error

                                    if df_score_evaluation is None:
                                        df_score_evaluation = df_temp.copy()
                                    else:
                                        df_score_evaluation = pd.merge(
                                            df_score_evaluation,
                                            df_temp,
                                            on=["dataset", "metric"],
                                        )

                        else:
                            if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                                (
                                    print(
                                        f"            >> ignore model: {model} and condvec: {condvec}"
                                    )
                                    if verbose
                                    else None
                                )
                                continue

                            df_temp = get_row(
                                df_score_full,
                                dataset,
                                model,
                                condvec,
                                evaluation,
                                None,
                                loss_version,
                            )

                            if not df_temp.empty:
                                model_name = df_temp["model_name"].iloc[0]
                                df_temp = df_temp.rename(
                                    columns={
                                        "value": model_name,
                                    }
                                )

                                if is_read_error:
                                    error = df_temp["error"].values[0]
                                df_temp = df_temp[["dataset", "metric", model_name]]
                                df_temp = df_temp.reset_index(drop=True)

                                if is_read_error and not pd.isna(error):
                                    df_temp.loc[0, model_name] = error

                                if df_score_evaluation is None:
                                    df_score_evaluation = df_temp.copy()
                                else:
                                    df_score_evaluation = pd.merge(
                                        df_score_evaluation,
                                        df_temp,
                                        on=["dataset", "metric"],
                                    )

        if df_score is None:
            df_score = df_score_evaluation.copy()
        else:
            df_score = pd.concat(
                [df_score, df_score_evaluation],
                axis=0,
                ignore_index=True,
            )

    return df_score


# ==========================================================================================
# Added end here
# ==========================================================================================
def get_ranking_per_metric(df_score, r):
    columns = list(df_score.columns)

    # Initialize an empty dictionary to store rankings for each column
    rankings_per_column = {}

    # Iterate over each column in axis 0 (each column of r)
    for i in range(r.shape[0]):
        # Extract the column data from r
        column_data = r[i, :]

        # Create a dictionary mapping column names to their values
        result_dict = dict(zip(columns, column_data))

        # Fixed - assign the average rank to all keys that share the same value
        # Step 1: Sort the dictionary by values in descending order
        sorted_items = sorted(
            result_dict.items(), key=lambda item: item[1], reverse=True
        )

        # Step 2: Assign ranks with ties handled (average rank for ties)
        rankings = {}
        current_rank = 1

        while current_rank <= len(sorted_items):
            # Find all items with the same value (ties)
            current_value = sorted_items[current_rank - 1][1]
            tie_group = []

            # Collect all keys with the same value
            while (
                current_rank - 1 < len(sorted_items)
                and sorted_items[current_rank - 1][1] == current_value
            ):
                tie_group.append(
                    sorted_items[current_rank - 1][0]
                )  # Add the key to the tie group
                current_rank += 1

            # Calculate the average rank for the tie group
            start_rank = current_rank - len(
                tie_group
            )  # Starting rank for the tie group
            avg_rank = (
                start_rank + current_rank - 1
            ) / 2  # Average of first and last rank in the group

            # Assign the average rank to all keys in the tie group
            for key in tie_group:
                rankings[key] = avg_rank

        # Store the rankings for this column
        rankings_per_column[f"{i}"] = rankings

    # Convert the rankings_per_column dictionary into a DataFrame
    rankings_df = pd.DataFrame(rankings_per_column)

    return rankings_df


def get_rankings(
    evaluations=[
        "statistics",
        "ml",
        "ml_augment",
        "dp",
    ],
    datasets=[
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
    ],
    path_df_score="database/results/biobank_compare_gmdp.csv",
    is_return_metric_ranking=False,
    list_fail_model_dataset=None,
):

    R = {}
    R_metric = {}

    for dataset in datasets:
        for models in [
            [
                "ctgan",
                "copulagan",
                "dpcgans",
                "ctab",
                "tvae",
                "tabddpm",
                "tabsyn",
            ]
        ]:
            for condvecs in [
                [0, 1],
            ]:

                for model_types in [
                    ["mlp", "resnet"],
                ]:
                    if models != ["tabddpm"] and model_types == ["mlp"]:
                        continue

                    d = {}

                    if condvecs == [1]:
                        d["method"] = models[0]
                    else:
                        d["method"] = models[0] + "*"

                    if models == ["tabddpm"]:
                        d["method"] += f"-{model_types[0]}"

                    # try:

                    df_score = get_df_score_comparison(
                        [dataset],
                        evaluations,
                        models=models,
                        condvecs=condvecs,
                        model_types=model_types,
                        path_df_score=path_df_score,
                        # verbose=True,
                        verbose=False,
                    )

                    df_score = df_score.iloc[:, 2:]

                    if list_fail_model_dataset is not None:
                        for model, _dataset in list_fail_model_dataset:
                            if _dataset == dataset:
                                df_score = df_score.loc[
                                    :, ~df_score.columns.str.startswith(model)
                                ]

                    result, result2, r = perform_test(
                        df_score,
                        verbose=0,
                        return_p_value=True,
                        return_rank=True,
                    )

                    result_dict = dict(zip(list(df_score.columns), np.mean(r, axis=0)))

                    sorted_dict = dict(
                        sorted(
                            result_dict.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    )

                    # Create rankings
                    rankings = {}
                    rank = 1
                    for key in sorted_dict:
                        rankings[key] = rank
                        rank += 1

                    print()
                    print()
                    print()
                    print(f"{dataset}")
                    print("Key\tRanking")
                    print("-" * 20)
                    for key, value in rankings.items():
                        print(f"{key}\t{value}")

                    R[dataset] = rankings

                    if is_return_metric_ranking:
                        rankings_df = get_ranking_per_metric(df_score, r)
                        R_metric[dataset] = rankings_df

    if is_return_metric_ranking:
        return R, R_metric
    else:
        return R


def get_runtime(
    evaluations=[
        "runtime",
    ],
    datasets=[
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
    ],
    path_df_score="database/results/biobank_compare_runtime.csv",
    is_read_error=False,
):

    df_merged = None

    #
    for dataset in datasets:
        for models in [
            [
                "ctgan",
                "copulagan",
                "dpcgans",
                "ctab",
                "tvae",
                "tabddpm",
                "tabsyn",
            ]
        ]:
            for condvecs in [
                [0, 1],
            ]:

                for model_types in [
                    ["mlp", "resnet"],
                ]:
                    if models != ["tabddpm"] and model_types == ["mlp"]:
                        continue

                    d = {}

                    if condvecs == [1]:
                        d["method"] = models[0]
                    else:
                        d["method"] = models[0] + "*"

                    if models == ["tabddpm"]:
                        d["method"] += f"-{model_types[0]}"

                    # try:

                    df_score = get_df_score_comparison(
                        [dataset],
                        evaluations,
                        models=models,
                        condvecs=condvecs,
                        model_types=model_types,
                        path_df_score=path_df_score,
                        # verbose=True,
                        verbose=False,
                        is_read_error=is_read_error,
                    )

                    df_score = df_score.iloc[:, 2:]
                    df_score.index = [dataset]

                    if df_merged is None:
                        df_merged = df_score.copy()
                    else:
                        df_merged = pd.concat([df_merged, df_score], axis=0)

    return df_merged


def write_df_runtime(
    datasets,
    evaluations=["statistics", "ml", "ml_augment", "dp"],
    models=None,
    condvecs=None,
    path_df_score=f"database/results/biobank_compare_runtime.csv",
    dir="database/20250211_optimization_biobank_phase1and2",
    verbose=True,
):
    def udpate_df(
        _df_score, _path_loss, _loss_version, dataset, model, condvec=1, model_type=""
    ):
        if os.path.exists(_path_loss) and os.path.exists(_path_loss):
            trials = hyperopt_utils.load_project(_path_loss, is_print=False)
            best_trial = trials.best_trial
            r = (
                (best_trial["refresh_time"] - best_trial["book_time"]).total_seconds()
                / 60
                / 60
            )

            df_temp = pd.DataFrame(
                [
                    {
                        "metric": "runtime",
                        "value": round(r, 2),
                    }
                ]
            )

            if model_type == "":
                v = f"{model}_{condvec}_{loss_version}"
            else:
                v = f"{model}_{model_type}_{loss_version}"

            df_temp.insert(0, "model_name", v)
            df_temp.insert(0, "loss_version", _loss_version)
            df_temp.insert(0, "evaluation", "runtime")
            df_temp.insert(0, "model_type", model_type)
            df_temp.insert(0, "condvec", condvec)
            df_temp.insert(0, "model", model)
            df_temp.insert(0, "dataset", dataset)

            if _df_score is None:
                _df_score = (
                    df_temp.copy()
                )  # Use copy to avoid referencing the same object
            else:
                _df_score = pd.concat([_df_score, df_temp], axis=0, ignore_index=True)

        return _df_score

    if os.path.exists(path_df_score):
        df_score = pd.read_csv(path_df_score, sep="\t", header=0, index_col=0)
        return df_score
    else:
        df_score = None

    for dataset in datasets:
        print(f"    >> dataset: {dataset}") if verbose else None
        for condvec in condvecs:
            print(f"        >> condvec: {condvec}") if verbose else None
            for model in models:
                print(f"            >> model: {model}") if verbose else None
                for loss_version in [0, 2]:
                    (
                        print(f"                >> loss version: {loss_version}")
                        if verbose
                        else None
                    )

                    if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                        (
                            print(
                                f"            >> ignore model: {model} and condvec: {condvec}"
                            )
                            if verbose
                            else None
                        )
                        continue

                    if model == "tabddpm":
                        for model_type in ["mlp", "resnet"]:
                            # for model_type in ["mlp"]:
                            (
                                print(f"                >> model_type: {model_type}")
                                if verbose
                                else None
                            )
                            path_loss = f"{dir}/{dataset}_{model}_loss_version-{loss_version}_model-{model_type}_module-gmdp.hyperopt"

                            (
                                print(f"                    >> {path_loss}")
                                if verbose
                                else None
                            )

                            df_score = udpate_df(
                                df_score,
                                path_loss,
                                loss_version,
                                dataset,
                                model,
                                condvec,
                                model_type=model_type,
                            )

                    elif model == "tabsyn":
                        path_loss = f"{dir}/{dataset}_{model}_loss_version-{loss_version}-{condvec}_module-gmdp.hyperopt"
                        (
                            print(f"                    >> {path_loss}")
                            if verbose
                            else None
                        )

                        df_score = udpate_df(
                            df_score,
                            path_loss,
                            loss_version,
                            dataset,
                            model,
                            condvec,
                            model_type="",
                        )

                    else:
                        if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                            (
                                print(
                                    f"            >> ignore model: {model} and condvec: {condvec}"
                                )
                                if verbose
                                else None
                            )
                            continue

                        path_loss = f"{dir}/{dataset}_{model}_loss_version-{loss_version}-{condvec}_module-gmdp.hyperopt"

                        print(f"                >> {path_loss}") if verbose else None

                        try:
                            df_score = udpate_df(
                                df_score,
                                path_loss,
                                loss_version,
                                dataset,
                                model,
                                condvec,
                                model_type="",
                            )
                        except:
                            pass

    df_score.to_csv(path_df_score, sep="\t", encoding="utf-8")
    return df_score


def run_ranking():
    write_df_ranking(
        datasets=[
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
        ],
        models=[
            "ctgan",
            "copulagan",
            "dpcgans",
            "ctab",
            "tvae",
            "tabddpm",
            "tabsyn",
        ],
        condvecs=[0, 1],
        path_df_score=f"database/results/biobank_compare_gmdp.csv",
        dir="database/optimization",
    )

    R = get_rankings()


def run_runtime():
    write_df_runtime(
        datasets=[
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
        ],
        models=[
            "ctgan",
            "copulagan",
            "dpcgans",
            "ctab",
            "tvae",
            "tabddpm",
            "tabsyn",
        ],
        condvecs=[0, 1],
        path_df_score=f"database/results/biobank_compare_runtime.csv",
        dir="database/20250211_optimization_biobank_phase1and2",
    )

    df_score = get_runtime(
        evaluations=["runtime"],
        path_df_score="database/results/biobank_compare_runtime.csv",
    )

    a = 2


def write_df_data_sufficient(
    datasets,
    evaluations=["statistics", "ml", "ml_augment", "dp"],
    models=None,
    condvecs=None,
    dir="database/20241003_optimization_biobank",
    path_df_score=f"database/results/biobank_compare_gmdp.csv",
    verbose=True,
):
    def udpate_df(
        _df_score,
        _path_loss,
        _loss_version,
        dataset,
        model,
        condvec=1,
        model_type="",
        row_number="",
    ):
        if os.path.exists(_path_loss) and os.path.exists(_path_loss):
            I0 = hyperopt_utils.IncrementalObjectiveOptimizationGenerativeModel(
                _path_loss, is_print=False
            )
            # I0.update_trials_losses(evaluations=evaluations)

            for evaluation in evaluations:
                if model_type == "":
                    v = f"{model}_{condvec}_{loss_version}"
                else:
                    v = f"{model}_{model_type}_{loss_version}"

                r0 = I0.get_row_fmin_evaluation("value", evaluation)
                df_temp = r0.copy()
                df_temp.insert(0, "row_number", row_number)
                df_temp.insert(0, "model_name", v)
                df_temp.insert(0, "loss_version", _loss_version)
                df_temp.insert(0, "evaluation", evaluation)
                df_temp.insert(0, "model_type", model_type)
                df_temp.insert(0, "condvec", condvec)
                df_temp.insert(0, "model", model)
                df_temp.insert(0, "dataset", dataset)

                if _df_score is None:
                    _df_score = (
                        df_temp.copy()
                    )  # Use copy to avoid referencing the same object
                else:
                    _df_score = pd.concat(
                        [_df_score, df_temp], axis=0, ignore_index=True
                    )

        return _df_score

    if os.path.exists(path_df_score):
        df_score = pd.read_csv(path_df_score, sep="\t", header=0, index_col=0)
        return df_score
    else:
        df_score = None

    for dataset in datasets:
        print(f"    >> dataset: {dataset}") if verbose else None
        for condvec in condvecs:
            print(f"        >> condvec: {condvec}") if verbose else None
            for model in models:
                print(f"            >> model: {model}") if verbose else None
                for loss_version in [0, 2]:
                    (
                        print(f"                >> loss version: {loss_version}")
                        if verbose
                        else None
                    )
                    for row_number in [
                        50,
                        100,
                        200,
                        300,
                        500,
                        1000,
                        1500,
                        2000,
                        3000,
                        5000,
                        7000,
                        10000,
                        15000,
                        20000,
                        30000,
                        40000,
                        50000,
                        60000,
                        70000,
                    ]:
                        (
                            print(f"                    >> row_number: {row_number}")
                            if verbose
                            else None
                        )

                        if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                            (
                                print(
                                    f"              >> ignore model: {model} and condvec: {condvec}"
                                )
                                if verbose
                                else None
                            )
                            continue

                        if model == "tabddpm":
                            for model_type in ["mlp", "resnet"]:
                                (
                                    print(
                                        f"                      >> model_type: {model_type}"
                                    )
                                    if verbose
                                    else None
                                )
                                path_loss = f"{dir}/{dataset}_rownum-{row_number}_{model}_loss_version-{loss_version}_model-{model_type}_module-gmdp.hyperopt"

                                if "biobank_phase3" in dataset:
                                    write_hyperopt_phase3(
                                        path_loss,
                                        loss_version,
                                        dataset,
                                        model,
                                        condvec,
                                        model_type="",
                                        row_number=row_number,
                                    )

                                (
                                    print(f"                            >> {path_loss}")
                                    if verbose
                                    else None
                                )

                                try:
                                    df_score = udpate_df(
                                        df_score,
                                        path_loss,
                                        loss_version,
                                        dataset,
                                        model,
                                        condvec,
                                        model_type=model_type,
                                        row_number=row_number,
                                    )
                                except:
                                    pass

                        elif model == "tabsyn":
                            path_loss = f"{dir}/{dataset}_rownum-{row_number}_{model}_loss_version-{loss_version}-{condvec}_module-gmdp.hyperopt"

                            if "biobank_phase3" in dataset:
                                write_hyperopt_phase3(
                                    path_loss,
                                    loss_version,
                                    dataset,
                                    model,
                                    condvec,
                                    model_type="",
                                    row_number=row_number,
                                )

                            (
                                print(f"                            >> {path_loss}")
                                if verbose
                                else None
                            )
                            try:
                                df_score = udpate_df(
                                    df_score,
                                    path_loss,
                                    loss_version,
                                    dataset,
                                    model,
                                    condvec,
                                    model_type="",
                                    row_number=row_number,
                                )
                            except:
                                pass

                        else:
                            if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                                (
                                    print(
                                        f"                  >> ignore model: {model} and condvec: {condvec}"
                                    )
                                    if verbose
                                    else None
                                )
                                continue

                            path_loss = f"{dir}/{dataset}_rownum-{row_number}_{model}_loss_version-{loss_version}-{condvec}_module-gmdp.hyperopt"

                            if "biobank_phase3" in dataset:
                                write_hyperopt_phase3(
                                    path_loss,
                                    loss_version,
                                    dataset,
                                    model,
                                    condvec,
                                    model_type="",
                                    row_number=row_number,
                                )

                            (
                                print(f"                        >> {path_loss}")
                                if verbose
                                else None
                            )

                            try:
                                df_score = udpate_df(
                                    df_score,
                                    path_loss,
                                    loss_version,
                                    dataset,
                                    model,
                                    condvec,
                                    model_type="",
                                    row_number=row_number,
                                )
                                a = 2
                            except:
                                pass

    df_score.to_csv(path_df_score, sep="\t", encoding="utf-8")
    return df_score


def run_data_sufficient():
    write_df_data_sufficient(
        datasets=[
            # "biobank_record_vital",
            # "biobank_record_dead",
            # "biobank_record_icd7",
            # "biobank_record_icd9",
            # "biobank_record_icdo2",
            # "biobank_record_icdo3",
            # "biobank_patient_dead",
            # "biobank_sen_meta",
            # "biobank_sen_prostate",
            # "biobank_sen_breast",
            # "biobank_sen_colorectal",
            # "biobank_sen_uroandkid",
            # "biobank_sen_lung",
            # "biobank_sen_pancreatic",
            # "biobank_sen_haematological",
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
        ],
        models=[
            "ctgan",
            "copulagan",
            "dpcgans",
            "ctab",
            "tvae",
            "tabddpm",
            "tabsyn",
        ],
        # condvecs=[0, 1],
        condvecs=[1],
        path_df_score=f"database/results/biobank_compare_gmdp_data_sufficient.csv",
        # dir="database/20250815_gan_optimize_phase3_only_score",
        dir="database/20250824_gan_optimize_phase3_only_score",
        # dir="database/20250218_optimization_biobank_phase1and2_data_sufficient",
        # dir="database/20250218_optimization_biobank_phase1and2_data_sufficient",
    )

    # R = get_rankings()


if __name__ == "__main__":
    # run_ranking()
    run_runtime()
    # run_data_sufficient()
