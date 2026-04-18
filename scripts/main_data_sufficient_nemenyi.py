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


# ==========================================================================================
# Added end here
# ==========================================================================================
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
        dir="database/20250218_optimization_biobank_phase1and2_data_sufficient",
    )

    # R = get_rankings()


if __name__ == "__main__":
    run_ranking()
    # run_data_sufficient()
