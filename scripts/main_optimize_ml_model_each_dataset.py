import argparse
import numpy as np
from rich import print
import gc
from termcolor import colored

from hyperopt import hp, STATUS_OK
from rich import print

from sklearn.model_selection import KFold

import engine.utils.hyperopt_utils as hyperopt_utils
import engine.utils.path_utils as path_utils
from engine.datasets import get_dataset

from engine.experiment_technical_paper import (
    perform_linear_regression,
    perform_svm,
    perform_bagging,
    perform_randomforest,
    perform_xgboost,
)


import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    try:
        print("Current device:", torch.cuda.get_device_name(0))
    except:
        pass

    try:
        print("Current device:", torch.cuda.get_device_name(1))
    except:
        pass

else:
    print("CUDA is not available.")


# arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--is_test",
    # default=1,
    default=0,
    type=int,
)
parser.add_argument(
    "--max_trials",
    type=int,
    default=30,
)
parser.add_argument(
    "--dataset",
    default="adult",
    # "adult",
    # "mnist12",
    # "mnist28",
    # "news",
    # "diabetes",
    # "diabetesbalanced",
    # "house",
    # "covertype",
    # "credit",
    # "intrusion",
    # "abalone",
    # "buddy",
    # "gesture",
    # "biobank_record_vital",
    # "biobank_record_dead",
    # "biobank_patient_dead",
    type=str,
)
parser.add_argument(
    "--ml_method",
    default="regression",
    # "regression"
    # "svm"
    # "randomforest"
    # "bagging"
    # "xgboost"
)
parser.add_argument(
    "--bo_method",
    default="ior",
    # default="sbo",
)
parser.add_argument(
    "--bo_method_agg",
    # default="mean",
    default="median",
)
args = parser.parse_args()


# ===========================================================================
# objective
# ===========================================================================


def objective(params):
    dict_method = {
        "svm": perform_svm,
        "randomforest": perform_randomforest,
        "bagging": perform_bagging,
        "xgboost": perform_xgboost,
    }

    dict_method["regression"] = perform_linear_regression

    if (
        args.dataset
        in [
            "biobank_record_icd7",
            "biobank_record_icd9",
            "biobank_record_icdo2",
            "biobank_record_icdo3",
        ]
        and args.ml_method == "randomforest"
    ):
        device = "cpu"
    else:
        device = "gpu"  # bug with gpu

    if args.dataset in [
        "biobank_phase3_cancer_all",
        "biobank_phase3_cancer_bio_all",
        "biobank_phase3_cancer_150",
        "biobank_phase3_cancer_151",
        "biobank_phase3_cancer_153",
        "biobank_phase3_cancer_154",
        "biobank_phase3_cancer_155",
        "biobank_phase3_cancer_156",
        "biobank_phase3_cancer_157",
        "biobank_phase3_cancer_162",
        "biobank_phase3_cancer_172",
        "biobank_phase3_cancer_173",
        "biobank_phase3_cancer_174",
        "biobank_phase3_cancer_180",
        "biobank_phase3_cancer_182",
        "biobank_phase3_cancer_183",
        "biobank_phase3_cancer_185",
        "biobank_phase3_cancer_188",
        "biobank_phase3_cancer_189",
        "biobank_phase3_cancer_191",
        "biobank_phase3_cancer_192",
        "biobank_phase3_cancer_193",
        "biobank_phase3_cancer_194",
        "biobank_phase3_cancer_199",
        "biobank_phase3_cancer_200",
        "biobank_phase3_cancer_203",
        "biobank_phase3_cancer_204",
        "biobank_phase3_cancer_205",
    ]:
        is_balanced = True  # we use cpu for RF,

    print(params)

    try:
        D = get_dataset(args.dataset)
        X = D.data_train[D.features]
        y = D.data_train[D.target]

        # print(f"features", D.features)
        # print(f"target", D.target)

        if args.is_test:
            X = X.head(100)
            y = y.head(100)

        f = dict_method[args.ml_method]

        # Number of folds
        n_splits = 5

        # Initialize KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Initialize lists to hold data for each fold
        folds = []

        # Split data
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            folds.append((X_train, y_train, X_test, y_test))

        loss = 0  # temporary loss, we will iteratively update these losses later
        scores = {}
        # Print the results for verification
        for i, (X_train, y_train, X_test, y_test) in enumerate(folds):
            print(f"Fold {i+1}:")
            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            print(f"y_test shape: {y_test.shape}\n")

            if D.output == "classification":
                acc, precision, recall, f1, gmean, roc = f(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    output=D.output,
                    params=params,
                    device=device,
                )
                scores[f"fold{i}_acc"] = acc
                scores[f"fold{i}_precision"] = precision
                scores[f"fold{i}_recall"] = recall
                scores[f"fold{i}_f1"] = f1
                scores[f"fold{i}_gmean"] = gmean
                scores[f"fold{i}_roc"] = roc
            else:
                mae, mse, r2 = f(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    output=D.output,
                    params=params,
                    device=device,
                )
                scores[f"fold{i}_mae"] = mae
                scores[f"fold{i}_mse"] = mse
                scores[f"fold{i}_r2"] = r2

        # Find best loss
        reason = "success"
        gc.collect()

    except Exception as e:
        reason = str(e)
        loss = np.inf

    gc.collect()
    print(f"reason: {reason}")

    d = {
        "loss": loss,
        "status": STATUS_OK,
        "reason": reason,
        "params": params,
        "scores_ml": scores,
        "device": device,
    }

    print(d)
    return d


def init_search_space(args, output):
    if args.ml_method == "regression":
        if output == "classification":  # Logistic Regression
            search_space = {
                "algorithm": hp.choice(
                    "algorithm", ["svd", "eig", "qr", "svd-qr", "svd-jacobi"]
                ),
                "C": hp.loguniform("C", -4, 4),
                "solver": hp.choice(
                    "solver",
                    [
                        "newton-cg",
                        "lbfgs",
                        "liblinear",
                        "sag",
                        "saga",
                    ],
                ),
                "max_iter": hp.quniform("max_iter", 50, 200, 1),
                "max_iter": hp.quniform("max_iter", 50, 200, 1),
                "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
                "class_weight": hp.choice("class_weight", ["balanced", None]),
                # "class_weight": hp.choice("class_weight", ["balanced"]),
            }
        else:  # ElasticNet
            search_space = {
                "alpha": hp.uniform("alpha", 0.1, 10),
                "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
                "fit_intercept": hp.choice("fit_intercept", [True, False]),
                "normalize": hp.choice("normalize", [True, False]),
                "max_iter": hp.choice("max_iter", list(range(100, 2000, 1))),
                "tol": hp.loguniform("tol", np.log(1e-5), np.log(1e-1)),
            }

    elif args.ml_method == "bagging":
        if output == "classification":  # Logistic Regression
            search_space = {
                "algorithm": hp.choice(
                    "algorithm", ["svd", "eig", "qr", "svd-qr", "svd-jacobi"]
                ),
                "C": hp.loguniform("C", -4, 4),
                "solver": hp.choice("solver", ["qn"]),
                "max_iter": hp.quniform("max_iter", 50, 200, 1),
                "max_iter": hp.quniform("max_iter", 50, 200, 1),
                "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
                "class_weight": hp.choice("class_weight", ["balanced", None]),
                # "class_weight": hp.choice("class_weight", ["balanced"]),
            }
        else:  # ElasticNet
            search_space = {
                "alpha": hp.uniform("alpha", 0.1, 10),
                "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
                "fit_intercept": hp.choice("fit_intercept", [True, False]),
                "normalize": hp.choice("normalize", [True, False]),
                "max_iter": hp.choice("max_iter", list(range(100, 2000, 1))),
                "tol": hp.loguniform("tol", np.log(1e-5), np.log(1e-1)),
            }

    elif args.ml_method == "svm":
        if output == "classification":  # LinearSVC
            search_space = {
                "C": hp.uniform("C", 0.1, 10),
                "max_iter": hp.choice("max_iter", list(range(100, 1500, 1))),
                "tol": hp.loguniform("tol", -5, -1),
                "penalty": hp.choice("penalty", ["l1", "l2"]),
                "loss": hp.choice("loss", ["hinge", "squared_hinge"]),
                "fit_intercept": hp.choice("fit_intercept", [True, False]),
                "penalized_intercept": hp.choice("penalized_intercept", [True, False]),
                "class_weight": hp.choice("class_weight", ["balanced", None]),
            }
        else:  # LinearSVR
            search_space = {
                "C": hp.uniform("C", 0.1, 10),
                "epsilon": hp.uniform("epsilon", 0.0, 1.0),
                "max_iter": hp.choice("max_iter", list(range(100, 1500, 1))),
                "tol": hp.loguniform("tol", -5, -1),
                "fit_intercept": hp.choice("fit_intercept", [True, False]),
                "penalized_intercept": hp.choice("penalized_intercept", [True, False]),
            }
    elif args.ml_method == "randomforest":
        if output == "classification":  # RandomForestClassifier
            search_space = {
                "n_estimators": hp.choice("n_estimators", list(range(50, 500, 1))),
                "max_depth": hp.choice("max_depth", list(range(10, 100, 1))),
                "min_samples_split": hp.choice(
                    "min_samples_split", list(range(2, 20, 1))
                ),
                "min_samples_leaf": hp.choice(
                    "min_samples_leaf", list(range(1, 20, 1))
                ),
                "max_features": hp.choice("max_features", ["sqrt", "log2"]),
            }
        else:  # RandomForestRegressor
            search_space = {
                "n_estimators": hp.choice("n_estimators", list(range(50, 500, 1))),
                "max_depth": hp.choice("max_depth", list(range(10, 100, 1))),
                "min_samples_split": hp.choice(
                    "min_samples_split", list(range(2, 20, 1))
                ),
                "min_samples_leaf": hp.choice(
                    "min_samples_leaf", list(range(1, 20, 1))
                ),
                "max_features": hp.choice("max_features", ["sqrt", "log2"]),
            }

    elif args.ml_method == "xgboost":
        if output == "classification":  # XGBClassifier
            search_space = {
                "n_estimators": hp.choice("n_estimators", list(range(50, 500, 1))),
                "max_depth": hp.choice("max_depth", list(range(3, 15, 1))),
                "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
                "subsample": hp.uniform("subsample", 0.5, 1.0),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
                "gamma": hp.uniform("gamma", 0, 5),
                "reg_alpha": hp.uniform("reg_alpha", 0, 1),
                "reg_lambda": hp.uniform("reg_lambda", 0, 1),
                "scale_pos_weight": hp.uniform("scale_pos_weight", 1, 10),
            }
        else:  # XGBRegressor
            search_space = {
                "n_estimators": hp.choice("n_estimators", list(range(50, 500, 1))),
                "max_depth": hp.choice("max_depth", list(range(3, 15, 1))),
                "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
                "subsample": hp.uniform("subsample", 0.5, 1.0),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
                "gamma": hp.uniform("gamma", 0, 5),
                "reg_alpha": hp.uniform("reg_alpha", 0, 1),
                "reg_lambda": hp.uniform("reg_lambda", 0, 1),
                "scale_pos_weight": hp.uniform("scale_pos_weight", 1, 10),
            }

    return search_space


# ===========================================================================
# main
# ===========================================================================
if __name__ == "__main__":
    # loop indefinitely and stop whenever you like
    max_successful_trials = args.max_trials

    # init search space
    D = get_dataset(args.dataset)
    search_space = init_search_space(args, D.output)
    print(search_space)

    # hyperopt
    database_path = "database"
    filename = f"{args.dataset}_{args.ml_method}"

    if args.bo_method == "ior":
        folder = "optimization_ml_method"
    elif args.bo_method == "sbo":
        folder = f"optimization_ml_method_sbo_{args.bo_method_agg}"

    hyperopt_project_path = path_utils.get_hyperopt_path(
        filename, database_path=database_path, folder=folder
    )

    trials = hyperopt_utils.load_project(hyperopt_project_path)
    algo = "tpe"
    is_continue = True

    if len(trials.trials) > 0:
        n_successful_trials = hyperopt_utils.get_number_successful_trials(
            trials, success_value="success"
        )
        n_trials = hyperopt_utils.get_number_trials(trials)
        if n_successful_trials >= max_successful_trials or (
            n_successful_trials == 0 and n_trials >= args.max_trials
        ):
            is_continue = False

    if args.bo_method == "ior":
        I = hyperopt_utils.IncrementalObjectiveOptimizationMLMethod(
            hyperopt_project_path,
        )
    else:
        agg = args.bo_method_agg
        I = hyperopt_utils.StandardObjectiveOptimizationMLMethod(
            hyperopt_project_path,
            agg=agg,
        )

    while is_continue:
        print(colored("=" * 100, "red"))
        n_trials, n_successful_trials = hyperopt_utils.run_trials(
            project_path=hyperopt_project_path,
            objective=objective,
            space=search_space,
            algo=algo,
        )
        I.update_trials_losses(evaluations=["ml"])
        if n_successful_trials >= max_successful_trials:
            is_continue = False
            print(colored("=" * 100, "red"))
            print(colored("Done", "red"))
            print(colored("=" * 100, "red"))
        if n_successful_trials == 0 and n_trials >= args.max_trials:
            is_continue = False
            print(colored("=" * 100, "red"))
            print(colored("Done but n_successful_trials = 0", "red"))
            print(colored("=" * 100, "red"))
