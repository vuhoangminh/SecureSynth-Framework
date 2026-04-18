import argparse
import pandas as pd
import gc
from termcolor import colored

from hyperopt import hp, STATUS_OK
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

import engine.utils.hyperopt_utils as hyperopt_utils
import engine.utils.path_utils as path_utils

from engine._legacy_experiment_biobank import (
    perform_xgboost,
    perform_svm,
    perform_linear_regression,
    perform_bagging,
    perform_randomforest,
    preprocess_record_dead,
    preprocess_record_vital,
    preprocess_record_icd,
    perform_ae,
)


# Try these: https://stackoverflow.com/questions/31681373/making-svm-run-faster-in-python
# CUDA_VISIBLE_DEVICES=0 nohup python scripts/main_optimize.py --method svm --is_test 0 --task dead --is_ae 0  > d0.log &
# CUDA_VISIBLE_DEVICES=0 nohup python scripts/main_optimize.py --method xgboost --is_test 0 --task dead --is_ae 0  > d0.log &


"""
CUDA_VISIBLE_DEVICES=0 nohup python scripts/main_optimize.py --method xgboost --is_test 0 --task dead --is_ae 1 --autoencoder_input 3 --autoencoder_output 2  > d2.log &
CUDA_VISIBLE_DEVICES=0 nohup python scripts/main_optimize.py --method xgboost --is_test 0 --task dead --is_ae 1 --autoencoder_input 4 --autoencoder_output 3  > d2.log &

CUDA_VISIBLE_DEVICES=1 nohup python scripts/main_optimize.py --method reg --is_test 0 --task dead --is_ae 1 --autoencoder_input 5 --autoencoder_output 4  > d0.log &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/main_optimize.py --method reg --is_test 0 --task dead --is_ae 1 --autoencoder_input 5 --autoencoder_output 4  > d1.log &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/main_optimize.py --method reg --is_test 0 --task dead --is_ae 1 --autoencoder_input 5 --autoencoder_output 4  > d2.log &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/main_optimize.py --method reg --is_test 0 --task dead --is_ae 1 --autoencoder_input 5 --autoencoder_output 4  > d3.log &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/main_optimize.py --method reg --is_test 0 --task dead --is_ae 1 --autoencoder_input 5 --autoencoder_output 4  > d4.log &
"""


# ===========================================================================
# arguments
# ===========================================================================
global args

# arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-t", "--tuner", type=str, default="tpe", choices=["tpe", "randomsearch"]
)
parser.add_argument(
    "-m",
    "--method",
    type=str,
    default="reg",
    choices=["reg", "svm", "xgboost", "nn", "bagging", "randomforest"],
)
parser.add_argument("-te", "--is_test", type=int, default=1)
parser.add_argument("-mt", "--max_trials", type=int, default=1000)
parser.add_argument(
    "--task",
    type=str,
    default="dead",
    choices=["dead", "vital", "icd7_code", "icd9_code", "icdo2_code", "icdo3_code"],
)
parser.add_argument(
    "--is_ae",
    type=int,
    default=0,
)
parser.add_argument(
    "--autoencoder_input",
    type=int,
    default=4,
    choices=[0, 3, 4, 5, 6, 7],
)
parser.add_argument(
    "--autoencoder_output",
    type=int,
    default=2,
    choices=[0, 2, 3, 4, 5, 6],
)
parser.add_argument(
    "-r",
    "--resource",
    type=str,
    default="desktop",
    choices=["desktop", "gauss", "laplace", "alvis"],
)
parser.add_argument(
    "--device",
    type=str,
    default="gpu",
    choices=["cpu", "gpu"],
)
args = parser.parse_args()

if args.is_ae:
    assert args.autoencoder_output < args.autoencoder_input


if args.resource in ["desktop", "gauss", "laplace"]:
    per_process_gpu_memory_fraction = 1.0 / 5.0  # 11 GB -> 200 MB
else:
    per_process_gpu_memory_fraction = 1.0 / 16.0  # 48 GB -> 300 MB

# gpu_options = tf.compat.v1.GPUOptions(
#     per_process_gpu_memory_fraction=per_process_gpu_memory_fraction
# )
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# ===========================================================================
# preprocess
# ===========================================================================
def prepare_data():
    df = pd.read_csv(f"database/20230524_biobank_data.tsv", sep="\t", header=0)
    df = df.fillna(-1)

    if args.is_test:
        frac = 0.1
    else:
        frac = 1

    print()
    print(f"FRAC: {frac}")
    print()

    if args.task == "vital":
        X, y = preprocess_record_vital(df, is_original=1, frac=frac)
    elif args.task == "dead":
        X, y = preprocess_record_dead(df, is_original=1, frac=frac)
    elif args.task in [
        "icd7_code",
        "icd9_code",
        "icdo2_code",
        "icdo3_code",
    ]:
        X, y = preprocess_record_icd(df, is_original=1, frac=frac, code=args.task)

    if args.is_ae:
        X_encoded = perform_ae(X, args.autoencoder_input, args.autoencoder_output)
        return X_encoded, y
    else:
        return X, y


# ===========================================================================
# objective
# ===========================================================================
def objective(params):
    print(params)

    try:
        X, y = prepare_data()

        if args.method == "reg":
            params = params_reg = {
                "penalty": params["penalty"],
                "solver": params["solver"],
                "multi_class": params["multi_class"],
                "class_weight": params["class_weight"],
                "C": params["C"],
                "l1_ratio": params["l1_ratio"],
                "max_iter": params["max_iter"],
            }
            acc = perform_linear_regression(X, y, params=params, device="gpu")
        elif args.method == "svm":
            if args.device == "cpu":
                params = params_svm = {
                    "C": params["C"],
                    "kernel": params["kernel"],
                    "degree": params["degree"],
                    "gamma": params["gamma"],
                    "class_weight": params["class_weight"],
                    "decision_function_shape": params["decision_function_shape"],
                }
            else:
                params = params_svm = {
                    "C": params["C"],
                    "lbfgs_memory": params["lbfgs_memory"],
                    "penalty": params["penalty"],
                    "loss": params["loss"],
                }
            acc = perform_svm(X, y, params=params, device="gpu")
        elif args.method == "xgboost":
            params = params_xgboost = {
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "max_bin": params["max_bin"],
                "learning_rate": params["learning_rate"],
                "gamma": params["gamma"],
                "eval_metric": params["eval_metric"],
            }
            acc = perform_xgboost(X, y, params=params, device="gpu")
        elif args.method == "bagging":
            params = params_bagging = {
                "C": params["C"],
                "kernel": params["kernel"],
                "degree": params["degree"],
                "gamma": params["gamma"],
                "class_weight": params["class_weight"],
                "decision_function_shape": params["decision_function_shape"],
            }
            acc = perform_bagging(X, y, params=params, device="gpu")
        elif args.method == "randomforest":
            params = params_randomforest = {
                "n_estimators": params["n_estimators"],
                "min_samples_split": params["min_samples_split"],
                "min_samples_leaf": params["min_samples_leaf"],
                "n_jobs": -1,
            }
            acc = perform_randomforest(X, y, params=params, device="cpu")
        else:
            params = {}
            acc = 0

        # Find best loss
        reason = "success"
        loss = -acc

    except Exception as e:
        reason = str(e)
        loss = 0

    gc.collect()
    print(f"reason: {reason}")

    return {
        "loss": loss,
        "status": STATUS_OK,
        "reason": reason,
        "params": params,
    }


# ===========================================================================
# main
# ===========================================================================
if __name__ == "__main__":
    # loop indefinitely and stop whenever you like
    max_successful_trials = args.max_trials

    # search space
    step_weight = 0.0001
    step_threshold = 0.01
    step_threshold_percent = 0.1

    # Search space for Logistic Regression - start
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    space_reg = {
        "penalty": hp.choice("penalty", ["l1", "l2", "elasticnet", None]),
        "solver": hp.choice(
            "solver",
            ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        ),
        "multi_class": hp.choice("multi_class", ["ovr", "multinomial"]),
        "class_weight": hp.choice("class_weight", ["balanced", None]),
        "C": hp.uniform("C", 0.1, 10),
        "l1_ratio": hp.uniform("l1_ratio", 0, 1),
        "max_iter": scope.int(hp.quniform("max_iter", 100, 1000, 1)),
    }
    # Search space for Logistic Regression - end

    # Search space for Support Vector Classification - start
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    space_svm = {
        "C": hp.uniform("C", 0.1, 10),
        "kernel": hp.choice(
            # "kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"]
            # "kernel", ["linear", "rbf", "sigmoid"]
            "kernel",
            ["rbf", "sigmoid"],
        ),
        # "degree": sample(scope.int(hp.quniform("degree", 3, 10, 1))),
        "degree": hp.choice("degree", [3]),
        "gamma": hp.choice("gamma", ["scale", "auto"]),
        "class_weight": hp.choice("class_weight", ["balanced", None]),
        "decision_function_shape": hp.choice("decision_function_shape", ["ovo", "ovr"]),
    }
    space_svm_gpu = {
        "C": hp.uniform("C", 0.1, 10),
        "lbfgs_memory": scope.int(hp.quniform("lbfgs_memory", 1, 10, 1)),
        "penalty": hp.choice("penalty", ["l1", "l2"]),
        "loss": hp.choice("loss", ["hinge", "squared_hinge"]),
    }
    # Search space for Support Vector Classification - end

    # Search space for XGBoost - start
    # https://xgboost.readthedocs.io/en/stable/python/python_api.html
    space_xgboost = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 5, 1000, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 2, 10, 1)),
        "max_bin": hp.choice("max_bin", [64, 128, 256, 512]),
        "learning_rate": hp.uniform("learning_rate", 0, 1),
        "gamma": hp.uniform("gamma", 0, 100),
        "eval_metric": hp.choice("eval_metric", ["error"]),
    }
    # Search space for XGBoost - end

    # Search space for Bagging - Support Vector Classification - start
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    space_bagging = {
        "C": hp.uniform("C", 0.1, 10),
        "kernel": hp.choice(
            # "kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"]
            # "kernel", ["linear", "rbf", "sigmoid"]
            "kernel",
            ["rbf", "sigmoid"],
        ),
        # "degree": sample(scope.int(hp.quniform("degree", 3, 10, 1))),
        "degree": hp.choice("degree", [3]),
        "gamma": hp.choice("gamma", ["scale", "auto"]),
        "class_weight": hp.choice("class_weight", ["balanced", None]),
        "decision_function_shape": hp.choice("decision_function_shape", ["ovo", "ovr"]),
        "n_jobs": hp.choice("n_jobs", [-1]),
    }
    # Search space for Bagging - Support Vector Classification - end

    # Search space for RandomForestClassifier - start
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    space_randomforest = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 5, 200, 1)),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 10, 1)),
        "n_jobs": hp.choice("n_jobs", [-1]),
    }
    # Search space for RandomForestClassifier - end

    if args.method == "reg":
        space = space_reg
    elif args.method == "svm" and args.device == "cpu":
        space = space_svm
    elif args.method == "svm" and args.device == "gpu":
        space = space_svm_gpu
    elif args.method == "xgboost":
        space = space_xgboost
    elif args.method == "bagging":
        space = space_bagging
    elif args.method == "randomforest":
        space = space_randomforest
    else:
        space = None

    database_path = "database_test" if args.is_test else "database"

    if args.is_ae:
        hyperopt_project_path = path_utils.get_hyperopt_path(
            f"{args.task}_{args.method}_with_ae_{args.autoencoder_input}_{args.autoencoder_output}",
            database_path=database_path,
        )
    else:
        hyperopt_project_path = path_utils.get_hyperopt_path(
            f"{args.task}_{args.method}_without_ae", database_path=database_path
        )

    trials = hyperopt_utils.load_project(hyperopt_project_path)
    is_continue = True
    if len(trials.trials) > 0:
        n_successful_trials = hyperopt_utils.get_number_successful_trials(trials)
        if n_successful_trials >= max_successful_trials:
            is_continue = False

    while is_continue:
        print(colored("=" * 100, "red"))

        if len(trials.trials) < max_successful_trials / 2:  # random
            n_trials, n_successful_trials = hyperopt_utils.run_trials(
                project_path=hyperopt_project_path,
                objective=objective,
                space=space,
                algo="random",
            )
        else:  # tpe
            n_trials, n_successful_trials = hyperopt_utils.run_trials(
                project_path=hyperopt_project_path,
                objective=objective,
                space=space,
                algo="tpe",
            )

        if n_successful_trials >= max_successful_trials:
            is_continue = False
            print(colored("=" * 100, "red"))
            print(colored("Done", "red"))
            print(colored("=" * 100, "red"))
