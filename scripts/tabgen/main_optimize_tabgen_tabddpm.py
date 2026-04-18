import os
import json
import time
import argparse
import subprocess
import numpy as np
from rich import print
import gc
from termcolor import colored
import random

from hyperopt import hp, STATUS_OK
from rich import print

import engine.utils.hyperopt_utils as hyperopt_utils
import engine.utils.path_utils as path_utils

from engine.evaluate_technical_paper import (
    compute_statistical_metrics,
    compute_ml_metrics_all_ml_methods,
    compute_dp_metrics,
)

from engine.datasets import get_dataset
from engine.utils.data_utils import get_epochs_max_and_max_trials
from engine.config import config
from models.tab_ddpm.tab_ddpm.utils import FoundNANsError
import models.tab_ddpm.lib as lib


# ===========================================================================
# objective
# ===========================================================================
def update_params(params):
    for key, value in params.items():
        if key in [
            "generator_lr",
            "generator_decay",
            "discriminator_lr",
            "discriminator_decay",
            "l2scale",
            "is_loss_corr",
            "is_loss_dwp",
            "is_loss_corr_num",
            "is_loss_dwp_num",
        ]:
            params[key] = 10**value
    return params


def update_cmd(params, cmd):
    updated_cmd = cmd
    for key, value in params.items():
        updated_cmd += f" --{key} {value}"
    return updated_cmd


def add_dict_to_args(parser, dictionary):
    """Adds a dictionary of arguments to a copy of the argparse parser.

    Args:
      parser: The argparse parser object.
      dictionary: The dictionary of arguments to add.

    Returns:
      A new argparse parser with the added arguments.
    """
    import copy

    new_parser = copy.deepcopy(parser)
    for key, value in dictionary.items():
        new_parser.add_argument(
            f"--{key}", type=type(value), default=value, help=f"Value for {key}"
        )
    return new_parser


def init_search_space(args):

    def suggest_dim(name):
        l = list(range(d_min, d_max + 1, 1))
        return hp.choice(name, l)

    search_space = {}

    if args.model_type == "mlp":
        min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
        search_space["n_layers"] = hp.choice(
            "n_layers", list(range(min_n_layers, max_n_layers + 1, 1))
        )
        search_space["d_first"] = suggest_dim("d_first")
        search_space["d_middle"] = suggest_dim("d_middle")
        search_space["d_last"] = suggest_dim("d_last")
    elif args.model_type == "resnet":
        min_n_layers, max_n_layers = 1, 8
        search_space["n_blocks"] = hp.choice(
            "n_blocks", list(range(min_n_layers, max_n_layers + 1, 1))
        )
        search_space["d_main"] = hp.choice("d_main", list(range(64, 1024 + 1, 1)))
        search_space["d_hidden"] = hp.choice("d_hidden", list(range(64, 1024 + 1, 1)))
        search_space["dropout_first"] = hp.uniform("dropout_first", 0.0, 0.5)
        search_space["dropout_second"] = hp.uniform("dropout_second", 0.0, 0.5)

        """
            d_in: the input size
            n_blocks: the number of Blocks
            d_main: the input size (or, equivalently, the output size) of each Block
            d_hidden: the output size of the first linear layer in each Block
            dropout_first: the dropout rate of the first dropout layer in each Block.
            dropout_second: the dropout rate of the second dropout layer in each Block.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021

        Layers (A) UniformInt[1, 8], (B) UniformInt[1, 16]
        Layer size (A) UniformInt[64, 512], (B) UniformInt[64, 1024]
        Hidden factor (A,B) Uniform[1, 4]
        Hidden dropout (A,B) Uniform[0, 0.5]
        Residual dropout (A,B) {0, Uniform[0, 0.5]}
        Learning rate (A,B) LogUniform[1e-5, 1e-2]
        Weight decay (A,B) {0, LogUniform[1e-6, 1e-3
        """

    search_space["lr"] = hp.uniform("lr", 0.00001, 0.003)
    search_space["weight_decay"] = hp.choice("weight_decay", [0.0])
    search_space["gaussian_loss_type"] = hp.choice("gaussian_loss_type", ["mse"])
    search_space["batch_size"] = hp.choice("batch_size", [256, 4096])
    search_space["num_timesteps"] = hp.choice("num_timesteps", [100, 1000])

    if args.is_test:
        search_space["steps"] = hp.choice("steps", [1000])
    else:
        search_space["steps"] = hp.choice("steps", [5000, 20000, 30000])

    if args.loss_version != 0:
        search_space.update(
            {
                "is_loss_corr": hp.uniform("is_loss_corr", -2, 4),
                "is_loss_dwp": hp.uniform("is_loss_dwp", -14, -2),
                "n_moment_loss_dwp": hp.choice("n_moment_loss_dwp", [1, 2, 3, 4]),
                "is_loss_num": hp.choice("is_loss_num", [0, 1]),
                "is_loss_corr_num": hp.uniform("is_loss_corr_num", -2, 4),
                "is_loss_dwp_num": hp.uniform("is_loss_dwp_num", -40, -6),
                "n_moment_loss_dwp_num": hp.choice("n_moment_loss_dwp_num", [1, 2]),
            }
        )

    return search_space


def generate_config_toml(base_config_path, params, dir_logs):
    base_config = lib.load_config(base_config_path)

    base_config["train"]["main"]["lr"] = params["lr"]
    base_config["train"]["main"]["steps"] = params["steps"]
    base_config["train"]["main"]["batch_size"] = params["batch_size"]
    base_config["train"]["main"]["weight_decay"] = params["weight_decay"]

    if args.loss_version == 2:
        base_config["train"]["main"]["loss_version"] = 2
        base_config["train"]["main"]["is_loss_corr"] = params["is_loss_corr"]
        base_config["train"]["main"]["is_loss_dwp"] = params["is_loss_dwp"]
        base_config["train"]["main"]["n_moment_loss_dwp"] = params["n_moment_loss_dwp"]
        base_config["train"]["main"]["is_loss_num"] = params["is_loss_num"]
        base_config["train"]["main"]["is_loss_corr_num"] = params["is_loss_corr_num"]
        base_config["train"]["main"]["is_loss_dwp_num"] = params["is_loss_dwp_num"]
        base_config["train"]["main"]["n_moment_loss_dwp_num"] = params[
            "n_moment_loss_dwp_num"
        ]
    ## Added by Minh - handle loss_version 4 and 5
    elif args.loss_version == 4:  # only correlation loss
        base_config["train"]["main"]["loss_version"] = 4
        base_config["train"]["main"]["is_loss_corr"] = params["is_loss_corr"]
        base_config["train"]["main"]["is_loss_dwp"] = 0
        base_config["train"]["main"]["n_moment_loss_dwp"] = 0
        base_config["train"]["main"]["is_loss_num"] = 0
        base_config["train"]["main"]["is_loss_corr_num"] = params["is_loss_corr_num"]
        base_config["train"]["main"]["is_loss_dwp_num"] = 0
        base_config["train"]["main"]["n_moment_loss_dwp_num"] = 0
    elif args.loss_version == 5:  # only distribution loss
        base_config["train"]["main"]["loss_version"] = 5
        base_config["train"]["main"]["is_loss_corr"] = 0
        base_config["train"]["main"]["is_loss_dwp"] = params["is_loss_dwp"]
        base_config["train"]["main"]["n_moment_loss_dwp"] = params["n_moment_loss_dwp"]
        base_config["train"]["main"]["is_loss_num"] = params["is_loss_num"]
        base_config["train"]["main"]["is_loss_corr_num"] = 0
        base_config["train"]["main"]["is_loss_dwp_num"] = params["is_loss_dwp_num"]
        base_config["train"]["main"]["n_moment_loss_dwp_num"] = params[
            "n_moment_loss_dwp_num"
        ]
    ## Added by Minh - handle loss_version 4 and 5
    else:
        base_config["train"]["main"]["loss_version"] = 0
        base_config["train"]["main"]["is_loss_corr"] = 0
        base_config["train"]["main"]["is_loss_dwp"] = 0
        base_config["train"]["main"]["n_moment_loss_dwp"] = 0
        base_config["train"]["main"]["is_loss_num"] = 0
        base_config["train"]["main"]["is_loss_corr_num"] = 0
        base_config["train"]["main"]["is_loss_dwp_num"] = 0
        base_config["train"]["main"]["n_moment_loss_dwp_num"] = 0

    if args.model_type == "mlp":
        n_layers = 2 * params["n_layers"]
        d_first = [2 ** params["d_first"]] if n_layers else []
        d_middle = [2 ** params["d_middle"]] * (n_layers - 2) if n_layers > 2 else []
        d_last = [2 ** params["d_last"]] if n_layers > 1 else []
        d_layers = d_first + d_middle + d_last
        base_config["model_params"]["rtdl_params"]["d_layers"] = d_layers
    elif args.model_type == "resnet":
        base_config["model_params"]["rtdl_params"]["n_blocks"] = params["n_blocks"]
        base_config["model_params"]["rtdl_params"]["d_main"] = params["d_main"]
        base_config["model_params"]["rtdl_params"]["d_hidden"] = params["d_hidden"]
        base_config["model_params"]["rtdl_params"]["dropout_first"] = params[
            "dropout_first"
        ]
        base_config["model_params"]["rtdl_params"]["dropout_second"] = params[
            "dropout_second"
        ]

    base_config["eval"]["type"]["eval_type"] = "synthetic"
    base_config["diffusion_params"]["gaussian_loss_type"] = params["gaussian_loss_type"]
    base_config["diffusion_params"]["num_timesteps"] = params["num_timesteps"]
    base_config["parent_dir"] = dir_logs

    lib.dump_config(base_config, f"{dir_logs}/config.toml")

    return f"{dir_logs}/config.toml"


def objective(params):
    def construct_return_dict(
        loss,
        reason,
        params,
        df_score,
        df_score_ml,
        df_score_augment,
        df_score_dp,
        dir_logs,
    ):
        return {
            "loss": loss,
            "status": STATUS_OK,
            "reason": reason,
            "params": params,
            "scores_statistics": (
                df_score.iloc[0].to_dict() if df_score is not None else {}
            ),
            "scores_ml": (
                df_score_ml.iloc[0].to_dict() if df_score_ml is not None else {}
            ),
            "scores_ml_augment": (
                df_score_augment.iloc[0].to_dict()
                if df_score_augment is not None
                else {}
            ),
            "scores_dp": (
                df_score_dp.iloc[0].to_dict() if df_score_dp is not None else {}
            ),
            "dir_logs": dir_logs if "dir_logs" in locals() else None,
        }

    params = update_params(params)
    params["epochs"] = params["steps"]

    print(params)

    try:
        new_parser = add_dict_to_args(parser, params)
        new_args = new_parser.parse_args()
        dir_logs = os.path.join(
            args.dir_logs, path_utils.get_folder_technical_paper(new_args)
        )
        path_utils.make_dir(dir_logs)

        D = get_dataset(new_args.dataset)

        if args.model_type == "mlp":
            base_config_path = f"database/dataset/{args.dataset}/config.toml"
        elif args.model_type == "resnet":
            base_config_path = f"database/dataset/{args.dataset}/config_resnet.toml"

        config_path = generate_config_toml(base_config_path, params, dir_logs)

        cmd = f"python -W ignore models/tab_ddpm/scripts/pipeline.py --config {config_path} --dataset {args.dataset}"
        print(f">> logging to {dir_logs}")
        print(f">> running {cmd}")

        # os.system(cmd)

        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # If you need to print or use the output, use result.stdout.decode()
        print(result.stdout.decode())

        folder = path_utils.get_filename(dir_logs)
        discrete_columns = D.discrete_columns
        continuous_columns = [
            c for c in list(D.data_train) if c not in discrete_columns
        ]

        print(">> compute metrics...")
        df_score = compute_statistical_metrics(
            None,
            folder,
            discrete_columns,
            continuous_columns,
            mode="last",
        )
        df_score = df_score.drop(df_score.columns[:6], axis=1)  # Drop first 6 columns
        df_score = df_score.drop(df_score.columns[-2:], axis=1)  # Drop last 2 columns

        df_score_ml = compute_ml_metrics_all_ml_methods(
            D,
            None,
            folder,
            mode="last",
            task="single",
        )
        df_score_ml = df_score_ml.drop(df_score_ml.columns[:6], axis=1)

        df_score_augment = compute_ml_metrics_all_ml_methods(
            D,
            None,
            folder,
            mode="last",
            task="augment",
        )
        df_score_augment = df_score_augment.drop(df_score_augment.columns[:6], axis=1)

        if args.module != "public":
            df_score_dp = compute_dp_metrics(
                D,
                None,
                folder,
                key_fields=D.key_fields,
                sensitive_fields=D.sensitive_fields,
            )
            df_score_dp = df_score_dp.drop(df_score_dp.columns[:6], axis=1)
        else:
            df_score_dp = None

        # Find best loss
        print(">> done in try...")
        loss = 0  # we will update later
        reason = "success"

        gc.collect()
        print(f"reason: {reason}")

        print(">> return at objective...")
        return construct_return_dict(
            loss,
            reason,
            params,
            df_score,
            df_score_ml,
            df_score_augment,
            df_score_dp,
            dir_logs,
        )

    except subprocess.CalledProcessError as e:
        reason = f"Command '{cmd}' failed with return code {e.returncode}"
        print(reason)
        print(e.stderr.decode())
        loss = np.inf

    except FoundNANsError as e:  # synthetic tensor contains nan values
        reason = str(e)
        loss = np.inf

    except AssertionError as e:  # synthetic tensor contains nan values
        reason = str(e)
        loss = np.inf

    except Exception as e:
        reason = str(e)
        loss = np.inf

    # In case of an error, return the failure response with loss set to infinity
    return construct_return_dict(loss, reason, params, None, None, None, None, None)


# ===========================================================================
# main
# ===========================================================================
if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dir_logs",
        type=str,
        default="database/gan_optimize/",
        help="dir logs",
    )
    parser.add_argument(
        "--is_test",
        default=1,
        # default=0,
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
        # LIST_DATASETS = [
        #     # small
        #     "abalone",  # done
        #     "wilt",  # done
        #     "churn2",  # done
        #     "diabetes_openml",  # done
        #     "buddy",  # done
        #     "gesture",  # done
        #     "insurance",  # done
        #     "king",  # done
        #     "adult",  # done
        #     "cardio",  # done
        #     "california",  # done
        #     "house_16h",  # done
        #     "house",  # done
        #     "news",  # done
        #     "diabetesbalanced",  # 2
        #     # large
        #     "diabetes",  # 4
        #     "mnist12",  # 5
        #     # super large
        #     "credit",  # 4
        #     "higgs-small",
        #     "miniboone",
        #     "fb-comments",
        #     "covertype",
        #     "intrusion",
        #     "mnist28",
        # ]
        type=str,
    )
    parser.add_argument(
        "-a",
        "--arch",
        default="tabddpm",
    )
    parser.add_argument(
        "--loss_version",
        default=2,
        choices=[
            0,  # conventional
            2,  # version 2: generalize mean loss to distribution loss and correct correlation loss
            4,  # version 4: only correlation loss
            5,  # version 5: only distribution loss
        ],
        type=int,
    )
    parser.add_argument(
        "--is_condvec",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--model_type",
        default="mlp",
        type=str,
        choices=["mlp", "resnet"],
    )
    parser.add_argument(
        "--bo_method",
        default="ior",
        # default="sbo",
    )
    parser.add_argument(
        "--bo_method_agg",
        default="mean",
        # default="median",
    )
    parser.add_argument(
        "--module",
        choices=[
            "public",
            "gm",
            "dp",
            "gmdp",
        ],  # dp and gmdp are only for sensitive data
        default="public",
        # default="gmdp",
    )

    # generate subsets of a pandas DataFrame by subsampling rows and shuffling columns before sampling
    parser.add_argument(
        "--row_number",
        default=None,
        type=int,
    )

    args = parser.parse_args()

    # loop indefinitely and stop whenever you like
    max_successful_trials = args.max_trials

    # init search space
    search_space = init_search_space(args)
    print(search_space)

    # dp for biobank
    if args.module in ["public", "gm"]:
        evaluations = ["statistics", "ml", "ml_augment"]
    elif args.module == "dp":
        evaluations = ["dp"]
    elif args.module == "gmdp":
        evaluations = ["statistics", "ml", "ml_augment", "dp"]

    if args.module == "public":
        module = ""
    else:
        module = f"_module-{args.module}"

    # hyperopt
    database_path = "database"
    filename = f"{args.dataset}_{args.arch}_loss_version-{args.loss_version}_model-{args.model_type}{module}"
    if args.is_test:
        filename = f"test_" + filename

    if args.bo_method == "ior":
        folder = "optimization"
    elif args.bo_method == "sbo":
        folder = f"optimization_sbo_{args.bo_method_agg}"
    hyperopt_project_path = path_utils.get_hyperopt_path(
        filename, database_path=database_path, folder=folder
    )

    trials = hyperopt_utils.load_project(hyperopt_project_path)
    algo = "tpe"
    is_continue = True

    if args.bo_method == "ior":
        I = hyperopt_utils.IncrementalObjectiveOptimizationGenerativeModel(
            hyperopt_project_path,
        )
    else:
        agg = args.bo_method_agg
        I = hyperopt_utils.StandardObjectiveOptimizationGenerativeModel(
            hyperopt_project_path,
            agg=agg,
        )

    if len(trials.trials) > 0:
        I.update_trials_losses(evaluations=evaluations)

        n_successful_trials = hyperopt_utils.get_number_successful_trials(
            trials, success_value="success"
        )
        n_trials = hyperopt_utils.get_number_trials(trials)
        if n_successful_trials >= max_successful_trials or (
            n_successful_trials == 0 and n_trials >= args.max_trials
        ):
            is_continue = False

    while is_continue:
        print(colored("=" * 100, "red"))
        n_trials, n_successful_trials = hyperopt_utils.run_trials(
            project_path=hyperopt_project_path,
            objective=objective,
            space=search_space,
            algo=algo,
        )
        I.update_trials_losses(evaluations=evaluations)

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
