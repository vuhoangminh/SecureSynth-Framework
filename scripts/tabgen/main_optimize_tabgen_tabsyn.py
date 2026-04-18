import os
import json
import time
import argparse
import subprocess
import shutil
import numpy as np
import pandas as pd
from rich import print
import gc
from termcolor import colored

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
        ]:
            params[key] = 10**value
    return params


def update_cmd(params, cmd):
    updated_cmd = cmd
    for key, value in params.items():
        updated_cmd += f" --{key} {value}"
    return updated_cmd


def get_folders_by_modified_time(directory, reverse=False):
    """
    Gets a list of folders in a directory sorted by modification time.

    Args:
      directory: The path to the directory.
      reverse: If True, sorts by newest first, otherwise oldest first.

    Returns:
      A list of folder paths sorted by modification time.
    """

    folders = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]
    folders.sort(key=os.path.getmtime, reverse=reverse)
    return folders


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

    def _legacy_save_preprocessed(_D, _dir_logs):
        df = _D.data_train

        # Convert all object dtype columns to int
        df = df.apply(
            lambda col: (
                pd.to_numeric(col, errors="ignore") if col.dtype == "object" else col
            )
        )
        print(df)
        path_utils.make_dir(_dir_logs)
        print(f"Save to {_dir_logs}")
        df.to_csv(
            os.path.join(_dir_logs, "preprocessed.csv"), sep="\t", encoding="utf-8"
        )

    def save_preprocessed(data_path, _dir_logs, row_number):
        print(f"Save to {_dir_logs}")

        if row_number is not None:
            path_from = f"{data_path}/preprocessed_rownum-{row_number}.csv"
        else:
            path_from = f"{data_path}/preprocessed.csv"
        path_to = os.path.join(_dir_logs, "preprocessed.csv")

        print(f">> Copying from {path_from} to {path_to}")
        shutil.copyfile(path_from, path_to)

    params = update_params(params)
    print(params)

    try:
        D = get_dataset(args.dataset)

        params["epochs"] = params["num_epochs"]

        batch_size_tvae = (
            512 if args.dataset == "biobank_phase3_cancer_bio_all" else 4096
        )

        if args.row_number is not None:
            cmd_tvae = f"python -W ignore models/tabsyn/main.py --method vae --dataname {args.dataset} --row_number {args.row_number}--batch_size_tvae {batch_size_tvae}"
        else:
            cmd_tvae = f"python -W ignore models/tabsyn/main.py --method vae --dataname {args.dataset} --batch_size_tvae {batch_size_tvae}"

        try:
            result_tvae = subprocess.run(
                cmd_tvae,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("TVAE Output:\n", result_tvae.stdout.decode())
        except subprocess.CalledProcessError as e:
            reason = f"Command '{cmd_tvae}' failed with return code {e.returncode}"
            print(reason)
            print(e.stderr.decode())
            loss = np.inf
            return construct_return_dict(
                loss, reason, params, None, None, None, None, None
            )

        if args.row_number is not None:
            _cmd = f"python -W ignore models/tabsyn/main.py --dir_logs {args.dir_logs} --is_test {args.is_test} --dataname {args.dataset} --arch {args.arch} --loss_version {args.loss_version}  --row_number {args.row_number}"
        else:
            _cmd = f"python -W ignore models/tabsyn/main.py --dir_logs {args.dir_logs} --is_test {args.is_test} --dataname {args.dataset} --arch {args.arch} --loss_version {args.loss_version}"

        cmd_train = update_cmd(params, _cmd)

        new_parser = add_dict_to_args(parser, params)
        new_args = new_parser.parse_args()
        dir_logs = os.path.join(
            args.dir_logs, path_utils.get_folder_technical_paper(new_args)
        )
        print(f">> logging to {dir_logs}")

        print(f">> running {cmd_train}")
        try:
            result = subprocess.run(
                cmd_train,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            reason = f"Command '{cmd_train}' failed with return code {e.returncode}"
            print(reason)
            print(e.stderr.decode())
            loss = np.inf
            return construct_return_dict(
                loss, reason, params, None, None, None, None, None
            )

        # If you need to print or use the output, use result.stdout.decode()
        print(result.stdout.decode())

        data_path = f"database/dataset/{args.dataset}/temp"
        save_preprocessed(data_path, dir_logs, args.row_number)

        # sample
        cmd_sample = cmd_train + " --mode sample"

        os.system(cmd_sample)

        folder = path_utils.get_filename(dir_logs)
        discrete_columns = D.discrete_columns
        continuous_columns = [
            c for c in list(D.data_train) if c not in discrete_columns
        ]

        loss = 0  # we will update later
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
        reason = "success"

        gc.collect()

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

    except Exception as e:
        reason = str(e)
        loss = np.inf

    except RuntimeError as e:  # synthetic tensor contains nan values
        reason = str(e)
        loss = np.inf

    return construct_return_dict(loss, reason, params, None, None, None, None, None)


def init_search_space(args):
    search_space = {}

    if args.is_test:
        search_space["batch_size"] = hp.choice("batch_size", [256])
        search_space["dim_t"] = hp.choice("dim_t", [256])
        search_space["num_epochs"] = hp.choice("num_epochs", [20])
        search_space["lr"] = hp.uniform("lr", 0.00001, 0.003)
        search_space["factor"] = hp.uniform("factor", 0.1, 0.9)
    else:
        search_space["batch_size"] = hp.choice(
            "batch_size", [256, 512, 1024, 2048, 4096]
        )
        search_space["dim_t"] = hp.choice("dim_t", [256, 512, 1024, 2048])
        search_space["num_epochs"] = hp.choice("num_epochs", [5000, 10000, 20000])
        search_space["lr"] = hp.uniform("lr", 0.00001, 0.003)
        search_space["factor"] = hp.uniform("factor", 0.1, 0.9)

    if args.loss_version != 0:
        search_space.update(
            {
                "is_loss_corr": hp.uniform("is_loss_corr", -2, 6),
                "is_loss_dwp": hp.uniform("is_loss_dwp", -14, -2),
                "n_moment_loss_dwp": hp.choice("n_moment_loss_dwp", [1, 2, 3, 4]),
            }
        )

    return search_space


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
        # default="biobank_patient_dead",
        default="biobank_phase3_dummy_pca",
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
        type=str,
    )
    parser.add_argument(
        "-a",
        "--arch",
        default="tabsyn",
    )
    parser.add_argument(
        "--loss_version",
        default=0,
        # default=0,
        choices=[
            0,  # conventional
            2,  # version 2: submitted to ICLR25
            4,  # version 2: submitted to ICLR25
        ],
        type=int,
    )
    parser.add_argument(
        "--is_condvec",
        default=1,
        type=int,
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
    parser.add_argument(
        "--module",
        choices=[
            "public",
            "gm",
            "dp",
            "gmdp",
        ],  # dp and gmdp are only for sensitive data
        default="gmdp",
        # default="gmdp",
    )

    # generate subsets of a pandas DataFrame by subsampling rows and shuffling columns before sampling
    parser.add_argument(
        "--row_number",
        default=None,
        # default=100,
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

    if args.row_number is not None:
        filename = f"{args.dataset}_rownum-{args.row_number}_{args.arch}_loss_version-{args.loss_version}-{args.is_condvec}{module}"
    else:
        filename = f"{args.dataset}_{args.arch}_loss_version-{args.loss_version}-{args.is_condvec}{module}"

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


"""

cmd_train
python -W ignore models/tabsyn/main.py --dir_logs database/gan_optimize/ --is_test 1 --dataname abalone --arch tabsyn --loss_version 0 --batch_size 256 --dim_t 256 --factor 0.14714594930252034 --lr 0.001975809591584522 --num_epochs 500 --epochs 500

cmd_sample
python -W ignore models/tabsyn/main.py --dir_logs database/gan_optimize/ --is_test 1 --dataname abalone --arch tabsyn --loss_version 0 --mode sample --batch_size 256 --dim_t 256 --factor 0.14714594930252034 --lr 0.001975809591584522 --num_epochs 500 --epochs 500

'python -W ignore models/tabsyn/main.py --dir_logs database/gan_optimize/ --is_test 1 --dataname biobank_patient_dead --arch tabsyn --loss_version 2 --batch_size 256 --dim_t 256 --factor 0.6955097744047156 --is_loss_corr 0.0133637027551146 --is_loss_dwp 0.00021516425054382953 --lr 0.0020678324349874735 --n_moment_loss_dwp 1 --num_epochs 20 --epochs 20 --mode sample'


sbatch scripts/jobs_bianca/biobank_phase3_cancer_all_tabsyn_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_bio_all_tabsyn_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_185_tabsyn_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_174_tabsyn_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_153_tabsyn_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_162_tabsyn_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_188_tabsyn_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_172_tabsyn_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_all_ctgan-0_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_all_ctgan-1_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_bio_all_ctgan-0_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_bio_all_ctgan-1_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_all_tvae-1_module-gmdp.sh
sbatch scripts/jobs_bianca/biobank_phase3_cancer_bio_all_tvae-1_module-gmdp.sh
"""
