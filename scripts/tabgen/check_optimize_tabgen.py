import os
import glob
import re
from rich import print
import numpy as np
import pickle
from hyperopt import fmin
import engine.utils.path_utils as path_utils


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
    "adult",  # done
    "cardio",  # done
    "california",  # done
    "house_16h",  # done
    "house",  # done
    "news",  # done
    "diabetesbalanced",  # done
    # large
    "diabetes",  # done
    "mnist12",  # done
    # super large
    "credit",  # done
    "higgs-small",  # done
    "miniboone",  # done
    "fb-comments",
    "covertype",
    "intrusion",
    "mnist28",
    # "biobank_record_vital",
    # "biobank_record_dead",
    # "biobank_record_icd7",
    # "biobank_record_icd9",
    # "biobank_record_icdo2",
    # "biobank_record_icdo3",
    # "biobank_patient_dead",
]


def remove_hyperopt_lock_files(folder_path):
    """Removes all files with ".hyperopt.lock" extension in a given folder.

    Args:
      folder_path: The path to the folder.
    """

    lock_files = glob.glob(os.path.join(folder_path, "*.hyperopt.lock"))
    for lock_file in lock_files:
        os.remove(lock_file)


def get_paths(paths, dataset=None):
    if dataset is not None:
        p = []
        for path in paths:
            if dataset in path:
                p.append(path)
        paths = p
    return paths


# dir = "database/optimization_ml_method/"
dir = "database/optimization/"
# dir = "database/optimization_phase2_data_sufficient"
# dir = "database/optimization_data_sufficient/"
# dir = "database/optimization_sbo_mean/"
# dir = "database/optimization_sbo_median/"
# remove_hyperopt_lock_files(dir)

paths = glob.glob(os.path.join(dir, "*.hyperopt"))


def general(paths):
    # paths = get_paths(paths, "wilt")
    # paths = get_paths(paths, "biobank")
    # paths = get_paths(paths, "mnist12_tabddpm_loss_version-2_model-mlp")
    # paths = get_paths(paths, "credit_tvae_loss_version-2-1.hyperopt")
    # paths = get_paths(paths, "mnist12_tabddpm_loss_version-2_model-mlp")
    # paths = get_paths(paths, "tabddpm")
    # paths = get_paths(paths, "ctgan")
    # paths = get_paths(paths, "tvae")
    # paths = get_paths(paths, "copulagan")
    # paths = get_paths(paths, "tabsyn")
    # paths = get_paths(paths, "dpcgans")
    # paths = get_paths(paths, "biobank_patient_dead_rownum-20000_")
    # paths = get_paths(paths, "biobank_patient_dead_rownum-")
    # paths = get_paths(paths, "tabsyn")
    # paths = get_paths(paths, "biobank_sen_uroandkid")
    # paths = get_paths(paths, "biobank_")
    # paths = get_paths(paths, "record_dead")
    # paths = get_paths(paths, "abalone")
    # paths = get_paths(paths, "news")
    # paths = get_paths(paths, "higgs-small")
    # paths = get_paths(paths, "loss_version-4")
    # paths = get_paths(paths, "biobank_sen")
    # paths = get_paths(paths, "biobank_phase3_cancer_153")
    # paths = get_paths(paths, "biobank_phase3_cancer_all")
    # paths = get_paths(paths, "biobank_phase3_cancer_bio_all")
    # paths = get_paths(paths, "xgboost")

    # paths = get_paths(paths)
    paths.sort()

    N = 20
    N = 30
    # N = 100

    not_done_list = []
    for path in paths:
        try:
            trials = pickle.load(open(path, "rb"))
        except:
            continue

        # print()
        # print(path_utils.get_filename_without_extension(path))

        count = 0
        count_success = 0
        fmin = np.inf
        trial_fmin = None

        # Extract durations
        durations = [
            (t["refresh_time"] - t["book_time"]).total_seconds() for t in trials.trials
        ]

        # print("Durations of each trial (hours):", sum(durations) / 60 / 60)

        for trial in trials.results:
            if trial["reason"] == "success":
                count_success += 1
                # print(trial)
                print(trial["loss"])
                a = 2
            if trial["loss"] < fmin:
                trial_fmin = trial
                fmin = trial["loss"]

            # if (
            #     trial["scores_ml"]["fold0_gmean"] > 0
            #     and trial["scores_ml"]["fold1_precision"] < 1
            # ):
            #     a = 2
            print(trial)
            count = count + 1

        print()
        print()

        """
        python -W ignore scripts/tabgen/main_tabgen.py --dir_logs database/gan_optimize/ --is_test 0 --dataset biobank_phase3_dummy --arch ctgan --loss_version 0 --checkpoint_freq 100 --is_condvec 0 --batch_size 200 --discriminator_decay 1.1335380377164272e-07 --discriminator_dim 128 --discriminator_lr 2.5529909452737364e-05 --dp_sigma 0.7808473588167365 --dp_weight_clip 0.31732184923126583 --embedding_dim 32 --epochs 400 --generator_decay 5.946743069779757e-07 --generator_dim 128 --generator_lr 0.00011308360035148551 --private 0
        """

        print(path_utils.get_filename_without_extension(path), count_success)

        # print(trial_fmin)
        # print(trial_fmin["dir_logs"])\

        print()
        print()

        if count_success < N:
            # if count_success >= 30:
            print()
            print(path_utils.get_filename_without_extension(path))

            # if not (count_success == 0 and count >= N):
            #     print(f"n_success/n_runs = {count_success} / {count}")
            #     print(trial_fmin["loss"])

            #     not_done_list.append(path)

            print(f"n_success/n_runs = {count_success} / {count}")

    print()
    print()
    print()
    print()
    for p in not_done_list:
        print(f"sbatch scripts/jobs/{path_utils.get_filename_without_extension(p)}.sh")
    print()
    print()
    print()
    print()


# Function to extract rownum (xxx)
def extract_rownum(path):
    match = re.search(r"rownum-(\d+)", path)
    return int(match.group(1)) if match else float("inf")  # Handle missing cases


def data_suff(paths):
    # paths = get_paths(paths, "wilt")
    # paths = get_paths(paths, "biobank")
    # paths = get_paths(paths, "mnist12_tabddpm_loss_version-2_model-mlp")
    # paths = get_paths(paths, "credit_tvae_loss_version-2-1.hyperopt")
    # paths = get_paths(paths, "mnist12_tabddpm_loss_version-2_model-mlp")
    # paths = get_paths(paths, "tabddpm")
    paths = get_paths(paths, "ctgan")
    # paths = get_paths(paths, "tvae")
    # paths = get_paths(paths, "copulagan")
    # paths = get_paths(paths, "tabsyn")
    # paths = get_paths(paths, "biobank_patient_dead_rownum-20000_")
    # paths = get_paths(paths, "biobank_patient_dead_rownum-")
    # paths = get_paths(paths, "biobank_record_dead_rownum-")
    # paths = get_paths(paths, "biobank_record_icd7_rownum-")
    # paths = get_paths(paths, "biobank_record_icd9_rownum-")
    # paths = get_paths(paths, "biobank_record_icdo2_rownum-")
    # paths = get_paths(paths, "biobank_record_icdo3_rownum-")
    # paths = get_paths(paths, "biobank_record_vital_rownum-")
    paths = get_paths(paths, "biobank_sen_meta_rownum-")

    # paths = get_paths(paths, "tabsyn")

    # paths = get_paths(paths)
    # paths.sort()
    # Sort paths based on extracted rownum
    paths = sorted(paths, key=extract_rownum)

    print(paths)

    N = 20
    N = 30
    N = 100

    not_done_list = []
    for path in paths:
        try:
            trials = pickle.load(open(path, "rb"))
        except:
            continue

        # print()
        print(path_utils.get_filename_without_extension(path))

        count = 0
        count_success = 0
        fmin = np.inf
        trial_fmin = None

        for trial in trials.results:
            if trial["reason"] == "success":
                count_success += 1
                # print(trial)
                # print(trial["loss"])
            if trial["loss"] < fmin:
                trial_fmin = trial
                fmin = trial["loss"]
            count = count + 1

        # print(path_utils.get_filename_without_extension(path), count_success)
        # print()
        # print()

        # print(trial_fmin)
        # print(trial_fmin["scores_dp"]["dp_k_anonymization_safe"])
        # print(trial_fmin["scores_dp"]["dp_l_diversity_safe"])

        # print(trial_fmin["scores_dp"]["dp_k_anonymization_synthetic"])
        try:
            print(trial_fmin["scores_dp"]["dp_k_anonymization_synthetic"])
            print(trial_fmin["scores_dp"]["dp_l_diversity_synthetic"])
            print(trial_fmin["scores_dp"]["dp_k_map"])
            # print(trial_fmin["scores_dp"]["dp_re_identification_score"])
            # print(trial_fmin["scores_dp"]["dp_domias_mia_accuracy"])
            # print(trial_fmin["scores_dp"]["dp_domias_mia_auc"])

            # print(
            #     "dp_k_anonymization_synthetic:",
            #     trial_fmin["scores_dp"]["dp_k_anonymization_synthetic"],
            # )
            # print(
            #     "dp_l_diversity_synthetic:",
            #     trial_fmin["scores_dp"]["dp_l_diversity_synthetic"],
            # )
            # print("dp_k_map:", trial_fmin["scores_dp"]["dp_k_map"])

            # print(
            #     "dp_re_identification_score:",
            #     trial_fmin["scores_dp"]["dp_re_identification_score"],
            # )
            # print(
            #     "dp_domias_mia_accuracy:",
            #     trial_fmin["scores_dp"]["dp_domias_mia_accuracy"],
            # )
            # print("dp_domias_mia_auc:", trial_fmin["scores_dp"]["dp_domias_mia_auc"])

        except:
            pass

        a = 2


def data_suff_check_tabsyn(paths):

    def extract_python_command_tabsyn(input_string):
        """
        Extracts the Python command from an error message string.

        Args:
            input_string: The string containing the error message.

        Returns:
            The Python command as a string, or None if not found.
        """
        match = re.search(r"python.*?(?=' failed)", input_string)
        if match:
            return match.group(0)
        else:
            return None

    def split_and_format_commands_tabsyn(
        commands, output_filename="scripts/bash/commands.txt"
    ):
        """
        Splits a list of Python commands into 8 portions, concatenates each portion
        with "apptainer exec --nv scripts/apptainer/biobank.sif " prefix and ";" separator,
        and saves the exported command strings to a text file.

        Args:
            commands: A list of Python command strings.
            output_filename: The name of the output text file.
        """

        num_portions = 8
        chunk_size = len(commands) // num_portions
        remainder = len(commands) % num_portions
        list_commands = []

        with open(output_filename, "w") as f:
            start_index = 0

            for i in range(num_portions):
                # Determine the size of the current chunk
                end_index = start_index + chunk_size
                if i < remainder:
                    end_index += 1  # Distribute the remainder

                current_chunk = commands[start_index:end_index]
                start_index = end_index

                # Add the prefix and concatenate with semicolon
                prefixed_commands = [
                    f"apptainer exec --nv scripts/apptainer/biobank.sif {cmd}"
                    for cmd in current_chunk
                ]
                concatenated_command = "; ".join(prefixed_commands)

                for prefixed_command in prefixed_commands:
                    list_commands.append(prefixed_command)

                f.write(f'export command{i + 1}="{concatenated_command}"\n')

        print()
        print()
        print()
        for command in list_commands:
            print(f'"{command}"')

    paths = sorted(paths, key=extract_rownum)

    # paths = get_paths(paths, "ctgan")
    # paths = get_paths(paths, "tvae")
    # paths = get_paths(paths, "copulagan")
    paths = get_paths(paths, "tabsyn")

    paths.sort()

    # print(paths)

    not_done_list = []
    count_not_done = 0
    list_python_commands = []
    for path in paths:
        try:
            trials = pickle.load(open(path, "rb"))
        except:
            continue

        # print()

        is_success = 0

        for trial in trials.results:
            if trial["reason"] == "success":
                is_success = 1
                break

        if not is_success:
            print(path_utils.get_filename_without_extension(path))
            count_not_done += 1
            # for trial in trials:
            #     print(trial["result"]["reason"])
            #     a = 2

            list_python_commands.append(
                extract_python_command_tabsyn(trials.results[0]["reason"])
            )

    exported_commands = split_and_format_commands_tabsyn(list_python_commands)

    print(count_not_done)
    print(exported_commands)


def data_suff_check_ctgan(paths):

    paths = sorted(paths, key=extract_rownum)

    paths = get_paths(paths, "ctgan")
    # paths = get_paths(paths, "tvae")
    # paths = get_paths(paths, "copulagan")
    # paths = get_paths(paths, "tabsyn")

    paths.sort()

    # print(paths)

    not_done_list = []
    count_not_done = 0
    list_python_commands = []
    for path in paths:
        try:
            trials = pickle.load(open(path, "rb"))
        except:
            continue

        # print()

        is_success = 0

        for trial in trials.results:
            if trial["reason"] == "success":
                is_success = 1
                break

        if not is_success:
            print(path_utils.get_filename_without_extension(path))
            count_not_done += 1
            for trial in trials:
                print(trial["result"]["reason"])
                a = 2

    print(count_not_done)


if __name__ == "__main__":
    general(paths)
    # data_suff(paths)
    # data_suff_check_tabsyn(paths)
    # data_suff_check_ctgan(paths)

"""
generate df_score with num_row column
read from df_score
plot to see the pattern
statistics + dp
read real+ synthetic and perform ml on test set
"""
