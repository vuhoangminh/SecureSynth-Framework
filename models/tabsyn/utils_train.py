import numpy as np
import os

import src
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]


class TabularDatasetOnlyContOrDis(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        # Use an empty tensor if X_num or X_cat is None
        if self.X_num is not None:
            this_num = self.X_num[index]
        else:
            this_num = torch.tensor([])  # Empty tensor for numerical data

        if self.X_cat is not None:
            this_cat = self.X_cat[index]
        else:
            this_cat = torch.tensor([])  # Empty tensor for categorical data

        sample = (this_num, this_cat)
        return sample

    def __len__(self):
        # Return the length of the available data
        if self.X_num is not None:
            return self.X_num.shape[0]
        else:
            return self.X_cat.shape[0]


def preprocess(
    dataset_path,
    task_type="binclass",
    inverse=False,
    cat_encoding=None,
    concat=True,
    row_number=None,
):

    T_dict = {}

    T_dict["normalization"] = "quantile"
    T_dict["num_nan_policy"] = "mean"
    T_dict["cat_nan_policy"] = None
    T_dict["cat_min_frequency"] = None
    T_dict["cat_encoding"] = cat_encoding
    T_dict["y_policy"] = "default"

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        data_path=dataset_path,
        T=T,
        task_type=task_type,
        change_val=False,
        concat=concat,
        row_number=row_number,
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        # Fixed here. We dont want to touch test set
        # X_train_num, X_test_num = X_num["train"], X_num["test"]
        # X_train_cat, X_test_cat = X_cat["train"], X_cat["test"]

        ## Added by Minh
        if X_num is not None:
            X_train_num, X_test_num = X_num["train"], X_num["val"]
            X_num = (X_train_num, X_test_num)
            d_numerical = X_train_num.shape[1]
            if inverse:
                num_inverse = dataset.num_transform.inverse_transform
        else:
            X_num = (None, None)
            d_numerical = 0
            num_inverse = None

        if X_cat is not None:
            X_train_cat, X_test_cat = X_cat["train"], X_cat["val"]
            # categories = src.get_categories(X_cat["train"])

            # Added by Minh -- fix n_classes
            categories = src.get_categories(
                np.vstack([X_cat["val"], X_cat["test"], X_cat["train"]])
            )
            # Added by Minh

            X_cat = (X_train_cat, X_test_cat)
            if inverse:
                cat_inverse = dataset.cat_transform.inverse_transform
        else:
            X_cat = (None, None)
            categories = None
            cat_inverse = None
        ## Added by Minh

        if inverse:
            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


## Added by Minh -- handle saving train.csv when subsampling
def save_preprocessed(
    data_path, X_num, X_cat, y, row_number, concat=True, task_type="binclass"
):
    import json
    from tabsyn.latent_utils import recover_data
    import engine.utils.path_utils as path_utils

    with open(f"{data_path}/tabsyn_info.json", "r") as f:
        info = json.load(f)

    if X_num is None:  # process X_cat
        if concat and "class" in task_type:
            syn_df = recover_data(
                None,
                X_cat["train"][:, 1:],
                y["train"][..., np.newaxis],
                info,
            )
        else:
            syn_df = recover_data(
                None,
                X_cat["train"],
                y["train"][..., np.newaxis],
                info,
            )
    elif X_cat is None:  # process X_num
        if concat and "regression" in task_type:
            syn_df = recover_data(
                X_num["train"][:, 1:],
                None,
                y["train"][..., np.newaxis],
                info,
            )
        else:
            syn_df = recover_data(
                X_num["train"],
                None,
                y["train"][..., np.newaxis],
                info,
            )
    else:
        if concat:
            if "class" in task_type:
                syn_df = recover_data(
                    X_num["train"],
                    X_cat["train"][:, 1:],
                    y["train"][..., np.newaxis],
                    info,
                )
            elif "regression" in task_type:
                syn_df = recover_data(
                    X_num["train"][:, 1:],
                    X_cat["train"],
                    y["train"][..., np.newaxis],
                    info,
                )
            else:
                raise ValueError(
                    f"{task_type} is neither classification or regression..."
                )
        else:
            syn_df = recover_data(
                X_num["train"], X_cat["train"], y["train"][..., np.newaxis], info
            )

    idx_name_mapping = info["idx_name_mapping"]
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)

    save_dir = f"{data_path}/temp"
    path_utils.make_dir(save_dir)

    if row_number is not None:
        syn_df.to_csv(
            f"{save_dir}/preprocessed_rownum-{row_number}.csv",
            sep="\t",
            encoding="utf-8",
        )
    else:
        syn_df.to_csv(
            f"{save_dir}/preprocessed.csv",
            sep="\t",
            encoding="utf-8",
        )


## Added by Minh -- handle saving train.csv when subsampling


def make_dataset(
    data_path: str,
    T: src.Transformations,
    task_type,
    change_val: bool,
    concat=True,
    ## Added by Minh
    row_number=None,
    ## Added by Minh
):

    # classification
    if task_type == "binclass" or task_type == "multiclass":
        X_cat = (
            {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        )
        X_num = (
            {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        )
        y = {} if os.path.exists(os.path.join(data_path, "y_train.npy")) else None

        ## Added by Minh -- handle no dis. columns
        if concat:
            X_cat = {}
        ## Added by Minh -- handle no dis. columns

        for split in ["train", "val", "test"]:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

            ## Added by Minh
            if row_number is not None and split == "train":
                # Determine the number of available rows (assuming all arrays have the same length)
                # Determine the valid indices based on non-None arrays
                if X_cat_t is not None and y_t is not None:
                    valid_indices = np.arange(
                        len(X_cat_t)
                    )  # Valid indices based on X_cat_t or y_t
                elif X_cat_t is not None:
                    valid_indices = np.arange(len(X_cat_t))
                elif y_t is not None:
                    valid_indices = np.arange(len(y_t))
                else:
                    raise ValueError("Both X_cat_t and y_t cannot be None.")
                # Randomly select indices for sampling
                np.random.seed(42)
                sampled_indices = np.random.choice(
                    valid_indices, size=row_number, replace=False
                )
            else:
                sampled_indices = None
            ## Added by Minh

            if X_num is not None:
                if sampled_indices is not None:
                    X_num[split] = X_num_t[sampled_indices]
                else:
                    X_num[split] = X_num_t

            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                if sampled_indices is not None:
                    X_cat[split] = X_cat_t[sampled_indices]
                else:
                    X_cat[split] = X_cat_t

            ## Added by Minh -- handle no dis. columns
            else:
                if concat:
                    if sampled_indices is not None:
                        X_cat[split] = concat_y_to_X(X_cat_t, y_t)[sampled_indices]
                    else:
                        X_cat[split] = concat_y_to_X(X_cat_t, y_t)

            ## Added by Minh
            if y is not None:
                if sampled_indices is not None:
                    y[split] = y_t[sampled_indices]
                else:
                    y[split] = y_t

    else:
        # regression
        X_cat = (
            {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        )
        X_num = (
            {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        )
        y = {} if os.path.exists(os.path.join(data_path, "y_train.npy")) else None

        for split in ["train", "val", "test"]:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

            ## Added by Minh
            if row_number is not None and split == "train":
                # Determine the number of available rows (assuming all arrays have the same length)
                # Determine the valid indices based on non-None arrays
                if X_cat_t is not None and y_t is not None:
                    valid_indices = np.arange(
                        len(X_cat_t)
                    )  # Valid indices based on X_cat_t or y_t
                elif X_cat_t is not None:
                    valid_indices = np.arange(len(X_cat_t))
                elif y_t is not None:
                    valid_indices = np.arange(len(y_t))
                else:
                    raise ValueError("Both X_cat_t and y_t cannot be None.")
                # Randomly select indices for sampling
                np.random.seed(42)
                sampled_indices = np.random.choice(
                    valid_indices, size=row_number, replace=False
                )
            else:
                sampled_indices = None
            ## Added by Minh

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                if sampled_indices is not None:
                    X_num[split] = X_num_t[sampled_indices]
                else:
                    X_num[split] = X_num_t

            ## Added by Minh -- handle no cont. columns
            else:
                if concat:
                    if sampled_indices is not None:
                        X_num[split] = y_t[sampled_indices]
                    else:
                        X_num[split] = y_t
            ## Added by Minh

            if X_cat is not None:
                if sampled_indices is not None:
                    X_cat[split] = X_cat_t[sampled_indices]
                else:
                    X_cat[split] = X_cat_t

            if y is not None:
                if sampled_indices is not None:
                    y[split] = y_t[sampled_indices]
                else:
                    y[split] = y_t

    info = src.load_json(os.path.join(data_path, "info.json"))

    ## Added by Minh
    save_preprocessed(data_path, X_num, X_cat, y, row_number, concat, task_type)
    ## Added by Minh

    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(info["task_type"]),
        n_classes=info.get("n_classes"),
    )

    if change_val:
        D = src.change_val(D)

    return src.transform_dataset(D, T, None)
