import os
import argparse
import numpy as np
import pandas as pd
from engine.config import config
import engine.utils.data_utils as data_utils
import engine.utils.path_utils as path_utils
from engine.utils.eval_utils import (
    compute_dwp,
    compute_diff_correlation,
)
from engine.utils.data_utils import drop_duplicates


parser = argparse.ArgumentParser(description="PyTorch CTGan Post-processing")
parser.add_argument(
    "--embedding_dim",
    default=128,
    type=int,
)
parser.add_argument(
    "--epochs",
    type=int,
    default=3000,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "-b",
    "--batch_size",
    default=64000,
    type=int,
    metavar="N",
)
parser.add_argument(
    "--private",
    default=1,
    type=int,
)
parser.add_argument(
    "--is_test",
    default=0,
    type=int,
)
parser.add_argument(
    "--dp_sigma",
    default=1.0,
    type=float,
)
parser.add_argument(
    "--dp_weight_clip",
    default=0.01,
    type=float,
)
parser.add_argument(
    "--is_loss_corr",
    default=0,
    type=int,
)
parser.add_argument(
    "--is_loss_dwp",
    default=0,
    type=int,
)
parser.add_argument(
    "--is_condvec",
    default=1,
    type=int,
)
parser.add_argument(
    "--is_drop_id",
    default=1,
    type=int,
)
parser.add_argument(
    "--dataset",
    default="record",
    choices=["record", "patient"],
    type=str,
)
args = parser.parse_args()


def compute_score(df, df_fake, weight=10):
    corr = compute_diff_correlation(df, df_fake)
    dwp, _, _ = compute_dwp(df, df_fake)
    d = corr + dwp * weight
    return d


def sample_all_classes(df, synthetic, folder):
    is_sample = True
    while is_sample:
        is_satisfied = True
        df_fake = synthetic.sample(len(df))
        for col in list(df):
            print(f"{col}: {df[col].nunique()}, {df_fake[col].nunique()}")
            if df[col].nunique() != df_fake[col].nunique():
                is_satisfied = False
        if is_satisfied:
            is_sample = False

    df_fake.reset_index(drop=True, inplace=True)
    df_fake.to_csv(
        f"database/synthetic/latest_{folder}.csv",
        sep="\t",
        encoding="utf-8",
    )

    return df_fake


def sample_optimal(df, df_fake, folder):
    best_score = np.inf
    best_df_sample = None
    for i in range(10_000):
        print(f">> processing {i} / 10000")
        df_sample = df_fake.sample(len(df))
        is_satisfied = True
        list_failed = []
        for col in list(df):
            # print(f"{col}: {df[col].nunique()}, {df_sample[col].nunique()}")
            if df[col].nunique() != df_sample[col].nunique():
                is_satisfied = False
                list_failed.append(col)
        if is_satisfied:
            score = compute_score(df, df_sample)
            if score < best_score:
                best_score = score
                best_df_sample = df_sample.copy()
            print(score)
        else:
            print(list_failed)

    best_df_sample.reset_index(drop=True, inplace=True)
    best_df_sample.to_csv(
        f"database/synthetic/latest_{folder}.csv",
        sep="\t",
        encoding="utf-8",
    )


def run_sample_all_classes():
    folder = path_utils.get_folder(args)

    df = pd.read_csv(
        f"database/gan/{folder}/preprocessed.csv",
        sep="\t",
        header=0,
        index_col=0,
    )
    df = data_utils.inverse_transform(
        df, label_encoder_path=f"database/{args.dataset}_label_encoder.pickle"
    )

    df_fake = pd.read_csv(
        f"database/synthetic/{folder}.csv",
        sep="\t",
        header=0,
        index_col=0,
    )

    df_fake = data_utils.inverse_transform(
        df_fake, label_encoder_path="database/record_label_encoder.pickle"
    )
    print(len(df_fake))

    df_fake = drop_duplicates(df_fake)
    print(len(df_fake))

    df_fake = drop_duplicates(df, df_fake)
    print(len(df_fake))

    latest_df_fake = sample_all_classes(df, df_fake, folder)


def run_sample(is_overwrite=False):
    folder = path_utils.get_folder(args)
    filename = config.LIST_BEST[folder]
    label_encoder_path = f"database/{args.dataset}_label_encoder.pickle"

    df = pd.read_csv(
        f"database/gan/{folder}/preprocessed.csv",
        sep="\t",
        header=0,
        index_col=0,
    )
    df = data_utils.inverse_transform(df, label_encoder_path=label_encoder_path)
    print(df)

    df_fake = pd.read_csv(
        f"database/gan/{folder}/{filename}",
        sep="\t",
        header=0,
        index_col=0,
    )
    df_fake = data_utils.inverse_transform(
        df_fake, label_encoder_path=label_encoder_path
    )
    print(df_fake)
    print(len(df_fake))

    df_fake = drop_duplicates(df_fake)
    print(len(df_fake))

    df_fake = drop_duplicates(df, df_fake)
    print(len(df_fake))

    if len(df_fake) > len(df):
        df_fake = df_fake.sample(len(df))

    real_path = f"database/synthetic/real-{args.dataset}.csv"
    if is_overwrite or not os.path.exists(real_path):
        df.to_csv(real_path, sep="\t", encoding="utf-8")

    synthetic_path = f"database/synthetic/synthetic-{folder}.csv"
    if is_overwrite or not os.path.exists(synthetic_path):
        df_fake.to_csv(synthetic_path, sep="\t", encoding="utf-8")


def main():
    # run_sample_all_classes()
    run_sample()


if __name__ == "__main__":
    main()
