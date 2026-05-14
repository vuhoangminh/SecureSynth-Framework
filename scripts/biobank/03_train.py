import os
import pandas as pd
import argparse

# from ctgan import CTGAN

from models.ctgan import CTGAN

import engine.logger as logger
import engine.utils.model_utils as model_utils
import engine.utils.path_utils as path_utils
import engine.utils.io_utils as io_utils
import engine.utils.print_utils as print_utils
import engine.utils.data_utils as data_utils


parser = argparse.ArgumentParser(description="PyTorch CTGan Training")
parser.add_argument("--dir_logs", type=str, default="database/gan/", help="dir logs")
parser.add_argument("-a", "--arch", metavar="ARCH")
parser.add_argument(
    "--epochs", type=int, default=100, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch_size",
    default=16000,
    type=int,
    metavar="N",
)
parser.add_argument(
    "--private",
    default=1,
    type=int,
)
parser.add_argument(
    "-p",
    "--print_freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--is_test",
    default=1,
    type=int,
)
parser.add_argument(
    "--embedding_dim",
    default=128,
    type=int,
)
parser.add_argument(
    "--generator_dim",
    default="(256,256)",
    type=str,
)
parser.add_argument(
    "--discriminator_dim",
    default="(256,256)",
    type=str,
)
parser.add_argument(
    "--generator_lr",
    default=2e-4,
    type=float,
)
parser.add_argument(
    "--generator_decay",
    default=1e-6,
    type=float,
)
parser.add_argument(
    "--discriminator_lr",
    default=2e-4,
    type=float,
)
parser.add_argument(
    "--discriminator_decay",
    default=1e-6,
    type=float,
)
parser.add_argument(
    "--discriminator_steps",
    default=1,
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
    default="patient",
    choices=["record", "patient"],
    type=str,
)
parser.add_argument(
    "--is_only_sample",
    default=0,
    type=int,
)
parser.add_argument(
    "--checkpoint_freq",
    default=50,
    type=int,
)
parser.add_argument(
    "--resume",
    default=1,
    type=int,
)
args = parser.parse_args()


# ===========================================================================
# main
# ===========================================================================
if __name__ == "__main__":
    if args.dataset == "record":
        df = pd.read_csv(
            f"database/dataset/biobank_record_vital/20230524_biobank_data.tsv",
            sep="\t",
            header=0,
        )
        df = data_utils.preprocess_record(df)

        # we remove id because of the number of patients -> too expensive
        if args.is_drop_id:
            df = df.drop(
                ["id"],
                axis=1,
            )

        print(df)
    elif args.dataset == "patient":
        df = pd.read_csv(
            f"database/dataset/biobank_patient_dead/20231115_predict_data_persons.tsv",
            sep="\t",
            header=0,
        )
        df = data_utils.preprocess_patient(df)
        print(df)

    if args.is_test:
        df = df.tail(4_000)
        args.batch_size = 1_000
        # args.checkpoint_freq = 2

    # get logs directory
    args.dir_logs = os.path.join(args.dir_logs, path_utils.get_folder(args))
    path_utils.make_dir(args.dir_logs)
    print_utils.print_separator()
    print(f"Save to {args.dir_logs}")
    df.to_csv(
        os.path.join(args.dir_logs, "preprocessed.csv"), sep="\t", encoding="utf-8"
    )

    # names of the columns that are discrete
    discrete_columns = list(df.columns)
    if "id" in discrete_columns:
        discrete_columns.remove("id")
    id_columns = ["id"]

    # set up experiment logger
    exp_logger = None
    if exp_logger is None:
        exp_name = os.path.basename(args.dir_logs)  # add timestamp
        exp_logger = logger.Experiment(exp_name, io_utils.convert_args_to_dict(args))
        exp_logger.add_meters("train", model_utils.make_meters_ctgan())

    # init CTGAN
    ctgan = CTGAN(
        args,
        embedding_dim=args.embedding_dim,
        generator_dim=io_utils.convert_string_to_tuple(args.generator_dim),
        discriminator_dim=io_utils.convert_string_to_tuple(args.discriminator_dim),
        generator_lr=args.generator_lr,
        generator_decay=args.generator_decay,
        discriminator_lr=args.discriminator_lr,
        discriminator_decay=args.discriminator_decay,
        discriminator_steps=args.discriminator_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        private=args.private,
        dp_sigma=args.dp_sigma,
        dp_weight_clip=args.dp_weight_clip,
        is_loss_corr=args.is_loss_corr,
        is_loss_dwp=args.is_loss_dwp,
        is_condvec=args.is_condvec,
        checkpoint_freq=args.checkpoint_freq,
        verbose=True,
    )

    # bug with dataloader when last batch is incomplete
    if args.is_only_sample:
        ctgan.load_and_sample(
            df,
            exp_logger,
            discrete_columns=discrete_columns,
            is_exluded_real_data=False,
            n_sample=len(df),
            # n_sample=1_000_000,
        )
    else:
        ctgan.fit(
            df,
            exp_logger,
            discrete_columns=discrete_columns,
            id_columns=id_columns,
        )
        # Create synthetic data
        synthetic_data = ctgan.sample(130_000)
        print(synthetic_data)
        synthetic_data.to_csv(
            os.path.join(args.dir_logs, "fake.csv"), sep="\t", encoding="utf-8"
        )
