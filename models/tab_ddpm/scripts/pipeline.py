import tomli
import shutil
import os
import argparse
from train import train
from sample import sample
import pandas as pd
import models.tab_ddpm.lib as lib
import torch


def load_config(path):
    with open(path, "rb") as f:
        return tomli.load(f)


def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        metavar="FILE",
        # default="database/gan_optimize/test-diabetesbalanced-tabddpm-lv_2-bs_4096-epochs_1000-df_9-dm_8-dl_9-nl_2-lr_2.79e-03-model_mlp-moment_3-losscorcorr_2.97e+02-lossdis_4.40e-04-condvec_1/config.toml",
        # default="database/gan_optimize/test-churn2-tabddpm-lv_2-bs_4096-epochs_1000-df_8-dm_9-dl_9-nl_3-lr_2.17e-03-model_mlp-moment_2-losscorcorr_7.41e-01-lossdis_3.57e-03-condvec_1/config.toml",
        # default="database/gan_optimize/test-churn2-tabddpm-lv_2-bs_256-epochs_1000-df_8-dm_7-dl_10-nl_4-lr_2.01e-04-model_mlp-moment_1-losscorcorr_2.09e-01-lossdis_3.67e-06-condvec_1/config.toml",
        # default="database/gan_optimize/test-abalone-tabddpm-lv_2-bs_4096-epochs_1000-dm_1013-dh_172-df_0.1-ds_0.1-nl_2-lr_3.68e-04-model_resnet-moment_2-losscorcorr_2.53e+01-lossdis_4.81e-05-condvec_1/config.toml",
        # default="database/gan_optimize/biobank_patient_dead-tabddpm/config.toml",
        # default="database/gan_optimize/test-wilt-tabddpm-lv_2-bs_256-epochs_1000-df_7-dm_9-dl_8-nl_1-lr_1.96e-03-model_mlp-moment_3-losscorcorr_7.95e-02-lossdis_3.15e-09-condvec_1/config.toml",
        # default="database/gan_optimize/test-diabetesbalanced-tabddpm-lv_2-bs_256-epochs_1000-df_10-dm_10-dl_9-nl_1-lr_3.33e-04-model_mlp-moment_4-losscorcorr_1.69e+00-lossdis_1.43e-08-condvec_1/config.toml",
        # default="database/gan_optimize/ddpm_cb_best/config.toml",
        # default="database/gan_optimize/adult-ddpm_cb_best/config.toml",
        # default="database/gan_optimize/abalone-ddpm_cb_best/config.toml",
        # default="database/gan_optimize/test-abalone-tabddpm-lv_2-bs_4096-epochs_1000-dm_676-dh_224-df_0.2-ds_0.5-nl_3-lr_1.68e-03-model_resnet-moment_3-losscorcorr_4.05e+03-lossdis_6.27e-10-condvec_1/config.toml",
        # default="database/gan_optimize/test-biobank_patient_dead-tabddpm-lv_2-bs_256-epochs_1000-df_8-dm_7-dl_7-nl_2-lr_1.17e-03-model_mlp-moment_4-losscorcorr_4.23e+00-lossdis_7.35e-08-condvec_1/config.toml",
        default="database/gan_optimize/test-adult-tabddpm-lv_2-bs_256-epochs_1000-df_10-dm_9-dl_7-nl_1-lr_2.52e-03-model_mlp-moment_2-losscorcorr_9.22e+03-lossdis_1.35e-09-condvec_1/config.toml",
    )
    parser.add_argument("--dataset", metavar="str", default="biobank_patient_dead")
    parser.add_argument("--change_val", action="store_true", default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    if "device" in raw_config:
        device = torch.device(raw_config["device"])
    else:
        device = torch.device("cuda:0")

    save_file(os.path.join(raw_config["parent_dir"], "config.toml"), args.config)

    train(
        **raw_config["train"]["main"],
        **raw_config["diffusion_params"],
        parent_dir=raw_config["parent_dir"],
        real_data_path=raw_config["real_data_path"],
        model_type=raw_config["model_type"],
        model_params=raw_config["model_params"],
        T_dict=raw_config["train"]["T"],
        num_numerical_features=raw_config["num_numerical_features"],
        device=device,
        change_val=args.change_val,
        ## Added
        num_samples=raw_config["sample"]["num_samples"],
        ## Added
    )

    sample(
        num_samples=raw_config["sample"]["num_samples"],
        batch_size=raw_config["sample"]["batch_size"],
        disbalance=raw_config["sample"].get("disbalance", None),
        **raw_config["diffusion_params"],
        parent_dir=raw_config["parent_dir"],
        real_data_path=raw_config["real_data_path"],
        model_path=os.path.join(raw_config["parent_dir"], "model.pt"),
        model_type=raw_config["model_type"],
        model_params=raw_config["model_params"],
        T_dict=raw_config["train"]["T"],
        num_numerical_features=raw_config["num_numerical_features"],
        device=device,
        seed=raw_config["sample"].get("seed", 0),
        change_val=args.change_val,
        steps=raw_config["train"]["main"]["steps"],
        dataset=args.dataset,
    )

    save_file(
        os.path.join(raw_config["parent_dir"], "info.json"),
        os.path.join(raw_config["real_data_path"], "info.json"),
    )


if __name__ == "__main__":
    main()
