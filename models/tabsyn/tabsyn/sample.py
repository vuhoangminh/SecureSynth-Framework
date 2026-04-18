import os
import torch

import argparse
import warnings
import time

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample

import engine.utils.path_utils as path_utils

warnings.filterwarnings("ignore")


def main(args):
    def gen_data(_num_samples, _save_path):
        sample_dim = in_dim

        x_next = sample(model.denoise_fn_D, _num_samples, sample_dim)
        x_next = x_next * 2 + mean.to(device)

        syn_data = x_next.float().cpu().numpy()
        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, info, num_inverse, cat_inverse, args.device
        )

        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info["idx_name_mapping"]
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns=idx_name_mapping, inplace=True)
        syn_df.to_csv(
            _save_path,
            sep="\t",
            encoding="utf-8",
        )

        end_time = time.time()
        print("Time:", end_time - start_time)

        print("Saving sampled data to {}".format(_save_path))

    dataname = args.dataname
    device = args.device
    steps = args.steps

    setattr(args, "dataset", getattr(args, "dataname"))
    setattr(args, "epochs", getattr(args, "num_epochs"))

    train_z, _, _, _, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, args.dim_t).to(device)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    dir_logs = os.path.join(args.dir_logs, path_utils.get_folder_technical_paper(args))
    print(dir_logs)
    save_path = dir_logs + f"/fake_{args.epochs:05}.csv"

    model.load_state_dict(torch.load(f"{dir_logs}/model.pt"))

    """
        Generating samples    
    """
    start_time = time.time()

    num_samples = int(train_z.shape[0] * 1.2)
    gen_data(num_samples, save_path)

    if args.row_number_full is not None:
        gen_data(args.row_number_full, dir_logs + f"/synthetic_full.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generation")

    parser.add_argument(
        "--dataname", type=str, default="adult", help="Name of dataset."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch.")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of function evaluations."
    )

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"
