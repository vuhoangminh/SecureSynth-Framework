import os
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time

from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train
import engine.utils.path_utils as path_utils

warnings.filterwarnings("ignore")


def main(args):
    device = args.device

    setattr(args, "dataset", getattr(args, "dataname"))
    setattr(args, "epochs", getattr(args, "num_epochs"))

    dir_logs = os.path.join(args.dir_logs, path_utils.get_folder_technical_paper(args))
    print(dir_logs)

    train_z, _, _, _, _ = get_input_train(args)

    if not os.path.exists(dir_logs):
        os.makedirs(dir_logs)

    in_dim = train_z.shape[1]

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z

    batch_size = args.batch_size
    # batch_size = 4096

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    # num_epochs = 10000 + 1
    # num_epochs = 100 + 1
    num_epochs = args.num_epochs

    denoise_fn = MLPDiffusion(in_dim, args.dim_t).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(
        denoise_fn=denoise_fn,
        hid_dim=train_z.shape[1],
        ## Added by Minh
        loss_version=args.loss_version,
        is_loss_corr=args.is_loss_corr,
        is_loss_dwp=args.is_loss_dwp,
        n_moment_loss_dwp=args.n_moment_loss_dwp,
        ## Added by Minh
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        # lr=1e-3,
        lr=args.lr,
        weight_decay=0,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        # factor=0.9,
        factor=args.factor,
        patience=20,
        verbose=True,
    )

    model.train()

    best_loss = float("inf")
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):

        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)

            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f"{dir_logs}/model.pt")
        else:
            patience += 1
            if patience == 500:
                print("Early stopping")
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f"{dir_logs}/model_{epoch}.pt")

    end_time = time.time()
    print("Time: ", end_time - start_time)

    # Write the string to a text file
    with open(f"{dir_logs}/runtime.txt", "w") as file:
        file.write(f"Time: {end_time - start_time}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training of TabSyn")

    parser.add_argument(
        "--dataname", type=str, default="adult", help="Name of dataset."
    )

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"
