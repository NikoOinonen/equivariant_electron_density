import argparse
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_iso_permuted_dataset, get_scalar_density_comparisons


def lossPerChannel(y_ml, y_target, Rs=[(12, 0), (5, 1), (4, 2), (2, 3), (1, 4)]):

    err = y_ml - y_target
    loss_perChannel_list = np.zeros(len(Rs))
    normalization = err.sum() / err.mean()

    counter = 0
    for mul, l in Rs:
        if l == 0:
            temp_loss = err[:, :mul].pow(2).sum().abs() / normalization
        else:
            temp_loss = err[:, counter : counter + mul * (2 * l + 1)].pow(2).sum().abs() / normalization

        loss_perChannel_list[l] += temp_loss.detach().cpu().numpy()

        counter += mul * (2 * l + 1)

    return loss_perChannel_list


def lr_schedule(i_batch, lr_init=1e-10, T_warm=1000, T_decay=10000):
    if i_batch <= T_warm:
        lr = lr_init + (1 - lr_init) * (i_batch / T_warm)
    else:
        lr = 1 / (1 + (i_batch - T_warm) / T_decay)
    return lr


def main():
    parser = argparse.ArgumentParser(description="train electron density")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--testset", type=str)
    parser.add_argument("--train_split", type=int)
    parser.add_argument("--test_split", type=int)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_average", type=int, default=1)
    parser.add_argument("--continue_run", type=str)
    parser.add_argument("--run_comment", type=str, default="")
    parser.add_argument("--ldep", type=bool, default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("What device am I using?", device)

    torch.set_default_dtype(torch.float32)

    # def2 basis set max irreps
    # WARNING. this is currently hard-coded for def2_universal
    Rs = [(19, 0), (5, 1), (5, 2), (3, 3), (1, 4)]

    train_split = args.train_split
    test_split = args.test_split
    train_data_path = Path(args.dataset)
    test_data_path = Path(args.testset)
    num_epochs = args.epochs
    batch_average = args.batch_average
    ldep_bool = args.ldep

    lr = 1e-2
    lr_warm = 4000
    lr_decay = 10000
    density_spacing = 0.25
    print_interval = 500
    model_kwargs = {
        "irreps_in": "10x 0e",  # irreps_in (= number of atom types)
        "irreps_hidden": [(mul, (l, p)) for l, mul in enumerate([125, 40, 25, 15]) for p in [-1, 1]],  # irreps_hidden
        "irreps_out": "19x0e + 5x1o + 5x2e + 3x3o + 1x4e",  # irreps_out (= Rs)
        "irreps_node_attr": None,  # irreps_node_attr
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(3),  # irreps_edge_attr
        "layers": 3,
        "max_radius": 3.5,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_neighbors": 12.2298,
        "num_nodes": 24,
        "reduce_output": False,
    }

    free_atom_data_path = Path(__file__).parent.parent / "data" / "free_atom_s_only"
    free_atom_densities = {
        1: free_atom_data_path / "H_def2-universal-jfit-decon_density.out",
        6: free_atom_data_path / "C_def2-universal-jfit-decon_density.out",
        7: free_atom_data_path / "N_def2-universal-jfit-decon_density.out",
        8: free_atom_data_path / "O_def2-universal-jfit-decon_density.out",
        9: free_atom_data_path / "F_def2-universal-jfit-decon_density.out",
        14: free_atom_data_path / "Si_def2-universal-jfit-decon_density.out",
        15: free_atom_data_path / "P_def2-universal-jfit-decon_density.out",
        16: free_atom_data_path / "S_def2-universal-jfit-decon_density.out",
        17: free_atom_data_path / "Cl_def2-universal-jfit-decon_density.out",
        35: free_atom_data_path / "Br_def2-universal-jfit-decon_density.out",
    }

    print("Loading train set")
    train_dataset = get_iso_permuted_dataset(train_data_path, free_atom_densities)
    random.shuffle(train_dataset)

    print("Loading test set")
    test_dataset = get_iso_permuted_dataset(test_data_path, free_atom_densities)

    if train_split is None:
        train_split = len(train_dataset)
    elif train_split > len(train_dataset):
        raise ValueError("Split is too large for the dataset.")

    if test_split is None:
        test_split = len(test_dataset)
    elif test_split > len(test_dataset):
        raise ValueError("Split is too large for the test set.")

    train_loader = torch_geometric.data.DataLoader(train_dataset[:train_split], batch_size=1, shuffle=True)
    test_loader = torch_geometric.data.DataLoader(test_dataset[:test_split], batch_size=1, shuffle=True)

    model = Network(**model_kwargs)
    model.to(device)

    optim = Adam(model.parameters(), lr=lr)
    optim.zero_grad()

    scheduler = lr_scheduler.LambdaLR(optim, lambda nb: lr_schedule(nb, T_warm=lr_warm, T_decay=lr_decay))

    if args.continue_run:

        run_dir = Path(args.continue_run)
        if not run_dir.exists():
            print(f"No existing run directory at {run_dir}")
            sys.exit(1)

        writer = SummaryWriter(log_dir=run_dir)

        # Load weights
        weights_paths = list(run_dir.glob("model_weights_epoch_*.pt"))
        if not weights_paths:
            print(f"No weights found in {run_dir}.")
            sys.exit(1)
        weights_paths = sorted(weights_paths, key=lambda p: int(p.name.split("_")[-1].split(".")[0]))
        weights_path = weights_paths[-1]
        state = torch.load(weights_path)
        model.load_state_dict(state)

        epoch_num = int(weights_path.name.split("_")[-1].split(".")[0])
        optimizer_state = torch.load(run_dir / f"optimizer_epoch_{epoch_num}.pt")
        optim.load_state_dict(optimizer_state["optimizer"])
        scheduler.load_state_dict(optimizer_state["scheduler"])

        print(f"Continuing training using weights from {weights_path}")

        epoch_start = int(weights_path.name.split("_")[-1].split(".")[0])
        i_batch = epoch_start * len(train_loader)

    else:
        comment = f"_{args.run_comment}" if args.run_comment else ""
        writer = SummaryWriter(comment=comment)
        run_dir = Path(writer.get_logdir())
        epoch_start = 0
        i_batch = 1

    print(f"Saving log to {run_dir}")

    # Dump run information to run directory
    with open(run_dir / "run_data.pickle", "wb") as f:
        run_data = model_kwargs | {
            "train_data_path": train_data_path.resolve(),
            "train_dataset_size": train_split,
            "test_data_path": test_data_path.resolve(),
            "test_dataset_size": test_split,
            "batch_average": batch_average,
            "lr": lr,
            "lr_warm": lr_warm,
            "lr_decay": lr_decay,
            "density_spacing": density_spacing,
            "Rs": Rs,
            "job_id": os.environ["SLURM_JOB_ID"],
            "job_name": os.environ["SLURM_JOB_NAME"],
        }
        pickle.dump(run_data, f)
    with open(run_dir / "environment.yaml", "w") as f:
        subprocess.run(["conda", "env", "export"], stdout=f)

    loss_cum = 0.0
    loss_perchannel_cum = np.zeros(len(Rs))
    mae_cum = 0.0
    mue_cum = 0.0

    for epoch in range(epoch_start, num_epochs):

        for step, data in enumerate(train_loader):

            mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
            output = model(data.to(device))
            y_ml = output * mask.to(device)
            err = y_ml - data.y.to(device)

            for mul, l in Rs:
                if l == 0:
                    num_ele = sum(sum(y_ml[:, :mul])).detach()

            mue_cum += num_ele / batch_average
            mae_cum += abs(num_ele) / batch_average

            # compute loss per channel
            if ldep_bool:
                loss_perchannel_cum += lossPerChannel(y_ml, data.y.to(device), Rs) / batch_average

            loss = err.pow(2).mean() / batch_average
            loss_cum += loss.detach()

            loss.backward()

            if (step + 1) % batch_average == 0 or (step + 1) == len(train_loader):

                optim.step()
                optim.zero_grad()
                scheduler.step()

                if i_batch % print_interval == 0:

                    print(f"Epoch {epoch + 1}, Train {step + 1}/{len(train_loader)}")

                    writer.add_scalar("Loss/Train", float(loss_cum) / print_interval, i_batch)
                    writer.add_scalar("Loss/Train l=0", float(loss_perchannel_cum[0]) / print_interval, i_batch)
                    writer.add_scalar("Loss/Train l=1", float(loss_perchannel_cum[1]) / print_interval, i_batch)
                    writer.add_scalar("Loss/Train l=2", float(loss_perchannel_cum[2]) / print_interval, i_batch)
                    writer.add_scalar("Loss/Train l=3", float(loss_perchannel_cum[3]) / print_interval, i_batch)
                    writer.add_scalar("Loss/Train l=4", float(loss_perchannel_cum[4]) / print_interval, i_batch)
                    writer.add_scalar("Metrics/Train_MAE", mae_cum / print_interval, i_batch)
                    writer.add_scalar("Metrics/Train_MUE", mue_cum / print_interval, i_batch)
                    writer.add_scalar("Other/Learning rate", scheduler.get_last_lr()[0], i_batch)
                    writer.flush()

                    loss_cum = 0.0
                    loss_perchannel_cum = np.zeros(len(Rs))
                    mae_cum = 0.0
                    mue_cum = 0.0

                i_batch += 1

        # now the test loop
        print(f"Epoch {epoch + 1} Test")
        with torch.no_grad():
            test_loss_cum = 0.0
            test_mae_cum = 0.0
            test_mue_cum = 0.0
            bigIs_cum = 0.0
            eps_cum = 0.0
            ep_per_l_cum = np.zeros(len(Rs))
            ele_diff_cum = 0.0
            for step, data in enumerate(test_loader):

                if (step + 1) % print_interval == 0:
                    print(f"Epoch {epoch + 1}, Test {step + 1}/{len(test_loader)}")

                mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
                y_ml = model(data.to(device)) * mask.to(device)
                err = y_ml - data.y.to(device)

                for mul, l in Rs:
                    if l == 0:
                        num_ele = torch.mean(y_ml[:, :mul]).detach()

                test_mue_cum += num_ele
                test_mae_cum += abs(num_ele)
                test_loss_cum += err.pow(2).mean().detach()

                if ldep_bool:
                    num_ele_target, _, bigI, ep, ep_per_l = get_scalar_density_comparisons(
                        data, y_ml, Rs, spacing=density_spacing, buffer=3.0, ldep=ldep_bool
                    )
                    ep_per_l_cum += ep_per_l
                else:
                    num_ele_target, _, bigI, ep = get_scalar_density_comparisons(
                        data, y_ml, Rs, spacing=density_spacing, buffer=3.0, ldep=ldep_bool
                    )

                n_ele = np.sum(data.z.cpu().detach().numpy())
                ele_diff_cum += np.abs(n_ele - num_ele_target)
                bigIs_cum += bigI
                eps_cum += ep

        # Save model
        save_path = run_dir / f"model_weights_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved model weights on epoch {epoch + 1} to {save_path}.")

        # Save optimizer state
        save_path = run_dir / f"optimizer_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            save_path,
        )
        print(f"Saved optimizer state on epoch {epoch + 1} to {save_path}.")

        # eps per l and loss per l hard coded for def2 below
        writer.add_scalar("Loss/Test", float(test_loss_cum.item()) / len(test_loader), i_batch)
        writer.add_scalar("Metrics/Test_MAE", test_mae_cum.item() / len(test_loader), i_batch)
        writer.add_scalar("Metrics/Test_MUE", test_mue_cum.item() / len(test_loader), i_batch)
        writer.add_scalar("Other/Test_Electron_Difference", ele_diff_cum.item() / len(test_loader), i_batch)
        writer.add_scalar("Other/Test_big_I", bigIs_cum.item() / len(test_loader), i_batch)
        writer.add_scalar("Other/Test_Epsilon", eps_cum.item() / len(test_loader), i_batch)
        writer.add_scalar("Other/Test_Epsilon l=0", ep_per_l_cum[0].item() / len(test_loader), i_batch)
        writer.add_scalar("Other/Test_Epsilon l=1", ep_per_l_cum[1].item() / len(test_loader), i_batch)
        writer.add_scalar("Other/Test_Epsilon l=2", ep_per_l_cum[2].item() / len(test_loader), i_batch)
        writer.add_scalar("Other/Test_Epsilon l=3", ep_per_l_cum[3].item() / len(test_loader), i_batch)
        writer.add_scalar("Other/Test_Epsilon l=4", ep_per_l_cum[4].item() / len(test_loader), i_batch)
        writer.add_scalar("Other/Epoch", epoch + 1, i_batch)
        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()
