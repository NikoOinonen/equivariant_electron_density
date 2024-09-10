import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
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


def main():
    parser = argparse.ArgumentParser(description="train electron density")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--testset", type=str)
    parser.add_argument("--train_split", type=int)
    parser.add_argument("--test_split", type=int)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--qm", type=str, default="pbe0")
    parser.add_argument("--ldep", type=bool, default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("What device am I using?", device)

    torch.set_default_dtype(torch.float32)

    test_dataset = args.testset
    num_epochs = args.epochs
    ldep_bool = args.ldep

    # def2 basis set max irreps
    # WARNING. this is currently hard-coded for def2_universal
    Rs = [(19, 0), (5, 1), (5, 2), (3, 3), (1, 4)]

    train_split = args.train_split
    test_split = args.test_split
    data_file = args.dataset
    lr = 1e-2
    density_spacing = 0.25
    save_interval = 1
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

    data_path = Path(__file__).parent.parent / "data"
    free_atom_densities = {
        1: data_path / "H_def2-universal-jfit-decon_density.out",
        6: data_path / "C_def2-universal-jfit-decon_density.out",
        7: data_path / "N_def2-universal-jfit-decon_density.out",
        8: data_path / "O_def2-universal-jfit-decon_density.out",
        9: data_path / "F_def2-universal-jfit-decon_density.out",
        14: data_path / "Si_def2-universal-jfit-decon_density.out",
        15: data_path / "P_def2-universal-jfit-decon_density.out",
        16: data_path / "S_def2-universal-jfit-decon_density.out",
        17: data_path / "Cl_def2-universal-jfit-decon_density.out",
        35: data_path / "Br_def2-universal-jfit-decon_density.out",
    }

    print("Loading train set")
    dataset = get_iso_permuted_dataset(data_file, free_atom_densities)
    random.shuffle(dataset)

    print("Loading test set")
    test_dataset = get_iso_permuted_dataset(args.testset, free_atom_densities)

    if train_split is None:
        train_split = len(dataset)
    elif train_split > len(dataset):
        raise ValueError("Split is too large for the dataset.")

    if test_split is None:
        test_split = len(test_dataset)
    elif test_split > len(test_dataset):
        raise ValueError("Split is too large for the test set.")

    batch_size = 1
    train_loader = torch_geometric.data.DataLoader(dataset[:train_split], batch_size=batch_size, shuffle=True)
    test_loader = torch_geometric.data.DataLoader(test_dataset[:test_split], batch_size=batch_size, shuffle=True)

    model = Network(**model_kwargs)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    optim.zero_grad()

    model.to(device)

    model_kwargs["train_dataset"] = data_file
    model_kwargs["train_dataset_size"] = train_split
    model_kwargs["lr"] = lr
    model_kwargs["density_spacing"] = density_spacing

    writer = SummaryWriter()
    run_dir = Path(writer.get_logdir())
    print(run_dir)

    i_batch = 1
    loss_cum = 0.0
    loss_perchannel_cum = np.zeros(len(Rs))
    mae_cum = 0.0
    mue_cum = 0.0
    for epoch in range(num_epochs):

        for step, data in enumerate(train_loader):

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
                writer.flush()

                loss_cum = 0.0
                loss_perchannel_cum = np.zeros(len(Rs))
                mae_cum = 0.0
                mue_cum = 0.0

            mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
            output = model(data.to(device))
            y_ml = output * mask.to(device)
            err = y_ml - data.y.to(device)

            for mul, l in Rs:
                if l == 0:
                    num_ele = sum(sum(y_ml[:, :mul])).detach()

            mue_cum += num_ele
            mae_cum += abs(num_ele)

            # compute loss per channel
            if ldep_bool:
                loss_perchannel_cum += lossPerChannel(y_ml, data.y.to(device), Rs)

            loss = err.pow(2).mean()
            loss_cum += loss.detach().abs()

            loss.backward()
            optim.step()
            optim.zero_grad()

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
                test_loss_cum += err.pow(2).mean().detach().abs()

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

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), run_dir / f"model_weights_epoch_{epoch + 1}.pt")

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
