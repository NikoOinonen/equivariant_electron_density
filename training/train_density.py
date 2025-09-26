import json
import math
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TrainConfig
from models import MaceNetwork
from utils import get_iso_permuted_dataset, get_scalar_density_comparisons

def load_dataset(data_list_path: Path) -> list[dict]:
    with open(data_list_path) as f:
        data_list = json.loads(f.read())
    dataset = []
    for dataset_path, cids in data_list.items():
        with open(data_list_path.parent / dataset_path, "rb") as f:
            data = pickle.load(f)
        dataset += [data[cid] for cid in cids]
    return dataset

def get_dataloader(
    data_path: Path,
    free_atom_densities: dict[int, Path],
    Rs: list[tuple[int, int]],
    exclude_elements: Optional[list[int]] = None,
    include_elements: Optional[list[int]] = None,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    world_size: int = 1,
    global_rank: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    
    dataset = load_dataset(data_path)
    dataset = get_iso_permuted_dataset(
        dataset,
        free_atom_densities,
        free_density_input=True,
        Rs=Rs,
        exclude_elements=exclude_elements,
        include_elements=include_elements,
    )

    if num_samples is None:
        num_samples = len(dataset)
    elif num_samples > len(dataset):
        raise ValueError("Split is too large for the dataset.")

    chunk = math.floor(num_samples / world_size)
    loader = DataLoader(
        dataset[global_rank * chunk : (global_rank + 1) * chunk],
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return loader


def lossPerChannel(y_ml: torch.Tensor, y_target: torch.Tensor, Rs: list[tuple[int, int]]) -> np.ndarray:

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


def lr_schedule(i_batch: int, lr_init: float = 1e-10, T_warm: int = 1000, T_decay: float = 10000) -> float:
    if i_batch <= T_warm:
        lr = lr_init + (1 - lr_init) * (i_batch / T_warm)
    else:
        lr = 1 / (1 + (i_batch - T_warm) / T_decay)
    return lr


def average_across_ranks(val: Any, device: str | int | torch.device, world_size: int) -> np.ndarray:
    val = torch.tensor(val).to(device)
    dist.all_reduce(val, dist.ReduceOp.SUM)
    val = val.cpu().numpy()
    val /= world_size
    return val


def main():

    # Initialize the distributed environment.
    dist.init_process_group("nccl")
    torch.set_default_dtype(torch.float32)

    config = TrainConfig.from_cmd_args()

    # def2 basis set max irreps
    # WARNING. this is currently hard-coded for def2_universal
    Rs = [(19, 0), (5, 1), (5, 2), (3, 3), (1, 4)]

    # The MACE layers only work with alternating parities for the irreps: https://github.com/ACEsuit/mace/discussions/42
    parity = ["e", "o"]
    irreps_hidden = "+".join(
        [f"{channels}x{l}{parity[l % 2]}" for l, channels in enumerate(config.irreps_hidden.split("-"))]
    )
    irreps_out = "+".join([f"{channels}x{l}{parity[l % 2]}" for channels, l in Rs])

    # https://github.com/ACEsuit/mace/issues/63
    assert (
        len(set(c for c in config.irreps_hidden.split("-"))) == 1
    ), "Hidden irreps must have the same number of channels for all l-orders."

    input_shape = Rs[0][0]  # Input is l=0 components of free densities
    density_spacing = 0.25

    model_kwargs = {
        "irreps_in": f"{input_shape}x0e",
        "irreps_hidden": irreps_hidden,
        "irreps_out": irreps_out,  # = Rs
        "node_attr_dim": None,
        "max_l_edges": len(config.irreps_hidden.split("-")) - 1,
        "message_correlation_order": config.correlation_order,
        "layers": config.num_layers,
        "max_radius": 3.5,
        "number_of_basis": 10,
        "num_neighbors": 12.2298,
        "num_nodes": 26,
        "reduce_output": False,
    }

    print(
        f"Starting on global rank {config.global_rank}, local rank {config.local_rank}. World size {config.world_size}\n",
        flush=True,
    )
    if config.global_rank == 0:
        print(f"Model kwargs: {model_kwargs}")

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

    if config.global_rank == 0:
        print("Loading datasets...")
    train_loader = get_dataloader(
        data_path=config.dataset,
        free_atom_densities=free_atom_densities,
        Rs=Rs,
        exclude_elements=config.exclude_elements,
        include_elements=config.include_elements,
        batch_size=config.batch_size,
        num_samples=config.train_samples,
        world_size=config.world_size,
        global_rank=config.global_rank,
        shuffle=True,
    )
    test_loader = get_dataloader(
        data_path=config.testset,
        free_atom_densities=free_atom_densities,
        Rs=Rs,
        exclude_elements=config.exclude_elements,
        include_elements=config.include_elements,
        batch_size=config.batch_size,
        num_samples=config.test_samples,
        world_size=config.world_size,
        global_rank=config.global_rank,
        shuffle=False,
    )

    model = MaceNetwork(**model_kwargs)
    model.to(config.local_rank)

    optim = Adam(model.parameters(), lr=config.lr)
    optim.zero_grad()

    scheduler = lr_scheduler.LambdaLR(optim, lambda nb: lr_schedule(nb, T_warm=config.lr_warm, T_decay=config.lr_decay))

    if config.run_dir:

        if not config.run_dir.exists():
            if config.global_rank == 0:
                print(f"No existing run directory at {config.run_dir}")
            sys.exit(1)

        # Load weights
        weights_paths = list(config.run_dir.glob("model_weights_epoch_*.pt"))
        if not weights_paths:
            print(f"No weights found in {config.run_dir}.")
            sys.exit(1)
        weights_paths = sorted(weights_paths, key=lambda p: int(p.name.split("_")[-1].split(".")[0]))
        weights_path = weights_paths[-1]
        state = torch.load(weights_path, map_location={"cuda:0": f"cuda:{config.local_rank}"})
        model.load_state_dict(state)

        epoch_num = int(weights_path.name.split("_")[-1].split(".")[0])
        optimizer_state = torch.load(
            config.run_dir / f"optimizer_epoch_{epoch_num}.pt", map_location={"cuda:0": f"cuda:{config.local_rank}"}
        )
        optim.load_state_dict(optimizer_state["optimizer"])
        scheduler.load_state_dict(optimizer_state["scheduler"])

        if config.global_rank == 0:
            print(f"Continuing training using weights from {weights_path}")

        epoch_start = int(weights_path.name.split("_")[-1].split(".")[0])
        i_batch = epoch_start * len(train_loader)

    else:
        epoch_start = 0
        i_batch = 1

    model = DistributedDataParallel(model, device_ids=[config.local_rank])

    if config.global_rank == 0:

        print(f"Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        if not config.run_dir:
            config.run_dir = Path("runs") / (
                f"{datetime.now().strftime('%y%m%d-%H%M%S')}"
                f"_bs{config.world_size * config.batch_size}"
                f"_ns{len(train_loader) * config.world_size}"
                f"_lr{config.lr:.1e}-{config.lr_warm}-{config.lr_decay:.1e}"
                f"_irreps{config.irreps_hidden}x{config.num_layers}"
                f"_corr{config.correlation_order}"
                f"_exc{'-'.join(str(e) for e in config.exclude_elements)}"
                if config.exclude_elements
                else "" f"_inc{'-'.join(str(e) for e in config.include_elements)}" if config.include_elements else ""
            )
        writer = SummaryWriter(str(config.run_dir))

        print(f"Saving log to {config.run_dir}")

        # Dump run information to run directory
        with open(config.run_dir / "run_data.json", "w") as f:
            run_data = config.into_json_dict() | {
                "model_kwargs": model_kwargs,
                "density_spacing": density_spacing,
                "Rs": Rs,
                "job_id": os.environ["SLURM_JOB_ID"],
                "job_name": os.environ["SLURM_JOB_NAME"],
            }
            f.write(json.dumps(run_data, indent=2))
        with open(config.run_dir / "environment.yaml", "w") as f:
            subprocess.run(["conda", "env", "export"], stdout=f)

    loss_cum = 0.0
    loss_per_channel = np.zeros(len(Rs))
    mae = 0.0
    mue = 0.0
    # print_interval = min(500, len(train_loader))
    print_interval = min(50, len(train_loader))

    for epoch in range(epoch_start, config.num_epochs):

        t0_train = time.perf_counter()

        if config.global_rank == 0:
            print(f"Epoch {epoch + 1} Train")

        for step, data in enumerate(train_loader):

            mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
            data = data.to(config.local_rank)

            y_ml = model(data) * mask.to(config.local_rank)
            loss = (y_ml - data.y).pow(2).mean()
            loss.backward()

            optim.step()
            optim.zero_grad()
            scheduler.step()

            for mul, l in Rs:
                if l == 0:
                    num_ele = sum(sum(y_ml[:, :mul])).detach()

            loss_cum += loss.detach()
            loss_per_channel += lossPerChannel(y_ml, data.y, Rs)
            mae += abs(num_ele)
            mue += num_ele

            if i_batch % print_interval == 0:

                # Take an average of the loss value across parallel ranks
                loss_cum = average_across_ranks(loss_cum, config.local_rank, config.world_size)
                loss_per_channel = average_across_ranks(loss_per_channel, config.local_rank, config.world_size)
                mae = average_across_ranks(mae, config.local_rank, config.world_size)
                mue = average_across_ranks(mue, config.local_rank, config.world_size)

                if config.global_rank == 0:

                    print(f"Epoch {epoch + 1}, Train {step + 1}/{len(train_loader)}")

                    writer.add_scalar("Loss/Train", float(loss_cum) / print_interval, i_batch)
                    writer.add_scalar("Loss/Train l=0", float(loss_per_channel[0]) / print_interval, i_batch)
                    writer.add_scalar("Loss/Train l=1", float(loss_per_channel[1]) / print_interval, i_batch)
                    writer.add_scalar("Loss/Train l=2", float(loss_per_channel[2]) / print_interval, i_batch)
                    writer.add_scalar("Loss/Train l=3", float(loss_per_channel[3]) / print_interval, i_batch)
                    writer.add_scalar("Loss/Train l=4", float(loss_per_channel[4]) / print_interval, i_batch)
                    writer.add_scalar("Metrics/Train_MAE", mae / print_interval, i_batch)
                    writer.add_scalar("Metrics/Train_MUE", mue / print_interval, i_batch)
                    writer.add_scalar("Other/Learning rate", scheduler.get_last_lr()[0], i_batch)
                    writer.flush()

                loss_cum = 0.0
                loss_per_channel = np.zeros(len(Rs))
                mae = 0.0
                mue = 0.0

            i_batch += 1

        if config.global_rank == 0:
            print(f"Train time: {time.perf_counter() - t0_train}")

        # Only test at intervals and on the last epoch
        if epoch % config.test_interval != 0 and epoch != (config.num_epochs - 1):
            continue

        if config.global_rank == 0:
            print(f"Epoch {epoch + 1} Test")
            t0_test = time.perf_counter()

        with torch.no_grad():
            test_loss = 0.0
            test_mae = 0.0
            test_mue = 0.0
            bigIs = 0.0
            eps = 0.0
            ep_per_l = np.zeros(len(Rs))
            ele_diff = 0.0
            for step, data in enumerate(test_loader):

                if config.global_rank == 0 and (step + 1) % print_interval == 0:
                    print(f"Epoch {epoch + 1}, Test {step + 1}/{len(test_loader)}")

                mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
                y_ml = model(data.to(config.local_rank)) * mask.to(config.local_rank)
                loss = (y_ml - data.y).pow(2).mean()

                for mul, l in Rs:
                    if l == 0:
                        num_ele = torch.mean(y_ml[:, :mul]).detach()

                test_mue += num_ele
                test_mae += abs(num_ele)
                test_loss += loss.detach()

                num_ele_target, _, bigI, ep, ep_per_l_ = get_scalar_density_comparisons(
                    data, y_ml, Rs, spacing=density_spacing, buffer=3.0, ldep=True
                )
                ep_per_l += ep_per_l_

                n_ele = np.sum(data.z.cpu().detach().numpy())
                ele_diff += np.abs(n_ele - num_ele_target)
                bigIs += bigI
                eps += ep

        test_loss = average_across_ranks(test_loss, config.local_rank, config.world_size)
        test_mae = average_across_ranks(test_mae, config.local_rank, config.world_size)
        test_mue = average_across_ranks(test_mue, config.local_rank, config.world_size)
        bigIs = average_across_ranks(bigIs, config.local_rank, config.world_size)
        eps = average_across_ranks(eps, config.local_rank, config.world_size)
        ep_per_l = average_across_ranks(ep_per_l, config.local_rank, config.world_size)
        ele_diff = average_across_ranks(ele_diff, config.local_rank, config.world_size)

        if config.global_rank == 0:

            print(f"Test time: {time.perf_counter() - t0_test}")

            # Save model
            save_path = config.run_dir / f"model_weights_epoch_{epoch + 1}.pt"
            torch.save(model.module.state_dict(), save_path)
            print(f"Saved model weights on epoch {epoch + 1} to {save_path}.")

            # Save optimizer state
            save_path = config.run_dir / f"optimizer_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "optimizer": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                save_path,
            )
            print(f"Saved optimizer state on epoch {epoch + 1} to {save_path}.")

            # eps per l and loss per l hard coded for def2 below
            writer.add_scalar("Loss/Test", float(test_loss) / len(test_loader), i_batch)
            writer.add_scalar("Metrics/Test_MAE", test_mae / len(test_loader), i_batch)
            writer.add_scalar("Metrics/Test_MUE", test_mue / len(test_loader), i_batch)
            writer.add_scalar("Other/Test_Electron_Difference", ele_diff / len(test_loader), i_batch)
            writer.add_scalar("Other/Test_big_I", bigIs / len(test_loader), i_batch)
            writer.add_scalar("Other/Test_Epsilon", eps / len(test_loader), i_batch)
            writer.add_scalar("Other/Test_Epsilon l=0", ep_per_l[0] / len(test_loader), i_batch)
            writer.add_scalar("Other/Test_Epsilon l=1", ep_per_l[1] / len(test_loader), i_batch)
            writer.add_scalar("Other/Test_Epsilon l=2", ep_per_l[2] / len(test_loader), i_batch)
            writer.add_scalar("Other/Test_Epsilon l=3", ep_per_l[3] / len(test_loader), i_batch)
            writer.add_scalar("Other/Test_Epsilon l=4", ep_per_l[4] / len(test_loader), i_batch)
            writer.add_scalar("Other/Epoch", epoch + 1, i_batch)
            writer.flush()

    if config.global_rank == 0:
        writer.close()


if __name__ == "__main__":
    main()
