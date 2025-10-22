from copy import deepcopy
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from train_density import (
    average_across_ranks,
    get_dataloader,
    get_lr_scheduler,
    lossPerChannel,
)

from config import TrainConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import MaceNetwork
from utils import DensityStatistics


def main():

    config = TrainConfig.from_cmd_args()
    with open(config.base_model / "run_data.json") as f:
        run_data = json.loads(f.read())

    # Initialize the distributed environment.
    dist.init_process_group("nccl")
    torch.set_default_dtype(torch.float32)
    mp.set_start_method("spawn")

    Rs = run_data["Rs"]
    density_spacing = run_data["density_spacing"]
    model_kwargs = run_data["model_kwargs"]

    if config.finetune_method == "elora":
        model_kwargs["r_lora"] = config.elora_rank

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

    optim = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optim.zero_grad()

    scheduler = get_lr_scheduler(
        scheduler_type=config.lr_scheduler,
        optim=optim,
        lr_warm=config.lr_warm,
        lr_decay=config.lr_decay,
        lr_mult=config.lr_mult,
        n_batch_per_epoch=len(train_loader),
    )

    weights_paths = list(config.base_model.glob("model_weights_epoch_*.pt"))
    if not weights_paths:
        print(f"No weights found in {config.base_model}.")
        sys.exit(1)
    weights_paths = sorted(weights_paths, key=lambda p: int(p.name.split("_")[-1].split(".")[0]))
    weights_path = weights_paths[-1]

    print(f"Using weights from {weights_path}")
    weights_init = torch.load(weights_path)
    model.load_state_dict(weights_init, strict=False)

    if config.finetune_method == "restart-all":
        # We just retrain all weights starting from the base model. Does not require any action.
        pass
    elif config.finetune_method == "restart-readout":
        # Freeze all layers except the readout layer
        for name, param in model.named_parameters():
            if "readout" not in name:
                param.requires_grad = False
    elif config.finetune_method == "elora":
        # We freeze all weights except the ELoRA weights
        # Also train "symmetric_contractions" following https://github.com/hyjwpk/ELoRA
        for name, param in model.named_parameters():
            if not ("LoRA" in name or ("symmetric_contractions" in name and "weights_max" not in name)):
                param.requires_grad = False
    else:
        raise ValueError(f"Unknown fine-tuning method {config.finetune_method}")

    model = DistributedDataParallel(model, device_ids=[config.local_rank])

    if config.global_rank == 0:

        print("Fine tuning the following layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

        print(f"Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        ft_str = config.finetune_method if config.finetune_method != "elora" else f"ELoRA-r{config.elora_rank}"
        l2_str = f"_l2-{config.weight_decay:.0e}" if config.weight_decay > 0 else ""
        config.run_dir = config.runs_base_dir / (
            f"{datetime.now().strftime('%y%m%d-%H%M%S')}"
            f"_base-{config.base_model.name.split('_')[0]}"
            f"_{ft_str}"
            f"_db-{config.dataset.stem}"
            f"_bs{config.world_size * config.batch_size}"
            f"_ns{len(train_loader) * config.world_size * config.batch_size}"
            f"_lr{config.lr:.1e}-{config.lr_warm}-{config.lr_decay:.1e}" + l2_str
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

        # Save initial weights state
        torch.save(weights_init, config.run_dir / f"model_weights_epoch_0.pt")

    loss_train = 0.0
    loss_per_channel = np.zeros(len(Rs))
    mae_train = 0.0
    mue_train = 0.0
    i_batch = 1
    print_interval = min(500, len(train_loader))
    density_stats = DensityStatistics(
        Rs=Rs,
        num_eps_per_l=len(model.module.irreps_out),
        spacing=density_spacing,
        buffer=3.0,
        num_proc=config.num_proc_test,
    )

    for epoch in range(config.num_epochs):

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
                    num_ele = torch.mean(y_ml[:, :mul]).detach()

            loss_train += loss.detach()
            loss_per_channel += lossPerChannel(y_ml, data.y, Rs)
            mae_train += abs(num_ele)
            mue_train += num_ele

            if i_batch % print_interval == 0:

                # Take an average of the loss value across parallel ranks
                loss_train = average_across_ranks(loss_train, config.local_rank, config.world_size)
                loss_per_channel = average_across_ranks(loss_per_channel, config.local_rank, config.world_size)
                mae_train = average_across_ranks(mae_train, config.local_rank, config.world_size)
                mue_train = average_across_ranks(mue_train, config.local_rank, config.world_size)

                if config.global_rank == 0:

                    print(f"Epoch {epoch + 1}, Train {step + 1}/{len(train_loader)}")

                    writer.add_scalar("Loss/Train", float(loss_train) / print_interval, i_batch)
                    for l, loss_ in enumerate(loss_per_channel):
                        writer.add_scalar(f"Loss/Train l={l}", float(loss_) / print_interval, i_batch)
                    writer.add_scalar("Metrics/Train_MAE", mae_train / print_interval, i_batch)
                    writer.add_scalar("Metrics/Train_MUE", mue_train / print_interval, i_batch)
                    writer.add_scalar("Other/Learning rate", scheduler.get_last_lr()[0], i_batch)
                    writer.flush()

                loss_train = 0.0
                loss_per_channel = np.zeros(len(Rs))
                mae_train = 0.0
                mue_train = 0.0

            i_batch += 1

        if config.global_rank == 0:
            print(f"Train time: {time.perf_counter() - t0_train}")

        # Only test at intervals and on the last epoch
        if epoch % config.test_interval != 0 and epoch != (config.num_epochs - 1):
            continue

        if config.global_rank == 0:
            print(f"Epoch {epoch + 1} Test")
            t0_test = time.perf_counter()

        loss_test = 0.0
        density_stats.start()

        with torch.no_grad():

            for step, data in enumerate(test_loader):

                if config.global_rank == 0 and (step + 1) % print_interval == 0:
                    print(f"Epoch {epoch + 1}, Test {step + 1}/{len(test_loader)}")

                mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
                y_ml = model(data.to(config.local_rank)) * mask.to(config.local_rank)
                loss = (y_ml - data.y).pow(2).mean()

                loss_test += loss.detach()
                density_stats.add_batch(data, y_ml)

        loss_test /= len(test_loader)
        mae_test, mue_test, ele_diff, bigIs, eps, eps_per_l = density_stats.get_results_mean()

        loss_test = average_across_ranks(loss_test, config.local_rank, config.world_size)
        mae_test = average_across_ranks(mae_test, config.local_rank, config.world_size)
        mue_test = average_across_ranks(mue_test, config.local_rank, config.world_size)
        ele_diff = average_across_ranks(ele_diff, config.local_rank, config.world_size)
        bigIs = average_across_ranks(bigIs, config.local_rank, config.world_size)
        eps = average_across_ranks(eps, config.local_rank, config.world_size)
        eps_per_l = average_across_ranks(eps_per_l, config.local_rank, config.world_size)

        if config.global_rank == 0:

            print(f"Test time: {time.perf_counter() - t0_test}")

            # Save model
            save_path = config.run_dir / f"model_weights_epoch_{epoch + 1}.pt"
            module = deepcopy(model.module)
            module.merge_LoRA()
            torch.save(module.state_dict(), save_path)
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

            writer.add_scalar("Loss/Test", loss_test, i_batch)
            writer.add_scalar("Metrics/Test_MAE", mae_test, i_batch)
            writer.add_scalar("Metrics/Test_MUE", mue_test, i_batch)
            writer.add_scalar("Other/Test_Electron_Difference", ele_diff, i_batch)
            writer.add_scalar("Other/Test_big_I", bigIs, i_batch)
            writer.add_scalar("Other/Test_Epsilon", eps, i_batch)
            for l, ep in enumerate(eps_per_l):
                writer.add_scalar(f"Other/Test_Epsilon l={l}", ep, i_batch)
            writer.add_scalar("Other/Epoch", epoch + 1, i_batch)
            writer.flush()

    if config.global_rank == 0:
        writer.close()


if __name__ == "__main__":
    main()
