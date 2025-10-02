import json
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from train_density import get_dataloader

from config import TestConfig
from models import MaceNetwork

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DensityStatistics


def main():

    mp.set_start_method("spawn")

    config = TestConfig.from_cmd_args()
    with open(config.run_dir / "run_data.json") as f:
        run_data = json.loads(f.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("What device am I using?", device)

    torch.set_default_dtype(torch.float32)

    print(f"Testing on dataset(s) {config.testset} using model from {config.run_dir}.")

    Rs = run_data["Rs"]
    density_spacing = 0.1
    test_start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    model = MaceNetwork(**run_data["model_kwargs"])
    model.to(device)

    if config.weights_epoch:
        weights_path = config.run_dir / f"model_weights_epoch_{config.weights_epoch}.pt"
        if not weights_path.exists():
            print(f"Weights for epoch {config.weights_epoch} not found in {config.run_dir}.")
            sys.exit(1)
    else:
        weights_paths = list(config.run_dir.glob("model_weights_epoch_*.pt"))
        if not weights_paths:
            print(f"No weights found in {config.run_dir}.")
            sys.exit(1)
        weights_paths = sorted(weights_paths, key=lambda p: int(p.name.split("_")[-1].split(".")[0]))
        weights_path = weights_paths[-1]

    print(f"Using weights from {weights_path}")
    state = torch.load(weights_path)
    model.load_state_dict(state)

    data_path = Path(__file__).parent.parent / "data" / "free_atom_s_only"
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

    print("Loading test set")
    test_loader = get_dataloader(
        data_path=config.testset,
        free_atom_densities=free_atom_densities,
        Rs=Rs,
        exclude_elements=config.exclude_elements,
        include_elements=config.include_elements,
        num_samples=config.test_samples,
        batch_size=config.batch_size,
    )

    eps = 0
    eps_per_l = np.zeros(len(Rs))
    test_loss = 0.0
    n_batch = len(test_loader)

    density_stats = DensityStatistics(
        Rs=Rs,
        num_eps_per_l=len(model.irreps_out),
        spacing=density_spacing,
        buffer=3.0,
        num_proc=config.num_proc_test,
    )
    density_stats.start()

    with torch.no_grad():

        for step, data in enumerate(test_loader):

            print(f"Sample {step + 1}/{n_batch}")

            mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
            y_ml = model(data.to(device)) * mask.to(device)
            loss = (y_ml - data.y).pow(2).mean()

            test_loss += loss.detach()
            density_stats.add_batch(data, y_ml)

    test_loss /= n_batch
    _, _, _, _, eps, eps_per_l = density_stats.get_results()

    print(f"\nTest loss: {test_loss}")
    print(f"Epsilon: {eps}")
    for l, ep in enumerate(eps_per_l):
        print(f"Epsilon l={l}: {ep}")

    with open(config.run_dir / f"test_{test_start_time}.results", "w") as f:
        f.write(f"Weights epoch: {config.weights_epoch}\n")
        f.write(f"Test set: {config.testset}\n")
        f.write(f"Number of samples: {n_batch}\n")
        f.write(f"Include elements: {config.include_elements}\n")
        f.write(f"Exclude elements: {config.exclude_elements}\n")
        f.write(f"Test loss: {test_loss}\n")
        f.write(f"Epsilon: {eps}\n")
        for l, ep in enumerate(eps_per_l):
            f.write(f"Epsilon l={l}: {ep}\n")


if __name__ == "__main__":
    main()
