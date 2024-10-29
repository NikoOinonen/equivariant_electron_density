import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_iso_permuted_dataset, get_scalar_density_comparisons


def main():

    parser = argparse.ArgumentParser(description="Test trained model")
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--weights_epoch", type=int)
    parser.add_argument("--dataset", type=str, nargs="+")
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--include_elements", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("What device am I using?", device)

    torch.set_default_dtype(torch.float32)

    run_dir = Path(args.run_dir)
    weights_epoch = args.weights_epoch
    dataset_paths = [Path(p) for p in args.dataset]
    num_samples = args.num_samples

    include_elements = args.include_elements
    if include_elements is not None:
        include_elements = [int(v) for v in include_elements.split("-")]

    print(f"Testing on dataset(s) {', '.join(args.dataset)} using model from {run_dir}.")

    # def2 basis set max irreps
    # WARNING. this is currently hard-coded for def2_universal
    Rs = [(19, 0), (5, 1), (5, 2), (3, 3), (1, 4)]

    density_spacing = 0.1

    with open(run_dir / "run_data.pickle", "rb") as f:
        params = pickle.load(f)

    model_kwargs = {
        "irreps_in": params["irreps_in"],
        "irreps_hidden": params["irreps_hidden"],
        "irreps_out": "19x0e + 5x1o + 5x2e + 3x3o + 1x4e",
        "irreps_node_attr": None,  # irreps_node_attr
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(3),  # irreps_edge_attr
        "layers": params["layers"],
        "max_radius": 3.5,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_neighbors": 12.2298,
        "num_nodes": 24,
        "reduce_output": False,
    }

    model = Network(**model_kwargs)
    model.to(device)

    if weights_epoch:
        weights_path = run_dir / f"model_weights_epoch_{weights_epoch}.pt"
        if not weights_path.exists():
            print(f"Weights for epoch {weights_epoch} not found in {run_dir}.")
            sys.exit(1)
    else:
        weights_paths = list(run_dir.glob("model_weights_epoch_*.pt"))
        if not weights_paths:
            print(f"No weights found in {run_dir}.")
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
    dataset = []
    for dataset_path in dataset_paths:
        dataset += get_iso_permuted_dataset(
            dataset_path,
            free_atom_densities,
            params["free_density_input"],
            Rs,
            include_elements=include_elements,
        )
    if num_samples is None:
        num_samples = len(dataset)
    test_loader = torch_geometric.data.DataLoader(dataset[:num_samples], batch_size=1, shuffle=False)

    eps_cum = 0
    eps_per_l_cum = np.zeros(len(Rs))

    with torch.no_grad():

        for step, data in enumerate(test_loader):

            print(f"Sample {step + 1}/{len(test_loader)}")

            mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
            y_ml = model(data.to(device)) * mask.to(device)

            _, _, _, eps, eps_per_l = get_scalar_density_comparisons(data, y_ml, Rs, spacing=density_spacing, buffer=3.0, ldep=True)
            eps_cum += eps
            eps_per_l_cum += eps_per_l

    print("\nEpsilon:", eps_cum / len(test_loader))
    print("Epsilon l=0", eps_per_l_cum[0] / len(test_loader))
    print("Epsilon l=1", eps_per_l_cum[1] / len(test_loader))
    print("Epsilon l=2", eps_per_l_cum[2] / len(test_loader))
    print("Epsilon l=3", eps_per_l_cum[3] / len(test_loader))
    print("Epsilon l=4", eps_per_l_cum[4] / len(test_loader))


if __name__ == "__main__":
    main()
