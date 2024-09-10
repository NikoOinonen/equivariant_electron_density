import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from ase import Atoms
from ase.io.xsf import write_xsf
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import gau2grid_density_kdtree, get_iso_permuted_dataset, find_min_max


def generate_grid(atom_pos: np.ndarray, spacing: float = 0.1, buffer: float = 2.0):

    origin = atom_pos.min(axis=0) - buffer
    n_points = ((atom_pos.max(axis=0) + buffer) - origin) // spacing + 1
    n_points = n_points.astype(np.int32)

    xyz = [np.linspace(origin[i], origin[i] + n_points[i] * spacing, n_points[i]) for i in range(3)]
    x, y, z = np.meshgrid(*xyz, indexing="ij")

    return x, y, z, origin


def save_to_xsf(file_path: Path, atom_pos: np.ndarray, atom_types: np.ndarray, density: np.ndarray, lattice_spacing: float):
    lattice = lattice_spacing * np.diag(density.shape)
    atoms = Atoms(numbers=atom_types, positions=atom_pos, cell=lattice, pbc=True)
    with open(file_path, "w") as f:
        write_xsf(f, [atoms], data=density)


def main():

    parser = argparse.ArgumentParser(description="Predict on trained model")
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--weights_epoch", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--num_samples", type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("What device am I using?", device)

    torch.set_default_dtype(torch.float32)

    run_dir = Path(args.run_dir)
    weights_epoch = args.weights_epoch
    dataset_path = Path(args.dataset)
    num_samples = args.num_samples
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = run_dir / "predictions"

    # def2 basis set max irreps
    # WARNING. this is currently hard-coded for def2_universal
    Rs = [(19, 0), (5, 1), (5, 2), (3, 3), (1, 4)]

    density_spacing = 0.1
    grid_buffer = 3.0
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

    print("Loading test set")
    dataset_path = get_iso_permuted_dataset(dataset_path, free_atom_densities)
    test_loader = torch_geometric.data.DataLoader(dataset_path[:num_samples], batch_size=1, shuffle=False)

    print(f"Saving predictions to {out_dir}")
    out_dir.mkdir(exist_ok=True)

    with torch.no_grad():

        for step, data in enumerate(test_loader):

            print(f"Prediction {step + 1}/{len(test_loader)}")

            mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
            y_ml = model(data.to(device)) * mask.to(device)

            atom_pos = data["pos_orig"].cpu().numpy()
            x, y, z, origin = generate_grid(atom_pos, spacing=density_spacing, buffer=grid_buffer)
            target_density, ml_density = gau2grid_density_kdtree(x.flatten(), y.flatten(), z.flatten(), data, y_ml, Rs, ldepb=False)

            target_density = target_density.reshape(x.shape)
            ml_density = ml_density.reshape(x.shape)
            density_diff = target_density - ml_density
            density_diff_rel = density_diff / target_density
            density_diff_rel[target_density < 1e-4] = 0

            atom_pos -= origin
            atom_types = data["z"][:, 0].int().cpu().numpy()

            save_to_xsf(out_dir / f"{step}_target.xsf", atom_pos, atom_types, target_density, density_spacing)
            save_to_xsf(out_dir / f"{step}_prediction.xsf", atom_pos, atom_types, ml_density, density_spacing)
            save_to_xsf(out_dir / f"{step}_diff.xsf", atom_pos, atom_types, density_diff, density_spacing)
            save_to_xsf(out_dir / f"{step}_relative_diff.xsf", atom_pos, atom_types, density_diff_rel, density_spacing)


if __name__ == "__main__":
    main()
