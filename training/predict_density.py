import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io.xsf import write_xsf
from train_density import get_dataloader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TestConfig
from models import MaceNetwork
from utils import gau2grid_density_kdtree


def generate_grid(atom_pos: np.ndarray, spacing: float = 0.1, buffer: float = 2.0):

    origin = atom_pos.min(axis=0) - buffer
    n_points = ((atom_pos.max(axis=0) + buffer) - origin) // spacing + 1
    n_points = n_points.astype(np.int32)

    xyz = [np.linspace(origin[i], origin[i] + n_points[i] * spacing, n_points[i]) for i in range(3)]
    x, y, z = np.meshgrid(*xyz, indexing="ij")

    return x, y, z, origin


def save_to_xsf(
    file_path: Path, atom_pos: np.ndarray, atom_types: np.ndarray, density: np.ndarray, lattice_spacing: float
):
    lattice = lattice_spacing * np.diag(density.shape)
    atoms = Atoms(numbers=atom_types, positions=atom_pos, cell=lattice, pbc=True)
    with open(file_path, "w") as f:
        write_xsf(f, [atoms], data=density)


def complete_coefficients(data, ml_y, rs):
    ml_coeffs_full = np.zeros(data.full_c.shape)
    for i_atom, (iso_coeffs, ml_coeffs, norm) in enumerate(
        zip(
            data.iso_c.cpu().detach().numpy(),
            ml_y.cpu().detach().numpy(),
            data.norm.cpu().detach().numpy(),
        )
    ):
        counter = 0
        for mul, l in rs:
            for _ in range(mul):
                normal = norm[counter]
                if normal != 0:
                    pop_ml = ml_coeffs[counter : counter + (2 * l + 1)]
                    c_ml = pop_ml * normal / (2 * np.sqrt(2))
                    ml_coeffs_full[i_atom, counter : counter + (2 * l + 1)] = (
                        c_ml + iso_coeffs[counter : counter + (2 * l + 1)]
                    )
                counter += 2 * l + 1
    return ml_coeffs_full


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("What device am I using?", device)

    torch.set_default_dtype(torch.float32)

    config = TestConfig.from_cmd_args()
    with open(config.run_dir / "run_data.json") as f:
        run_data = json.loads(f.read())

    out_dir = config.run_dir / f"predictions_{datetime.now().strftime('%y%m%d-%H%M%S')}"
    density_spacing = 0.1
    grid_buffer = 3.0
    Rs = run_data["Rs"]

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
    state = torch.load(weights_path, map_location=device)
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
    )

    print(f"Saving predictions to {out_dir}")
    out_dir.mkdir(exist_ok=True)

    with torch.no_grad():

        for step, data in enumerate(test_loader):

            print(f"Prediction {step + 1}/{len(test_loader)}")

            mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
            y_ml = model(data.to(device)) * mask.to(device)

            atom_pos = data["pos_orig"].cpu().numpy()
            atom_types = data["z"][:, 0].int().cpu().numpy()
            x, y, z, origin = generate_grid(atom_pos, spacing=density_spacing, buffer=grid_buffer)
            target_density, ml_density = gau2grid_density_kdtree(
                x.flatten(), y.flatten(), z.flatten(), data, y_ml, Rs, ldepb=False
            )

            target_density = target_density.reshape(x.shape)
            ml_density = ml_density.reshape(x.shape)
            density_diff = target_density - ml_density
            density_diff_rel = density_diff / target_density
            density_diff_rel[target_density < 1e-4] = 0

            ml_coeffs = complete_coefficients(data, y_ml, Rs)
            for coeffs, file_name in [(ml_coeffs, "prediction"), (data.full_c.cpu().numpy(), "target")]:
                sample = {
                    "atom_pos": atom_pos,
                    "atom_types": atom_types,
                    "coeffs": coeffs,
                    "norm": data.norm.cpu().numpy(),
                    "exp": data.exp.cpu().numpy(),
                    "basis_ls": Rs,
                }
                with open(out_dir / f"{step}_{file_name}_coeffs.pickle", "wb") as f:
                    pickle.dump(sample, f)

            atom_pos -= origin
            save_to_xsf(out_dir / f"{step}_target.xsf", atom_pos, atom_types, target_density, density_spacing)
            save_to_xsf(out_dir / f"{step}_prediction.xsf", atom_pos, atom_types, ml_density, density_spacing)
            save_to_xsf(out_dir / f"{step}_diff.xsf", atom_pos, atom_types, density_diff, density_spacing)
            save_to_xsf(out_dir / f"{step}_relative_diff.xsf", atom_pos, atom_types, density_diff_rel, density_spacing)

            print(f"Epsilon: {100 * np.abs(density_diff).sum() / target_density.sum():.3f}%")


if __name__ == "__main__":
    main()
