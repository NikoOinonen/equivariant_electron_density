#!/usr/bin/env python3

import json
from pathlib import Path
import shutil
import sys
import psi4
import numpy as np


def get_norms_exponents(atom_types: list[int], ls: list[int], fit_basis: str) -> tuple[dict, dict]:

    psi4.core.set_global_option("df_basis_scf", fit_basis)

    mol_str = ""
    z = 0
    for z_atom in atom_types:
        mol_str += f"{z_atom} 0 0 {z}\n"
        z += 2

    mol = psi4.geometry(mol_str)
    aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JFIT", fit_basis)

    numfuncatom = np.zeros(mol.natom())
    shells = []
    for func in range(0, aux_basis.nbf()):
        current = aux_basis.function_to_center(func)
        shell = aux_basis.function_to_shell(func)
        numfuncatom[current] += 1
        shells.append(shell)

    shellmap = []
    for shell in range(0, aux_basis.nshell()):
        count = shells.count(shell)
        shellmap.append((count - 1) // 2)

    counter = 0
    ls_total = sum([mul * (2 * l + 1) for l, mul in enumerate(ls)])
    norms = {}
    exponents = {}
    for i_atom in range(0, mol.natom()):
        z = atom_types[i_atom]
        l_prev = 0
        i_func = 0
        norms[z] = np.zeros(ls_total)
        exponents[z] = np.zeros(ls_total)
        for j in range(counter, counter + int(numfuncatom[i_atom])):
            shell_num = aux_basis.function_to_shell(j)
            shell = aux_basis.shell(shell_num)
            l_current = shellmap[shell_num]
            if l_current > l_prev:
                i_func = sum([mul * (2 * l + 1) for l, mul in enumerate(ls[:l_current])])
                l_prev = l_current
            norms[z][i_func] = shell.coef(0)
            exponents[z][i_func] = shell.exp(0)
            counter += 1
            i_func += 1
        norms[z] = list(norms[z])
        exponents[z] = list(exponents[z])

    return norms, exponents


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Give path to model directory as argument")
        sys.exit(1)

    model_directory = Path(sys.argv[1])
    out_dir = Path("packed_models") / model_directory.name

    fit_basis = "def2-universal-jfit-decon"

    weights_paths = list(model_directory.glob("model_weights_epoch_*.pt"))
    if not weights_paths:
        print(f"No weights found in {model_directory}.")
        sys.exit(1)
    weights_paths = sorted(weights_paths, key=lambda p: int(p.name.split("_")[-1].split(".")[0]))
    weights_path = weights_paths[-1]

    with open(model_directory / "run_data.json") as f:
        run_data = json.load(f)

    with open(model_directory / "free_atom_coeffs.json") as f:
        free_atom_coeffs = json.load(f)

    ls = [mul for mul, _ in run_data["Rs"]]
    norms, exponents = get_norms_exponents([int(z) for z in free_atom_coeffs], ls=ls, fit_basis=fit_basis)

    model_info = {
        "model_kwargs": run_data["model_kwargs"],
        "basis_info": {
            "basis_name": fit_basis,
            "l_multiplicities": ls,
            "free_atom_coeffs": free_atom_coeffs,
            "norms": norms,
            "exponents": exponents,
        },
    }

    out_dir.mkdir(exist_ok=True, parents=True)

    shutil.copy(weights_path, out_dir / "model_weights.pt")
    with open(out_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=4)
