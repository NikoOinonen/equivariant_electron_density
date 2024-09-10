# pylint: disable=invalid-name, no-member, arguments-differ, missing-docstring, line-too-long

import os
import pickle
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
import periodictable as pt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import flatten_list


def get_densities(dens_file, elements):
    """
    inputs:
    - density file
    - number of atoms

    returns:
    - shape [N, X] list of basis function coefficients, where X is the number of basis functions per atom
    """
    ## get density coefficients for each atom
    ## ordered in ascending l order
    ## also get Rs_out for each atoms

    basis_coeffs = []
    basis_exponents = []
    basis_norms = []
    num_basis_func = []
    Rs_outs = []
    for l in range(0, 20):
        flag = 0
        atom_index = -1
        counter = 0
        multiplicity = 0
        with open(dens_file, "r") as density_file:
            for line in density_file:
                if flag == 1:
                    split = line.split()
                    if int(split[0]) == l:
                        basis_coeffs[atom_index].append(float(split[1]))
                        basis_exponents[atom_index].append(float(split[2]))
                        basis_norms[atom_index].append(float(split[3]))
                        multiplicity += 1
                    counter += 1
                    if counter == num_lines:
                        flag = 0
                        if multiplicity != 0:
                            Rs_outs[atom_index].append((multiplicity // (2 * l + 1), l))
                if "functions" in line:
                    num_lines = int(line.split()[3])
                    num_basis_func.append(num_lines)
                    counter = 0
                    multiplicity = 0
                    flag = 1
                    atom_index += 1
                    if l == 0:
                        basis_coeffs.append([])
                        basis_exponents.append([])
                        basis_norms.append([])
                        Rs_outs.append([])

    # break coefficients list up into l-based vectors
    newbasis_coeffs = []
    newbasis_exponents = []
    newbasis_norms = []
    atom_index = -1
    for atom in Rs_outs:
        atom_index += 1
        counter = 0
        newbasis_coeffs.append([])
        newbasis_exponents.append([])
        newbasis_norms.append([])
        for Rs in atom:
            number = Rs[0]
            l = Rs[1]
            for i in range(0, number):
                newbasis_coeffs[atom_index].append(basis_coeffs[atom_index][counter : counter + (2 * l + 1)])
                newbasis_exponents[atom_index].append(basis_exponents[atom_index][counter : counter + (2 * l + 1)])
                newbasis_norms[atom_index].append(basis_norms[atom_index][counter : counter + (2 * l + 1)])
                counter += 2 * l + 1

    Rs_out_list = []
    elementdict = {}
    for i, elem in enumerate(elements):
        if elem not in elementdict:
            elementdict[elem] = Rs_outs[i]
            Rs_out_list.append(Rs_outs[i])

    psi4_2_e3nn = [
        [0],  # s
        [2, 0, 1],  # p
        [4, 2, 0, 1, 3],  # d
        [6, 4, 2, 0, 1, 3, 5],  # f
        [8, 6, 4, 2, 0, 1, 3, 5, 7],  # g
        [10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9],  # h
        [12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11],  # i
    ]

    # change convention from psi4 to e3nn
    for i, atom in enumerate(newbasis_coeffs):
        for j, item in enumerate(atom):
            l = (len(item) - 1) // 2
            if l > 6:
                raise ValueError("L is too high. Currently only supports L<7")
            newbasis_coeffs[i][j] = [item[k] for k in psi4_2_e3nn[l]]

    return newbasis_coeffs, newbasis_exponents, newbasis_norms, Rs_outs


def get_coordinates(inputfile):
    """
    reads in coordinates and atomic number from psi4 input file

    returns:
    -shape [N, 3] numpy array of points
    -shape [N] numpy array of masses
    -shape [N] list of element symbols
    """

    points = np.loadtxt(inputfile, usecols=range(1, 4))
    numatoms = len(points)
    atomic_numbers = np.genfromtxt(inputfile, usecols=0, dtype="str")
    atomic_numbers = [int(i) for i in atomic_numbers]
    elements = [pt.elements[i].symbol for i in atomic_numbers]
    unique_elements = len(np.unique(atomic_numbers))
    onehot = np.zeros((numatoms, unique_elements))

    # get one hot vector
    weighted_onehot = onehot
    typedict = {}
    counter = -1
    for i, num in enumerate(atomic_numbers):
        if num not in typedict:
            # dictionary: key = atomic number
            # value = 0,1,2,3 (ascending types)
            counter += 1
            typedict[num] = counter
        weighted_onehot[i, typedict[num]] = num

    # print(weighted_onehot)

    return points, numatoms, atomic_numbers, elements, weighted_onehot


def get_dataset(data_dir, xyzs_dir):

    data_dir = Path(data_dir)
    density_file_paths = data_dir.glob("*.dat")
    density_file_paths = sorted(density_file_paths)

    # coeff_by_type_list = []
    dataset = []
    for i, density_file in enumerate(density_file_paths):

        print(f"({i + 1}/{len(density_file_paths)}) {density_file.name}")

        cid = int(density_file.name.split(".dat")[0])
        xyz_path = xyzs_dir / f"molecule_{cid}_0" / "GEOM-B3LYP.xyz"

        # read in xyz file
        # get number of atoms
        # get onehot encoding
        points, _, atomic_numbers, elements, weighted_onehot = get_coordinates(xyz_path)

        # construct one hot encoding
        onehot = weighted_onehot
        # replace all nonzero values with 1
        onehot[onehot > 0.001] = 1

        # read in density file
        coefficients, exponents, norms, Rs_out_list = get_densities(density_file, elements)

        # compute Rs_out_max
        # this is necessary because O and H have different Rs_out
        # to deal with this, I set the global Rs_out to be the maximum
        # basically, for each L,
        # i take whichever entry that has the higher multiplicity

        # print(Rs_out_list)

        a = list(zip_longest(*Rs_out_list))
        # remove Nones
        b = [[v if v is not None else (0, 0) for v in nested] for nested in a]

        Rs_out_max = []
        for rss in b:
            Rs_out_max.append(max(rss))

        # HACKY: MANUAL OVERRIDE OF RS_OUT_MAX
        # set manual Rs_out_max (comment out if desired)
        Rs_out_max = [(19, 0), (5, 1), (5, 2), (3, 3), (1, 4)]
        print("Using manual Rs_out_max:", Rs_out_max)

        ## now construct coefficient, exponent and norm arrays
        ## from Rs_out_max
        ## pad with zeros

        coeff_dim = 0
        for mul, l in Rs_out_max:
            coeff_dim += mul * ((2 * l) + 1)

        rect_coeffs = torch.zeros(len(Rs_out_list), coeff_dim)
        rect_expos = torch.zeros(len(Rs_out_list), coeff_dim)
        rect_norms = torch.zeros(len(Rs_out_list), coeff_dim)

        for i, (atom, coeff_list, expo_list, norm_list) in enumerate(zip(Rs_out_list, coefficients, exponents, norms)):
            counter = 0
            list_counter = 0
            for (mul, l), (max_mul, max_l) in zip(atom, Rs_out_max):
                n = mul * ((2 * l) + 1)
                rect_coeffs[i, counter : counter + n] = torch.Tensor(list(flatten_list(coeff_list[list_counter : list_counter + mul])))
                rect_expos[i, counter : counter + n] = torch.Tensor(list(flatten_list(expo_list[list_counter : list_counter + mul])))
                rect_norms[i, counter : counter + n] = torch.Tensor(list(flatten_list(norm_list[list_counter : list_counter + mul])))
                list_counter += mul
                max_n = max_mul * ((2 * max_l) + 1)
                counter += max_n

        cluster_dict = {
            "type": torch.Tensor(atomic_numbers),
            "pos": torch.Tensor(points),
            "onehot": torch.Tensor(onehot),
            "coefficients": rect_coeffs,
            "exponents": rect_expos,
            "norms": rect_norms,
            "rs_max": Rs_out_max,
            "cid": cid,
        }
        dataset.append(cluster_dict)

    # reset onehot based on whole dataset
    # need list of unique atomic numbers, in ascending order
    # then iterate through atoms
    # onehot[i, index_of_matching_atom_in_unique_elements] = 1
    # look up how to get index- np.where

    all_anum = []
    for item in dataset:
        anum = item["type"]
        all_anum.extend(anum)

    unique_elements = np.unique(all_anum)
    num_unique_elements = len(unique_elements)

    for item in dataset:
        numatoms = item["pos"].shape[0]
        onehot = np.zeros((numatoms, num_unique_elements))
        for i, num in enumerate(item["type"]):
            n = num.item()
            index = np.where(unique_elements == n)
            onehot[i, index] = 1
        item["onehot"] = torch.Tensor(onehot)

    # now get Rs_out_max for whole dataset
    all_rs = []
    for item in dataset:
        rs = item["rs_max"]
        all_rs.append(rs)

    # print(all_rs)

    a = list(zip_longest(*all_rs))
    # remove Nones
    b = [[v if v is not None else (0, 0) for v in nested] for nested in a]

    Rs_out_max = []
    for rss in b:
        Rs_out_max.append(max(rss))

    print("irreps_out", Rs_out_max)

    return dataset


if __name__ == "__main__":

    db_dir = Path("/scratch/work/oinonen1/density_db")
    # db_dir = Path("/mnt/triton/density_db")
    densities_dir = db_dir / "fitted_densities"
    xyzs_dir = db_dir / "CCSD-CID"
    pickle_path = Path("./dataset.pickle")

    dataset = get_dataset(densities_dir, xyzs_dir)

    with open(pickle_path, "wb") as f:
        pickle.dump(dataset, f)
