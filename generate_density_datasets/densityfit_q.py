from pathlib import Path
import sys
import numpy as np
import psi4


def load_molecule(xyz_path: Path):
    with open(xyz_path) as f:
        molstr = " ".join(f.readlines())
    molstr = molstr + "\n symmetry c1 \n no_reorient \n no_com \n"
    mol = psi4.geometry(molstr)
    return mol


def fit_density(mol_dir: Path, out_path: Path, fit_basis: str):

    mol = load_molecule(mol_dir / "GEOM-B3LYP.xyz")
    density_matrix = 2 * np.load(mol_dir / "D-CCSD.npy") # x2 because the saved density matrix only has alpha electrons

    orbital_basis = psi4.core.BasisSet.build(mol)
    aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JFIT", fit_basis)
    # aux_basis.print_detail_out()

    numfuncatom = np.zeros(mol.natom())
    funcmap = []
    shells = []

    # note: atoms are 0 indexed
    for func in range(0, aux_basis.nbf()):
        current = aux_basis.function_to_center(func)
        shell = aux_basis.function_to_shell(func)
        shells.append(shell)

        funcmap.append(current)
        numfuncatom[current] += 1

    shellmap = []
    for shell in range(0, aux_basis.nshell()):
        count = shells.count(shell)
        shellmap.append((count - 1) // 2)

    # print(numfuncatom)

    zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(orbital_basis)

    orbital_overlap = mints.ao_overlap()
    # total_electrons = (density_matrix @ orbital_overlap).trace()
    total_electrons = (density_matrix * orbital_overlap).sum()
    total_nuclear_charge = sum([mol.Z(i) for i in range(mol.natom())])
    assert np.allclose(total_electrons, total_nuclear_charge), (total_electrons, total_nuclear_charge)

    #
    # Check normalization of the aux basis
    #
    # Saux = np.array(mints.ao_overlap(aux_basis, aux_basis))
    # print("Saux", Saux)

    #
    # Form 3 center integrals (P|mn)
    #
    Jabj = np.squeeze(mints.ao_eri(aux_basis, zero_basis, orbital_basis, orbital_basis))

    #
    # Form metric (P|Q) and invert, filtering out small eigenvalues for stability
    #
    Mij = np.squeeze(mints.ao_eri(aux_basis, zero_basis, aux_basis, zero_basis))
    evals, evecs = np.linalg.eigh(Mij)
    evals = np.where(evals < 1e-10, 0.0, 1.0 / evals)
    Mij_inv = np.einsum("ik,k,jk->ij", evecs, evals, evecs)

    ## THIS IS SLOW
    #
    # Recompute the integrals, as a simple sanity check (mn|rs) = (mn|P) PQinv[P,Q] (Q|rs)
    # where PQinv[P,Q] is the P,Qth element of the invert of the matrix (P|Q) (a Coulomb integral)
    #
    # approx = np.einsum("Pmn,PQ,Qrs->mnrs", Jabj.astype(np.float32), Mij_inv.astype(np.float32), Jabj.astype(np.float32), optimize=True)
    # exact = mints.ao_eri()
    # print("checking how good the fit is")
    # print(approx - exact)

    #
    # Finally, compute and print the fit coefficients.  From the density matrix, D, the
    # coefficients of the vector of basis aux basis funcions |P) is given by
    #
    # D_P = Sum_mnQ D_mn (mn|Q) PQinv[P,Q]
    #

    # compute q from equations 15-17 in Dunlap paper
    # "Variational fitting methods for electronic structure calculations"
    q = []
    counter = 0
    for i in range(0, mol.natom()):
        for j in range(counter, counter + int(numfuncatom[i])):
            # print(D_P[j])
            shell_num = aux_basis.function_to_shell(j)
            shell = aux_basis.shell(shell_num)
            # assumes that each shell only has 1 primitive. true for a2 basis
            normalization = shell.coef(0)
            exponent = shell.exp(0)
            if shellmap[shell_num] == 0:
                integral = (1 / (4 * exponent)) * np.sqrt(np.pi / exponent)
                q.append(4 * np.pi * normalization * integral)
            else:
                q.append(0.0)
            counter += 1

    q = np.array(q)

    # these are the old coefficients
    D_P = np.einsum("mn,Pmn,PQ->Q", density_matrix, Jabj, Mij_inv, optimize=True)

    # compute lambda
    numer = total_nuclear_charge - np.dot(q, D_P)
    denom = np.dot(np.dot(q, Mij_inv), q)
    lambchop = numer / denom

    new_D_P = D_P + np.dot(Mij_inv, lambchop * q)

    with open(out_path, "w+") as f:
        counter = 0
        totalq = 0.0
        newtotalq = 0.0
        for i in range(0, mol.natom()):
            f.write("Atom number: %i \n" % i)
            f.write("number of functions: %i \n" % int(numfuncatom[i]))
            for j in range(counter, counter + int(numfuncatom[i])):
                shell_num = aux_basis.function_to_shell(j)
                shell = aux_basis.shell(shell_num)
                # assumes that each shell only has 1 primitive. true for a2 basis
                normalization = shell.coef(0)
                exponent = shell.exp(0)
                integral = (1 / (4 * exponent)) * np.sqrt(np.pi / exponent)

                if shellmap[shell_num] == 0:
                    totalq += D_P[j] * 4 * np.pi * normalization * integral
                    newtotalq += new_D_P[j] * 4 * np.pi * normalization * integral

                f.write(f"{shellmap[shell_num]} {np.array2string(new_D_P[j])} {exponent} {normalization}\n")
                counter += 1

    print("Total nuclear charge:", total_nuclear_charge)
    print("Total electrons (fitted basis):", totalq)
    print("Total electrons (fitted basis, corrected):", newtotalq)
    assert np.allclose(total_nuclear_charge, newtotalq)


if __name__ == "__main__":

    # db_dir = Path("/scratch/work/oinonen1/density_db")
    db_dir = Path("/mnt/triton/density_db")
    input_dir = db_dir / "CCSD-CID"
    out_dir = db_dir / "fitted_densities"

    original_basis = "cc-pvdz"
    fit_basis = "def2-universal-jfit-decon"

    if len(sys.argv) < 3:
        print("Not enough arguments")
        sys.exit(1)

    # Division over multiple processes
    n_proc = int(sys.argv[1])
    i_proc = int(sys.argv[2])

    print(f"Performing density fit from {original_basis} to {fit_basis} basis set...")
    psi4.core.set_global_option("basis", original_basis)
    psi4.core.set_global_option("df_basis_scf", fit_basis)
    psi4.core.set_output_file(f'output_{i_proc}.dat', False)

    out_dir.mkdir(exist_ok=True)

    mol_dirs = list(input_dir.glob("molecule_*_0"))
    mol_dirs = sorted(mol_dirs)
    mol_dirs = mol_dirs[i_proc::n_proc]

    n_mols = len(mol_dirs)
    print("Total number of molecules:", n_mols)

    # cid = 123083
    # mol_dir = input_dir / f"molecule_{cid}_0"
    # out_path = Path("test") / "test.dat"
    # fit_density(mol_dir, out_path, fit_basis)

    for i, mol_dir in enumerate(mol_dirs):

        cid = mol_dir.name.split('_')[1]
        out_path = out_dir / f"{cid}.dat"
        if out_path.exists():
            print(f"{cid} already done.")
            continue

        print(f"Molecule {i+1} / {n_mols}, CID: {cid}")

        try:
            fit_density(mol_dir, out_path, fit_basis)
        except Exception as e:
            print(f"Ran into an error:\n{e}")
