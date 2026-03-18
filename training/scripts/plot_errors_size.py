#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    run_dir = Path(
        "../runs/251029-135929_db-ccsd-cid_25_train_bs16_ns33616_lr-wd-1.5e-03-8000-3.5e+05_irreps128-128-128-128x6_corr3"
    )
    outdir = Path("plots")

    eps = {}
    for path in run_dir.glob("test*.results"):
        print(path)
        with open(path) as f:
            for line in f:
                if "Epsilon mean:" in line:
                    eps_mean = float(line.split()[2])
                if "Epsilon std:" in line:
                    eps_std = float(line.split()[2])
                if "Number of samples:" in line:
                    n_sample = int(line.split()[3])
                if "Test set:" in line:
                    testset_path = Path(line.split()[2])
                    try:
                        size_bin = testset_path.stem.split("_")[1][1:]
                    except ValueError:
                        size_bin = 0.0
                if "Weights epoch:" in line:
                    epoch = int(line.split()[2])
            if size_bin == "26-100":
                continue
            if size_bin not in eps:
                eps[size_bin] = []
            eps[size_bin].append((epoch, eps_mean, eps_std, n_sample))

    for size_bin in eps:
        epoch_eps = np.array(eps[size_bin])
        inds = np.argsort(epoch_eps[:, 0])
        eps[size_bin] = epoch_eps[inds]

    size_bins = sorted(eps.keys(), key=lambda k: int(k.split("-")[0]))
    for size_bin in size_bins:
        epoch_eps = eps[size_bin]
        std = epoch_eps[:, 2] / np.sqrt(epoch_eps[:, 3])
        plt.semilogy(epoch_eps[:, 0], epoch_eps[:, 1], label=f"n_atom = {size_bin}")
        plt.fill_between(epoch_eps[:, 0], epoch_eps[:, 1] - 2 * std, epoch_eps[:, 1] + 2 * std, alpha=0.2)

    ind1 = run_dir.stem.find("db-")
    ind2 = run_dir.stem.find("bs")
    train_set = run_dir.stem[ind1 + 3 : ind2 - 1]

    plt.xlabel("Epoch")
    plt.ylabel("Epsilon error (%)")
    plt.title(f"Train set: {train_set}")
    plt.legend()
    plt.savefig(outdir / f"test_error_size_{run_dir.stem.split('_')[0]}.png", dpi=200)
    plt.show()

    eps_last = [eps[s][-1, 1] for s in size_bins]
    eps_last_error = [eps[s][-1, 2] / np.sqrt(eps[s][-1, 3]) for s in size_bins]
    print(eps["1-5"])
    print(size_bins)
    print(eps_last)
    plt.bar(size_bins, eps_last, width=0.7)
    plt.errorbar(size_bins, eps_last, yerr=eps_last_error, fmt="o", color="r", ms=0)
    plt.xticks()
    plt.xlabel("n_atoms")
    plt.ylabel("Epsilon error (%)")
    plt.savefig(outdir / f"test_error_size_bar_{run_dir.stem.split('_')[0]}.png", dpi=200)
    plt.show()
