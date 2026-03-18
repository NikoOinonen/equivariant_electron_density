#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    run_dir = Path(
        "../runs_ft_perturbed/251022-140615_base-251001-060742_ELoRA-r24_db-perturbed_0.05-0.10_mix_train_bs8_ns1072_lr5.0e-04-8000-3.5e+05"
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
                        sigma = float(testset_path.stem.split("_")[1])
                    except ValueError:
                        sigma = 0.0
                if "Weights epoch:" in line:
                    epoch = int(line.split()[2])
            if sigma not in eps:
                eps[sigma] = []
            eps[sigma].append((epoch, eps_mean, eps_std, n_sample))

    for sigma in eps:
        epoch_eps = np.array(eps[sigma])
        inds = np.argsort(epoch_eps[:, 0])
        eps[sigma] = epoch_eps[inds]

    for sigma in sorted(eps.keys()):
        epoch_eps = eps[sigma]
        std = epoch_eps[:, 2] / np.sqrt(epoch_eps[:, 3])
        plt.plot(epoch_eps[:, 0], epoch_eps[:, 1], label=f"sigma = {sigma:.2f}")
        plt.fill_between(epoch_eps[:, 0], epoch_eps[:, 1] - 2 * std, epoch_eps[:, 1] + 2 * std, alpha=0.2)

    ind1 = run_dir.stem.find("db-")
    ind2 = run_dir.stem.find("bs")
    train_set = run_dir.stem[ind1 + 3 : ind2 - 1]

    plt.xlabel("Epoch")
    plt.ylabel("Epsilon error (%)")
    plt.title(f"Train set: {train_set}")
    plt.legend()
    plt.savefig(outdir / f"test_error_perturb_{run_dir.stem.split('_')[0]}.png", dpi=200)
    plt.show()

    # plt.bar(sigmas, eps, width=0.03)
    # plt.xticks()
    # plt.xlabel("sigma (Å)")
    # plt.ylabel("Epsilon error (%)")
    # plt.title("Error on perturbed molecules")
    # plt.savefig(f"test_error_perturb_{run_dir.stem.split('_')[0]}.png", dpi=200)
    # plt.show()
