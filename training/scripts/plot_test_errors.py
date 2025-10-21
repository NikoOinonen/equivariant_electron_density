#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    
    run_dir = Path("runs_ft/250929-125049_base-250923-165703_restart-all_bs8_ns528")

    results_paths = run_dir.glob("test*.results")

    eps_inc = []
    eps_exc = []
    for path in results_paths:
        
        with open(path) as f:
            inc = False
            for line in f:
                if "Epsilon:" in line:
                    eps = float(line.split()[1])
                if "Include elements:" in line:
                    e = line.split()[2]
                    if e != "None":
                        inc = True
                if "Weights epoch:" in line:
                    epoch = int(line.split()[2])
            if inc:
                eps_inc.append((epoch, eps))
            else:
                eps_exc.append((epoch, eps))
    
    eps_inc = np.array(eps_inc)
    eps_exc = np.array(eps_exc)

    ind = np.argsort(eps_inc[:, 0])
    eps_inc = eps_inc[ind]
    ind = np.argsort(eps_exc[:, 0])
    eps_exc = eps_exc[ind]

    print(eps_inc[:, 1])

    plt.figure(figsize=(10, 8))
    plt.plot(eps_inc[:, 0], eps_inc[:, 1], label="Fine-tune data (with P)")
    plt.plot(eps_exc[:, 0], eps_exc[:, 1], label="Original data (no P)")
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon error (%)")
    plt.legend()
    plt.savefig("ft_test_error.png")
    plt.show()