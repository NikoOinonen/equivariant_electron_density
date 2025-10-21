#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    log_path = Path(sys.argv[1])

    data = np.loadtxt(log_path, delimiter=",", skiprows=1)

    lr = data[:-1, 1]
    loss = data[:-1, 2]

    fig = plt.figure(figsize=(10, 7))
    plt.loglog(lr, loss)
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.title(log_path.stem)

    plt.savefig(f"lr_{log_path.stem}.png")
    plt.show()
