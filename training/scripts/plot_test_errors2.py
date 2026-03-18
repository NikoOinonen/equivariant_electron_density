#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard(path: Path):
    ea = event_accumulator.EventAccumulator(
        str(path),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    return {s: ea.Scalars(s) for s in ea.Tags()["scalars"]}


if __name__ == "__main__":

    run_dir = Path(
        "../runs/251029-135929_db-ccsd-cid_25_train_bs16_ns33616_lr-wd-1.5e-03-8000-3.5e+05_irreps128-128-128-128x6_corr3"
    )
    print(run_dir.exists())

    tb_events_path = next(run_dir.glob("events.out.tfevents.*"))
    tb_data = parse_tensorboard(tb_events_path)
    epochs_train = np.array([l.value for l in tb_data["Other/Epoch"]])
    eps_train = np.array([l.value for l in tb_data["Other/Test_Epsilon"]])

    with open(run_dir / "run_data.json") as f:
        run_data = json.load(f)
    trainset = Path(run_data["dataset"]).stem

    eps = {}
    for path in run_dir.glob("test*.results"):
        print(path)
        with open(path) as f:
            for line in f:
                if "Epsilon mean:" in line:
                    eps_mean = float(line.split()[2])
                if "Test set:" in line:
                    testset = Path(line.split()[2]).stem
                if "Weights epoch:" in line:
                    epoch = int(line.split()[2])
            if testset not in eps:
                eps[testset] = []
            eps[testset].append((epoch, eps_mean))

    for testset in eps:
        epoch_eps = np.array(eps[testset])
        inds = np.argsort(epoch_eps[:, 0])
        eps[testset] = epoch_eps[inds]
        print(testset, eps[testset][:, 1])

    plt.semilogy(epochs_train, eps_train, label=f"{trainset} (Original)")
    for testset in sorted(eps.keys()):
        epoch_eps = eps[testset]
        plt.semilogy(epoch_eps[:, 0], epoch_eps[:, 1], label=testset)

    plt.xlabel("Epoch")
    plt.ylabel("Epsilon error (%)")
    plt.legend()
    plt.savefig(f"plots/test_error_{run_dir.stem.split('_')[0]}.png", dpi=200)
    plt.show()
