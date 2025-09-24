#!/usr/bin/env python3


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard(path):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    return {s: ea.Scalars(s) for s in ea.Tags()["scalars"]}


if __name__ == "__main__":

    base_dir = Path("runs")

    loss_train = []
    loss_val = []
    epsilon_val = []
    n_train = []

    run_times = [
        "Nov14_05-40-57",
        "Sep09_15-14-12",
        "Sep11_14-58-56",
        "Sep16_15-35-19",
        "Sep18_06-47-12",
        "Sep22_15-49-40",
    ]

    for run_time in run_times:

        run_dir = next(base_dir.glob(f"{run_time}*"))

        split_ind = run_dir.name.find("split")
        if split_ind > 0:
            n_train.append(int(run_dir.name[split_ind:].split("_")[0][5:]))
        else:
            n_train.append(74968)

        events_path = next(run_dir.glob("events.*triton*"))
        scalars = parse_tensorboard(str(events_path))

        loss_train.append(np.array([l.value for l in scalars["Loss/Train"]])[-100:].mean())
        loss_val.append(np.array([l.value for l in scalars["Loss/Test"]])[-1])
        epsilon_val.append(np.array([l.value for l in scalars["Other/Test_Epsilon"]])[-3:].mean())

    n_train = np.array(n_train)
    inds = np.argsort(n_train)
    n_train = n_train[inds]

    loss_train = np.array(loss_train)[inds]
    loss_val = np.array(loss_val)[inds]
    epsilon_val = np.array(epsilon_val)[inds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.loglog(n_train, loss_train, marker="x", label="Loss (train)")
    ax1.loglog(n_train, loss_val, marker="x", label="Loss (val)")
    ax2.semilogx(n_train, epsilon_val, marker="x", label="Epsilon")
    ax1.set_xlabel("N_train")
    ax1.set_ylabel("Loss")
    ax2.set_xlabel("N_train")
    ax2.set_ylabel("Error(%)")
    ax1.legend()
    ax2.legend()

    fig.savefig("learning_curve.png")
    plt.close()
