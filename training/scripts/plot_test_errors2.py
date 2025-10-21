#!/usr/bin/env python3

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
        "runs_ft/251008-130430_base-250923-165703_restart-all_db-P_mix_train_bs16_ns4224_lr8.0e-04-8000-3.5e+05"
    )

    tb_events_path = next(run_dir.glob("events.out.tfevents.*"))
    tb_data = parse_tensorboard(tb_events_path)
    epochs_train = np.array([l.value for l in tb_data["Other/Epoch"]])
    eps_train = np.array([l.value for l in tb_data["Other/Test_Epsilon"]])

    eps_test = []
    epochs_test = []
    for path in run_dir.glob("test*.results"):
        print(path)
        with open(path) as f:
            for line in f:
                if "Epsilon:" in line:
                    eps = float(line.split()[1])
                    eps_test.append(eps)
                    break
                if "Weights epoch:" in line:
                    epoch = int(line.split()[2])
                    epochs_test.append(epoch)

    eps_test = np.array(eps_test)
    epochs_test = np.array(epochs_test)

    ind = np.argsort(epochs_test)
    eps_test = eps_test[ind]
    epochs_test = epochs_test[ind]

    plt.plot(epochs_train, eps_train, label="Fine-tune data (with P)")
    plt.plot(epochs_test, eps_test, label="Original data (no P)")
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon error (%)")
    plt.legend()
    plt.savefig(f"ft_test_error_{run_dir.stem.split('_')[0]}.png", dpi=200)
    plt.show()
