#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard(path: Path):
    ea = event_accumulator.EventAccumulator(
        str(path),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    return {s: ea.Scalars(s) for s in ea.Tags()["scalars"]}


if __name__ == "__main__":

    run_dir = Path("runs/251001-060742_bs16_ns74976_lr1.5e-03-8000-3.5e+05_irreps128-128-128-128x6_corr3")

    tb_events_path = next(run_dir.glob("events.out.tfevents.*"))
    tb_data = parse_tensorboard(tb_events_path)
    eps_train = np.array([l.value for l in tb_data["Other/Test_Epsilon"]])

    eps_test = []
    sigma_test = []
    for path in run_dir.glob("test*.results"):
        print(path)
        with open(path) as f:
            for line in f:
                if "Epsilon:" in line:
                    eps = float(line.split()[1])
                    eps_test.append(eps)
                    break
                if "Test set:" in line:
                    testset_path = Path(line.split()[2])
                    sigma_test.append(float(testset_path.stem.split("_")[-1]))

    eps_test = [eps_train[-1]] + eps_test
    sigma = [0.0] + sigma_test

    plt.bar(sigma, eps_test, width=0.03)
    plt.xticks(sigma)
    plt.xlabel("sigma (Å)")
    plt.ylabel("Epsilon error (%)")
    plt.title("Error on perturbed molecules")
    plt.savefig(f"test_error_perturb_{run_dir.stem.split('_')[0]}.png", dpi=200)
    plt.show()
