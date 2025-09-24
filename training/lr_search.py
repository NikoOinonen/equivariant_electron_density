import os
import sys
from pathlib import Path

import torch
from torch.optim import Adam, lr_scheduler
from train_density import get_dataloader

from config import TrainConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import MaceNetwork

if __name__ == "__main__":

    config = TrainConfig.from_cmd_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("What device am I using?", device)

    torch.set_default_dtype(torch.float32)

    Rs = [(19, 0), (5, 1), (5, 2), (3, 3), (1, 4)]

    # The MACE layers only work with alternating parities for the irreps: https://github.com/ACEsuit/mace/discussions/42
    parity = ["e", "o"]
    irreps_hidden = "+".join(
        [f"{channels}x{l}{parity[l % 2]}" for l, channels in enumerate(config.irreps_hidden.split("-"))]
    )
    irreps_out = "+".join([f"{channels}x{l}{parity[l % 2]}" for channels, l in Rs])

    # https://github.com/ACEsuit/mace/issues/63
    assert (
        len(set(c for c in config.irreps_hidden.split("-"))) == 1
    ), "Hidden irreps must have the same number of channels for all l-orders."

    max_l_edges = len(irreps_hidden) - 1
    input_shape = Rs[0][0]

    loss_log_path = Path(f"loss_log_lr_avg{config.batch_average}_{config.irreps_hidden}x{config.num_layers}.csv")
    with open(loss_log_path, "w") as f:
        f.write("i_batch,lr,loss\n")

    model_kwargs = {
        "irreps_in": f"{input_shape}x0e",
        "irreps_hidden": irreps_hidden,
        "irreps_out": irreps_out,  # = Rs
        "node_attr_dim": None,
        "max_l_edges": len(config.irreps_hidden.split("-")) - 1,
        "message_correlation_order": config.correlation_order,
        "layers": config.num_layers,
        "max_radius": 3.5,
        "number_of_basis": 10,
        "num_neighbors": 12.2298,
        "num_nodes": 26,
        "reduce_output": False,
    }

    print(model_kwargs)

    free_atom_data_path = Path(__file__).parent.parent / "data" / "free_atom_s_only"
    free_atom_densities = {
        1: free_atom_data_path / "H_def2-universal-jfit-decon_density.out",
        6: free_atom_data_path / "C_def2-universal-jfit-decon_density.out",
        7: free_atom_data_path / "N_def2-universal-jfit-decon_density.out",
        8: free_atom_data_path / "O_def2-universal-jfit-decon_density.out",
        9: free_atom_data_path / "F_def2-universal-jfit-decon_density.out",
        14: free_atom_data_path / "Si_def2-universal-jfit-decon_density.out",
        15: free_atom_data_path / "P_def2-universal-jfit-decon_density.out",
        16: free_atom_data_path / "S_def2-universal-jfit-decon_density.out",
        17: free_atom_data_path / "Cl_def2-universal-jfit-decon_density.out",
        35: free_atom_data_path / "Br_def2-universal-jfit-decon_density.out",
    }

    print("Loading train set")
    train_loader = get_dataloader(
        data_path=config.dataset,
        free_atom_densities=free_atom_densities,
        Rs=Rs,
        exclude_elements=config.exclude_elements,
        include_elements=config.include_elements,
        num_samples=config.train_samples,
        world_size=1,
        global_rank=0,
    )

    model = MaceNetwork(**model_kwargs)
    model.to(device)

    optim = Adam(model.parameters(), lr=1e-6)
    optim.zero_grad()

    scheduler = lr_scheduler.LambdaLR(optim, lambda nb: 1.05**nb)

    model.train()
    losses = []
    i_batch = 0
    for data in train_loader:

        mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
        output = model(data.to(device))
        y_ml = output * mask.to(device)
        err = y_ml - data.y.to(device)
        loss = err.pow(2).mean() / config.batch_average

        loss.backward()
        losses.append(loss.item())

        if len(losses) == config.batch_average:

            optim.step()
            optim.zero_grad()
            scheduler.step()

            loss = torch.tensor(losses).mean()
            losses = []

            lr = scheduler.get_last_lr()[0]
            print(f"Batch {i_batch}, learning rate: {lr}, loss: {loss}")

            # Save loss to file
            with open(loss_log_path, "a") as f:
                f.write(f"{i_batch},{lr},{loss}\n")

            if i_batch == 0:
                loss_init = loss
            elif loss > loss_init * 4:
                break

            i_batch += 1
