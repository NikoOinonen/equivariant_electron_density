from pathlib import Path

import torch
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
from torch.optim import Adam, lr_scheduler
from train_density import get_args, get_dataloader

if __name__ == "__main__":

    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("What device am I using?", device)

    torch.set_default_dtype(torch.float32)

    Rs = [(19, 0), (5, 1), (5, 2), (3, 3), (1, 4)]

    train_data_path = Path(args.dataset)
    batch_average = args.batch_average
    irreps_hidden = [int(v) for v in args.irreps_hidden.split("-")]
    num_layers = args.num_layers
    free_density_input = args.free_density_input
    input_shape = Rs[0][0] if free_density_input else 10

    loss_log_path = Path(f"loss_log_lr_avg{batch_average}_{args.irreps_hidden}x{num_layers}.csv")
    with open(loss_log_path, "w") as f:
        f.write("i_batch,lr,loss\n")

    model_kwargs = {
        "irreps_in": f"{input_shape}x0e",
        "irreps_hidden": [(mul, (l, p)) for l, mul in enumerate(irreps_hidden) for p in [-1, 1]],  # irreps_hidden
        "irreps_out": "19x0e + 5x1o + 5x2e + 3x3o + 1x4e",  # irreps_out (= Rs)
        "irreps_node_attr": None,  # irreps_node_attr
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(3),  # irreps_edge_attr
        "layers": num_layers,
        "max_radius": 3.5,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_neighbors": 12.2298,
        "num_nodes": 24,
        "reduce_output": False,
    }

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
        train_data_path,
        free_atom_densities,
        free_density_input,
        Rs,
        split=None,
        world_size=1,
        global_rank=0,
    )

    model = Network(**model_kwargs)
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
        loss = err.pow(2).mean() / batch_average

        loss.backward()
        losses.append(loss.item())

        if len(losses) == batch_average:

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
