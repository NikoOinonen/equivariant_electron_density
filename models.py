#!/usr/bin/env python3

import math
from typing import Callable, Dict, Optional, Union

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.gate_points_2101 import scatter as scatter2
from e3nn.nn.models.gate_points_2101 import smooth_cutoff, tp_path_exists
from e3nn.util.jit import compile_mode
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_scatter import scatter

from mace_layer.blocks import (
    EquivariantProductBasisBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace_layer.e3nn_elora.nn import FullyConnectedNet, Gate
from mace_layer.e3nn_elora.o3 import FullyConnectedTensorProduct, Linear, TensorProduct


@compile_mode("script")
class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_out : `e3nn.o3.Irreps` or None
        representation of the output node features

    number_of_basis : int
        number of basis on which the edge length are projected

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        number_of_basis,
        radial_layers,
        radial_neurons,
        num_neighbors,
        r_lora=None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out, r_lora=r_lora)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in, r_lora=r_lora)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
            r_lora=r_lora,
        )
        self.fc = FullyConnectedNet(
            [number_of_basis] + radial_layers * [radial_neurons] + [tp.weight_numel],
            torch.nn.functional.silu,
            r_lora=r_lora,
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out, r_lora=r_lora)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded) -> torch.Tensor:
        weight = self.fc(edge_length_embedded)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter2(edge_features, edge_dst, dim_size=x.shape[0]).div(self.num_neighbors**0.5)

        x = self.lin2(x, node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x


class MACE_layer(torch.nn.Module):
    r"""A MACE layer from the `"MACE: Higher Order Equivariant Message Passing Neural Networks
    for Fast and Accurate Force Fields, Neurips 2022"
    <https://arxiv.org/abs/2206.07697>`_ paper
    Construct a single layer of the MACE architecture for efficient higher order equivariant message
    passing.

    Args:
        max_ell (int): Maximum angular momentum in the spherical expansion on edges, :math:`l = 0, 1, \dots`.
        Controls the resolution of the spherical expansion.
        correlation (int): The maximum correlation order of the messages, :math:`\nu = 0, 1, \dots`.
        n_dims_in (int): The number of input node attributes.
        hidden_irreps (str): The hidden irreps defining the node features to construct.
        node_feats_irreps (str): The irreps of the node features in the input.
        edge_feats_irreps (str): The irreps of the edge features in the input.
        avg_num_neighbors (float): A normalization factor for the pooling operation,
        usually taken as the average number of neighbors.
        interaction_cls (Callable, optional): The type of interaction block to use.
        Defaults to RealAgnosticResidualInteractionBlock.
        element_dependent (bool, optional): Whether to use element dependent basis functions.
        Defaults to False.
        use_sc (bool, optional): Whether to use the self connection. Defaults to True.
    """

    def __init__(
        self,
        correlation: int,
        node_attr_dim: int,
        edge_attr_irreps: int,
        hidden_irreps: str,
        node_feats_irreps: str,
        edge_feats_irreps: str,
        avg_num_neighbors: float,
        interaction_cls: Callable = RealAgnosticResidualInteractionBlock,
        use_sc: bool = True,
        r_lora: Optional[int] = None,
    ):
        super().__init__()

        node_attr_irreps = o3.Irreps([(node_attr_dim, (0, 1))])
        hidden_irreps = o3.Irreps(hidden_irreps)
        node_feats_irreps = o3.Irreps(node_feats_irreps)
        edge_feats_irreps = o3.Irreps(edge_feats_irreps)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (edge_attr_irreps * num_features).sort()[0].simplify()

        self.interaction = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attr_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            r_lora=r_lora,
        )
        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=self.interaction.target_irreps,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=node_attr_dim,
            use_sc=use_sc,
            r_lora=r_lora,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        node_feats, sc = self.interaction(
            node_feats=node_feats,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )
        node_feats = self.product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
        return node_feats


class MaceNetwork(torch.nn.Module):

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        irreps_out: o3.Irreps,
        node_attr_dim: Optional[int],
        max_l_edges: int,
        message_correlation_order: int,
        layers: int,
        max_radius: float,
        number_of_basis: int,
        num_neighbors: float,
        num_nodes: float,
        reduce_output: bool = True,
        r_lora: Optional[int] = None,
    ):

        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output
        self.r_lora = r_lora

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.node_attr_dim = node_attr_dim if node_attr_dim is not None else 1
        self.max_l_edges = max_l_edges
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(max_l_edges)
        self.message_correlation_order = message_correlation_order

        self.input_has_node_attr = node_attr_dim is not None

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.node_embed = Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_hidden, r_lora=r_lora)
        self.layers = torch.nn.ModuleList()
        node_feats_irreps = self.irreps_hidden

        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and tp_path_exists(node_feats_irreps, self.irreps_edge_attr, ir)
                ]
            )
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l > 0 and tp_path_exists(node_feats_irreps, self.irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(node_feats_irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
                r_lora=r_lora,
            )
            conv = MACE_layer(
                correlation=message_correlation_order,
                node_attr_dim=self.node_attr_dim,
                edge_attr_irreps=self.irreps_edge_attr,
                hidden_irreps=self.irreps_hidden,
                node_feats_irreps=node_feats_irreps,
                edge_feats_irreps=o3.Irreps(f"{number_of_basis}x0e"),
                avg_num_neighbors=num_neighbors,
                r_lora=r_lora,
            )
            linear = Linear(irreps_in=self.irreps_hidden, irreps_out=gate.irreps_in, r_lora=r_lora)
            node_feats_irreps = gate.irreps_out
            self.layers.append(torch.nn.ModuleList([conv, linear, gate]))

        self.readout = Convolution(
            irreps_in=node_feats_irreps,
            irreps_node_attr=o3.Irreps([(self.node_attr_dim, (0, 1))]),
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_out=self.irreps_out,
            number_of_basis=number_of_basis,
            radial_layers=1,
            radial_neurons=64,
            num_neighbors=num_neighbors,
            r_lora=r_lora,
        )

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Evaluate the network

        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        edge_index = radius_graph(data["pos"], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization="component")
        edge_length = edge_vec.norm(dim=1)
        edge_feats = soft_one_hot_linspace(
            x=edge_length, start=0.0, end=self.max_radius, number=self.number_of_basis, basis="gaussian", cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_attr and "z" in data:
            z = data["z"]
        else:
            assert self.node_attr_dim == 1
            z = data["pos"].new_ones((data["pos"].shape[0], 1))

        x = self.node_embed(data["x"])
        for conv, linear, gate in self.layers:
            x = conv(x, z, edge_attr, edge_feats, edge_index)
            x = linear(x)
            x = gate(x)

        x = self.readout(x, z, edge_src, edge_dst, edge_attr, edge_feats)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return x

    def merge_LoRA(self):
        if self.r_lora is None:
            return
        for name, module in self.named_modules():
            if hasattr(module, "merge_LoRA") and module is not self:
                module.merge_LoRA()
        self.r_lora = None


if __name__ == "__main__":

    from torch_geometric.loader import DataLoader

    model = MaceNetwork(
        irreps_in="10x0e",
        # irreps_hidden=[(mul, (l, p)) for l, mul in enumerate([125, 40, 25, 15]) for p in [-1, 1]],
        irreps_hidden="20x0e + 20x1o + 20x2e + 20x3o + 20x4e",
        irreps_out="19x0e + 5x1o + 5x2e + 3x3o + 1x4e",
        node_attr_dim=None,
        max_l_edges=4,
        message_correlation_order=3,
        layers=3,
        max_radius=3.5,
        number_of_basis=10,
        num_neighbors=12,
        num_nodes=24,
        reduce_output=False,
        r_lora=16,
    )

    device = "cuda"
    model = model.to(device)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    data = [
        {
            "pos": torch.rand(5, 3).to(device),
            "x": torch.rand(5, 10).to(device),
        },
        {
            "pos": torch.rand(4, 3).to(device),
            "x": torch.rand(4, 10).to(device),
        },
    ]
    ys = []
    for d in data:
        y = model(d)
        print(y.shape)
        ys.append(y)

    data_ = [Data(pos=d["pos"], x=d["x"]) for d in data]
    dataloader = DataLoader(data_, batch_size=len(data))

    for d in dataloader:
        y = model(d)
        print(d, y.shape)

        for i in range(len(data)):
            y0 = y[d.batch == i]
            print((y0 - ys[i]).abs().max(), y0.abs().mean())
            assert torch.allclose(y0, ys[i], rtol=1e-4, atol=1e-7)

    model.merge_LoRA()
