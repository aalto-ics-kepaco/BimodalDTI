import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, global_add_pool
from torch_geometric.nn.conv import MessagePassing

class ResidueGIGN(nn.Module):
    def __init__(self, atom_dim: int, residue_dim: int, hidden_dim: int):
        super().__init__()

        self.lin_atom = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.SiLU()
        )

        self.lin_residue = nn.Sequential(
            nn.Linear(residue_dim, hidden_dim),
            nn.SiLU()
        )

        self.convs = nn.ModuleList()
        for _ in range(3):
            self.convs.append(HeteroConv({
                ('atom', 'bond', 'atom'): GIGNConv(hidden_dim, hidden_dim),
                ('residue', 'bond', 'residue'): GIGNConv(hidden_dim, hidden_dim),
                ('atom', 'near', 'residue'): GIGNConv((hidden_dim, hidden_dim), hidden_dim),
                ('residue', 'near', 'atom'): GIGNConv((hidden_dim, hidden_dim), hidden_dim)
            }, aggr='sum'))

        self.fc = GIGNFC(hidden_dim, hidden_dim)

    def embed(self, data):
        x_dict = data.x_dict
        x_dict = {'atom': self.lin_atom(x_dict['atom']), 'residue': self.lin_residue(x_dict['residue'])}
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict, data.edge_attr_dict)
        x_atom = global_add_pool(x_dict['atom'], data['atom'].batch)
        x_residue = global_add_pool(x_dict['residue'], data['residue'].batch)
        return x_atom + x_residue
    
    def get_latent(self, data):
        x = self.embed(data)
        return self.fc.get_latent(x)

    def forward(self, data):
        x = self.embed(data)
        return self.fc(x)

class GIGNConv(MessagePassing):
    def __init__(self, in_channels: int | tuple[int, int], out_channels: int):
        super().__init__(aggr='add')
        
        self.mlp_node = nn.Sequential(
            nn.Linear(in_channels[0] if isinstance(in_channels, tuple) else in_channels, out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels)
        )

        self.mlp_coord = nn.Sequential(
            nn.Linear(9, in_channels[1] if isinstance(in_channels, tuple) else in_channels),
            nn.SiLU()
        )

    def forward(
            self,
            x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            size = None
        ):
        radial = self.mlp_coord(edge_attr)
        if isinstance(x, tuple):
            out_node = self.propagate(
                edge_index=edge_index,
                x=x,
                radial=radial,
                size=(x[0].shape[0], x[1].shape[0])
            )
            out = self.mlp_node(x[1] + out_node)
            return out
        else:
            out_node = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
            out = self.mlp_node(x + out_node)
            return out
    
    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, radial: torch.Tensor, index: torch.Tensor):
        return x_j * radial
    
class GIGNFC(nn.Module):
    def __init__(self, d_graph_layer: int, d_FC_layer: int, n_FC_layer = 3, dropout = 0.1, n_tasks = 1):
        super().__init__()
        self.layers = nn.Sequential()
        for _ in range(n_FC_layer - 1):
            self.layers.append(nn.Linear(d_graph_layer if n_FC_layer == 1 else d_FC_layer, d_FC_layer))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.BatchNorm1d(d_FC_layer))
        self.final_layer = nn.Linear(d_graph_layer if n_FC_layer == 1 else d_FC_layer, n_tasks)

    def get_latent(self, x):
        return self.layers(x)
    
    def forward(self, x):
        x = self.layers(x)
        return self.final_layer(x).squeeze()
