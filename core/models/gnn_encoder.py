import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from typing import Dict, Any

class GraphEncoder(nn.Module):
    def __init__(self, node_in_dim: int = 3, hidden_dim: int = 64, embed_dim: int = 128):
        """
        node_in_dim: 3 (e.g., [length, occupancy, disrupted])
        """
        super().__init__()
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)
        self.activation = nn.ReLU()
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        data: PyTorch Geometric Data object containing node features (x) and edge indices (edge_index)
        Returns a global graph embedding of shape (1, embed_dim) by mean pooling.
        """
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        
        # Global mean pooling to get a fixed-size embedding for the whole graph
        global_embed = torch.mean(x, dim=0, keepdim=True)
        return global_embed

def state_to_pyg_data(state: Dict[str, Any]) -> Data:
    """
    Helper function to convert the environment state dictionary into a PyTorch Geometric Data object.
    """
    nodes = state["topology"]["nodes"]
    edges = state["topology"]["edges"]
    
    # Create node mapping to integer indices
    node_list = list(nodes.keys())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # Build node features tensor: [length, occupancy, disrupted]
    x_features = []
    for node in node_list:
        data = nodes[node]
        length = float(data.get("length", 1.0))
        occupancy = float(data.get("occupancy", 0))
        disrupted = float(data.get("disrupted", 0))
        x_features.append([length, occupancy, disrupted])
        
    x = torch.tensor(x_features, dtype=torch.float)
    
    # Build edge index tensor
    source_nodes = []
    target_nodes = []
    for u, v in edges:
        source_nodes.append(node_to_idx[u])
        target_nodes.append(node_to_idx[v])
        
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)
