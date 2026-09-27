import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dim: int, embed_dim: int, action_dim: int = 4):
        # action_dim: 1 (speed continuous) + 3 (signal logits) = 4
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim + embed_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, obs: torch.Tensor, graph_embed: torch.Tensor) -> torch.Tensor:
        # obs: local agent observation (e.g., speed, position) [batch_size, obs_dim]
        # graph_embed: GNN embedding [batch_size, embed_dim]
        x = torch.cat([obs, graph_embed], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        
        # Split out into speed and signal logits
        speed_raw = out[..., 0:1]
        signal_logits = out[..., 1:4]
        
        # Speed should be 0-160
        speed = torch.sigmoid(speed_raw) * 160.0
        
        return torch.cat([speed, signal_logits], dim=-1)

class Critic(nn.Module):
    def __init__(self, embed_dim: int, num_agents: int, action_dim: int = 4):
        super(Critic, self).__init__()
        # Critic takes global graph embedding and ALL agents' actions
        self.fc1 = nn.Linear(embed_dim + (num_agents * action_dim), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, graph_embed: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        # joint_action: [batch_size, num_agents * action_dim]
        x = torch.cat([graph_embed, joint_action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
