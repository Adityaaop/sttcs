import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from typing import List, Dict, Any, Tuple

from core.env.rail_env import RailEnv
from core.safety.failsafe_layer import FailsafeWrapper
from core.models.gnn_encoder import GraphEncoder, state_to_pyg_data
from core.rl.maddpg_agent import Actor, Critic
from core.rl.replay_buffer import ReplayBuffer

class MADDPGTrainer:
    def __init__(self, num_agents: int = 3, action_dim: int = 4, obs_dim: int = 2, embed_dim: int = 128, lr: float = 1e-3, gamma: float = 0.95):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.gamma = gamma
        
        self.actors = [Actor(obs_dim, embed_dim, action_dim) for _ in range(num_agents)]
        self.target_actors = [Actor(obs_dim, embed_dim, action_dim) for _ in range(num_agents)]
        self.critics = [Critic(embed_dim, num_agents, action_dim) for _ in range(num_agents)]
        self.target_critics = [Critic(embed_dim, num_agents, action_dim) for _ in range(num_agents)]
        
        self.actor_opts = [optim.Adam(a.parameters(), lr=lr) for a in self.actors]
        self.critic_opts = [optim.Adam(c.parameters(), lr=lr) for c in self.critics]
        
        self.gnn = GraphEncoder(node_in_dim=3, hidden_dim=64, embed_dim=embed_dim)
        self.gnn_opt = optim.Adam(self.gnn.parameters(), lr=lr)
        
        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
            
        self.buffer = ReplayBuffer(capacity=10000)
        
    def extract_obs(self, state: Dict[str, Any], agent_idx: int) -> torch.Tensor:
        trains = list(state["trains"].values())
        if agent_idx < len(trains):
            t = trains[agent_idx]
            return torch.tensor([t["speed"], t["position"]], dtype=torch.float32)
        else:
            return torch.zeros(self.obs_dim, dtype=torch.float32)
            
    def compute_reward(self, env: RailEnv, safe_actions: Dict[str, Dict[str, Any]]) -> float:
        # R = 0.3 * Throughput - 0.3 * Delay + 0.2 * SafetyMargin + 0.2 * Utilization
        throughput = 0.0
        delay = 0.0
        safety_margin = 0.0
        utilization = 0.0
        
        for t in env.trains.values():
            throughput += (t.speed / 160.0)
            if t.speed < 10.0:
                delay += 1.0
            utilization += 0.1 # simplified
            
        for sa in safe_actions.values():
            if sa.get("_violation"):
                safety_margin -= 1.0
            else:
                safety_margin += 0.1
                
        return 0.3 * throughput - 0.3 * delay + 0.2 * safety_margin + 0.2 * utilization
        
    def step(self, batch_size: int = 32):
        if len(self.buffer) < batch_size:
            return 0.0, 0.0
            
        states, joint_actions, joint_rewards, next_states, dones, indices, weights = self.buffer.sample(batch_size)
        weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        joint_rewards_t = torch.tensor(joint_rewards, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        # We need GNN embeddings for batch
        # For simplicity in this demo implementation, we iterate through batch
        # A real implementation would batch the graphs using PyTorch Geometric DataLoader
        graph_embeds = []
        next_graph_embeds = []
        obs_batch = [[] for _ in range(self.num_agents)]
        next_obs_batch = [[] for _ in range(self.num_agents)]
        
        for i in range(batch_size):
            data = state_to_pyg_data(states[i])
            next_data = state_to_pyg_data(next_states[i])
            graph_embeds.append(self.gnn(data))
            next_graph_embeds.append(self.gnn(next_data))
            for a_idx in range(self.num_agents):
                obs_batch[a_idx].append(self.extract_obs(states[i], a_idx))
                next_obs_batch[a_idx].append(self.extract_obs(next_states[i], a_idx))
                
        graph_embeds = torch.cat(graph_embeds, dim=0) # [batch, embed_dim]
        next_graph_embeds = torch.cat(next_graph_embeds, dim=0)
        
        obs_t = [torch.stack(o) for o in obs_batch]
        next_obs_t = [torch.stack(o) for o in next_obs_batch]
        
        joint_actions_t = torch.tensor(joint_actions, dtype=torch.float32).view(batch_size, -1)
        
        target_joint_actions = []
        for a_idx in range(self.num_agents):
            a_next_action = self.target_actors[a_idx](next_obs_t[a_idx], next_graph_embeds)
            # Differentiate discrete logic via Gumbel Softmax if needed, here we just use raw logits/probs for simplicity
            target_joint_actions.append(a_next_action)
        target_joint_actions = torch.cat(target_joint_actions, dim=-1)
        
        td_errors = np.zeros(batch_size)
        
        for a_idx in range(self.num_agents):
            # Critic update
            target_q = self.target_critics[a_idx](next_graph_embeds, target_joint_actions.detach())
            y = joint_rewards_t + self.gamma * (1 - dones_t) * target_q
            q = self.critics[a_idx](graph_embeds, joint_actions_t)
            
            critic_loss = (weights_t * F.mse_loss(q, y.detach(), reduction='none')).mean()
            
            self.critic_opts[a_idx].zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_opts[a_idx].step()
            
            td_errors += (q.detach() - y.detach()).abs().squeeze().numpy()
            total_critic_loss += critic_loss.item()
            
            # Actor update
            # Get actions for all agents from current policy for Q evaluation
            curr_joint_actions = []
            for j in range(self.num_agents):
                if j == a_idx:
                    curr_joint_actions.append(self.actors[j](obs_t[j], graph_embeds.detach()))
                else:
                    curr_joint_actions.append(self.actors[j](obs_t[j], graph_embeds.detach()).detach())
                    
            curr_joint_actions_cat = torch.cat(curr_joint_actions, dim=-1)
            actor_loss = -self.critics[a_idx](graph_embeds.detach(), curr_joint_actions_cat).mean()
            
            self.actor_opts[a_idx].zero_grad()
            actor_loss.backward()
            self.actor_opts[a_idx].step()
            
            total_actor_loss += actor_loss.item()
            
        # Update GNN parameters
        self.gnn_opt.step()
        self.gnn_opt.zero_grad()
            
        self.buffer.update_priorities(indices, td_errors / self.num_agents)
        
        # Soft update targets
        tau = 0.01
        for a_idx in range(self.num_agents):
            for target_param, param in zip(self.target_actors[a_idx].parameters(), self.actors[a_idx].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            for target_param, param in zip(self.target_critics[a_idx].parameters(), self.critics[a_idx].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                
        return total_actor_loss / self.num_agents, total_critic_loss / self.num_agents
        
    def save_weights(self, path: str = "weights"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.gnn.state_dict(), os.path.join(path, "gnn.pth"))
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), os.path.join(path, f"actor_{i}.pth"))
            torch.save(self.critics[i].state_dict(), os.path.join(path, f"critic_{i}.pth"))
