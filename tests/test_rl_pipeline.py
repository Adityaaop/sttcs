import pytest
import torch
import numpy as np
from core.rl.maddpg_agent import Actor, Critic
from core.rl.replay_buffer import ReplayBuffer
from core.rl.trainer import MADDPGTrainer
from core.env.rail_env import RailEnv
from core.models.gnn_encoder import state_to_pyg_data

def test_actor_forward_pass():
    obs_dim = 2
    embed_dim = 128
    action_dim = 4
    
    actor = Actor(obs_dim, embed_dim, action_dim)
    
    obs = torch.randn(1, obs_dim)
    graph_embed = torch.randn(1, embed_dim)
    
    action = actor(obs, graph_embed)
    
    assert action.shape == (1, action_dim)
    
    # Speed is [0, 160]
    speed = action[0, 0].item()
    assert 0.0 <= speed <= 160.0

def test_critic_forward_pass():
    embed_dim = 128
    num_agents = 3
    action_dim = 4
    
    critic = Critic(embed_dim, num_agents, action_dim)
    
    graph_embed = torch.randn(1, embed_dim)
    joint_action = torch.randn(1, num_agents * action_dim)
    
    q_value = critic(graph_embed, joint_action)
    
    assert q_value.shape == (1, 1)

def test_replay_buffer_sampling():
    buffer = ReplayBuffer(capacity=100)
    
    # Dummy data
    state = {"topology": {"nodes": {"a": {"length": 1, "occupancy": 0, "disrupted": 0}}, "edges": []}, "trains": {}}
    next_state = state
    joint_action = np.random.randn(3 * 4)
    joint_reward = 1.0
    done = False
    
    for _ in range(10):
        buffer.add(state, joint_action, joint_reward, next_state, done)
        
    states, joint_actions, joint_rewards, next_states, dones, indices, weights = buffer.sample(batch_size=5)
    
    assert len(states) == 5
    assert joint_actions.shape == (5, 12)
    assert joint_rewards.shape == (5,)
    assert dones.shape == (5,)
    assert indices.shape == (5,)
    assert weights.shape == (5,)

def test_trainer_step():
    env = RailEnv(num_stations=4)
    state = env.reset()
    num_agents = len(env.trains)
    
    trainer = MADDPGTrainer(num_agents=num_agents)
    
    # Add dummy experiences
    for _ in range(35): # Need more than batch size (32)
        joint_action = np.random.randn(num_agents * 4)
        trainer.buffer.add(state, joint_action, 1.0, state, False)
        
    # Verify no dimension mismatches during step
    actor_loss, critic_loss = trainer.step(batch_size=32)
    
    assert not np.isnan(actor_loss)
    assert not np.isnan(critic_loss)
