import argparse
import numpy as np
import torch
import random
from core.env.rail_env import RailEnv
from core.safety.failsafe_layer import FailsafeWrapper
from core.rl.trainer import MADDPGTrainer
from core.models.gnn_encoder import state_to_pyg_data

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_episode(env: RailEnv, safe_env: FailsafeWrapper, trainer: MADDPGTrainer, max_steps: int = 100, train_mode: bool = True):
    state = env.reset()
    total_reward = 0.0
    
    train_ids = list(env.trains.keys())
    num_agents = len(train_ids)
    
    for step in range(max_steps):
        # Generate actions
        raw_actions = {}
        joint_action_list = []
        
        with torch.no_grad():
            data = state_to_pyg_data(state)
            graph_embed = trainer.gnn(data)
            
            for i in range(num_agents):
                obs = trainer.extract_obs(state, i)
                action_tensor = trainer.actors[i](obs.unsqueeze(0), graph_embed)
                action_np = action_tensor.squeeze(0).numpy()
                
                if train_mode:
                    # Add simple exploration noise
                    action_np += np.random.normal(0, 0.1, size=action_np.shape)
                    
                joint_action_list.append(action_np)
                
                speed = np.clip(action_np[0], 0, 160)
                signal_state = np.argmax(action_np[1:4])
                
                raw_actions[train_ids[i]] = {
                    "speed_advisory": float(speed),
                    "signal_state": int(signal_state)
                }
                
        # Environment step via failsafe wrapper
        safe_actions = safe_env.apply_safe_actions(raw_actions)
        next_state, base_reward, done, info = env.step(safe_actions)
        
        # Calculate custom RL reward
        reward = trainer.compute_reward(env, safe_actions)
        total_reward += reward
        
        if train_mode:
            joint_action_arr = np.concatenate(joint_action_list)
            trainer.buffer.add(state, joint_action_arr, reward, next_state, done)
            
            # Train step
            trainer.step(batch_size=32)
            
        state = next_state
        if done:
            break
            
    return total_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--eval-interval', type=int, default=2)
    args = parser.parse_args()
    
    set_seed(42)
    
    env = RailEnv(num_stations=4, section_length=10.0)
    env.reset()
    safe_env = FailsafeWrapper(env)
    
    num_agents = len(env.trains)
    
    trainer = MADDPGTrainer(num_agents=num_agents, lr=args.learning_rate)
    
    for ep in range(1, args.episodes + 1):
        # Training
        train_reward = run_episode(env, safe_env, trainer, train_mode=True)
        print(f"Episode {ep} | Train Reward: {train_reward:.2f}")
        
        if ep % args.eval_interval == 0:
            eval_reward = run_episode(env, safe_env, trainer, train_mode=False)
            print(f"  > Eval Reward: {eval_reward:.2f}")
            trainer.save_weights()

if __name__ == "__main__":
    main()
