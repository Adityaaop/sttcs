import numpy as np
import random
from typing import Dict, List, Tuple, Any

class ReplayBuffer:
    """
    Simplified Prioritized Experience Replay (PER).
    Stores (state, joint_action, joint_reward, next_state, done).
    """
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def add(self, state: Dict[str, Any], joint_action: np.ndarray, joint_reward: float, next_state: Dict[str, Any], done: bool):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, joint_action, joint_reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, joint_action, joint_reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        if len(self.buffer) == 0:
            return None
            
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        batch = list(zip(*samples))
        # Returns: states, joint_actions, joint_rewards, next_states, dones, indices, weights
        return batch[0], np.array(batch[1]), np.array(batch[2], dtype=np.float32), batch[3], np.array(batch[4], dtype=np.float32), indices, weights
        
    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = max(prio, 1e-5) # Ensure non-zero priority

    def __len__(self):
        return len(self.buffer)
