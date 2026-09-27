import onnxruntime as ort
import numpy as np
from typing import Dict, Any

from core.safety.failsafe_layer import FailsafeWrapper
from core.env.rail_env import RailEnv

class EdgeSectionAgent:
    def __init__(self, gnn_path: str, actor_path: str, env: RailEnv):
        """
        Lightweight ONNX agent running independently of PyTorch.
        """
        # Specify execution providers prioritizing CUDA then CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.gnn_session = ort.InferenceSession(gnn_path, providers=providers)
        self.actor_session = ort.InferenceSession(actor_path, providers=providers)
        
        # Attach Failsafe wrapper to guarantee constraints
        self.failsafe = FailsafeWrapper(env)
        self.env = env
        
    def _extract_obs(self, train_id: str, state: Dict[str, Any]) -> np.ndarray:
        """
        Extract observation [speed, position, direction, priority, distance_to_end]
        """
        train = self.env.trains[train_id]
        
        current_node = self.env.graph.nodes[train.current_track]
        track_length = current_node.get("length", 1.0)
        dist_to_end = track_length - train.position
        
        obs = np.array([
            train.speed,
            train.position,
            train.direction,
            train.priority,
            dist_to_end
        ], dtype=np.float32)
        
        return obs
        
    def _state_to_onnx_graph(self, state: Dict[str, Any]):
        """
        Convert topology to node features and edge list for ONNX GNN.
        """
        nodes = state["topology"]["nodes"]
        edges = state["topology"]["edges"]
        
        node_list = list(nodes.keys())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        x_features = []
        for node in node_list:
            data = nodes[node]
            x_features.append([
                float(data.get("length", 1.0)),
                float(data.get("occupancy", 0.0)),
                float(data.get("disrupted", 0.0))
            ])
            
        x = np.array(x_features, dtype=np.float32)
        
        source_nodes = []
        target_nodes = []
        for u, v in edges:
            source_nodes.append(node_to_idx[u])
            target_nodes.append(node_to_idx[v])
            
        edge_index = np.array([source_nodes, target_nodes], dtype=np.int64)
        return x, edge_index
        
    def act(self, state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        raw_actions = {}
        
        # 1. Run GNN once globally
        x, edge_index = self._state_to_onnx_graph(state)
        gnn_inputs = {"x": x, "edge_index": edge_index}
        
        graph_embed = self.gnn_session.run(None, gnn_inputs)[0] # Shape: [1, embed_dim]
        
        # 2. Run Actor for each train
        for train_id in self.env.trains.keys():
            obs = self._extract_obs(train_id, state) # Shape: [5]
            obs = np.expand_dims(obs, axis=0) # Shape: [1, 5]
            
            actor_inputs = {
                "obs": obs,
                "graph_embed": graph_embed
            }
            
            action = self.actor_session.run(None, actor_inputs)[0] # Shape: [1, 4]
            action = action[0]
            
            speed = np.clip(action[0], 0, 160)
            signal = np.argmax(action[1:4])
            
            raw_actions[train_id] = {
                "speed_advisory": float(speed),
                "signal_state": int(signal)
            }
            
        # 3. Deterministic Safety Wrapper
        safe_actions = self.failsafe.apply_safe_actions(raw_actions)
        return safe_actions
