from typing import Dict, Any, List
from core.env.rail_env import RailEnv

class BaselineController:
    def __init__(self, env: RailEnv):
        self.env = env
        
    def _get_blocks_ahead_clear(self, train, max_lookahead=3) -> int:
        """
        Count how many blocks ahead are clear.
        """
        clear_blocks = 0
        current_node = train.current_track
        projected_position = train.position
        
        for _ in range(max_lookahead):
            node_data = self.env.graph.nodes[current_node]
            track_length = node_data.get("length", 1.0)
            
            # Move to next block if we are near the end, but since we just want to count blocks,
            # we iterate through the graph.
            edges = list(self.env.graph.out_edges(current_node))
            if not edges:
                break
                
            # Baseline just looks at the default straight route
            next_node = edges[0][1]
            
            # Check occupancy
            is_occupied = any(t.current_track == next_node for t in self.env.trains.values())
            is_disrupted = next_node in self.env.disruptions
            
            if is_occupied or is_disrupted:
                break
                
            clear_blocks += 1
            current_node = next_node
            
        return clear_blocks

    def compute_actions(self) -> Dict[str, Dict[str, Any]]:
        actions = {}
        
        # Sort by priority for strict precedence
        trains_by_priority = sorted(self.env.trains.values(), key=lambda t: t.priority)
        
        # Simple reserved set for the current step
        reserved = set()
        
        for train in trains_by_priority:
            if train.current_track in reserved:
                # If a higher priority train somehow reserved our current block, halt
                actions[train.id] = {"speed_advisory": 0.0, "signal_state": 0}
                continue
                
            clear_ahead = self._get_blocks_ahead_clear(train, max_lookahead=3)
            
            # 4-Aspect logic mapped to speed
            if clear_ahead >= 3:
                speed = 160.0
                signal = 2 # Green
            elif clear_ahead == 2:
                speed = 80.0
                signal = 2 # Double Yellow equivalent
            elif clear_ahead == 1:
                speed = 30.0
                signal = 1 # Yellow
            else:
                speed = 0.0
                signal = 0 # Red
                
            edges = list(self.env.graph.out_edges(train.current_track))
            if edges and speed > 0:
                reserved.add(edges[0][1])
                
            actions[train.id] = {
                "speed_advisory": speed,
                "signal_state": signal
            }
            
        return actions
