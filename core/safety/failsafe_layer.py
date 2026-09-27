from typing import Dict, Any, List
from core.env.rail_env import RailEnv, Train

class FailsafeWrapper:
    def __init__(self, env: RailEnv):
        self.env = env
        self.min_headway = 2.0  # km
        
    def check_headway_violation(self, train: Train, action: Dict[str, Any]) -> bool:
        """
        Check if the proposed speed would cause the train to breach the 2.0 km headway.
        """
        target_speed = action.get("speed_advisory", 0.0)
        if target_speed <= 0.0:
            return False
            
        distance_to_move = target_speed * self.env.step_size
        projected_position = train.position + distance_to_move
        
        # Check all other trains
        for other_id, other_train in self.env.trains.items():
            if other_id == train.id:
                continue
                
            # If on the same track and moving in same direction
            if other_train.current_track == train.current_track and other_train.direction == train.direction:
                # If other train is ahead
                if other_train.position > train.position:
                    # Calculate projected distance
                    # Assuming other train moves at its current speed (worst case it stops, so we use its current position or a pessimistic projection)
                    # For a strict headway, if we project our position and it's within 2.0km of their CURRENT position, that's a violation risk.
                    distance_between = other_train.position - projected_position
                    if distance_between < self.min_headway:
                        return True
                        
            # Simplification: In a full routing graph, we'd also check trains on the *next* block if we're close to the boundary.
            # We'll implement a basic next-block check.
            current_node = self.env.graph.nodes[train.current_track]
            track_length = current_node.get("length", 1.0)
            
            if projected_position >= track_length:
                # We are projecting into the next block
                edges = list(self.env.graph.out_edges(train.current_track))
                if edges:
                    valid_next_nodes = [e[1] for e in edges]
                    next_node = action.get("route", valid_next_nodes[0])
                    if next_node not in valid_next_nodes:
                        next_node = valid_next_nodes[0]
                        
                    if other_train.current_track == next_node:
                        # Other train is on the next block
                        distance_into_next = projected_position - track_length
                        distance_between = other_train.position - distance_into_next
                        if distance_between < self.min_headway:
                            return True
                            
        return False
        
    def check_spad_violation(self, train: Train, action: Dict[str, Any]) -> bool:
        """
        Signal Passed at Danger protection.
        If signal_state is 0 (Red) and train is about to enter an occupied block, halt.
        """
        signal_state = action.get("signal_state", 0) # Default to Red if missing
        target_speed = action.get("speed_advisory", 0.0)
        
        if target_speed <= 0.0:
            return False
            
        if signal_state == 0:
            distance_to_move = target_speed * self.env.step_size
            projected_position = train.position + distance_to_move
            
            current_node = self.env.graph.nodes[train.current_track]
            track_length = current_node.get("length", 1.0)
            
            # Are we crossing a block boundary?
            if projected_position >= track_length:
                # Check if next block is occupied or disrupted
                edges = list(self.env.graph.out_edges(train.current_track))
                if edges:
                    valid_next_nodes = [e[1] for e in edges]
                    next_node = action.get("route", valid_next_nodes[0])
                    if next_node not in valid_next_nodes:
                        next_node = valid_next_nodes[0]
                    
                    # Check occupancy
                    is_occupied = any(t.current_track == next_node for t in self.env.trains.values())
                    is_disrupted = next_node in self.env.disruptions.keys() # assuming keys are node names
                    
                    if is_occupied or is_disrupted:
                        return True
                        
        return False
        
    def apply_safe_actions(self, raw_actions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        safe_actions = {}
        for train_id, action in raw_actions.items():
            if train_id not in self.env.trains:
                continue
                
            train = self.env.trains[train_id]
            safe_action = dict(action)
            
            # Check Headway
            if self.check_headway_violation(train, safe_action):
                safe_action["speed_advisory"] = 0.0
                safe_action["_violation"] = "HEADWAY"
                
            # Check SPAD
            if self.check_spad_violation(train, safe_action):
                safe_action["speed_advisory"] = 0.0
                safe_action["_violation"] = "SPAD"
                
            safe_actions[train_id] = safe_action
            
        return safe_actions
        
    def step(self, raw_actions: Dict[str, Dict[str, Any]]):
        safe_actions = self.apply_safe_actions(raw_actions)
        return self.env.step(safe_actions)
