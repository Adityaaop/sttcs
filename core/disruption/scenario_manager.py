from typing import Dict, Any, Optional
from core.env.rail_env import RailEnv

class Disruption:
    def __init__(self, id: str, type: str, location: str, duration_steps: int, severity: float = 1.0):
        self.id = id
        self.type = type # "track_blockage", "signal_failure", "speed_restriction", "train_breakdown"
        self.location = location # node id or train id
        self.duration_steps = duration_steps
        self.severity = severity # 1.0 = full block, <1.0 = speed reduction
        self.active = True

class ScenarioManager:
    def __init__(self, env: RailEnv):
        self.env = env
        self.active_disruptions: Dict[str, Disruption] = {}
        self.disruption_counter = 0
        
    def step(self):
        to_remove = []
        for d_id, disruption in self.active_disruptions.items():
            if disruption.duration_steps > 0:
                disruption.duration_steps -= 1
            
            if disruption.duration_steps == 0:
                to_remove.append(d_id)
                
        for d_id in to_remove:
            self.remove_disruption(d_id)
            
    def inject_disruption(self, type: str, location: str, duration_steps: int, severity: float = 1.0) -> str:
        self.disruption_counter += 1
        d_id = f"disruption_{self.disruption_counter}"
        
        disruption = Disruption(d_id, type, location, duration_steps, severity)
        self.active_disruptions[d_id] = disruption
        
        # Apply to environment state
        if type == "track_blockage" or type == "train_breakdown":
            self.env.disruptions[location] = type
        elif type == "speed_restriction":
            self.env.disruptions[location] = f"{type}_{severity}"
        elif type == "signal_failure":
            self.env.disruptions[location] = type
            
        return d_id
        
    def remove_disruption(self, d_id: str):
        if d_id in self.active_disruptions:
            d = self.active_disruptions[d_id]
            d.active = False
            
            if d.location in self.env.disruptions:
                del self.env.disruptions[d.location]
                
            del self.active_disruptions[d_id]
            
    def get_max_speed_for_segment(self, location: str) -> float:
        # Check if there is an active speed restriction on this segment
        max_speed = 160.0
        for d in self.active_disruptions.values():
            if d.type == "speed_restriction" and d.location == location:
                # severity could represent the max speed directly or a factor. Let's say it's the speed in km/h.
                max_speed = min(max_speed, d.severity)
        return max_speed
