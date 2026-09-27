from typing import Dict, Any, List, Tuple
import networkx as nx

from core.env.rail_env import RailEnv, Train
from core.disruption.scenario_manager import ScenarioManager

class RecoveryEngine:
    def __init__(self, env: RailEnv, scenario_manager: ScenarioManager):
        self.env = env
        self.scenario_manager = scenario_manager
        
    def resolve_precedence(self, trains: List[Train]) -> List[Train]:
        # Sort trains by priority (lower number = higher priority)
        return sorted(trains, key=lambda t: t.priority)
        
    def get_alternative_paths(self, source: str, target: str) -> List[List[str]]:
        try:
            # We don't want to use disrupted nodes
            disrupted_nodes = [d.location for d in self.scenario_manager.active_disruptions.values() if d.type in ["track_blockage", "train_breakdown"]]
            
            # Create a subgraph without disrupted nodes
            safe_nodes = [n for n in self.env.graph.nodes if n not in disrupted_nodes]
            safe_graph = self.env.graph.subgraph(safe_nodes)
            
            paths = list(nx.shortest_simple_paths(safe_graph, source, target))
            return paths
        except nx.NetworkXNoPath:
            return []
            
    def compute_recovery_plan(self) -> Dict[str, Dict[str, Any]]:
        # Returns a dict of safe/advisory actions for each train
        plan = {}
        
        # Sort all trains by priority
        prioritized_trains = self.resolve_precedence(list(self.env.trains.values()))
        
        reserved_nodes = set()
        
        for train in prioritized_trains:
            # Check if current path has disruption ahead
            # For simplicity, we just look 1-2 blocks ahead
            edges = list(self.env.graph.out_edges(train.current_track))
            
            if not edges:
                plan[train.id] = {"speed_advisory": 0.0, "signal_state": 0} # end of line
                continue
                
            next_default_node = edges[0][1] # Simple default
            
            # Check if next default node is disrupted or reserved by higher priority
            is_disrupted = any(d.location == next_default_node and d.type in ["track_blockage", "train_breakdown"] 
                               for d in self.scenario_manager.active_disruptions.values())
                               
            if is_disrupted or next_default_node in reserved_nodes:
                # Need alternative route
                paths = self.get_alternative_paths(train.current_track, train.destination)
                
                valid_path = None
                for path in paths:
                    if len(path) > 1:
                        next_alt_node = path[1]
                        if next_alt_node not in reserved_nodes:
                            valid_path = path
                            break
                            
                if valid_path:
                    next_node = valid_path[1]
                    reserved_nodes.add(next_node)
                    reserved_nodes.add(train.current_track)
                    
                    # Compute safe speed
                    max_speed = self.scenario_manager.get_max_speed_for_segment(train.current_track)
                    plan[train.id] = {
                        "speed_advisory": min(80.0, max_speed), # Diverted, slower
                        "signal_state": 1, # Yellow
                        "route": next_node
                    }
                else:
                    # No path, must hold
                    reserved_nodes.add(train.current_track)
                    plan[train.id] = {
                        "speed_advisory": 0.0,
                        "signal_state": 0 # Red
                    }
            else:
                # Path is clear
                reserved_nodes.add(next_default_node)
                reserved_nodes.add(train.current_track)
                
                max_speed = self.scenario_manager.get_max_speed_for_segment(train.current_track)
                plan[train.id] = {
                    "speed_advisory": min(160.0, max_speed),
                    "signal_state": 2 # Green
                }
                
        return plan
