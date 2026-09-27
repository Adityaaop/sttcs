import networkx as nx
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel

class Train(BaseModel):
    id: str
    position: float  # Current position in km from the start of the current track
    speed: float  # Current speed in km/h
    direction: int  # 1 for forward (e.g., Up), -1 for backward (e.g., Down)
    current_track: str # Node ID of the track block
    destination: str
    priority: int = 5 # Lower is higher priority (e.g., 1=Rajdhani, 8=Freight)
    
class RailEnv:
    def __init__(self, num_stations: int = 4, section_length: float = 10.0, step_size: float = 1.0/60.0):
        # step_size is in hours (e.g., 1 minute = 1/60 hours)
        self.num_stations = num_stations
        self.section_length = section_length
        self.step_size = step_size
        self.graph = nx.DiGraph()
        self.trains: Dict[str, Train] = {}
        self.disruptions: Dict[str, str] = {} # location -> type
        self._build_topology()
        
    def _build_topology(self):
        # Simplified topology: linear tracks between stations
        # For a more complex layout, we'd add loop lines and platform bays.
        # Nodes represent track blocks or stations.
        # Edges represent connectivity.
        
        for i in range(self.num_stations):
            station_id = f"Station_{i}"
            self.graph.add_node(station_id, type="station", length=1.0)
            
            # Loop lines / Platforms at station
            loop_id = f"Loop_{i}"
            self.graph.add_node(loop_id, type="loop", length=1.0)
            
            # Connect mainline to station and loop
            if i > 0:
                prev_station = f"Station_{i-1}"
                prev_loop = f"Loop_{i-1}"
                
                # Forward track
                track_fwd = f"Track_{i-1}_{i}_Fwd"
                self.graph.add_node(track_fwd, type="track", length=self.section_length)
                self.graph.add_edge(prev_station, track_fwd)
                self.graph.add_edge(prev_loop, track_fwd) # Can enter forward track from previous loop
                self.graph.add_edge(track_fwd, station_id)
                self.graph.add_edge(track_fwd, loop_id)
                
                # Backward track
                track_bwd = f"Track_{i}_{i-1}_Bwd"
                self.graph.add_node(track_bwd, type="track", length=self.section_length)
                self.graph.add_edge(station_id, track_bwd)
                self.graph.add_edge(loop_id, track_bwd)
                self.graph.add_edge(track_bwd, prev_station)
                self.graph.add_edge(track_bwd, prev_loop) # Can enter backward track to previous loop

    def reset(self) -> Dict[str, Any]:
        self.trains = {}
        self.disruptions = {}
        # Add some initial trains matching the mock TMS ingest stream
        self.add_train("T_Rajdhani_01", "Track_0_1_Fwd", 0.0, direction=1, speed=100.0, priority=1)
        self.add_train("T_Freight_02", "Track_1_2_Fwd", 2.0, direction=1, speed=50.0, priority=8)
        self.add_train("T_Express_03", "Track_2_3_Fwd", 4.0, direction=1, speed=80.0, priority=3)
        return self.get_state()
        
    def add_train(self, train_id: str, track_id: str, position: float, direction: int, speed: float = 0.0, priority: int = 5):
        self.trains[train_id] = Train(
            id=train_id,
            position=position,
            speed=speed,
            direction=direction,
            current_track=track_id,
            destination=f"Station_{self.num_stations - 1}" if direction == 1 else "Station_0",
            priority=priority
        )
        
    def step(self, actions: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # actions format: {train_id: {"speed_advisory": float, "signal_state": int, "platform_assignment": str}}
        # 0: Red, 1: Yellow, 2: Green
        
        reward = 0.0
        done = False
        info = {}
        
        for train_id, action in actions.items():
            if train_id not in self.trains:
                continue
                
            train = self.trains[train_id]
            target_speed = action.get("speed_advisory", 0.0)
            target_speed = min(max(target_speed, 0.0), 160.0)
            
            # Simple physics: instant acceleration for now, bounded by advisory
            train.speed = target_speed
            
            # If there's a blockage, we should ideally stop, but the environment 
            # will just process physics. The failsafe wrapper should handle stopping.
            # We'll allow the train to move.
            distance_to_move = train.speed * self.step_size
            
            current_track_node = self.graph.nodes[train.current_track]
            track_length = current_track_node.get("length", 1.0)
            
            train.position += distance_to_move
            
            # Very basic track transition logic
            if train.position >= track_length:
                # Transition to next track
                edges = list(self.graph.out_edges(train.current_track))
                if edges:
                    # Pick next edge. In a real system, use platform_assignment / routing.
                    next_node = edges[0][1]
                    train.current_track = next_node
                    train.position = train.position - track_length
                    reward += 1.0 # Reward for advancing
                else:
                    # End of line
                    train.position = track_length
                    train.speed = 0.0
                    
        return self.get_state(), reward, done, info

    def get_state(self) -> Dict[str, Any]:
        node_features = {}
        for node, data in self.graph.nodes(data=True):
            node_features[node] = {
                "type": data.get("type"),
                "length": data.get("length"),
                "occupancy": 0,
                "disrupted": 1 if node in self.disruptions.values() else 0
            }
            
        for train in self.trains.values():
            if train.current_track in node_features:
                node_features[train.current_track]["occupancy"] += 1
                
        return {
            "topology": {
                "nodes": node_features,
                "edges": list(self.graph.edges())
            },
            "trains": {t_id: t.model_dump() for t_id, t in self.trains.items()}
        }
        
    def inject_disruption(self, disruption_type: str, location: str):
        self.disruptions[location] = disruption_type
