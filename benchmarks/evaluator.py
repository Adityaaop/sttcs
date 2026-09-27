import time
import copy
import numpy as np
import torch
from typing import Dict, Any, List, Tuple

from core.env.rail_env import RailEnv
from core.safety.failsafe_layer import FailsafeWrapper
from core.disruption.scenario_manager import ScenarioManager
from benchmarks.baseline_controller import BaselineController
from core.rl.trainer import MADDPGTrainer
from core.models.gnn_encoder import state_to_pyg_data

class Evaluator:
    def __init__(self, num_stations=4):
        self.num_stations = num_stations
        
    def setup_scenario(self, scenario_type: str) -> Tuple[RailEnv, ScenarioManager]:
        env = RailEnv(num_stations=self.num_stations)
        sm = ScenarioManager(env)
        
        # Scenario A: Normal (40 trains)
        # We will stagger them on track 0_1 and loop lines to fit
        if scenario_type == "A":
            num_trains = 20 # Scaled down for quick test, can be 40
            for i in range(num_trains):
                pos = (i * 2.5) % 10.0
                track = f"Track_{min(i % 3, 2)}_{min(i % 3 + 1, 3)}_Fwd"
                priority = 1 if i % 5 == 0 else 5
                env.add_train(f"T_{i}", track, pos, direction=1, priority=priority)
                
        elif scenario_type == "B": # Peak Surge (150% traffic)
            num_trains = 30 
            for i in range(num_trains):
                pos = (i * 1.5) % 10.0
                track = f"Track_{min(i % 3, 2)}_{min(i % 3 + 1, 3)}_Fwd"
                priority = 1 if i % 5 == 0 else 5
                env.add_train(f"T_{i}", track, pos, direction=1, priority=priority)
                
        elif scenario_type == "C": # High Disruption
            num_trains = 20
            for i in range(num_trains):
                pos = (i * 2.5) % 10.0
                track = f"Track_{min(i % 3, 2)}_{min(i % 3 + 1, 3)}_Fwd"
                priority = 1 if i % 5 == 0 else 5
                env.add_train(f"T_{i}", track, pos, direction=1, priority=priority)
            
        return env, sm

    def evaluate(self, scenario_type: str, controller_type: str, steps: int = 100) -> Dict[str, Any]:
        env, sm = self.setup_scenario(scenario_type)
        safe_env = FailsafeWrapper(env)
        
        if controller_type == "AI":
            num_agents = len(env.trains)
            ai_trainer = MADDPGTrainer(num_agents=num_agents)
            # In a real scenario, we'd load weights here
            # ai_trainer.load_weights("weights/")
            
        elif controller_type == "Baseline":
            baseline = BaselineController(env)
            
        metrics = {
            "throughput_count": 0,
            "total_delay_mins": 0.0,
            "min_headway_recorded": 999.0,
            "safety_violations": 0,
            "total_latency_ms": 0.0,
            "delays_list": [],
            "trajectories": {t_id: [] for t_id in env.trains.keys()}
        }
        
        for step_idx in range(steps):
            if scenario_type == "C" and step_idx == 30:
                sm.inject_disruption("track_blockage", "Station_1", duration_steps=50)
                sm.inject_disruption("signal_failure", "Station_2", duration_steps=50)
                
            sm.step()
            state = env.get_state()
            
            start_time = time.time()
            raw_actions = {}
            
            if controller_type == "AI":
                with torch.no_grad():
                    data = state_to_pyg_data(state)
                    graph_embed = ai_trainer.gnn(data)
                    train_ids = list(env.trains.keys())
                    for i in range(num_agents):
                        if i < len(train_ids):
                            obs = ai_trainer.extract_obs(state, i)
                            action_tensor = ai_trainer.actors[i](obs.unsqueeze(0), graph_embed)
                            action_np = action_tensor.squeeze(0).numpy()
                            speed = np.clip(action_np[0], 0, 160)
                            signal = np.argmax(action_np[1:4])
                            raw_actions[train_ids[i]] = {"speed_advisory": speed, "signal_state": int(signal)}
                            
            elif controller_type == "Baseline":
                raw_actions = baseline.compute_actions()
                
            metrics["total_latency_ms"] += (time.time() - start_time) * 1000.0
            
            # Apply failsafe
            safe_actions = safe_env.apply_safe_actions(raw_actions)
            
            # Count violations
            for a in safe_actions.values():
                if a.get("_violation"):
                    metrics["safety_violations"] += 1
                    
            # Record headway (simplified)
            positions = sorted([t.position for t in env.trains.values() if t.current_track == "Track_0_1_Fwd"])
            for i in range(len(positions) - 1):
                hw = positions[i+1] - positions[i]
                metrics["min_headway_recorded"] = min(metrics["min_headway_recorded"], hw)
                
            # Step env
            _, reward, done, _ = env.step(safe_actions)
            
            for train in env.trains.values():
                metrics["trajectories"][train.id].append(train.position)
                if train.speed > 0:
                    metrics["throughput_count"] += train.speed / 160.0
                if train.speed < 10.0:
                    metrics["total_delay_mins"] += 1.0 # 1 step = 1 min approx
                    metrics["delays_list"].append(1.0)
                else:
                    metrics["delays_list"].append(0.0)
                    
        # Averages
        num_trains = len(env.trains) if len(env.trains) > 0 else 1
        metrics["avg_delay_per_train"] = metrics["total_delay_mins"] / num_trains
        metrics["avg_latency_ms"] = metrics["total_latency_ms"] / steps
        metrics["throughput_per_hr"] = metrics["throughput_count"] / (steps / 60.0) if steps > 0 else 0
        
        return metrics
