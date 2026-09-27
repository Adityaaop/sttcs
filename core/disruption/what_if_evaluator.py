import copy
from typing import Dict, Any, Tuple, List
from core.env.rail_env import RailEnv
from core.safety.failsafe_layer import FailsafeWrapper
from core.disruption.scenario_manager import ScenarioManager
from core.disruption.recovery_engine import RecoveryEngine

class WhatIfEvaluator:
    def __init__(self, env: RailEnv, scenario_manager: ScenarioManager):
        self.base_env = env
        self.base_scenario_manager = scenario_manager
        
    def evaluate_plan(self, plan: Dict[str, Dict[str, Any]], steps: int = 15) -> Dict[str, Any]:
        """
        Fast-forward simulation cloning the current state and executing the plan.
        Steps could represent minutes if step_size is 1/60 hr.
        """
        # Deep clone the environment and scenario manager
        sim_env = copy.deepcopy(self.base_env)
        sim_sm = copy.deepcopy(self.base_scenario_manager)
        sim_sm.env = sim_env # Update reference
        
        safe_env = FailsafeWrapper(sim_env)
        
        throughput = 0.0
        delays = 0.0
        safety_interventions = 0
        
        # Fast forward N steps
        for step in range(steps):
            # Step the scenarios
            sim_sm.step()
            
            # Use the provided plan as the base action, but we might need to route it.
            # The plan tells us advisory speed and route. Let's just apply it statically for this simulation,
            # or dynamically recalculate. A static plan evaluation assumes trains hold these advisories.
            
            # Apply failsafe
            safe_actions = safe_env.apply_safe_actions(plan)
            
            for t_id, action in safe_actions.items():
                if action.get("_violation"):
                    safety_interventions += 1
                    
            state, reward, done, info = sim_env.step(safe_actions)
            
            for train in sim_env.trains.values():
                throughput += train.speed / 160.0
                if train.speed < 10.0:
                    delays += 1.0
                    
        # Metrics calculation
        metrics = {
            "throughput_score": throughput,
            "delay_penalty": delays,
            "safety_interventions": safety_interventions,
            "recovery_score": throughput - delays - (safety_interventions * 10)
        }
        
        return metrics
        
    def get_top_k_plans(self, k: int = 2) -> List[Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]]:
        # Generate candidate plans.
        # Plan 1: Default dynamic recovery engine plan
        engine = RecoveryEngine(self.base_env, self.base_scenario_manager)
        plan_1 = engine.compute_recovery_plan()
        
        # Plan 2: "All Stop" extremely conservative plan
        plan_2 = {}
        for t_id in self.base_env.trains.keys():
            plan_2[t_id] = {"speed_advisory": 0.0, "signal_state": 0}
            
        # Plan 3: "Aggressive" attempt (ignore priority, push max speed)
        # We can implement this by just setting max speeds and relying on failsafe.
        plan_3 = {}
        for t_id in self.base_env.trains.keys():
            plan_3[t_id] = {"speed_advisory": 160.0, "signal_state": 2}
            
        candidates = [plan_1, plan_2, plan_3]
        results = []
        
        for p in candidates:
            metrics = self.evaluate_plan(p)
            results.append((p, metrics))
            
        # Sort by recovery_score descending
        results.sort(key=lambda x: x[1]["recovery_score"], reverse=True)
        
        return results[:k]
