import pytest
from core.env.rail_env import RailEnv
from core.disruption.scenario_manager import ScenarioManager
from core.disruption.recovery_engine import RecoveryEngine
from core.safety.failsafe_layer import FailsafeWrapper

def test_disruption_recovery_and_safety():
    env = RailEnv(num_stations=4)
    sm = ScenarioManager(env)
    
    # Train 1: Rajdhani (Priority 1)
    # Train 2: Freight (Priority 8)
    env.trains.clear()
    env.add_train("T1_Rajdhani", "Track_0_1_Fwd", 8.0, direction=1, speed=80.0, priority=1)
    env.add_train("T2_Freight", "Track_0_1_Fwd", 5.0, direction=1, speed=80.0, priority=8)
    
    # Inject blockage at Station_1 (main line)
    sm.inject_disruption("track_blockage", "Station_1", duration_steps=100)
    
    engine = RecoveryEngine(env, sm)
    safe_env = FailsafeWrapper(env)
    
    plan = engine.compute_recovery_plan()
    
    # Check plan priorities and routing
    assert plan["T1_Rajdhani"]["route"] == "Loop_1" # Routed through loop line
    assert plan["T1_Rajdhani"]["speed_advisory"] > 0.0 # Allowed to move
    
    assert plan["T2_Freight"]["speed_advisory"] == 0.0 # Held back
    assert plan["T2_Freight"]["signal_state"] == 0 # Red signal
    
    # Apply plan and verify safety wrapper
    safe_actions = safe_env.apply_safe_actions(plan)
    
    # T1_Rajdhani should not have a violation
    assert "_violation" not in safe_actions["T1_Rajdhani"]
    
    # Now simulate T1 moving into the loop and stopping, and T2 approaching from behind.
    # We want to verify headway violation prevention.
    # Let's say T1 is stopped at Loop_1.
    env.trains["T1_Rajdhani"].current_track = "Loop_1"
    env.trains["T1_Rajdhani"].position = 0.5
    env.trains["T1_Rajdhani"].speed = 0.0
    
    # T2 is on Track_0_1_Fwd approaching Loop_1 at pos 9.5 (track length 10.0)
    env.trains["T2_Freight"].position = 9.5
    env.trains["T2_Freight"].speed = 80.0
    
    # Even if T2 is erroneously given a green signal and 80km/h speed
    unsafe_plan = {
        "T2_Freight": {"speed_advisory": 80.0, "signal_state": 2, "route": "Loop_1"}
    }
    
    safe_actions_2 = safe_env.apply_safe_actions(unsafe_plan)
    
    # T2 should be halted by failsafe because T1 is in Loop_1 and distance is 0.5 + 0.5 = 1.0km < 2.0km min headway
    # Wait, check_headway_violation only checks same track or next track.
    # Let's verify check_headway_violation triggers.
    
    assert safe_actions_2["T2_Freight"]["speed_advisory"] == 0.0
    assert "_violation" in safe_actions_2["T2_Freight"]
