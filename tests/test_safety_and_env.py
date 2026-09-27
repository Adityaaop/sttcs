import pytest
from core.env.rail_env import RailEnv
from core.safety.failsafe_layer import FailsafeWrapper

@pytest.fixture
def env():
    e = RailEnv(num_stations=4, section_length=10.0, step_size=1.0/60.0) # 1 minute step
    e.reset()
    return e

@pytest.fixture
def safe_env(env):
    return FailsafeWrapper(env)

def test_normal_advancement(safe_env):
    # Initial state: T1 at Station_0, T2 at Track_0_1_Fwd (pos 5.0)
    safe_env.env.trains.clear()
    safe_env.env.add_train("T1", "Station_0", 0.0, 1)
    safe_env.env.add_train("T2", "Track_0_1_Fwd", 5.0, 1)
    
    # Both are facing forward (direction=1)
    # T1 is behind T2.
    # We will advance both at 60 km/h. Distance is 60 * (1/60) = 1.0 km.
    actions = {
        "T1": {"speed_advisory": 60.0, "signal_state": 2}, # Green
        "T2": {"speed_advisory": 60.0, "signal_state": 2}  # Green
    }
    
    state, reward, done, info = safe_env.step(actions)
    
    # Train 1 should advance by 1 km (length of Station_0), so it transitions to Track_0_1_Fwd at pos 0.0
    assert safe_env.env.trains["T1"].speed == 60.0
    assert safe_env.env.trains["T1"].position == 0.0
    assert safe_env.env.trains["T1"].current_track == "Track_0_1_Fwd"
    
    # Train 2 should advance by 1 km
    assert safe_env.env.trains["T2"].speed == 60.0
    assert safe_env.env.trains["T2"].position == 6.0

def test_headway_violation(safe_env):
    # Setup: T1 on Track_0_1_Fwd at pos 0.0, T2 on Track_0_1_Fwd at pos 2.5
    safe_env.env.trains.clear()
    safe_env.env.add_train("T1", "Track_0_1_Fwd", 0.0, direction=1)
    safe_env.env.add_train("T2", "Track_0_1_Fwd", 2.5, direction=1)
    
    # T1 tries to move 1.0 km (at 60 km/h) -> projected pos 1.0. Distance to T2 is 1.5 < 2.0. Headway violation!
    actions = {
        "T1": {"speed_advisory": 60.0, "signal_state": 2},
        "T2": {"speed_advisory": 0.0, "signal_state": 2} # T2 stops
    }
    
    # Check manual
    assert safe_env.check_headway_violation(safe_env.env.trains["T1"], actions["T1"]) == True
    
    state, reward, done, info = safe_env.step(actions)
    
    # T1 should have been halted
    assert safe_env.env.trains["T1"].speed == 0.0
    assert safe_env.env.trains["T1"].position == 0.0
    
    # T2 should be stopped as requested
    assert safe_env.env.trains["T2"].speed == 0.0
    assert safe_env.env.trains["T2"].position == 2.5

def test_spad_protection(safe_env):
    # Setup: T1 near end of Track_0_1_Fwd. T2 at Station_1.
    safe_env.env.trains.clear()
    safe_env.env.add_train("T1", "Track_0_1_Fwd", 9.5, direction=1)
    safe_env.env.add_train("T2", "Station_1", 0.0, direction=1)
    
    # Track_0_1_Fwd length is 10.0.
    # T1 tries to advance at 60 km/h (1.0 km distance). projected pos = 10.5 > 10.0.
    # Next node is Station_1, which is occupied by T2.
    # Signal is 0 (Red).
    
    actions = {
        "T1": {"speed_advisory": 60.0, "signal_state": 0},
        "T2": {"speed_advisory": 0.0, "signal_state": 2}
    }
    
    assert safe_env.check_spad_violation(safe_env.env.trains["T1"], actions["T1"]) == True
    
    state, reward, done, info = safe_env.step(actions)
    
    # T1 should be halted
    assert safe_env.env.trains["T1"].speed == 0.0
    assert safe_env.env.trains["T1"].position == 9.5

def test_disruption_recovery(safe_env):
    # Setup: T1 near end of Track_0_1_Fwd. Disruption at Station_1.
    safe_env.env.trains.clear()
    safe_env.env.add_train("T1", "Track_0_1_Fwd", 9.5, direction=1)
    
    safe_env.env.inject_disruption("track_blockage", "Station_1")
    
    # Signal is Red.
    actions = {
        "T1": {"speed_advisory": 60.0, "signal_state": 0}
    }
    
    # SPAD should catch it due to disruption
    assert safe_env.check_spad_violation(safe_env.env.trains["T1"], actions["T1"]) == True
    
    state, reward, done, info = safe_env.step(actions)
    
    # T1 halted
    assert safe_env.env.trains["T1"].speed == 0.0
    assert safe_env.env.trains["T1"].position == 9.5
