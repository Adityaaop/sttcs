import os
import pytest
import numpy as np
from core.env.rail_env import RailEnv
from deploy.edge_inference import EdgeSectionAgent

@pytest.fixture
def dummy_env():
    env = RailEnv(num_stations=4)
    # Add a couple of trains
    env.add_train("T_Rajdhani_01", "Track_0_1_Fwd", 2.0, direction=1, speed=100.0, priority=1)
    env.add_train("T_Freight_02", "Track_0_1_Fwd", 1.0, direction=1, speed=60.0, priority=8)
    return env

def test_onnx_models_exist():
    # Expects deploy/export_onnx.py to have been run
    assert os.path.exists("deploy/gnn_encoder.onnx"), "GNN ONNX model missing"
    assert os.path.exists("deploy/actor.onnx"), "Actor ONNX model missing"

def test_edge_inference_shapes(dummy_env):
    agent = EdgeSectionAgent("deploy/gnn_encoder.onnx", "deploy/actor.onnx", dummy_env)
    state = dummy_env.get_state()
    
    # 1. Test graph encoding shape
    x, edge_index = agent._state_to_onnx_graph(state)
    assert x.ndim == 2 and x.shape[1] == 3
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    
    gnn_inputs = {"x": x, "edge_index": edge_index}
    embed = agent.gnn_session.run(None, gnn_inputs)[0]
    assert embed.shape == (1, 128)
    
    # 2. Test actor shape
    obs = agent._extract_obs("T_Rajdhani_01", state)
    assert obs.shape == (5,)
    
    obs_batch = np.expand_dims(obs, axis=0)
    actor_inputs = {"obs": obs_batch, "graph_embed": embed}
    action = agent.actor_session.run(None, actor_inputs)[0]
    
    assert action.shape == (1, 4)

def test_failsafe_fallback_mock_ingest(dummy_env):
    agent = EdgeSectionAgent("deploy/gnn_encoder.onnx", "deploy/actor.onnx", dummy_env)
    state = dummy_env.get_state()
    
    # Intentionally force a dangerous situation to test failsafe fallback
    # T2 (pos 1.0) is closely following T1 (pos 2.0) -> headway is 1.0 km, minimum is 2.0 km.
    # Therefore T2 MUST be halted.
    
    safe_actions = agent.act(state)
    
    # Assert T2 is halted due to headway violation
    assert safe_actions["T_Freight_02"]["speed_advisory"] == 0.0
    assert "_violation" in safe_actions["T_Freight_02"]
    assert safe_actions["T_Freight_02"]["_violation"] == "HEADWAY"
