import pytest
from fastapi.testclient import TestClient
from api.server import app
from api.websocket_stream import manual_overrides

@pytest.fixture
def client():
    return TestClient(app)

def test_websocket_stream(client):
    with client.websocket_connect("/ws/stream") as websocket:
        # Request state update (wait for the server to broadcast)
        # Since simulation_loop runs in background, we might receive an update immediately
        # But TestClient's websocket doesn't run background tasks of `lifespan` out of the box in some setups.
        # Wait, Starlette's TestClient might not run lifespan background tasks reliably if not entered.
        # Let's send a manual override message to test that logic.
        
        override_msg = {
            "type": "manual_override",
            "train_id": "T1",
            "speed": 120.0,
            "signal_state": 1
        }
        websocket.send_json(override_msg)
        
        import time
        time.sleep(0.1)
        
        # Verify it updated the global dict
        assert "T1" in manual_overrides
        assert manual_overrides["T1"]["speed_advisory"] == 120.0
        assert manual_overrides["T1"]["signal_state"] == 1
        
        # Test clear override
        clear_msg = {
            "type": "clear_override",
            "train_id": "T1"
        }
        websocket.send_json(clear_msg)
        
        time.sleep(0.1)
        
        # Verify it was cleared
        assert "T1" not in manual_overrides
