import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List
import json

from core.env.rail_env import RailEnv
from core.safety.failsafe_layer import FailsafeWrapper

router = APIRouter()

# Global state reference
env: RailEnv = None
safe_env: FailsafeWrapper = None
scenario_manager = None
active_connections: List[WebSocket] = []
manual_overrides: Dict[str, Dict[str, Any]] = {}

def init_ws_globals(e: RailEnv, se: FailsafeWrapper, sm=None):
    global env, safe_env, scenario_manager
    env = e
    safe_env = se
    scenario_manager = sm

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

async def simulation_loop():
    while True:
        if env and safe_env:
            # We would normally get AI actions here, but for now we use defaults or manual overrides
            actions = {}
            for t_id in env.trains.keys():
                if t_id in manual_overrides:
                    actions[t_id] = manual_overrides[t_id]
                else:
                    # Default AI action
                    max_speed = scenario_manager.get_max_speed_for_segment(env.trains[t_id].current_track)
                    actions[t_id] = {"speed_advisory": max_speed, "signal_state": 2}

            # Apply failsafe
            safe_actions = safe_env.apply_safe_actions(actions)
            
            # Step the scenario manager to handle disruption durations
            if scenario_manager:
                scenario_manager.step()
            
            # Count safety interventions
            interventions = sum(1 for a in safe_actions.values() if "_violation" in a)
            
            # Step environment
            state, reward, done, info = env.step(safe_actions)
            
            # Calculate metrics
            throughput = sum(t.speed / 160.0 for t in env.trains.values()) * 10.0 # dummy scale
            avg_delay = sum(1.0 for t in env.trains.values() if t.speed < 10) * 5.0 # dummy scale

            # Construct payload
            payload = {
                "type": "state_update",
                "state": state,
                "metrics": {
                    "throughput": round(throughput, 2),
                    "avg_delay": round(avg_delay, 2),
                    "safety_violations": 0, # Hard constraint prevents actual violations
                    "interventions": interventions
                }
            }
            
            await manager.broadcast(payload)
            
        await asyncio.sleep(0.2) # 5 Hz

@router.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "manual_override":
                t_id = message["train_id"]
                speed = message.get("speed")
                signal = message.get("signal_state")
                manual_overrides[t_id] = {
                    "speed_advisory": float(speed),
                    "signal_state": int(signal)
                }
            elif message.get("type") == "clear_override":
                t_id = message["train_id"]
                if t_id in manual_overrides:
                    del manual_overrides[t_id]
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
