import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from core.env.rail_env import RailEnv
from core.safety.failsafe_layer import FailsafeWrapper
from core.disruption.scenario_manager import ScenarioManager
from core.disruption.what_if_evaluator import WhatIfEvaluator
from api.disruption_routes import router as disruption_router, init_globals
from api.websocket_stream import router as ws_router, init_ws_globals, simulation_loop

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the websocket simulation loop
    loop_task = asyncio.create_task(simulation_loop())
    yield
    loop_task.cancel()

app = FastAPI(title="Train Traffic Control Simulation API", lifespan=lifespan)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize environment and safety wrapper
env = RailEnv(num_stations=4, section_length=10.0)
env.reset()
safe_env = FailsafeWrapper(env)
scenario_manager = ScenarioManager(env)
what_if_evaluator = WhatIfEvaluator(env, scenario_manager)

init_globals(env, scenario_manager, what_if_evaluator)
init_ws_globals(env, safe_env, scenario_manager)

app.include_router(disruption_router)
app.include_router(ws_router)

class ActionRequest(BaseModel):
    actions: Dict[str, Dict[str, Any]]
    
class DisruptionRequest(BaseModel):
    type: str
    location: str

@app.post("/simulate/step")
async def simulate_step(req: ActionRequest):
    try:
        state, reward, done, info = safe_env.step(req.actions)
        return {
            "state": state,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/simulate/state")
async def get_state():
    return env.get_state()

@app.post("/simulate/reset")
async def reset_env():
    state = env.reset()
    return state

@app.get("/healthz")
async def healthz():
    if env is not None and safe_env is not None:
        return {"status": "ok", "failsafe": "ready"}
    raise HTTPException(status_code=503, detail="Service not ready")
