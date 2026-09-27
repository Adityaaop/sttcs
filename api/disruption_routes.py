from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from core.env.rail_env import RailEnv
from core.disruption.scenario_manager import ScenarioManager
from core.disruption.what_if_evaluator import WhatIfEvaluator

router = APIRouter()

# Global instances (to be injected/bound by server.py)
env: RailEnv = None
scenario_manager: ScenarioManager = None
what_if_evaluator: WhatIfEvaluator = None
active_plan: Dict[str, Dict[str, Any]] = {}

class DisruptionRequest(BaseModel):
    type: str
    location: str
    duration_steps: int = 10
    severity: float = 1.0

class EvaluateRequest(BaseModel):
    steps: int = 15
    
class ApplyRequest(BaseModel):
    plan: Dict[str, Dict[str, Any]]

def init_globals(e: RailEnv, sm: ScenarioManager, wif: WhatIfEvaluator):
    global env, scenario_manager, what_if_evaluator
    env = e
    scenario_manager = sm
    what_if_evaluator = wif

@router.post("/disruption/inject")
async def inject_disruption(req: DisruptionRequest):
    try:
        d_id = scenario_manager.inject_disruption(req.type, req.location, req.duration_steps, req.severity)
        return {"status": "success", "disruption_id": d_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disruption/remove")
async def remove_disruption(req: DisruptionRequest):
    try:
        # Find active disruption by location
        to_remove = [d_id for d_id, d in scenario_manager.active_disruptions.items() if d.location == req.location]
        for d_id in to_remove:
            scenario_manager.remove_disruption(d_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disruption/evaluate")
async def evaluate_disruptions(req: EvaluateRequest):
    try:
        top_plans = what_if_evaluator.get_top_k_plans(k=2)
        formatted = []
        for i, (plan, metrics) in enumerate(top_plans):
            formatted.append({
                "plan_id": f"plan_{i+1}",
                "plan": plan,
                "metrics": metrics
            })
        return {"status": "success", "top_plans": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disruption/apply")
async def apply_plan(req: ApplyRequest):
    global active_plan
    active_plan = req.plan
    return {"status": "success", "message": "Plan committed to active controllers."}
