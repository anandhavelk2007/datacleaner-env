import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
from env.environment import DataCleanerEnv
from env.models import Action

app = FastAPI(title="DataCleaner Environment")

_env = None

def get_env():
    global _env
    if _env is None:
        _env = DataCleanerEnv()
    return _env

class ResetRequest(BaseModel):
    task_id: str = "easy"

class StepRequest(BaseModel):
    action: Action

def safe_serialize(obj):
    """Convert any pandas/numpy type to Python native, recursively."""
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(i) for i in obj]
    if isinstance(obj, (np.ndarray, pd.Series)):
        return safe_serialize(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return safe_serialize(obj.to_dict(orient="records"))
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    return obj

@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    try:
        env = get_env()
        obs = env.reset(request.task_id)
        return safe_serialize(obs.model_dump())
    except Exception as e:
        detail = f"Error: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=detail)

@app.post("/step")
async def step(request: StepRequest):
    try:
        env = get_env()
        obs, reward, done, info = env.step(request.action)
        return safe_serialize({
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        })
    except Exception as e:
        detail = f"Error: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=detail)

@app.get("/state")
async def get_state():
    try:
        env = get_env()
        if env.state is None:
            raise HTTPException(status_code=400, detail="Environment not reset")
        state_dict = {
            "task_id": env.state.task_id,
            "data": safe_serialize(env.state.data),
            "history": safe_serialize(env.state.history)
        }
        return {"state": state_dict}
    except Exception as e:
        detail = f"Error: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=detail)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "DataCleaner Environment is running. Use /health, /reset, /step, /state endpoints."}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)