from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any

class Observation(BaseModel):
    task_id: str
    description: str
    dataset_summary: Dict[str, Any]
    column_issues: List[str]

class Action(BaseModel):
    type: Literal["impute", "fix_date", "normalize_cat", "skip"]
    column: Optional[str] = None
    method: Optional[str] = None
    target: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str