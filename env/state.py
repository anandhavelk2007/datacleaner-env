import pandas as pd
from dataclasses import dataclass, field

@dataclass
class DataState:
    task_id: str
    data: pd.DataFrame
    history: list = field(default_factory=list)