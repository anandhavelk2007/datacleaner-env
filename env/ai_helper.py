import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import numpy as np
from typing import Dict

class AIHelper:
    def __init__(self):
        self.classifier = None
        self.country_mapping = {
            "usa": "USA", "u.s.a": "USA", "u.s.a.": "USA",
            "united states": "USA", "united states of america": "USA",
            "us": "USA", "us.": "USA"
        }

    def suggest_action(self, df: pd.DataFrame, column: str) -> Dict:
        col_data = df[column]
        if col_data.isnull().all():
            return {"type": "skip", "reason": "Column is empty", "confidence": 0.0}
        dtype = self.detect_type(col_data)
        if dtype == "numeric":
            return {"type": "impute", "method": "median", "confidence": 0.8}
        elif dtype == "date":
            return {"type": "fix_date", "confidence": 0.9}
        elif dtype == "categorical":
            unique_vals = set(col_data.dropna().astype(str).str.lower())
            if any(k in unique_vals for k in self.country_mapping.keys()):
                return {"type": "normalize_cat", "target": "USA", "confidence": 0.85}
            return {"type": "impute", "method": "mode", "confidence": 0.6}
        else:
            return {"type": "skip", "reason": "Unknown type", "confidence": 0.0}

    def detect_type(self, series: pd.Series) -> str:
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"
        try:
            pd.to_numeric(non_null)
            return "numeric"
        except:
            pass
        try:
            pd.to_datetime(non_null)
            return "date"
        except:
            pass
        if len(non_null.unique()) / len(non_null) < 0.5:
            return "categorical"
        return "text"