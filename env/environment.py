import os
import pandas as pd
import numpy as np
from typing import Tuple, Any
from .models import Observation, Action, Reward
from .state import DataState
from .tasks import TASKS, get_grader, NORMALIZATION_MAP
from .ai_helper import AIHelper

class DataCleanerEnv:
    def __init__(self):
        self.state = None
        self.task_id = None
        self.grader = None
        self.last_action = None
        self.ai_helper = AIHelper()
        self._bonus_history = []

    def reset(self, task_id: str = "easy") -> Observation:
        self.task_id = task_id
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "data", f"{task_id}_raw.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.getcwd(), "env", "data", f"{task_id}_raw.csv")
        raw_df = pd.read_csv(csv_path)
        self.state = DataState(data=raw_df, task_id=task_id)
        self.grader = get_grader(task_id)
        self.last_action = None
        self._bonus_history = []
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        if not self.state:
            raise RuntimeError("Environment not reset")
        try:
            reward_value, message = self._apply_action(action)
        except Exception as e:
            reward_value, message = -0.1, f"Invalid action: {str(e)}"
        if self.last_action and action.type != "skip":
            if action.type == self.last_action.type:
                if hasattr(action, 'column') and hasattr(self.last_action, 'column') and action.column == self.last_action.column:
                    reward_value -= 0.05
                    message += " (repeated action penalty)"
        self.last_action = action
        score = self.grader(self.state.data)
        done = (score >= 0.99)
        if done:
            reward_value += 1.0
            message += " Task completed!"
        reward_value = max(-1.0, min(2.0, reward_value))
        reward = Reward(value=reward_value, reason=message)
        obs = self._make_observation()
        info = {"score": score, "steps": len(self.state.history)}
        return obs, reward, done, info

    def state(self) -> Any:
        return self.state

    def _apply_action(self, action: Action) -> Tuple[float, str]:
        old_score = self.grader(self.state.data)
        df = self.state.data
        if action.type == "impute":
            if action.column is None or action.method is None:
                return -0.1, "Impute requires column and method"
            if action.column not in df.columns:
                return -0.1, f"Column {action.column} not found"
            col = df[action.column]
            if action.method not in ["mean", "median", "mode"]:
                return -0.2, f"Unknown imputation method {action.method}"
            if action.method == "mean":
                val = col.mean()
            elif action.method == "median":
                val = col.median()
            else:
                val = col.mode()[0] if not col.mode().empty else None
            if val is None:
                return -0.1, "No valid value to impute"
            df[action.column] = df[action.column].fillna(val)
        elif action.type == "fix_date":
            if action.column is None:
                return -0.1, "fix_date requires column"
            try:
                df[action.column] = pd.to_datetime(df[action.column], errors='coerce')
            except Exception as e:
                return -0.2, f"Date conversion failed: {e}"
        elif action.type == "normalize_cat":
            if action.column is None:
                return -0.1, "normalize_cat requires column"
            target = action.target if action.target else "USA"
            if self.task_id == "hard" and action.column == "country":
                df[action.column] = df[action.column].astype(str).str.lower().map(NORMALIZATION_MAP).fillna(df[action.column])
            else:
                df[action.column] = target
        elif action.type == "skip":
            pass
        else:
            return -0.5, f"Unknown action type {action.type}"
        new_score = self.grader(self.state.data)
        delta = new_score - old_score
        reward = delta * 0.5
        if delta > 0 and action.column:
            suggestion = self.ai_helper.suggest_action(self.state.data, action.column)
            if suggestion.get("type") == action.type:
                reward += 0.1
                message_extra = " +0.1 (column match)"
            else:
                message_extra = ""
            if action.type == "impute" and suggestion.get("method") == action.method:
                reward += 0.1
                message_extra += " +0.1 (method match)"
            col_key = f"{action.column}_{action.type}"
            if col_key not in self._bonus_history:
                self._bonus_history.append(col_key)
            else:
                reward -= 0.05
                message_extra += " -0.05 (repeated column action)"
        else:
            message_extra = ""
        message = f"Score improved by {delta:.3f}" + message_extra
        return reward, message

    def _make_observation(self) -> Observation:
        df = self.state.data
        missing = df.isnull().sum().to_dict()
        sample = df.head(3).to_dict(orient='records')
        column_types = {}
        column_stats = {}
        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            if pd.api.types.is_numeric_dtype(series):
                col_type = "numeric"
                col_stats = {"mean": series.mean(), "std": series.std(), "min": series.min(), "max": series.max()}
            elif pd.api.types.is_datetime64_any_dtype(series):
                col_type = "date"
                col_stats = {"min": series.min(), "max": series.max()}
            else:
                col_type = "categorical" if len(non_null.unique()) < 20 else "text"
                col_stats = {"unique_count": len(non_null.unique())}
            column_types[col] = col_type
            column_stats[col] = col_stats
        summary = {
            "shape": df.shape,
            "missing": missing,
            "sample": sample,
            "column_types": column_types,
            "column_stats": column_stats
        }
        issues = [col for col in df.columns if df[col].isnull().any()]
        for col, col_type in column_types.items():
            if col_type == "date" and not pd.api.types.is_datetime64_any_dtype(df[col]):
                issues.append(f"{col} (date format)")
        if self.task_id == "hard":
            country_col = "country"
            if country_col in df.columns:
                unique_vals = set(df[country_col].dropna().astype(str).str.lower())
                if any(v not in ["usa", "united states", "u.s.a"] for v in unique_vals if v):
                    issues.append(f"{country_col} (inconsistent categories)")
        return Observation(
            task_id=self.task_id,
            description=TASKS[self.task_id]["description"],
            dataset_summary=summary,
            column_issues=issues
        )