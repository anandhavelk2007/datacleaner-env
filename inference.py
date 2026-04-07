import warnings
warnings.filterwarnings("ignore")

import os
import json
import re
import sys
import numpy as np
import pandas as pd
from openai import OpenAI
from env.environment import DataCleanerEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

def deterministic_action(env, obs):
    df = env.state.data
    clean_issues = []
    for issue in obs.column_issues:
        col = re.sub(r'\s*\(.*?\)', '', issue).strip()
        if col in df.columns:
            clean_issues.append(col)
    if not clean_issues:
        clean_issues = list(df.columns)
    if "date" in clean_issues:
        return Action(type="fix_date", column="date")
    if "country" in clean_issues:
        return Action(type="normalize_cat", column="country", target="USA")
    if "age" in clean_issues or "salary" in clean_issues:
        col = clean_issues[0]
        return Action(type="impute", column=col, method="median")
    return Action(type="skip")

def call_openai(prompt):
    if not client:
        return None
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None

def parse_action_from_response(response_text):
    if not response_text:
        return None
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end != 0:
            action_dict = json.loads(response_text[start:end])
            action_type = action_dict.get("type")
            if action_type in ["impute", "fix_date", "normalize_cat", "skip"]:
                return Action(
                    type=action_type,
                    column=action_dict.get("column"),
                    method=action_dict.get("method"),
                    target=action_dict.get("target")
                )
    except:
        pass
    return None

def run_task(env, task_id):
    max_steps = {"easy": 2, "medium": 1, "hard": 1}[task_id]
    print(f"[START] task={task_id}", flush=True)
    obs = env.reset(task_id)
    done, step = False, 0
    while not done and step < max_steps:
        step += 1
        prompt = f"""
Task: {obs.description}
Current dataset issues: {obs.column_issues}
Choose an action and output it as JSON.
Example: {{"type": "impute", "column": "age", "method": "median"}}
Action:"""
        action = None
        if client:
            resp_text = call_openai(prompt)
            if resp_text:
                action = parse_action_from_response(resp_text)
        if action is None:
            action = deterministic_action(env, obs)
        obs, reward, done, info = env.step(action)
        print(f"[STEP] task={task_id} step={step} reward={reward.value:.3f}", flush=True)
    final_score = info.get("score", 0.0)
    print(f"[END] task={task_id} score={final_score:.3f} steps={step}", flush=True)
    return final_score

def main():
    env = DataCleanerEnv()
    for task in ["easy", "medium", "hard"]:
        run_task(env, task)

if __name__ == "__main__":
    main()