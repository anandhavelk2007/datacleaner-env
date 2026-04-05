import warnings
warnings.filterwarnings("ignore")

import os
import json
import re
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from env.environment import DataCleanerEnv
from env.models import Action

# Environment variables (required by hackathon)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN", "")   # No default – must be set by user

# Initialize OpenAI client (will be used if HF_TOKEN is valid)
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

def deterministic_action(env, obs):
    """Fallback deterministic action (same as before) – used when API fails."""
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
    """Call OpenAI API using the client; returns action string or None."""
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
    """Parse JSON action from API response; fallback to deterministic on failure."""
    if not response_text:
        return None
    try:
        # Try to extract JSON
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
    print(f"\n{'='*60}\n🚀 Starting task: {task_id.upper()}")
    obs = env.reset(task_id)
    print(f"📝 Description: {obs.description}\n{'='*60}\n")
    done, step = False, 0
    while not done and step < max_steps:
        step += 1
        print(f"\n--- Step {step} ---")
        # Build prompt for LLM
        prompt = f"""
Task: {obs.description}
Current dataset issues: {obs.column_issues}
Choose an action and output it as JSON.
Example: {{"type": "impute", "column": "age", "method": "median"}}
Action:"""
        # Try OpenAI first
        action = None
        if client:
            resp_text = call_openai(prompt)
            if resp_text:
                action = parse_action_from_response(resp_text)
        # Fallback to deterministic if OpenAI failed or returned invalid
        if action is None:
            action = deterministic_action(env, obs)
            print(f"⚠️ API fallback – using deterministic action")
        print(f"🤖 Agent action: {action.type}", end="")
        if action.column: print(f" on column '{action.column}'", end="")
        if action.method: print(f" using {action.method}", end="")
        if action.target: print(f" target '{action.target}'", end="")
        print()
        obs, reward, done, info = env.step(action)
        print(f"💰 Reward: {reward.value:.3f} - {reward.reason}")
        print(f"📊 Current score: {info['score']:.3f}")
        df = env.state.data
        print(f"📄 Data shape: {df.shape}")
        if not df.empty:
            print("Sample rows:")
            print(df.head(2).to_string())
    print(f"\n✅ Task {task_id} finished. Final score: {info['score']:.3f} in {step} steps.")
    return info['score']

def main():
    env = DataCleanerEnv()
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_task(env, task)
    print("\n" + "="*60)
    print("🏆 FINAL SCORES")
    print("="*60)
    for t, s in scores.items():
        print(f"{t.capitalize()}: {s:.3f}")
    with open("baseline_scores.txt", "w") as f:
        json.dump(scores, f, indent=2)

if __name__ == "__main__":
    main()