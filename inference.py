import warnings
warnings.filterwarnings("ignore")
import json
import re
import numpy as np
import pandas as pd
from env.environment import DataCleanerEnv
from env.models import Action

def clean_column_name(issue, df):
    col = re.sub(r'\s*\(.*?\)', '', issue).strip()
    return col if col in df.columns else issue

def get_action(env, obs):
    df = env.state.data
    clean_issues = []
    for issue in obs.column_issues:
        col = clean_column_name(issue, df)
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

def run_task(env, task_id):
    max_steps = {"easy": 2, "medium": 1, "hard": 1}[task_id]
    print(f"\n{'='*60}\n🚀 Starting task: {task_id.upper()}")
    obs = env.reset(task_id)
    print(f"📝 Description: {obs.description}\n{'='*60}\n")
    done, step = False, 0
    while not done and step < max_steps:
        step += 1
        print(f"\n--- Step {step} ---")
        action = get_action(env, obs)
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