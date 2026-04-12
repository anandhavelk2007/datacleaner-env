---
title: DataCleaner Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "1.0"
pinned: false
license: mit
---

# 🧹 DataCleaner Environment

[![Open In Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/Anandh-235/datacleaner-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A real‑world reinforcement learning environment where an AI agent learns to clean messy datasets – a task that consumes 80% of a data scientist’s time.**

## 📌 Why This Environment?

Data cleaning is the foundation of every data‑driven project. Yet it remains largely manual, tedious, and error‑prone. This environment provides a **controlled, reproducible, and challenging testbed** for training and evaluating agents on realistic data cleaning tasks – bridging the gap between AI research and practical application.

## 🎯 Problem Addressed

| **Problem** | **Impact** |
|-------------|-------------|
| Missing values | Skewed analysis, broken pipelines |
| Inconsistent date formats | Parsing errors, silent failures |
| Unstandardised categorical values | Duplicate entries, wrong aggregations |

The agent must learn to **detect** issues and **apply** the right cleaning operation (impute, fix date, normalise category) to maximise a dense reward signal.

## 🧠 Architecture
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ AI Agent │ │ DataCleanerEnv │ │ Dataset │
│ (OpenAI Client)│────▶│ (OpenEnv spec) │────▶│ (CSV) │
└─────────────────┘ └──────────────────┘ └─────────────────┘
│ │ │
│ action │ reward / observation │
│ │ │
▼ ▼ ▼
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ AI Helper │ │ Reward Engine │ │ Grader │
│ (optional) │ │ (partial │ │ (score 0–1) │
└─────────────────┘ │ progress) │ └─────────────────┘
└──────────────────┘


## 📊 Tasks & Difficulty

| Task   | Description                                      | Grading Metric                     | Difficulty |
|--------|--------------------------------------------------|------------------------------------|------------|
| **Easy**   | Fill missing numeric values using median imputation | 1 – (missing cells / total cells) | ⭐ |
| **Medium** | Convert date column to standard YYYY‑MM‑DD format | Proportion of valid dates          | ⭐⭐ |
| **Hard**   | Normalize inconsistent country names to "USA"    | Proportion of rows with "USA"      | ⭐⭐⭐ |

## 🔍 Real-World Relevance

Data cleaning is not just an academic problem:

- Data scientists spend **~80% of their time** cleaning data  
- Poor data quality costs businesses **billions annually**  
- Inconsistent preprocessing leads to unreliable ML models  

This environment simulates **real-world data issues** such as:
- Missing values in numeric columns  
- Messy date formats from different sources  
- Noisy categorical labels from user input  

👉 Making it highly relevant for:
- Machine Learning pipelines  
- Data Engineering workflows  
- AI automation systems  

## 🧪 Agent Workflow (Step‑by‑Step)

1. **Reset** – Load the raw dataset for a task (easy/medium/hard).
2. **Observe** – Receive a structured observation containing:
   - Dataset summary (shape, missing counts, sample rows)
   - Detected column types (numeric, date, categorical)
   - Column‑wise statistics (mean, std, unique count)
   - List of columns that still have issues.
3. **Act** – Choose one of four actions:
   - `impute(column, method)` – fill missing values with mean/median/mode.
   - `fix_date(column)` – convert column to datetime (flexible parsing).
   - `normalize_cat(column, target)` – replace all values in a categorical column with a target string.
   - `skip` – do nothing.
4. **Reward** – The environment:
   - Computes improvement in the task‑specific grader score.
   - Gives **bonus** for correct column/method selection (using an AIHelper).
   - Applies **penalty** for repeating the same action on the same column.
   - Adds **completion bonus** when score ≥ 0.99.
5. **Repeat** until task is completed or step limit reached.

## 💡 Why Reinforcement Learning?

Data cleaning is a **sequential decision‑making problem** – each action changes the dataset and affects future actions. RL allows an agent to learn optimal cleaning strategies through trial and error, balancing immediate improvements (e.g., imputing a single column) against long‑term data quality.

## 🌟 Novelty & Research Gap

While prior research has explored reinforcement learning for data cleaning, existing approaches are limited in scope and usability.

For example, systems like :contentReference[oaicite:0]{index=0} apply RL to data preprocessing tasks, and frameworks such as :contentReference[oaicite:1]{index=1} focus on domain-specific pipelines.

However, these approaches:
- Do not provide a **standardized Gym/OpenEnv-style environment**
- Lack **step-by-step interactive agent training**
- Are not designed as reusable **benchmark environments**
- Do not expose clear **observation-action-reward interfaces**

To the best of our knowledge, there are **no widely available RL environments for data cleaning tasks** that follow a structured, reproducible API.

👉 **This project fills that gap** by introducing a fully interactive, OpenEnv-compliant environment where agents learn data cleaning through sequential decision-making and reward feedback.

This makes the environment:
- Extensible for future research  
- Compatible with RL libraries (Stable-Baselines, RLlib)  
- Suitable as a **benchmark for AI agents in real-world data tasks**

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Docker (optional)
- Hugging Face account (for deployment)

### Installation
```bash
git clone https://github.com/anandhavelk2007/datacleaner-env.git
cd datacleaner-env
python -m venv venv
source venv/bin/activate      # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

Run Baseline Inference

HF_TOKEN=your_openai_api_key   # optional – fallback works without
python inference.py

Expected output:

[START] task=easy
[STEP] task=easy step=1 reward=0.267
[STEP] task=easy step=2 reward=1.233
[END] task=easy score=1.000 steps=2
[START] task=medium
[STEP] task=medium step=1 reward=1.300
[END] task=medium score=1.000 steps=1
[START] task=hard
[STEP] task=hard step=1 reward=0.000
[END] task=hard score=0.800 steps=1

Test the API Server Locally
python server/app.py
Then another terminal:

curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": {"type": "impute", "column": "age", "method": "median"}}'
curl http://localhost:7860/state

Deploy to Hugging Face Spaces

openenv push --repo-id Anandh-235/datacleaner-env

📈 Baseline Scores (GPT‑4 + Fallback)
Task	Score
Easy	1.000
Medium	1.000
Hard	0.800
These scores are reproducible and demonstrate that the environment is solvable by current LLMs and even by simple deterministic rules.

🧰 Built With
OpenEnv – Framework RL environments.

FastAPI – API server.

Hugging Face Spaces – Hosting.

Docker – Containerisation.

Pandas / NumPy – Data manipulation.

📁 Repository Structure

datacleaner-env/
├── env/                 # Core environment classes
├── server/              # FastAPI server (OpenEnv multi‑mode)
├── tests/               # Unit tests for graders and environment
├── optional_ui/         # Streamlit UI for interactive demos
├── inference.py         # Baseline inference script (structured output)
├── Dockerfile           # Container definition
├── openenv.yaml         # OpenEnv metadata
├── pyproject.toml       # Project configuration
├── uv.lock              # Dependency lock file
└── README.md            # This file

🏆 Why This Project Stands Out

✔ Real-world problem (not synthetic toy environment)  
✔ Fully OpenEnv-compliant (passes validation)  
✔ Dense reward system (better learning signal)  
✔ Multi-step decision making (true RL problem)  
✔ Extensible design (easy to add new tasks/actions)  

👉 Unlike typical hackathon projects, this is not just a model —  
👉 it is a complete environment training future AI systems.

🙏 Acknowledgements
Meta, PyTorch, Hugging Face, and Scaler School of Technology organising the hackathon.
The OpenEnv community the excellent framework.

📜 License
MIT © Anandhavel K
