---
title: DataCleaner Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# DataCleaner Environment

> An OpenEnv‑compliant environment where AI agents learn to clean messy datasets – a critical real‑world task.

## 🎯 Why This Environment?
Data cleaning consumes 80% of data scientists' time. This environment trains agents to automate it.

## 🔧 Tasks & Difficulty
| Task   | Description                                      | Grading Metric                     |
|--------|--------------------------------------------------|------------------------------------|
| Easy   | Fill missing numeric values using median imputation | 1 – (missing cells / total cells) |
| Medium | Convert date column to standard YYYY‑MM‑DD format | Proportion of valid dates          |
| Hard   | Normalize inconsistent country names to "USA"    | Proportion of rows with "USA"      |

## 🚀 Getting Started
```bash
pip install -r requirements.txt
python inference.py"# datacleaner-env" 
