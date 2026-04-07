import pandas as pd

NORMALIZATION_MAP = {
    "usa": "USA",
    "u.s.a": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "us": "USA",
    "us.": "USA"
}

TASKS = {
    "easy": {"description": "Fill missing numeric values using median imputation.", "target_file": "data/easy_target.csv"},
    "medium": {"description": "Convert date columns to standard YYYY-MM-DD format.", "target_file": "data/medium_target.csv"},
    "hard": {"description": "Normalize inconsistent country names to 'USA'.", "target_file": "data/hard_target.csv"}
}

def grader_easy(df: pd.DataFrame) -> float:
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    return 1.0 - (total_missing / total_cells) if total_cells else 0.0

def grader_medium(df: pd.DataFrame) -> float:
    col = 'date'
    if col not in df.columns:
        return 0.0
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return 1.0
    parsed = pd.to_datetime(df[col], errors='coerce')
    return parsed.notna().mean()

def grader_hard(df: pd.DataFrame) -> float:
    col = 'country'
    if col not in df.columns:
        return 0.0
    cleaned = df[col].astype(str).str.upper().str.strip()
    correct = (cleaned == "USA") | (cleaned == "UNITED STATES") | (cleaned == "U.S.A")
    return correct.mean()

def get_grader(task_id: str):
    mapping = {"easy": grader_easy, "medium": grader_medium, "hard": grader_hard}
    return mapping.get(task_id, lambda df: 0.0)
