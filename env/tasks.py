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
    """Score = 1 - (missing cells / total cells), clamped to (0,1)."""
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    if total_cells == 0:
        return 0.001
    raw_score = 1.0 - (total_missing / total_cells)
    # Clamp to (0,1) exclusive
    if raw_score <= 0.0:
        return 0.001
    if raw_score >= 1.0:
        return 0.999
    return raw_score

def grader_medium(df: pd.DataFrame) -> float:
    """Score based on how many dates are correctly parsed, clamped to (0,1)."""
    col = 'date'
    if col not in df.columns:
        return 0.001
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return 0.999
    parsed = pd.to_datetime(df[col], errors='coerce')
    valid = parsed.notna()
    raw_score = valid.mean()
    if raw_score <= 0.0:
        return 0.001
    if raw_score >= 1.0:
        return 0.999
    return raw_score

def grader_hard(df: pd.DataFrame) -> float:
    """Check proportion of normalized country names, clamped to (0,1)."""
    col = 'country'
    if col not in df.columns:
        return 0.001
    cleaned = df[col].astype(str).str.upper().str.strip()
    correct = (cleaned == "USA") | (cleaned == "UNITED STATES") | (cleaned == "U.S.A")
    raw_score = correct.mean()
    if raw_score <= 0.0:
        return 0.001
    if raw_score >= 1.0:
        return 0.999
    return raw_score

def get_grader(task_id: str):
    mapping = {"easy": grader_easy, "medium": grader_medium, "hard": grader_hard}
    return mapping.get(task_id, lambda df: 0.001)