from __future__ import annotations
from pathlib import Path
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def safe_divide(num, den, default=0.0):
    try:
        return num / den if den not in (0, None) else default
    except ZeroDivisionError:
        return default


def add_year_month(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["year_month"] = out[date_col].dt.to_period("M").astype(str)
    return out
