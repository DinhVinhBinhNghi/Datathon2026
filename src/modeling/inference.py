from __future__ import annotations
from pathlib import Path
import pandas as pd


def make_submission(predictions: pd.DataFrame, sample_submission: pd.DataFrame, revenue_col: str = "Revenue", cogs_col: str = "COGS") -> pd.DataFrame:
    sub = sample_submission.copy()
    sub["Date"] = pd.to_datetime(sub["Date"])
    pred = predictions.copy()
    pred["Date"] = pd.to_datetime(pred["Date"])
    sub = sub[["Date"]].merge(pred, on="Date", how="left")
    if revenue_col not in sub.columns:
        raise ValueError(f"Missing prediction column: {revenue_col}")
    if cogs_col not in sub.columns:
        sub[cogs_col] = sub[revenue_col] * 0.85
    return sub[["Date", revenue_col, cogs_col]]


def save_submission(submission: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = submission.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return path
