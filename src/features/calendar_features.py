from __future__ import annotations
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    d = out[date_col]
    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["day"] = d.dt.day
    out["dayofweek"] = d.dt.dayofweek
    out["dayofyear"] = d.dt.dayofyear
    out["weekofyear"] = d.dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["quarter"] = d.dt.quarter
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    return out
