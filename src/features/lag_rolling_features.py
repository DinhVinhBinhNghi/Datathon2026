from __future__ import annotations
import pandas as pd


def add_lag_features(df: pd.DataFrame, target_col: str = "Revenue", lags: tuple[int, ...] = (1, 7, 14, 28)) -> pd.DataFrame:
    out = df.sort_values("Date").copy()
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, target_col: str = "Revenue", windows: tuple[int, ...] = (7, 14, 28)) -> pd.DataFrame:
    out = df.sort_values("Date").copy()
    for w in windows:
        shifted = out[target_col].shift(1)
        out[f"{target_col}_roll_mean_{w}"] = shifted.rolling(w).mean()
        out[f"{target_col}_roll_std_{w}"] = shifted.rolling(w).std()
        out[f"{target_col}_roll_min_{w}"] = shifted.rolling(w).min()
        out[f"{target_col}_roll_max_{w}"] = shifted.rolling(w).max()
    return out
