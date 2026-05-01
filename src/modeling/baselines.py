from __future__ import annotations
import pandas as pd


def seasonal_naive_forecast(train: pd.DataFrame, future_dates: pd.Series, target_col: str = "Revenue", date_col: str = "Date", lag_days: int = 365) -> pd.DataFrame:
    hist = train.copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    lookup = hist.set_index(date_col)[target_col]
    out = pd.DataFrame({date_col: pd.to_datetime(future_dates)})
    out[target_col] = out[date_col].map(lambda d: lookup.get(d - pd.Timedelta(days=lag_days), lookup.tail(28).mean()))
    return out


def moving_average_forecast(train: pd.DataFrame, future_dates: pd.Series, target_col: str = "Revenue", window: int = 28, date_col: str = "Date") -> pd.DataFrame:
    value = float(train.sort_values(date_col)[target_col].tail(window).mean())
    return pd.DataFrame({date_col: pd.to_datetime(future_dates), target_col: value})
