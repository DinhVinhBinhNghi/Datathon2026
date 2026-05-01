from __future__ import annotations
import pandas as pd


def time_train_valid_split(df: pd.DataFrame, date_col: str = "Date", valid_start: str = "2022-01-01"):
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    train = data[data[date_col] < pd.Timestamp(valid_start)].copy()
    valid = data[data[date_col] >= pd.Timestamp(valid_start)].copy()
    return train, valid


def rolling_origin_splits(df: pd.DataFrame, date_col: str = "Date", n_splits: int = 3, valid_days: int = 90):
    data = df.sort_values(date_col).copy()
    dates = pd.to_datetime(data[date_col])
    max_date = dates.max()
    for i in range(n_splits, 0, -1):
        valid_end = max_date - pd.Timedelta(days=(i - 1) * valid_days)
        valid_start = valid_end - pd.Timedelta(days=valid_days - 1)
        train_idx = dates < valid_start
        valid_idx = (dates >= valid_start) & (dates <= valid_end)
        yield data.index[train_idx].to_numpy(), data.index[valid_idx].to_numpy()
