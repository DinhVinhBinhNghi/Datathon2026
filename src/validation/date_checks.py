from __future__ import annotations
import pandas as pd


def date_range_report(tables: dict[str, pd.DataFrame], date_cols: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for table, cols in date_cols.items():
        key = table.replace(".csv", "")
        if key not in tables:
            continue
        df = tables[key]
        for col in cols:
            if col not in df.columns:
                continue
            s = pd.to_datetime(df[col], errors="coerce")
            rows.append({
                "table": key,
                "date_col": col,
                "min_date": s.min(),
                "max_date": s.max(),
                "null_dates": int(s.isna().sum()),
                "rows": int(len(df)),
            })
    return pd.DataFrame(rows)


def assert_no_future_dates(df: pd.DataFrame, date_col: str, max_date: str = "2024-07-01") -> bool:
    s = pd.to_datetime(df[date_col], errors="coerce")
    return bool((s <= pd.Timestamp(max_date)).all())
