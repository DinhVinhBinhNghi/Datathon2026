from __future__ import annotations
import pandas as pd


def revenue_kpis(daily: pd.DataFrame) -> dict:
    return {
        "total_revenue": float(daily["Revenue"].sum()),
        "total_cogs": float(daily["COGS"].sum()),
        "gross_profit": float((daily["Revenue"] - daily["COGS"]).sum()),
        "gross_margin_pct": float((daily["Revenue"].sum() - daily["COGS"].sum()) / daily["Revenue"].sum()),
        "start_date": str(pd.to_datetime(daily["Date"]).min().date()),
        "end_date": str(pd.to_datetime(daily["Date"]).max().date()),
    }


def monthly_revenue_summary(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["year_month"] = df["Date"].dt.to_period("M").astype(str)
    return df.groupby("year_month", as_index=False).agg(
        revenue=("Revenue", "sum"), cogs=("COGS", "sum"), avg_daily_revenue=("Revenue", "mean")
    )
