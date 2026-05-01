from __future__ import annotations
import pandas as pd


def build_daily_business_panel(sales: pd.DataFrame, web_traffic: pd.DataFrame | None = None, orders: pd.DataFrame | None = None) -> pd.DataFrame:
    df = sales.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["gross_profit"] = df["Revenue"] - df["COGS"]
    df["gross_margin_pct"] = df["gross_profit"] / df["Revenue"].replace(0, pd.NA)

    if web_traffic is not None:
        wt = web_traffic.copy()
        wt["date"] = pd.to_datetime(wt["date"])
        wt_daily = wt.groupby("date", as_index=False).agg(
            sessions=("sessions", "sum"),
            unique_visitors=("unique_visitors", "sum"),
            page_views=("page_views", "sum"),
            bounce_rate=("bounce_rate", "mean"),
            avg_session_duration_sec=("avg_session_duration_sec", "mean"),
        )
        df = df.merge(wt_daily, left_on="Date", right_on="date", how="left").drop(columns=["date"])

    if orders is not None:
        o = orders.copy()
        o["order_date"] = pd.to_datetime(o["order_date"])
        od = o.groupby("order_date", as_index=False).agg(
            n_orders=("order_id", "nunique"),
            cancelled_orders=("order_status", lambda s: (s == "cancelled").sum()),
        )
        od["cancel_rate"] = od["cancelled_orders"] / od["n_orders"].replace(0, pd.NA)
        df = df.merge(od, left_on="Date", right_on="order_date", how="left").drop(columns=["order_date"])

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    return df
