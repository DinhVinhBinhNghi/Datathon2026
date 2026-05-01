from __future__ import annotations
import pandas as pd


def daily_traffic_features(web_traffic: pd.DataFrame) -> pd.DataFrame:
    wt = web_traffic.copy()
    wt["date"] = pd.to_datetime(wt["date"])
    out = wt.groupby("date", as_index=False).agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_session_duration_sec=("avg_session_duration_sec", "mean"),
        n_traffic_sources=("traffic_source", "nunique") if "traffic_source" in wt.columns else ("sessions", "size"),
    )
    out["views_per_session"] = out["page_views"] / out["sessions"].replace(0, pd.NA)
    out["visitors_per_session"] = out["unique_visitors"] / out["sessions"].replace(0, pd.NA)
    return out.rename(columns={"date": "Date"})
