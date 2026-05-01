from __future__ import annotations
import pandas as pd


def build_sku_action_matrix(product_mart: pd.DataFrame) -> pd.DataFrame:
    df = product_mart.copy()
    rev_med = df["revenue"].median() if "revenue" in df.columns else 0
    stock_med = df["avg_stockout_days"].median() if "avg_stockout_days" in df.columns else 0

    def label(row):
        high_rev = row.get("revenue", 0) >= rev_med
        high_stockout = row.get("avg_stockout_days", 0) >= stock_med
        if high_rev and high_stockout:
            return "Priority Replenish"
        if high_rev and not high_stockout:
            return "Protect Winners"
        if not high_rev and high_stockout:
            return "Selective Reorder"
        return "Reduce/Markdown"

    df["sku_action"] = df.apply(label, axis=1)
    return df


def action_summary(action_matrix: pd.DataFrame) -> pd.DataFrame:
    return action_matrix.groupby("sku_action", as_index=False).agg(
        sku_count=("product_id", "nunique"),
        revenue=("revenue", "sum"),
        avg_stockout_days=("avg_stockout_days", "mean") if "avg_stockout_days" in action_matrix.columns else ("revenue", "mean"),
    ).sort_values("revenue", ascending=False)
