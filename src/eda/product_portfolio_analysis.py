from __future__ import annotations
import numpy as np
import pandas as pd


def category_summary(order_line_fact: pd.DataFrame) -> pd.DataFrame:
    return order_line_fact.groupby("category", as_index=False).agg(
        revenue=("line_revenue", "sum"),
        units=("quantity", "sum"),
        orders=("order_id", "nunique"),
        margin_pct=("line_margin_pct", "mean") if "line_margin_pct" in order_line_fact.columns else ("line_revenue", "mean"),
        promo_line_rate=("has_promo", "mean"),
    ).sort_values("revenue", ascending=False)


def sku_pareto(product_mart: pd.DataFrame, revenue_col: str = "revenue") -> pd.DataFrame:
    df = product_mart.sort_values(revenue_col, ascending=False).copy()
    total = df[revenue_col].sum()
    df["revenue_share"] = df[revenue_col] / total if total else 0
    df["cum_revenue_share"] = df["revenue_share"].cumsum()
    df["sku_rank"] = np.arange(1, len(df) + 1)
    df["abc_class"] = pd.cut(df["cum_revenue_share"], bins=[-0.01, 0.80, 0.95, 1.01], labels=["A", "B", "C"])
    return df
