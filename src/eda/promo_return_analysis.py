from __future__ import annotations
import pandas as pd


def promo_summary(order_line_fact: pd.DataFrame) -> pd.DataFrame:
    df = order_line_fact.copy()
    return df.groupby("has_promo", as_index=False).agg(
        lines=("order_id", "size"),
        revenue=("line_revenue", "sum"),
        units=("quantity", "sum"),
        avg_line_revenue=("line_revenue", "mean"),
        margin_pct=("line_margin_pct", "mean") if "line_margin_pct" in df.columns else ("line_revenue", "mean"),
    )


def return_reason_summary(returns: pd.DataFrame, products: pd.DataFrame | None = None) -> pd.DataFrame:
    df = returns.copy()
    if products is not None and "category" not in df.columns:
        df = df.merge(products[["product_id", "category", "size"]], on="product_id", how="left")
    group_cols = ["return_reason"] + (["category"] if "category" in df.columns else [])
    return df.groupby(group_cols, as_index=False).agg(
        return_records=("return_quantity", "size"),
        return_units=("return_quantity", "sum"),
        refund_amount=("refund_amount", "sum"),
    ).sort_values("return_records", ascending=False)
