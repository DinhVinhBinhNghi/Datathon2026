from __future__ import annotations
import pandas as pd


def build_product_mart(products: pd.DataFrame, order_line_fact: pd.DataFrame, inventory: pd.DataFrame | None = None) -> pd.DataFrame:
    agg = order_line_fact.groupby("product_id", as_index=False).agg(
        units_sold=("quantity", "sum"),
        revenue=("line_revenue", "sum"),
        orders=("order_id", "nunique"),
        promo_line_rate=("has_promo", "mean"),
    )
    if "return_quantity" in order_line_fact.columns:
        ret = order_line_fact.groupby("product_id", as_index=False).agg(return_units=("return_quantity", "sum"))
        agg = agg.merge(ret, on="product_id", how="left")
        agg["return_units"] = agg["return_units"].fillna(0)
        agg["unit_return_rate"] = agg["return_units"] / agg["units_sold"].replace(0, pd.NA)
    out = products.merge(agg, on="product_id", how="left")
    for c in ["units_sold", "revenue", "orders", "promo_line_rate", "return_units", "unit_return_rate"]:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    out["gross_margin_pct"] = (out["price"] - out["cogs"]) / out["price"]

    if inventory is not None:
        inv = inventory.groupby("product_id", as_index=False).agg(
            avg_stockout_days=("stockout_days", "mean"),
            stockout_months=("stockout_flag", "sum"),
            avg_days_of_supply=("days_of_supply", "mean"),
            avg_fill_rate=("fill_rate", "mean"),
        )
        out = out.merge(inv, on="product_id", how="left")
    return out
