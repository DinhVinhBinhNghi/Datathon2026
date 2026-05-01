from __future__ import annotations
import pandas as pd


def monthly_inventory_features(inventory: pd.DataFrame) -> pd.DataFrame:
    inv = inventory.copy()
    inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"])
    inv["year_month"] = inv["snapshot_date"].dt.to_period("M").astype(str)
    out = inv.groupby("year_month", as_index=False).agg(
        stockout_days=("stockout_days", "sum"),
        stockout_product_rate=("stockout_flag", "mean"),
        overstock_product_rate=("overstock_flag", "mean"),
        reorder_product_rate=("reorder_flag", "mean"),
        avg_days_of_supply=("days_of_supply", "mean"),
        avg_fill_rate=("fill_rate", "mean"),
        total_units_sold_inventory=("units_sold", "sum"),
    )
    return out


def add_inventory_month_to_daily(daily: pd.DataFrame, monthly_inv: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = daily.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["year_month"] = out[date_col].dt.to_period("M").astype(str)
    return out.merge(monthly_inv, on="year_month", how="left")
