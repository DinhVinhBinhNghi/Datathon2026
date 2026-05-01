from __future__ import annotations
import pandas as pd


def build_customer_mart(customers: pd.DataFrame, orders: pd.DataFrame, order_line_fact: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    order_revenue = order_line_fact.groupby("order_id", as_index=False).agg(order_revenue=("line_revenue", "sum"))
    o = orders.merge(order_revenue, on="order_id", how="left")
    agg = o.groupby("customer_id", as_index=False).agg(
        n_orders=("order_id", "nunique"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
        total_revenue=("order_revenue", "sum"),
        avg_order_value=("order_revenue", "mean"),
    )
    agg["active_days"] = (agg["last_order_date"] - agg["first_order_date"]).dt.days + 1
    agg["orders_per_active_day"] = agg["n_orders"] / agg["active_days"].replace(0, pd.NA)
    out = customers.merge(agg, on="customer_id", how="left")
    out["n_orders"] = out["n_orders"].fillna(0).astype(int)
    out["total_revenue"] = out["total_revenue"].fillna(0)
    return out
