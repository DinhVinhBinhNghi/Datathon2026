from __future__ import annotations
import pandas as pd


def build_order_line_fact(
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    payments: pd.DataFrame | None = None,
    shipments: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    reviews: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = order_items.merge(orders, on="order_id", how="left", validate="many_to_one")
    df = df.merge(products, on="product_id", how="left", validate="many_to_one", suffixes=("", "_product"))

    if payments is not None:
        pay_cols = [c for c in ["order_id", "payment_value", "installments"] if c in payments.columns]
        df = df.merge(payments[pay_cols], on="order_id", how="left", validate="many_to_one")

    if shipments is not None:
        ship = shipments.copy()
        for c in ["ship_date", "delivery_date"]:
            if c in ship.columns:
                ship[c] = pd.to_datetime(ship[c])
        if {"ship_date", "delivery_date"}.issubset(ship.columns):
            ship["delivery_lead_days"] = (ship["delivery_date"] - ship["ship_date"]).dt.days
        df = df.merge(ship[[c for c in ["order_id", "shipping_fee", "delivery_lead_days"] if c in ship.columns]], on="order_id", how="left")

    if returns is not None:
        ret = returns.groupby(["order_id", "product_id"], as_index=False).agg(
            return_quantity=("return_quantity", "sum"),
            refund_amount=("refund_amount", "sum"),
            return_records=("return_id", "nunique") if "return_id" in returns.columns else ("return_quantity", "size"),
        )
        df = df.merge(ret, on=["order_id", "product_id"], how="left")

    if reviews is not None:
        rev = reviews.groupby(["order_id", "product_id"], as_index=False).agg(
            avg_rating=("rating", "mean"), review_records=("rating", "size")
        )
        df = df.merge(rev, on=["order_id", "product_id"], how="left")

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["line_revenue"] = df["quantity"] * df["unit_price"]
    if "cogs" in df.columns:
        df["line_cogs"] = df["quantity"] * df["cogs"]
        df["line_gross_profit"] = df["line_revenue"] - df["line_cogs"]
        df["line_margin_pct"] = df["line_gross_profit"] / df["line_revenue"].replace(0, pd.NA)
    df["has_promo"] = df.get("promo_id", pd.Series(index=df.index, dtype=object)).notna()
    if "promo_id_2" in df.columns:
        df["has_double_promo"] = df["promo_id"].notna() & df["promo_id_2"].notna()
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    return df
