from __future__ import annotations
import pandas as pd


def null_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        for col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                rows.append({"table": name, "column": col, "null_count": n, "null_rate": n / len(df)})
    return pd.DataFrame(rows).sort_values(["table", "null_rate"], ascending=[True, False]) if rows else pd.DataFrame(columns=["table","column","null_count","null_rate"])


def product_margin_rule(products: pd.DataFrame) -> pd.DataFrame:
    out = products.copy()
    out["gross_margin_pct"] = (out["price"] - out["cogs"]) / out["price"]
    out["rule_cogs_lt_price"] = out["cogs"] < out["price"]
    return out[["product_id", "price", "cogs", "gross_margin_pct", "rule_cogs_lt_price"]]


def order_payment_reconciliation(orders: pd.DataFrame, payments: pd.DataFrame) -> dict:
    order_ids = set(orders["order_id"])
    payment_ids = set(payments["order_id"])
    return {
        "orders": len(order_ids),
        "payments": len(payment_ids),
        "orders_without_payment": len(order_ids - payment_ids),
        "payments_without_order": len(payment_ids - order_ids),
        "status": "PASS" if order_ids == payment_ids else "WARN",
    }


def shipment_status_rule(orders: pd.DataFrame, shipments: pd.DataFrame) -> pd.DataFrame:
    eligible_status = {"shipped", "delivered", "returned"}
    tmp = shipments[["order_id"]].drop_duplicates().merge(
        orders[["order_id", "order_status"]], on="order_id", how="left"
    )
    tmp["allowed_status"] = tmp["order_status"].isin(eligible_status)
    return tmp
