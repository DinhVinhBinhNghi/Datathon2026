from __future__ import annotations
import pandas as pd


def daily_promo_features(order_line_fact: pd.DataFrame) -> pd.DataFrame:
    df = order_line_fact.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    out = df.groupby("order_date", as_index=False).agg(
        promo_line_rate=("has_promo", "mean"),
        promo_revenue=("line_revenue", lambda s: s[df.loc[s.index, "has_promo"]].sum()),
        total_revenue_items=("line_revenue", "sum"),
        total_discount=("discount_amount", "sum") if "discount_amount" in df.columns else ("line_revenue", lambda s: 0),
    )
    out["promo_revenue_share"] = out["promo_revenue"] / out["total_revenue_items"].replace(0, pd.NA)
    return out.rename(columns={"order_date": "Date"})
