from __future__ import annotations
import pandas as pd
from src.utils.constants import PRIMARY_KEYS


def duplicate_key_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for filename, keys in PRIMARY_KEYS.items():
        key = filename.replace(".csv", "")
        if key not in tables:
            continue
        df = tables[key]
        missing_cols = [c for c in keys if c not in df.columns]
        if missing_cols:
            rows.append({"table": key, "keys": keys, "status": "MISSING_COLUMNS", "duplicate_rows": None})
            continue
        dup = int(df.duplicated(keys).sum())
        rows.append({"table": key, "keys": ",".join(keys), "status": "PASS" if dup == 0 else "FAIL", "duplicate_rows": dup})
    return pd.DataFrame(rows)


def foreign_key_coverage(child: pd.DataFrame, parent: pd.DataFrame, child_key: str, parent_key: str) -> dict:
    child_non_null = child[child_key].dropna()
    parent_vals = set(parent[parent_key].dropna().unique())
    missing_mask = ~child_non_null.isin(parent_vals)
    missing = int(missing_mask.sum())
    return {
        "child_key": child_key,
        "parent_key": parent_key,
        "child_non_null_rows": int(len(child_non_null)),
        "missing_fk_rows": missing,
        "coverage_rate": 1 - missing / len(child_non_null) if len(child_non_null) else 1.0,
        "status": "PASS" if missing == 0 else "WARN",
    }


def relationship_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    checks = []
    pairs = [
        ("orders", "customers", "customer_id", "customer_id"),
        ("orders", "geography", "zip", "zip"),
        ("order_items", "orders", "order_id", "order_id"),
        ("order_items", "products", "product_id", "product_id"),
        ("payments", "orders", "order_id", "order_id"),
        ("shipments", "orders", "order_id", "order_id"),
        ("returns", "orders", "order_id", "order_id"),
        ("returns", "products", "product_id", "product_id"),
        ("reviews", "orders", "order_id", "order_id"),
        ("reviews", "products", "product_id", "product_id"),
        ("reviews", "customers", "customer_id", "customer_id"),
        ("inventory", "products", "product_id", "product_id"),
    ]
    for child_name, parent_name, child_key, parent_key in pairs:
        if child_name in tables and parent_name in tables and child_key in tables[child_name] and parent_key in tables[parent_name]:
            row = foreign_key_coverage(tables[child_name], tables[parent_name], child_key, parent_key)
            row.update({"child_table": child_name, "parent_table": parent_name})
            checks.append(row)
    return pd.DataFrame(checks)
