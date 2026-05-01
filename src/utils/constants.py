from pathlib import Path

RAW_FILES = [
    "products.csv", "customers.csv", "promotions.csv", "geography.csv",
    "orders.csv", "order_items.csv", "payments.csv", "shipments.csv",
    "returns.csv", "reviews.csv", "sales.csv", "sample_submission.csv",
    "inventory.csv", "web_traffic.csv",
]

DATE_COLUMNS = {
    "orders.csv": ["order_date"],
    "customers.csv": ["signup_date"],
    "promotions.csv": ["start_date", "end_date"],
    "shipments.csv": ["ship_date", "delivery_date"],
    "returns.csv": ["return_date"],
    "reviews.csv": ["review_date"],
    "sales.csv": ["Date"],
    "sample_submission.csv": ["Date"],
    "inventory.csv": ["snapshot_date"],
    "web_traffic.csv": ["date"],
}

PRIMARY_KEYS = {
    "products.csv": ["product_id"],
    "customers.csv": ["customer_id"],
    "promotions.csv": ["promo_id"],
    "geography.csv": ["zip"],
    "orders.csv": ["order_id"],
    "payments.csv": ["order_id"],
    "returns.csv": ["return_id"],
    "reviews.csv": ["review_id"],
}

DEFAULT_PROJECT_ROOT = Path(".")
