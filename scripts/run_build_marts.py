from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.load_data import load_raw_data
from pathlib import Path
from src.io.load_data import load_raw_data
from src.io.save_data import save_many
from src.joins.build_order_line_fact import build_order_line_fact
from src.joins.build_customer_mart import build_customer_mart
from src.joins.build_product_mart import build_product_mart
from src.joins.build_daily_business_panel import build_daily_business_panel

DATA_RAW = Path("data/raw")
DATA_INTERIM = Path("data/interim")
DATA_MARTS = Path("data/marts")


def main():
    d = load_raw_data(DATA_RAW)
    order_line = build_order_line_fact(
        d["order_items"], d["orders"], d["products"],
        payments=d.get("payments"), shipments=d.get("shipments"), returns=d.get("returns"), reviews=d.get("reviews")
    )
    customer = build_customer_mart(d["customers"], d["orders"], order_line)
    product = build_product_mart(d["products"], order_line, d.get("inventory"))
    daily = build_daily_business_panel(d["sales"], d.get("web_traffic"), d.get("orders"))
    save_many({"1_fact_order_item_enriched": order_line}, DATA_INTERIM)
    save_many({"3_customer_360": customer, "4_product_360": product, "5_daily_business_panel": daily}, DATA_MARTS)


if __name__ == "__main__":
    main()
