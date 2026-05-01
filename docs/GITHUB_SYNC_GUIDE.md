# GitHub sync guide — final EDA storyline

## Files to copy into repo

```text
src/eda/eda_utils.py
src/eda/final_story_charts.py
scripts/run_final_story_charts.py
scripts/collect_appendix_figures.py
notebooks/DATATHON_final_story.ipynb
outputs/figures/main/.gitkeep
outputs/figures/appendix_unused_for_main_report/README.md
requirements-final-eda.txt
```

## Run final figures

From repo root:

```bash
python scripts/run_final_story_charts.py --data-dir data --out-dir outputs/figures/main
```

The script searches these locations automatically:

```text
data/raw/
data/interim/
data/processed/
data/marts/
```

It prioritizes processed files if available:

```text
1_fact_order_item_enriched.csv
2_fact_order_enriched.csv
```

and falls back to raw files:

```text
sales.csv, orders.csv, order_items.csv, products.csv, payments.csv, inventory.csv
```

## Final main report figures

```text
outputs/figures/main/A1_overlay_revenue_margin_vi.png
outputs/figures/main/A2_customer_health_compact_vi.png
outputs/figures/main/A3_promo_refined_vi.png
outputs/figures/main/A5_sku_action_matrix_vi_no_summary.png
```

## Appendix figures

Use appendix only as supporting evidence. Do not overload the main report.

Suggested appendix folder:

```text
outputs/figures/appendix_unused_for_main_report/
```

Optional helper:

```bash
python scripts/collect_appendix_figures.py --source reports/figures --out outputs/figures/appendix_unused_for_main_report
```

## README snippet

Add this to README.md:

```markdown
### Reproduce final EDA figures

```bash
pip install -r requirements-final-eda.txt
python scripts/run_final_story_charts.py --data-dir data --out-dir outputs/figures/main
```

Main EDA storyline:

1. A1 — Doanh thu suy giảm mang tính cấu trúc, không chỉ mùa vụ.
2. A2 — Suy giảm đến từ customer base và repeat behavior, không phải AOV.
3. A3 — Khuyến mãi bảo vệ volume ngắn hạn nhưng không tạo profitable repeat.
4. A5 — Chuyển từ discount đại trà sang SKU action matrix cho tồn kho.
```
