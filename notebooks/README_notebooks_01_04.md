# Completed Datathon notebooks 01–04

Copy these files into `datathon-2026/notebooks/`:

```text
01_data_audit.ipynb
02_join_validation.ipynb
03_eda_main_storyline.ipynb
04_eda_supporting_evidence.ipynb
```

Recommended run order:

1. `01_data_audit.ipynb` — audit raw data, nulls, keys, business rules.
2. `02_join_validation.ipynb` — validate joins and build `data/interim/` + `data/marts/`.
3. `03_eda_main_storyline.ipynb` — main EDA figures and business storyline for report.
4. `04_eda_supporting_evidence.ipynb` — extra evidence for GitHub/appendix.

Expected repo folders:

```text
data/raw/          # original CSV files
data/interim/      # generated joined tables
data/marts/        # generated analytical marts
reports/figures/   # saved PNG charts
reports/tables/    # saved CSV summary tables
```

The notebooks also include a `/mnt/data` fallback so they can run inside this ChatGPT/Colab-style environment while you are testing.
