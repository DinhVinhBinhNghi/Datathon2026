# Datathon 2026 Round 1 — The Gridbreakers

Repository for the Datathon 2026 Round 1 submission. It contains the EDA/storytelling code, final figures, forecasting pipeline, and final Kaggle submission.

## What is included

- `notebooks/`: audit, join validation, EDA storyline, and final forecasting notebook.
- `src/`: reusable code for data loading, validation, feature engineering, EDA, and modeling.
- `scripts/`: command-line scripts to reproduce figures and the final forecasting submission.
- `outputs/figures/main/`: final EDA figures used in the report.
- `outputs/modeling/`: model explanation outputs such as grouped SHAP importance.
- `submissions/submission.csv`: final Kaggle submission file.
- `data/raw/`: local copy of the official competition files for reproducibility. This folder is ignored by Git to avoid pushing data accidentally.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Required data files

Place the official competition CSV files under:

```text
data/raw/
```

Required for the final forecasting pipeline:

```text
sales.csv
sample_submission.csv
inventory.csv
web_traffic.csv
returns.csv
shipments.csv
promotions.csv
```

## Reproduce the final forecasting submission

From the repository root:

```bash
python scripts/run_forecast_final.py --data-dir data/raw --out submissions/submission.csv
```

This writes:

```text
submissions/submission.csv
outputs/modeling/shap_group_comparison.csv
outputs/modeling/shap_group_comparison.png
```

The script validates that the output has the required columns, row count, date order, non-null predictions, and non-negative predictions.

## Reproduce model explainability figure

```bash
python scripts/run_feature_importance_figure.py --csv outputs/modeling/shap_group_comparison.csv --out outputs/modeling/shap_group_comparison.png
```

This lightweight script recreates the report-ready grouped SHAP figure from the saved table; it does not retrain the full model.

## Reproduce EDA figures

```bash
python scripts/run_final_story_charts.py
```

The main report figures are stored in:

```text
outputs/figures/main/
```

## Final forecasting model summary

The final model is an ensemble of LightGBM, XGBoost, and CatBoost regressors. It uses 33 features grouped into calendar basics, data-derived promotion flags, Fourier seasonality, seasonal profiles/medians, and auxiliary operational signals from inventory, web traffic, returns, and shipments. Revenue prediction combines 58% model signal with 42% seasonal profiles. COGS is modeled through margin estimation and targeted COGS post-processing based only on historical training data patterns.

## Leakage and reproducibility notes

- The model does **not** use actual `Revenue` or `COGS` from the hidden test period.
- All training targets come from `sales.csv` only.
- Auxiliary features are created from the official competition files listed above.
- The final submission preserves the exact row order of `sample_submission.csv`.
- Random seeds are fixed in the modeling script.

## GitHub note

The `.gitignore` excludes `data/` so the repository can be pushed safely without large local CSVs. Before judging or reproduction, place the official CSVs back into `data/raw/` and run the commands above.
