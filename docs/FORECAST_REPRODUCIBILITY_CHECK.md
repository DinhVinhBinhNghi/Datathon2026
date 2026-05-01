# Forecasting reproducibility and rule check

This note documents the final Part 3 pipeline against the competition constraints.

## Format check

Final submission path:

```text
submissions/submission.csv
```

Expected format:

```text
Date,Revenue,COGS
```

The script `scripts/run_forecast_final.py` checks:

- row count equals `sample_submission.csv`
- column order is exactly `Date, Revenue, COGS`
- date order matches `sample_submission.csv`
- predictions are non-null
- predictions are non-negative

## Leakage check

The forecasting pipeline uses:

- `sales.csv` as the only source of historical target values
- `sample_submission.csv` only for future dates and required row order
- operational/history-side files: `inventory.csv`, `web_traffic.csv`, `returns.csv`, `shipments.csv`, `promotions.csv`

The model does not use hidden test `Revenue` or hidden test `COGS` as features.

## Explainability output

The final repository contains grouped SHAP explanation artifacts at:

```text
outputs/modeling/shap_group_comparison.csv
outputs/modeling/shap_group_comparison.png
```

These files are used in the report to explain the main business drivers learned by the model. The values are grouped absolute SHAP contributions aggregated into five business-readable feature groups.
