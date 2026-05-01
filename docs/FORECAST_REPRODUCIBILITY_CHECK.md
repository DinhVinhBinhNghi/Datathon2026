# Forecasting reproducibility and rule check

This note documents the final Part 3 pipeline against the competition constraints.

## Format check

Final submission path:

```text
submissions/submission_634K.csv
```

Expected format:

```text
Date,Revenue,COGS
```

The script `scripts/run_forecast_634k.py` checks:

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

The final run writes feature importance to:

```text
outputs/modeling/feature_importance_634K.csv
```

This file can be referenced in the report when explaining the main business drivers learned by the model.
