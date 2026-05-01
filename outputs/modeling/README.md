# Modeling explanation outputs

This folder contains pre-generated explainability artifacts used in the report.

- `shap_group_comparison.png`: grouped SHAP contribution comparison for Model A (2013-2018) and Model B (2019-2022).
- `shap_group_comparison.csv`: values used in the grouped SHAP figure.
- `feature_group_importance_comparison.png` and `.csv`: backward-compatible copies of the same report-ready artifact.

The report uses grouped absolute SHAP contribution (`|SHAP|`) aggregated into 5 business-readable feature groups: Seasonal medians, Data-derived flags, Calendar basics, Fourier, and AUX aggregates. No external data or hidden test Revenue/COGS values are used.
