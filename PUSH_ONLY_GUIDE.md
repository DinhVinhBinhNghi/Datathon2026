# Push-only guide

This bundle is already prepared for GitHub submission. You do not need to install Python packages or rerun the model before pushing.

## What is already included

- Final forecasting notebook: `notebooks/05_sales_forecast_final.ipynb`
- Final forecasting script: `scripts/run_forecast_final.py`
- Final submission file: `submissions/submission.csv`
- Pre-generated explainability figure: `outputs/modeling/feature_group_importance_comparison.png`
- README and reproducibility notes

## Push to GitHub from VS Code terminal

```powershell
git init
git add .
git status
git commit -m "Finalize Datathon Round 1 repository"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

Replace `<username>/<repo-name>` with your actual GitHub repository.

## Notes

- The `data/` folder is ignored by `.gitignore`, so raw CSV files will not be pushed. This keeps the repository lightweight.
- The final submission is kept in `submissions/submission.csv`, which will be pushed.
- The explainability figure is already included, so the report can reference it without rerunning anything.
