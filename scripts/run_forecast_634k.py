"""
Reproduce the final Datathon 2026 Round 1 forecasting submission.

This script trains the final 634K forecasting ensemble from the official
competition files only and writes a Kaggle-ready submission with columns:
Date, Revenue, COGS.

Usage:
    python scripts/run_forecast_634k.py --data-dir data/raw --out submissions/submission_634K.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
except ImportError as exc:  # clearer error for a fresh VS Code environment
    raise SystemExit(
        "Missing modeling dependency. Please run: pip install -r requirements.txt"
    ) from exc


SEEDS = [42, 1337, 2024, 7, 99]
np.random.seed(42)

# Final blend weights selected from internal time-aware validation and sensitivity checks.
W_A = 0.54
ALPHA_UB = 0.70
ALPHA_YE = 0.50
REV_WEIGHTS = {"model": 0.58, "doy": 0.02, "month_dow": 0.04, "month_day": 0.36}

# Deterministic Vietnamese holiday calendar features. These use date information only;
# no target values from the test period are used.
TET = {
    2012: pd.Timestamp("2012-01-23"), 2013: pd.Timestamp("2013-02-10"),
    2014: pd.Timestamp("2014-01-31"), 2015: pd.Timestamp("2015-02-19"),
    2016: pd.Timestamp("2016-02-08"), 2017: pd.Timestamp("2017-01-28"),
    2018: pd.Timestamp("2018-02-16"), 2019: pd.Timestamp("2019-02-05"),
    2020: pd.Timestamp("2020-01-25"), 2021: pd.Timestamp("2021-02-12"),
    2022: pd.Timestamp("2022-02-01"), 2023: pd.Timestamp("2023-01-22"),
    2024: pd.Timestamp("2024-02-10"),
}


def days_to_tet(date: pd.Timestamp) -> int:
    cands = [TET.get(y) for y in (date.year - 1, date.year, date.year + 1) if TET.get(y) is not None]
    return min(((date - t).days for t in cands), key=abs) if cands else 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Quarter"] = df["Date"].dt.quarter
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsOddYear"] = (df["Year"] % 2).astype(int)

    df["Is_MegaSale"] = 0
    for m, d in [(11, 11), (12, 12), (2, 14), (3, 8)]:
        df.loc[(df["Month"] == m) & (df["Day"] == d), "Is_MegaSale"] = 1
    df.loc[(df["Month"] == 11) & (df["DayOfWeek"] == 4) & (df["Day"] >= 22) & (df["Day"] <= 28), "Is_MegaSale"] = 1

    is_odd = df["IsOddYear"] == 1
    df["Is_UrbanBlowout"] = (is_odd & (((df["Month"] == 7) & (df["Day"] >= 30)) | (df["Month"] == 8) | ((df["Month"] == 9) & (df["Day"] <= 2)))).astype(int)
    df["Is_RuralSpecial"] = (is_odd & (df["Month"] == 1)).astype(int)
    df["Is_SpringSale"] = (df["Month"] == 3).astype(int)
    df["Is_MidYearSale"] = (df["Month"] == 6).astype(int)
    df["Is_FallLaunch"] = (((df["Month"] == 8) & (df["Day"] >= 30)) | (df["Month"] == 9) | ((df["Month"] == 10) & (df["Day"] <= 2))).astype(int)
    df["Is_YearEnd"] = (((df["Month"] == 11) & (df["Day"] >= 18)) | (df["Month"] == 12)).astype(int)
    df["Is_ReunificationLabor"] = (((df["Month"] == 4) & (df["Day"] == 30)) | ((df["Month"] == 5) & (df["Day"] == 1))).astype(int)

    df["DaysToTet"] = df["Date"].apply(days_to_tet)
    df["IsTetWindow"] = (df["DaysToTet"].abs() <= 7).astype(int)
    df["IsPreTet"] = ((df["DaysToTet"] >= -14) & (df["DaysToTet"] < 0)).astype(int)
    df["IsPostTet"] = ((df["DaysToTet"] > 0) & (df["DaysToTet"] <= 14)).astype(int)

    df["doy_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)
    df["dom_sin"] = np.sin(2 * np.pi * df["Day"] / 31)
    df["dom_cos"] = np.cos(2 * np.pi * df["Day"] / 31)
    df["dow_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    return df


def build_aux_aggregates(data_dir: Path):
    inventory = pd.read_csv(data_dir / "inventory.csv", parse_dates=["snapshot_date"])
    web_traffic = pd.read_csv(data_dir / "web_traffic.csv", parse_dates=["date"])
    returns = pd.read_csv(data_dir / "returns.csv", parse_dates=["return_date"])
    shipments = pd.read_csv(data_dir / "shipments.csv", parse_dates=["ship_date", "delivery_date"])

    inventory["Month"] = inventory["snapshot_date"].dt.month
    inv_m = inventory.groupby("Month").agg(
        inv_stockout=("stockout_days", "mean"),
        inv_fillrate=("fill_rate", "mean"),
        inv_sellthru=("sell_through_rate", "mean"),
    ).reset_index()

    web_traffic["Month"] = web_traffic["date"].dt.month
    web_traffic["Day"] = web_traffic["date"].dt.day
    wt_md = web_traffic.groupby(["Month", "Day"]).agg(
        sessions_md=("sessions", "mean"),
        visitors_md=("unique_visitors", "mean"),
    ).reset_index()

    returns["Month"] = returns["return_date"].dt.month
    returns["Day"] = returns["return_date"].dt.day
    ret_md = returns.groupby(["Month", "Day"]).size().reset_index(name="returns_md")

    shipments["Month"] = shipments["ship_date"].dt.month
    shipments["Day"] = shipments["ship_date"].dt.day
    shipments["delivery_days"] = (shipments["delivery_date"] - shipments["ship_date"]).dt.days
    ship_md = shipments.groupby(["Month", "Day"]).agg(
        avg_ship_fee=("shipping_fee", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
    ).reset_index()
    return inv_m, wt_md, ret_md, ship_md


def add_aux_aggregates(df: pd.DataFrame, inv_m, wt_md, ret_md, ship_md) -> pd.DataFrame:
    df = df.merge(inv_m, on="Month", how="left")
    df = df.merge(ret_md, on=["Month", "Day"], how="left")
    df = df.merge(wt_md, on=["Month", "Day"], how="left")
    df = df.merge(ship_md, on=["Month", "Day"], how="left")
    for c in ["inv_stockout", "inv_fillrate", "inv_sellthru", "returns_md", "sessions_md", "visitors_md", "avg_ship_fee", "avg_delivery_days"]:
        df[c] = df[c].fillna(df[c].mean())
    return df


def build_seasonal_profiles(train_df: pd.DataFrame, target_col: str = "Revenue") -> dict:
    return {
        "global_median": train_df[target_col].median(),
        "doy_median": train_df.groupby("DayOfYear")[target_col].median(),
        "month_dow_mean": train_df.groupby(["Month", "DayOfWeek"])[target_col].mean(),
        "month_day_median": train_df.groupby(["Month", "Day"])[target_col].median(),
    }


def apply_profiles(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    df["rev_doy_median"] = df["DayOfYear"].map(p["doy_median"]).fillna(p["global_median"])
    df["rev_month_dow_mean"] = df.set_index(["Month", "DayOfWeek"]).index.map(p["month_dow_mean"]).fillna(p["global_median"])
    df["rev_month_day_median"] = df.set_index(["Month", "Day"]).index.map(p["month_day_median"]).fillna(p["global_median"])
    return df


FEATURES = [
    "Month", "Day", "DayOfWeek", "DayOfYear", "Quarter", "WeekOfYear", "IsWeekend", "IsOddYear",
    "Is_MegaSale", "Is_UrbanBlowout", "Is_RuralSpecial", "Is_SpringSale", "Is_MidYearSale",
    "Is_FallLaunch", "Is_YearEnd", "Is_ReunificationLabor",
    "DaysToTet", "IsTetWindow", "IsPreTet", "IsPostTet",
    "doy_sin", "doy_cos", "dom_sin", "dom_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "rev_doy_median", "rev_month_dow_mean", "rev_month_day_median",
    "inv_stockout", "inv_fillrate", "inv_sellthru", "returns_md", "sessions_md", "visitors_md",
    "avg_ship_fee", "avg_delivery_days",
]

LGB_PARAMS = dict(objective="regression", metric="rmse", n_estimators=1200, learning_rate=0.015, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbosity=-1)
XGB_PARAMS = dict(objective="reg:squarederror", n_estimators=1000, learning_rate=0.015, max_depth=5, subsample=0.8, colsample_bytree=0.8, verbosity=0)
CAT_PARAMS = dict(iterations=1200, learning_rate=0.015, depth=6, subsample=0.8, rsm=0.8, loss_function="RMSE", verbose=0, allow_writing_files=False)


def train_predict_3way(train: pd.DataFrame, test: pd.DataFrame, target: str = "Revenue", log_transform: bool = True) -> np.ndarray:
    if log_transform:
        y = np.log1p(train[target])
        invert = np.expm1
    else:
        y = train[target]
        invert = lambda x: x

    lgb_preds, xgb_preds, cat_preds = [], [], []
    for s in SEEDS:
        p = dict(LGB_PARAMS, random_state=s)
        m = lgb.LGBMRegressor(**p)
        m.fit(train[FEATURES], y)
        lgb_preds.append(invert(m.predict(test[FEATURES])))

        p = dict(XGB_PARAMS, random_state=s)
        m = xgb.XGBRegressor(**p)
        m.fit(train[FEATURES], y)
        xgb_preds.append(invert(m.predict(test[FEATURES])))

        p = dict(CAT_PARAMS, random_seed=s)
        m = CatBoostRegressor(**p)
        m.fit(train[FEATURES], y)
        cat_preds.append(invert(m.predict(test[FEATURES])))

    return (np.mean(lgb_preds, axis=0) + np.mean(xgb_preds, axis=0) + np.mean(cat_preds, axis=0)) / 3


def train_margin_lgb(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    preds = []
    for s in SEEDS:
        p = dict(LGB_PARAMS, random_state=s)
        m = lgb.LGBMRegressor(**p)
        m.fit(train[FEATURES], train["Margin"])
        preds.append(m.predict(test[FEATURES]))
    return np.clip(np.mean(preds, axis=0), 0.02, 0.35)


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert len(submission) == len(sample), "Submission row count does not match sample_submission."
    assert list(submission.columns) == ["Date", "Revenue", "COGS"], "Submission columns must be Date, Revenue, COGS."
    assert submission["Date"].tolist() == sample["Date"].dt.strftime("%Y-%m-%d").tolist(), "Date order differs from sample_submission."
    assert submission[["Revenue", "COGS"]].notna().all().all(), "Submission contains missing predictions."
    assert (submission[["Revenue", "COGS"]] >= 0).all().all(), "Submission contains negative predictions."


def run(data_dir: Path, out_path: Path, importance_out: Path | None = None) -> None:
    sales_raw = pd.read_csv(data_dir / "sales.csv", parse_dates=["Date"])
    sample = pd.read_csv(data_dir / "sample_submission.csv", parse_dates=["Date"])

    inv_m, wt_md, ret_md, ship_md = build_aux_aggregates(data_dir)

    sales_raw["Year"] = sales_raw["Date"].dt.year
    train_full = add_aux_aggregates(engineer_features(sales_raw), inv_m, wt_md, ret_md, ship_md)
    test_df = add_aux_aggregates(engineer_features(sample), inv_m, wt_md, ret_md, ship_md)

    profile = build_seasonal_profiles(train_full)
    train_full = apply_profiles(train_full, profile)
    test_df = apply_profiles(test_df, profile)

    train_A = train_full[(train_full["Year"] >= 2013) & (train_full["Year"] <= 2018)].copy()
    train_B = train_full[(train_full["Year"] >= 2019) & (train_full["Year"] <= 2022)].copy()

    print("Training revenue Model A...")
    pred_A_revenue = train_predict_3way(train_A, test_df, target="Revenue")
    print("Training revenue Model B...")
    pred_B_revenue = train_predict_3way(train_B, test_df, target="Revenue")

    seasonal_part = (
        REV_WEIGHTS["doy"] * test_df["rev_doy_median"].values
        + REV_WEIGHTS["month_dow"] * test_df["rev_month_dow_mean"].values
        + REV_WEIGHTS["month_day"] * test_df["rev_month_day_median"].values
    )
    final_A_revenue = REV_WEIGHTS["model"] * pred_A_revenue + seasonal_part
    final_B_revenue = REV_WEIGHTS["model"] * pred_B_revenue + seasonal_part

    train_full["Margin"] = ((train_full["Revenue"] - train_full["COGS"]) / train_full["Revenue"]).clip(0.02, 0.35)
    margin_train_A = train_full[(train_full["Year"] >= 2013) & (train_full["Year"] <= 2018)].copy()
    margin_train_B = train_full[(train_full["Year"] >= 2019) & (train_full["Year"] <= 2022)].copy()

    print("Training margin Model A...")
    pred_marg_A = train_margin_lgb(margin_train_A, test_df)
    print("Training margin Model B...")
    pred_marg_B = train_margin_lgb(margin_train_B, test_df)

    final_revenue = W_A * final_A_revenue + (1 - W_A) * final_B_revenue
    final_revenue = np.maximum(0, final_revenue)
    final_margin = np.clip(W_A * pred_marg_A + (1 - W_A) * pred_marg_B, 0.02, 0.35)
    final_cogs = final_revenue * (1 - final_margin)

    sales_with_ratio = sales_raw.copy()
    sales_with_ratio["Year"] = sales_with_ratio["Date"].dt.year
    sales_with_ratio["Month"] = sales_with_ratio["Date"].dt.month
    sales_with_ratio["Day"] = sales_with_ratio["Date"].dt.day
    sales_with_ratio["ratio"] = sales_with_ratio["COGS"] / sales_with_ratio["Revenue"]

    odd_year_ratio = sales_with_ratio[sales_with_ratio["Year"] % 2 == 1].groupby(["Month", "Day"])["ratio"].mean()
    post_2018_ratio = sales_with_ratio[sales_with_ratio["Year"] >= 2019].groupby(["Month", "Day"])["ratio"].mean()

    months = test_df["Date"].dt.month.values
    days = test_df["Date"].dt.day.values
    years = test_df["Date"].dt.year.values

    ub_window = (years % 2 == 1) & (((months == 7) & (days >= 30)) | (months == 8) | ((months == 9) & (days <= 2)))
    final_cogs_after_ub = final_cogs.copy()
    for i in np.where(ub_window)[0]:
        hist_ratio = odd_year_ratio.get((months[i], days[i]), 0.85)
        fix = final_revenue[i] * hist_ratio
        final_cogs_after_ub[i] = (1 - ALPHA_UB) * final_cogs[i] + ALPHA_UB * fix

    ye_window = (((months == 11) & (days >= 18)) | (months == 12)) & ~ub_window
    final_cogs_final = final_cogs_after_ub.copy()
    for i in np.where(ye_window)[0]:
        hist_ratio = post_2018_ratio.get((months[i], days[i]), 0.95)
        fix = final_revenue[i] * hist_ratio
        final_cogs_final[i] = (1 - ALPHA_YE) * final_cogs_after_ub[i] + ALPHA_YE * fix

    submission = pd.DataFrame({
        "Date": test_df["Date"].dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(final_revenue, 2),
        "COGS": np.round(final_cogs_final, 2),
    })
    validate_submission(submission, sample)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    # Feature importance for report explainability.
    p = dict(LGB_PARAMS, random_state=42)
    m = lgb.LGBMRegressor(**p)
    m.fit(train_A[FEATURES], np.log1p(train_A["Revenue"]))
    importances = pd.DataFrame({"feature": FEATURES, "importance": m.feature_importances_}).sort_values("importance", ascending=False)
    if importance_out is not None:
        importance_out.parent.mkdir(parents=True, exist_ok=True)
        importances.to_csv(importance_out, index=False)

    print(f"Saved {len(submission)} rows to {out_path}")
    print(f"Revenue mean: {submission['Revenue'].mean():,.0f}")
    print(f"COGS mean:    {submission['COGS'].mean():,.0f}")
    print("Top 10 features:")
    print(importances.head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final Datathon 2026 sales forecast model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Folder containing official competition CSV files.")
    parser.add_argument("--out", type=Path, default=Path("submissions/submission_634K.csv"), help="Output submission CSV path.")
    parser.add_argument("--importance-out", type=Path, default=Path("outputs/modeling/feature_importance_634K.csv"), help="Output feature importance CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.out, args.importance_out)
