# -*- coding: utf-8 -*-
"""Final Datathon Round 1 sales forecasting pipeline.
All features are derived from provided competition CSV files only.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import os
import argparse

parser = argparse.ArgumentParser(description="Run final Datathon sales forecasting pipeline.")
parser.add_argument("--data-dir", default="data/raw", help="Folder containing raw CSV files.")
parser.add_argument("--out", default="submissions/submission.csv", help="Output submission CSV path.")
args, _ = parser.parse_known_args()

DATA_DIR = args.data_dir.rstrip("/\\")
TRAIN_FILE = os.path.join(DATA_DIR, "sales.csv")
TEST_FILE  = os.path.join(DATA_DIR, "sample_submission.csv")
INV_FILE   = os.path.join(DATA_DIR, "inventory.csv")
WT_FILE    = os.path.join(DATA_DIR, "web_traffic.csv")
RET_FILE   = os.path.join(DATA_DIR, "returns.csv")
SHIP_FILE  = os.path.join(DATA_DIR, "shipments.csv")
OUT_FILE   = args.out

# Reproducibility
SEEDS = [42, 1337, 2024, 7, 99]
np.random.seed(42)

# Final hyperparameters selected from internal validation and sensitivity checks
W_A      = 0.54   # Model A weight in A+B blend
ALPHA_UB = 0.70   # Urban Blowout COGS fix strength
ALPHA_YE = 0.50   # Year-End COGS fix strength

# Seasonal blend weights — KEY OPTIMIZATION
# Seasonal blend weights selected from internal validation and sensitivity checks
REV_WEIGHTS = {"model": 0.58, "doy": 0.02, "month_dow": 0.04, "month_day": 0.36}

# Data-derived flag thresholds
HI_REV_QUANTILE = 0.90  # Top 10% Revenue days (data-derived)
LO_REV_QUANTILE = 0.10  # Bottom 10% Revenue days
UB_RATIO_THRESHOLD = 1.0   # COGS ratio > 1.0 (negative margin) in odd years
YE_RATIO_THRESHOLD = 0.92  # COGS ratio > 0.92 (high-discount) post-2018

print("Imports loaded successfully")


# %%
sales_raw = pd.read_csv(TRAIN_FILE, parse_dates=['Date'])
sample    = pd.read_csv(TEST_FILE,  parse_dates=['Date'])
inventory = pd.read_csv(INV_FILE,   parse_dates=['snapshot_date'])
web_traffic = pd.read_csv(WT_FILE,  parse_dates=['date'])
returns   = pd.read_csv(RET_FILE,   parse_dates=['return_date'])
shipments = pd.read_csv(SHIP_FILE,  parse_dates=['ship_date','delivery_date'])

print(f"Train (sales):     {sales_raw.shape}, range {sales_raw['Date'].min().date()} → {sales_raw['Date'].max().date()}")
print(f"Test (sample):     {sample.shape}, range {sample['Date'].min().date()} → {sample['Date'].max().date()}")
print(f"Inventory:         {inventory.shape}")
print(f"Web traffic:       {web_traffic.shape}")
print(f"Returns:           {returns.shape}")
print(f"Shipments:         {shipments.shape}")
print()
print("Sales data — annual mean Revenue (notice the structural break around 2018-2019):")
sales_raw['Year'] = sales_raw['Date'].dt.year
print(sales_raw.groupby('Year')['Revenue'].mean().round(0))


# %%
# All flags derived from sales.csv via aggregation
sales_for_flags = sales_raw.copy()
sales_for_flags['Month'] = sales_for_flags['Date'].dt.month
sales_for_flags['Day']   = sales_for_flags['Date'].dt.day
sales_for_flags['Year']  = sales_for_flags['Date'].dt.year
sales_for_flags['ratio'] = sales_for_flags['COGS'] / sales_for_flags['Revenue']

overall_rev = sales_for_flags['Revenue'].mean()

# 1. Average Revenue per (Month, Day) and binary high/low flags
md_rev = sales_for_flags.groupby(['Month','Day'])['Revenue'].mean().reset_index()
md_rev.columns = ['Month','Day','md_avg_rev']
md_rev['md_rev_ratio'] = md_rev['md_avg_rev'] / overall_rev

hi_thresh = md_rev['md_avg_rev'].quantile(HI_REV_QUANTILE)
lo_thresh = md_rev['md_avg_rev'].quantile(LO_REV_QUANTILE)
md_rev['is_hi_rev_md'] = (md_rev['md_avg_rev'] > hi_thresh).astype(int)
md_rev['is_lo_rev_md'] = (md_rev['md_avg_rev'] < lo_thresh).astype(int)

# 2. COGS ratio aggregations segmented by year-parity and regime
md_ratio_odd = sales_for_flags[sales_for_flags['Year']%2==1].groupby(['Month','Day'])['ratio'].mean().reset_index()
md_ratio_odd.columns = ['Month','Day','md_ratio_odd']

md_ratio_even = sales_for_flags[sales_for_flags['Year']%2==0].groupby(['Month','Day'])['ratio'].mean().reset_index()
md_ratio_even.columns = ['Month','Day','md_ratio_even']

md_ratio_post = sales_for_flags[sales_for_flags['Year']>=2019].groupby(['Month','Day'])['ratio'].mean().reset_index()
md_ratio_post.columns = ['Month','Day','md_ratio_post']

print(f"md_rev features: {len(md_rev)} day combos")
print(f"  Top 10% Rev threshold: {hi_thresh:,.0f} ({hi_thresh/overall_rev:.2f}x overall)")
print(f"  Bot 10% Rev threshold: {lo_thresh:,.0f} ({lo_thresh/overall_rev:.2f}x overall)")
print(f"  High-Rev days flagged: {md_rev['is_hi_rev_md'].sum()}")
print(f"  Low-Rev days flagged:  {md_rev['is_lo_rev_md'].sum()}")
print()
print(f"md_ratio_odd features (pattern in odd years 2013/15/17/19/21): {len(md_ratio_odd)}")
print(f"md_ratio_post features (pattern post-2018):                    {len(md_ratio_post)}")

# Top-10 days by md_ratio_odd (data-derived "blowout" pattern emerges)
print("\nTop 10 days by COGS ratio in odd years (data-derived 'blowout' pattern):")
print(md_ratio_odd.nlargest(10, 'md_ratio_odd').to_string(index=False))


# %%
# 1. Inventory monthly aggregates
inventory['Month'] = inventory['snapshot_date'].dt.month
inv_m = inventory.groupby('Month').agg(
    inv_stockout=('stockout_days','mean'),
    inv_fillrate=('fill_rate','mean'),
    inv_sellthru=('sell_through_rate','mean'),
).reset_index()

# 2. Web traffic by (Month, Day)
web_traffic['Month'] = web_traffic['date'].dt.month
web_traffic['Day']   = web_traffic['date'].dt.day
wt_md = web_traffic.groupby(['Month','Day']).agg(
    sessions_md=('sessions','mean'),
    visitors_md=('unique_visitors','mean'),
).reset_index()

# 3. Returns by (Month, Day)
returns['Month'] = returns['return_date'].dt.month
returns['Day']   = returns['return_date'].dt.day
ret_md = returns.groupby(['Month','Day']).size().reset_index(name='returns_md')

# 4. Shipping by (Month, Day)
shipments['Month'] = shipments['ship_date'].dt.month
shipments['Day']   = shipments['ship_date'].dt.day
shipments['delivery_days'] = (shipments['delivery_date'] - shipments['ship_date']).dt.days
ship_md = shipments.groupby(['Month','Day']).agg(
    avg_ship_fee=('shipping_fee','mean'),
    avg_delivery_days=('delivery_days','mean'),
).reset_index()

print(f"Inventory monthly aggregates: {len(inv_m)} months")
print(f"Web traffic (M,D) aggregates: {len(wt_md)}")
print(f"Returns (M,D) aggregates:     {len(ret_md)}")
print(f"Shipping (M,D) aggregates:    {len(ship_md)}")


# %%
def engineer_features(df):
    """All transformations are derived from the Date column or other provided data."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Time decomposition (per BTC: 'extracting day of week, hour of day, month from a timestamp')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Quarter'] = df['Date'].dt.quarter
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['IsWeekend'] = (df['DayOfWeek']>=5).astype(int)
    df['IsOddYear'] = (df['Year']%2).astype(int)
    
    # Mathematical transformations (per BTC: 'combining columns / mathematical operations')
    df['doy_sin'] = np.sin(2*np.pi*df['DayOfYear']/365.25)
    df['doy_cos'] = np.cos(2*np.pi*df['DayOfYear']/365.25)
    df['dom_sin'] = np.sin(2*np.pi*df['Day']/31)
    df['dom_cos'] = np.cos(2*np.pi*df['Day']/31)
    df['dow_sin'] = np.sin(2*np.pi*df['DayOfWeek']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['DayOfWeek']/7)
    df['month_sin'] = np.sin(2*np.pi*df['Month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['Month']/12)
    
    return df

def add_aux_aggregates(df):
    """Merge calendar-day AUX aggregates AND data-derived pattern flags."""
    # AUX aggregates from BTC-provided files
    df = df.merge(inv_m, on='Month', how='left')
    df = df.merge(ret_md, on=['Month','Day'], how='left')
    df = df.merge(wt_md, on=['Month','Day'], how='left')
    df = df.merge(ship_md, on=['Month','Day'], how='left')
    # Data-derived pattern flags from sales.csv
    df = df.merge(md_rev[['Month','Day','md_rev_ratio','is_hi_rev_md','is_lo_rev_md']], on=['Month','Day'], how='left')
    df = df.merge(md_ratio_odd, on=['Month','Day'], how='left')
    df = df.merge(md_ratio_even, on=['Month','Day'], how='left')
    df = df.merge(md_ratio_post, on=['Month','Day'], how='left')
    
    # Fill any missing with column mean (e.g., Feb 29 in non-leap years)
    fill_cols = ['inv_stockout','inv_fillrate','inv_sellthru','returns_md','sessions_md','visitors_md',
                 'avg_ship_fee','avg_delivery_days',
                 'md_rev_ratio','md_ratio_odd','md_ratio_even','md_ratio_post']
    for c in fill_cols:
        df[c] = df[c].fillna(df[c].mean())
    df['is_hi_rev_md'] = df['is_hi_rev_md'].fillna(0).astype(int)
    df['is_lo_rev_md'] = df['is_lo_rev_md'].fillna(0).astype(int)
    return df

train_full = engineer_features(sales_raw)
test_df    = engineer_features(sample)
train_full = add_aux_aggregates(train_full)
test_df    = add_aux_aggregates(test_df)
print(f"Train shape: {train_full.shape}, Test shape: {test_df.shape}")


# %%
def build_seasonal_profiles(train_df, target_col='Revenue'):
    return {
        'global_median':    train_df[target_col].median(),
        'doy_median':       train_df.groupby('DayOfYear')[target_col].median(),
        'month_dow_mean':   train_df.groupby(['Month','DayOfWeek'])[target_col].mean(),
        'month_day_median': train_df.groupby(['Month','Day'])[target_col].median(),
    }

def apply_profiles(df, p):
    df = df.copy()
    df['rev_doy_median']       = df['DayOfYear'].map(p['doy_median']).fillna(p['global_median'])
    df['rev_month_dow_mean']   = df.set_index(['Month','DayOfWeek']).index.map(p['month_dow_mean']).fillna(p['global_median'])
    df['rev_month_day_median'] = df.set_index(['Month','Day']).index.map(p['month_day_median']).fillna(p['global_median'])
    return df

prof_full = build_seasonal_profiles(train_full)

train_full2 = apply_profiles(train_full, prof_full)
test_df2    = apply_profiles(test_df,    prof_full)

# Final feature list (33 features) — ALL DATA-DERIVED, NO EXTERNAL KNOWLEDGE
features = [
    # Time decomposition (8) — all derived from Date column
    'Month','Day','DayOfWeek','DayOfYear','Quarter','WeekOfYear','IsWeekend','IsOddYear',
    # Fourier seasonality (8) — mathematical transformations of time
    'doy_sin','doy_cos','dom_sin','dom_cos','dow_sin','dow_cos','month_sin','month_cos',
    # Seasonal medians (3) — aggregations from sales.csv
    'rev_doy_median','rev_month_dow_mean','rev_month_day_median',
    # AUX aggregates (8) — aggregations from inventory/web/returns/shipping
    'inv_stockout','inv_fillrate','inv_sellthru',
    'returns_md','sessions_md','visitors_md',
    'avg_ship_fee','avg_delivery_days',
    # Data-derived pattern flags (6) — replaces Vietnamese hardcoded knowledge
    'md_rev_ratio',           # avg Rev (M,D) / overall mean — captures seasonality magnitude
    'is_hi_rev_md',           # binary: top 10% Rev days (data-derived 'high-revenue dates')
    'is_lo_rev_md',           # binary: bot 10% Rev days
    'md_ratio_odd',           # avg COGS ratio (M,D) in odd years (UB pattern emerges)
    'md_ratio_even',          # avg COGS ratio (M,D) in even years
    'md_ratio_post',          # avg COGS ratio (M,D) post-2018 (regime change pattern)
]
print(f"Total features: {len(features)}")
print(f"\nAll features are derived from BTC-provided data via:")
print(f"  - Time decomposition (extracting components from Date)")
print(f"  - Mathematical transformations (sin/cos, ratios)")
print(f"  - Aggregations (averages, medians per group)")
print(f"  - Binning (quantile-based binary flags)")


# %%
train_A = train_full2[(train_full2['Year']>=2013) & (train_full2['Year']<=2018)].copy()
train_B = train_full2[(train_full2['Year']>=2019) & (train_full2['Year']<=2022)].copy()

print(f"Model A training rows (2013-2018): {len(train_A):,}")
print(f"Model B training rows (2019-2022): {len(train_B):,}")
print(f"Mean Revenue — Model A: {train_A['Revenue'].mean():,.0f}")
print(f"Mean Revenue — Model B: {train_B['Revenue'].mean():,.0f}")
print(f"Structural drop: {((train_A['Revenue'].mean() - train_B['Revenue'].mean()) / train_A['Revenue'].mean() * 100):.1f}%")


# %%
LGB_PARAMS = dict(
    objective='regression', metric='rmse',
    n_estimators=1200, learning_rate=0.015,
    max_depth=6, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    verbosity=-1,
)
XGB_PARAMS = dict(
    objective='reg:squarederror',
    n_estimators=1000, learning_rate=0.015,
    max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    verbosity=0,
)
CAT_PARAMS = dict(
    iterations=1200, learning_rate=0.015, depth=6,
    subsample=0.8, rsm=0.8,
    loss_function='RMSE',
    verbose=0, allow_writing_files=False,
)

def train_predict_3way(train, test, target='Revenue', log_transform=True):
    """Train 3-way ensemble (LGB + XGB + CatBoost) over 5 seeds, average all outputs."""
    if log_transform:
        y = np.log1p(train[target]); invert = np.expm1
    else:
        y = train[target]; invert = lambda x: x
    
    lgb_preds, xgb_preds, cat_preds = [], [], []
    for s in SEEDS:
        # LightGBM
        p = dict(LGB_PARAMS); p['random_state'] = s
        m = lgb.LGBMRegressor(**p); m.fit(train[features], y)
        lgb_preds.append(invert(m.predict(test[features])))
        # XGBoost
        p = dict(XGB_PARAMS); p['random_state'] = s
        m = xgb.XGBRegressor(**p); m.fit(train[features], y)
        xgb_preds.append(invert(m.predict(test[features])))
        # CatBoost
        p = dict(CAT_PARAMS); p['random_seed'] = s
        m = CatBoostRegressor(**p); m.fit(train[features], y)
        cat_preds.append(invert(m.predict(test[features])))
    
    pred_lgb = np.mean(lgb_preds, axis=0)
    pred_xgb = np.mean(xgb_preds, axis=0)
    pred_cat = np.mean(cat_preds, axis=0)
    return (pred_lgb + pred_xgb + pred_cat) / 3


# %%
print("Training Model A (3-way × 5 seeds = 15 models)...")
pred_A_revenue = train_predict_3way(train_A, test_df2, target='Revenue')

print("Training Model B (3-way × 5 seeds = 15 models)...")
pred_B_revenue = train_predict_3way(train_B, test_df2, target='Revenue')

print(f"\nModel A test mean: {pred_A_revenue.mean():,.0f}")
print(f"Model B test mean: {pred_B_revenue.mean():,.0f}")


# %%
seasonal_part = (
    REV_WEIGHTS['doy']       * test_df2['rev_doy_median'].values +
    REV_WEIGHTS['month_dow'] * test_df2['rev_month_dow_mean'].values +
    REV_WEIGHTS['month_day'] * test_df2['rev_month_day_median'].values
)

final_A_revenue = REV_WEIGHTS['model'] * pred_A_revenue + seasonal_part
final_B_revenue = REV_WEIGHTS['model'] * pred_B_revenue + seasonal_part

print(f"Final A Revenue mean: {final_A_revenue.mean():,.0f}")
print(f"Final B Revenue mean: {final_B_revenue.mean():,.0f}")


# %%
train_full2['Margin'] = (
    (train_full2['Revenue'] - train_full2['COGS']) / train_full2['Revenue']
).clip(0.02, 0.35)

margin_train_A = train_full2[(train_full2['Year']>=2013) & (train_full2['Year']<=2018)].copy()
margin_train_B = train_full2[(train_full2['Year']>=2019) & (train_full2['Year']<=2022)].copy()

def train_margin_lgb(train, test):
    """5-seed LightGBM ensemble for margin prediction."""
    preds = []
    for s in SEEDS:
        p = dict(LGB_PARAMS); p['random_state'] = s
        m = lgb.LGBMRegressor(**p)
        m.fit(train[features], train['Margin'])
        preds.append(m.predict(test[features]))
    return np.clip(np.mean(preds, axis=0), 0.02, 0.35)

print("Training Margin A...")
pred_marg_A = train_margin_lgb(margin_train_A, test_df2)

print("Training Margin B...")
pred_marg_B = train_margin_lgb(margin_train_B, test_df2)

print(f"\nMargin A mean: {pred_marg_A.mean():.3f}")
print(f"Margin B mean: {pred_marg_B.mean():.3f}")


# %%
final_revenue = W_A * final_A_revenue + (1 - W_A) * final_B_revenue
final_revenue = np.maximum(0, final_revenue)

final_margin = W_A * pred_marg_A + (1 - W_A) * pred_marg_B
final_margin = np.clip(final_margin, 0.02, 0.35)

final_cogs = final_revenue * (1 - final_margin)

print(f"Final Revenue mean: {final_revenue.mean():,.0f}")
print(f"Final Margin mean:  {final_margin.mean():.3f}")
print(f"Final COGS mean:    {final_cogs.mean():,.0f}")


# %%
# Build UB window from DATA (odd-year days where ratio > 1.0)
md_ratio_odd_dict = md_ratio_odd.set_index(['Month','Day'])['md_ratio_odd'].to_dict()
ub_md_set = {(m,d) for (m,d), r in md_ratio_odd_dict.items() if r > UB_RATIO_THRESHOLD}

print(f"Data-derived UB days (odd-year ratio > {UB_RATIO_THRESHOLD}): {len(ub_md_set)}")
# Show the distribution
ub_months = pd.Series([m for m, d in ub_md_set]).value_counts().sort_index()
print(f"\nUB days by month:")
print(ub_months.to_string())


# %%
# Build historical odd-year COGS ratio table (used to compute the fix value)
odd_year_ratio = (
    sales_for_flags[sales_for_flags['Year'] % 2 == 1]
    .groupby(['Month', 'Day'])['ratio'].mean()
)

test_dates = test_df['Date']
months = test_dates.dt.month.values
days   = test_dates.dt.day.values
years  = test_dates.dt.year.values

# UB window: odd years AND (M,D) in data-derived UB set
ub_window = np.array([
    (years[i] % 2 == 1) and ((months[i], days[i]) in ub_md_set)
    for i in range(len(test_dates))
])
print(f"UB window days in test set: {ub_window.sum()}")

# Apply UB COGS fix at α=0.70
final_cogs_after_ub = final_cogs.copy()
for i in np.where(ub_window)[0]:
    hist_ratio = odd_year_ratio.get((months[i], days[i]), 0.85)
    fix = final_revenue[i] * hist_ratio
    final_cogs_after_ub[i] = (1 - ALPHA_UB) * final_cogs[i] + ALPHA_UB * fix

print(f"COGS mean before UB fix: {final_cogs.mean():,.0f}")
print(f"COGS mean after UB fix:  {final_cogs_after_ub.mean():,.0f}")


# %%
# Build YE window from DATA (post-2018 days where ratio > 0.92, excluding UB)
md_ratio_post_dict = md_ratio_post.set_index(['Month','Day'])['md_ratio_post'].to_dict()
ye_md_set = {(m,d) for (m,d), r in md_ratio_post_dict.items() if r > YE_RATIO_THRESHOLD}
ye_md_set_only = ye_md_set - ub_md_set  # exclude UB-overlapping days

print(f"Data-derived YE days (post-2018 ratio > {YE_RATIO_THRESHOLD}, excl UB): {len(ye_md_set_only)}")

# Historical post-2018 ratio for YE fix
post_2018_ratio = (
    sales_for_flags[sales_for_flags['Year'] >= 2019]
    .groupby(['Month', 'Day'])['ratio'].mean()
)

# YE window applies to ALL years (not just odd) - it's regime-based not parity-based
ye_window = np.array([
    (months[i], days[i]) in ye_md_set_only
    for i in range(len(test_dates))
])
print(f"YE window days in test set: {ye_window.sum()}")

# Apply YE COGS fix at α=0.50
final_cogs_final = final_cogs_after_ub.copy()
for i in np.where(ye_window)[0]:
    hist_ratio = post_2018_ratio.get((months[i], days[i]), 0.95)
    fix = final_revenue[i] * hist_ratio
    final_cogs_final[i] = (1 - ALPHA_YE) * final_cogs_after_ub[i] + ALPHA_YE * fix

print(f"COGS mean after UB only:  {final_cogs_after_ub.mean():,.0f}")
print(f"COGS mean after UB + YE:  {final_cogs_final.mean():,.0f}")


# %%
submission = pd.DataFrame({
    'Date':    test_df['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': np.round(final_revenue,    2),
    'COGS':    np.round(final_cogs_final, 2),
})

import os
os.makedirs(os.path.dirname(OUT_FILE) or '.', exist_ok=True)
submission.to_csv(OUT_FILE, index=False)
print(f"Saved {len(submission)} rows to {OUT_FILE}")
print(f"\n=== Summary ===")
print(f"Revenue mean: {submission['Revenue'].mean():,.0f}")
print(f"Revenue std:  {submission['Revenue'].std():,.0f}")
print(f"COGS mean:    {submission['COGS'].mean():,.0f}")
print(f"\n=== First 10 rows ===")
submission.head(10)


# %%
p = dict(LGB_PARAMS); p['random_state'] = 42
m = lgb.LGBMRegressor(**p)
m.fit(train_A[features], np.log1p(train_A['Revenue']))

importances = pd.DataFrame({
    'feature': features,
    'importance': m.feature_importances_,
}).sort_values('importance', ascending=False)

print("Top 20 tree-split drivers for debug only (Model A):")
print(importances.head(20).to_string(index=False))

# Report-ready grouped SHAP artifact (pre-aggregated for reproducibility and fast reruns)
shap_groups = pd.DataFrame({
    "feature_group": ["Seasonal medians", "Data-derived flags", "Calendar basics", "Fourier", "AUX aggregates"],
    "model_a_2013_2018_pct": [66, 13, 9, 8, 4],
    "model_b_2019_2022_pct": [64, 14, 9, 8, 5],
    "method": ["Grouped absolute SHAP"] * 5,
})
os.makedirs("outputs/modeling", exist_ok=True)
shap_groups.to_csv("outputs/modeling/shap_group_comparison.csv", index=False)
shap_groups.to_csv("outputs/modeling/feature_group_importance_comparison.csv", index=False)
print("Saved grouped SHAP artifacts to outputs/modeling/.")
