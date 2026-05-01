from __future__ import annotations
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.modeling.metrics import regression_report


def get_feature_columns(df: pd.DataFrame, target_col: str = "Revenue", date_col: str = "Date") -> list[str]:
    blocked = {target_col, "COGS", date_col}
    return [c for c in df.columns if c not in blocked]


def build_sklearn_model(df: pd.DataFrame, feature_cols: list[str] | None = None, model_type: str = "hist_gb") -> Pipeline:
    feature_cols = feature_cols or get_feature_columns(df)
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    model = HistGradientBoostingRegressor(random_state=42) if model_type == "hist_gb" else RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    return Pipeline([("preprocess", pre), ("model", model)])


def train_and_evaluate(train: pd.DataFrame, valid: pd.DataFrame, target_col: str = "Revenue", date_col: str = "Date", model_type: str = "hist_gb"):
    feature_cols = get_feature_columns(train, target_col=target_col, date_col=date_col)
    model = build_sklearn_model(train, feature_cols, model_type=model_type)
    X_train, y_train = train[feature_cols], train[target_col]
    X_valid, y_valid = valid[feature_cols], valid[target_col]
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return model, regression_report(y_valid, pred), pd.DataFrame({date_col: valid[date_col].values, "actual": y_valid.values, "pred": pred})
