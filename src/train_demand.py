"""
src/train_demand.py
──────────────────────────────────────────────────────────
Trains a Random Forest Regressor to predict monthly unit
quantity demand per Sub-Category.

Saves:
  models/demand_rf_model.pkl
  models/demand_scaler.pkl
  models/demand_feature_cols.pkl
  data/processed/monthly_demand.parquet
  data/processed/demand_tier.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import load_clean, run_pipeline, PROCESSED_DIR
from feature_engineering import build_monthly_demand

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _demand_tier(mean_qty: float, p33: float, p66: float) -> str:
    if mean_qty >= p66:
        return "High Demand"
    elif mean_qty >= p33:
        return "Medium Demand"
    else:
        return "Low Demand"


def train() -> dict:
    # ── Load ──────────────────────────────────────────────────────
    try:
        df = load_clean()
    except Exception:
        df = run_pipeline()

    demand_df = build_monthly_demand(df)
    demand_df.to_parquet(PROCESSED_DIR / "monthly_demand.parquet", index=False)

    # ── Features / target ─────────────────────────────────────────
    drop_cols = ["Quantity", "Date", "YearMonthStr", "Month", "Year", "Sales"]
    feat_cols = [c for c in demand_df.columns if c not in drop_cols]
    X = demand_df[feat_cols].select_dtypes(include=[np.number])
    y = demand_df["Quantity"]

    # Drop rows with NaN features (early lag rows)
    valid = X.notna().all(axis=1)
    X, y = X[valid], y[valid]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ── Model ─────────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    mae  = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2   = r2_score(y_test, pred)

    joblib.dump(rf,                  MODELS_DIR / "demand_rf_model.pkl")
    joblib.dump(list(X.columns),     MODELS_DIR / "demand_feature_cols.pkl")

    # ── Demand tier classification per Sub-Category ───────────────
    # Use original (pre-OHE) dataframe for this
    raw_demand = (
        df.groupby(["YearMonthStr", "Sub-Category"])["Quantity"]
        .sum()
        .reset_index()
    )
    sub_cat_means = raw_demand.groupby("Sub-Category")["Quantity"].mean()
    p33 = sub_cat_means.quantile(0.33)
    p66 = sub_cat_means.quantile(0.66)

    tier_df = sub_cat_means.reset_index()
    tier_df.columns = ["Sub-Category", "Mean_Monthly_Qty"]
    tier_df["Demand_Tier"] = tier_df["Mean_Monthly_Qty"].apply(
        lambda q: _demand_tier(q, p33, p66)
    )
    tier_df.to_parquet(PROCESSED_DIR / "demand_tier.parquet", index=False)

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
    print(f"[train_demand] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


if __name__ == "__main__":
    train()


