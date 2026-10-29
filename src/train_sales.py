"""
src/train_sales.py
──────────────────────────────────────────────────────────
Trains three regression models for monthly sales forecasting:
  • Linear Regression  (baseline)
  • Random Forest Regressor
  • XGBoost Regressor

Saves:
  models/sales_lr_model.pkl
  models/sales_rf_model.pkl
  models/sales_xgb_model.pkl
  models/sales_scaler.pkl
  models/sales_feature_cols.pkl
  data/processed/monthly_sales.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from preprocessing import load_clean, run_pipeline, PROCESSED_DIR
from feature_engineering import build_monthly_sales

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _feature_target_split(df: pd.DataFrame):
    """Return X (feature matrix) and y (Sales target)."""
    drop_cols = [
        "Sales", "YearMonthStr", "Date", "Profit",
        "Month", "Year",           # keep cyclical versions
    ]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].select_dtypes(include=[np.number])
    y = df["Sales"]
    return X, y, list(X.columns)


def train(force_reprocess: bool = False) -> dict:
    """Train all three models and return a metrics dict."""
    # ── Load data ──────────────────────────────────────────────────
    try:
        df = load_clean()
    except Exception:
        df = run_pipeline()

    monthly = build_monthly_sales(df)
    monthly.to_parquet(PROCESSED_DIR / "monthly_sales.parquet", index=False)

    X, y, feat_cols = _feature_target_split(monthly)

    # Drop any rows where features still contain NaN (early lag rows)
    valid = X.notna().all(axis=1)
    X, y = X[valid], y[valid]

    # ── Train / Test split (time-aware) ───────────────────────────
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ── Scale ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = {}

    # ── 1. Linear Regression ──────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    pred_lr = lr.predict(X_test_s)
    results["Linear Regression"] = _metrics(y_test, pred_lr)
    joblib.dump(lr, MODELS_DIR / "sales_lr_model.pkl")

    # ── 2. Random Forest ──────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)          # RF doesn't need scaled data
    pred_rf = rf.predict(X_test)
    results["Random Forest"] = _metrics(y_test, pred_rf)
    joblib.dump(rf, MODELS_DIR / "sales_rf_model.pkl")

    # ── 3. XGBoost ────────────────────────────────────────────────
    if XGB_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        results["XGBoost"] = _metrics(y_test, pred_xgb)
        joblib.dump(xgb, MODELS_DIR / "sales_xgb_model.pkl")
    else:
        results["XGBoost"] = {"MAE": None, "RMSE": None, "R2": None}

    # ── Save shared artefacts ─────────────────────────────────────
    joblib.dump(scaler,    MODELS_DIR / "sales_scaler.pkl")
    joblib.dump(feat_cols, MODELS_DIR / "sales_feature_cols.pkl")

    # ── Feature importance from RF ────────────────────────────────
    fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    fi.to_csv(MODELS_DIR / "sales_feature_importance.csv")

    print("[train_sales] Results:")
    for name, m in results.items():
        print(f"  {name}: MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R²={m['R2']:.4f}")

    return results


def _metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


if __name__ == "__main__":
    train()


