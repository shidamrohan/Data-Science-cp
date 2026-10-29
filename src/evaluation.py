"""
src/evaluation.py
──────────────────────────────────────────────────────────
Central evaluation utilities and a one-shot function that
loads all trained models and returns a combined metrics report
for the Model Performance dashboard page.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def regression_metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    return {
        "MAE":  round(mae,  2),
        "RMSE": round(rmse, 2),
        "R2":   round(r2,   4),
        "WAPE": round(wape, 2),
    }


def load_sales_metrics() -> pd.DataFrame:
    """
    Re-evaluate all saved sales models on held-out test data
    and return a tidy DataFrame of metrics.
    """
    parquet = PROCESSED_DIR / "monthly_sales.parquet"
    if not parquet.exists():
        return pd.DataFrame()

    monthly = pd.read_parquet(parquet)
    drop_cols = ["Sales", "YearMonthStr", "Date", "Profit", "Month", "Year"]
    feat_cols  = [c for c in monthly.columns if c not in drop_cols]
    X = monthly[feat_cols].select_dtypes(include=[np.number])
    y = monthly["Sales"]

    # Drop rows with NaN features (early lag rows) — same as done in train_sales.py
    valid = X.notna().all(axis=1)
    X, y = X[valid], y[valid]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    rows = []
    scaler = joblib.load(MODELS_DIR / "sales_scaler.pkl")
    X_test_filled = X_test.fillna(0)          # safety: fill any residual NaN
    X_test_s = scaler.transform(X_test_filled)

    model_files = {
        "Linear Regression": "sales_lr_model.pkl",
        "Random Forest":     "sales_rf_model.pkl",
        "XGBoost":           "sales_xgb_model.pkl",
    }
    for name, fname in model_files.items():
        path = MODELS_DIR / fname
        if not path.exists():
            continue
        model = joblib.load(path)
        X_input = X_test_s if name == "Linear Regression" else X_test.fillna(0)
        pred = model.predict(X_input)
        m = regression_metrics(y_test, pred)
        rows.append({"Model": name, **m})

    return pd.DataFrame(rows)


def load_feature_importance() -> pd.DataFrame:
    """Return RF feature importances (top 10) for the dashboard."""
    path = MODELS_DIR / "sales_feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    fi = pd.read_csv(path, index_col=0)
    fi.columns = ["Importance"]
    return fi.head(10).reset_index().rename(columns={"index": "Feature"})


def load_cluster_silhouette() -> float:
    """Return the K-Means silhouette score from saved cluster data."""
    path = PROCESSED_DIR / "rfm_clusters.parquet"
    if not path.exists():
        return 0.0
    rfm = pd.read_parquet(path)
    scaler = joblib.load(MODELS_DIR / "rfm_scaler.pkl")
    X = scaler.transform(rfm[["R_log", "F_log", "M_log"]])
    labels = rfm["Cluster"].values
    try:
        return round(float(silhouette_score(X, labels)), 4)
    except Exception:
        return 0.0


if __name__ == "__main__":
    df = load_sales_metrics()
    print(df.to_string(index=False))
    print("\nFeature Importance:")
    print(load_feature_importance().to_string(index=False))
    print(f"\nCluster Silhouette: {load_cluster_silhouette()}")


