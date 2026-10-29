"""
src/train_all.py
──────────────────────────────────────────────────────────
One-shot script that runs the full ML pipeline:
  1. Preprocess raw data
  2. Train Sales Forecasting models (LR, RF, XGBoost)
  3. Train Demand Forecasting model (RF)
  4. Train Customer Segmentation (RFM + KMeans)

Run once before launching the Streamlit app.
Usage:  uv run python src/train_all.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from preprocessing import run_pipeline
from train_sales import train as train_sales
from train_demand import train as train_demand
from clustering import train as train_clusters


def main():
    print("=" * 55)
    print(" Smart Retail Analytics - Full Training Pipeline")
    print("=" * 55)

    print("\n[1/4] Preprocessing raw data...")
    df = run_pipeline()
    print(f"      -> {len(df):,} clean rows ready.\n")

    print("[2/4] Training Sales Forecasting models...")
    sales_metrics = train_sales()
    print()

    print("[3/4] Training Demand Forecasting model...")
    demand_metrics = train_demand()
    print()

    print("[4/4] Training Customer Segmentation (RFM + K-Means)...")
    cluster_metrics = train_clusters()
    print()

    print("=" * 55)
    print(" Training complete! All models saved to models/")
    print("=" * 55)
    print("\nSales Forecasting Results:")
    for model, m in sales_metrics.items():
        print(f"  {model:20s} MAE={m['MAE']:>10.2f}  RMSE={m['RMSE']:>10.2f}  R2={m['R2']:.4f}")
    print(f"\nDemand Forecasting:     MAE={demand_metrics['MAE']:.2f}  R²={demand_metrics['R2']:.4f}")
    print(f"Customer Segmentation:  Silhouette = {cluster_metrics['Silhouette']:.4f}")
    print("\nRun the dashboard:  uv run streamlit run app.py")


if __name__ == "__main__":
    main()


