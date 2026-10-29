"""
pages/6_🧠_Model_Performance.py
─────────────────────────────────────────────────────────
Academic Model Evaluation & Benchmarking page.
  • Sales forecasting: LR vs RF vs XGBoost (MAE, RMSE, R², WAPE)
  • Best model callout badge
  • RF feature importance chart
  • KMeans silhouette score + cluster inertia
  • Demand model metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path as _Path

from evaluation import load_sales_metrics, load_feature_importance, load_cluster_silhouette

st.set_page_config(page_title="Model Performance", page_icon="🧠", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  div[data-testid="stMetric"], [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
    border: 1px solid #0f3460; border-radius: 12px; padding: 12px 14px;
    min-width: 0 !important;
  }
  div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] p, [data-testid="metric-container"] label {
    color: #a0aec0 !important; font-size: 0.82rem !important; white-space: nowrap !important;
  }
  div[data-testid="stMetricValue"],
  div[data-testid="stMetricValue"] > div,
  div[data-testid="stMetricValue"] *,
  [data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e2e8f0 !important; font-size: 1.15rem !important; font-weight: 700 !important;
    white-space: nowrap !important; overflow: visible !important; text-overflow: unset !important;
    letter-spacing: -0.3px !important;
  }
  div[data-testid="stMetricDelta"], [data-testid="metric-container"] [data-testid="metric-delta"] {
    font-size: 0.82rem !important;
  }
</style>
""", unsafe_allow_html=True)

BASE_DIR   = _Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
PROC_DIR   = BASE_DIR / "data" / "processed"


def main():
    st.markdown("## 🧠 Model Performance & Evaluation")
    st.caption("Real metrics computed from held-out test data — not illustrative numbers.")
    st.divider()

    # ── Sales Forecasting Metrics ─────────────────────────────────
    st.subheader("📊 Sales Forecasting — Model Comparison")

    metrics_df = load_sales_metrics()

    if metrics_df.empty:
        st.warning("⚠️ No model metrics found. Run `uv run python src/train_all.py` first.")
    else:
        # Determine best model by highest R²
        best_row  = metrics_df.loc[metrics_df["R2"].idxmax()]
        best_name = best_row["Model"]
        best_r2   = best_row["R2"]
        best_mae  = best_row["MAE"]
        best_rmse = best_row["RMSE"]

        # ── KPI strip ────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Best Model",   best_name)
        c2.metric("📉 Best MAE",     inr(best_mae))
        c3.metric("📉 Best RMSE",    inr(best_rmse))
        c4.metric("📈 Best R²",      f"{best_r2:.4f}")

        st.divider()

        col_table, col_chart = st.columns([1, 2])

        with col_table:
            st.markdown("#### 📋 Full Benchmark Table")
            styled = metrics_df.copy()
            styled["MAE"]  = styled["MAE"].apply(lambda x: inr(x))
            styled["RMSE"] = styled["RMSE"].apply(lambda x: inr(x))
            styled["R2"]   = styled["R2"].apply(lambda x: f"{x:.4f}")
            if "WAPE" in styled.columns:
                styled["WAPE"] = styled["WAPE"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Highlight best model
            st.markdown(f"""
            <div style="background:#f6ad5522; border:2px solid #f6ad55; border-radius:10px;
                        padding:14px; text-align:center; margin-top:10px;">
              <div style="color:#f6ad55; font-size:1rem; font-weight:700;">🏆 Best Performing Model</div>
              <div style="color:#e2e8f0; font-size:1.6rem; font-weight:700;">{best_name}</div>
              <div style="color:#a0aec0; font-size:0.85rem;">R² = {best_r2:.4f} · MAE = {inr(best_mae)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            st.markdown("#### 📊 Model Comparison Charts")
            metrics_raw = load_sales_metrics()

            # R² bar chart
            fig_r2 = px.bar(
                metrics_raw, x="Model", y="R2",
                color="Model", text=metrics_raw["R2"].apply(lambda v: f"{v:.4f}"),
                color_discrete_sequence=["#4299e1", "#68d391", "#f6ad55"],
            )
            fig_r2.update_traces(textposition="outside")
            fig_r2.update_layout(
                title="R² Score (higher = better)", height=260,
                plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
                font_color="#a0aec0", showlegend=False,
                yaxis=dict(range=[0, 1.1], gridcolor="#1e3a5f"),
            )
            st.plotly_chart(fig_r2, use_container_width=True)

            # MAE & RMSE grouped bar
            melted = metrics_raw.melt(id_vars="Model", value_vars=["MAE", "RMSE"],
                                       var_name="Metric", value_name="Value")
            fig_err = px.bar(
                melted, x="Model", y="Value", color="Metric",
                barmode="group",
                color_discrete_sequence=["#e53e3e", "#f6ad55"],
                text=melted["Value"].apply(lambda v: inr(v)),
            )
            fig_err.update_traces(textposition="outside")
            fig_err.update_layout(
                title="MAE & RMSE (lower = better)", height=260,
                plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
                font_color="#a0aec0",
                yaxis=dict(gridcolor="#1e3a5f"),
            )
            st.plotly_chart(fig_err, use_container_width=True)

    st.divider()

    # ── Feature Importance ────────────────────────────────────────
    st.subheader("🔍 Feature Importance (Random Forest — Sales Forecasting)")
    fi_df = load_feature_importance()

    if not fi_df.empty:
        fi_df["Importance%"] = (fi_df["Importance"] / fi_df["Importance"].sum() * 100).round(2)
        fig_fi = px.bar(
            fi_df.sort_values("Importance%"),
            x="Importance%", y="Feature", orientation="h",
            color="Importance%", color_continuous_scale="Blues",
            text=fi_df.sort_values("Importance%")["Importance%"].apply(lambda v: f"{v:.1f}%"),
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(
            title="Top Feature Importances (% of total importance)",
            height=420, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.dataframe(fi_df[["Feature", "Importance%"]].sort_values("Importance%", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Feature importance file not found. Train models first.")

    st.divider()

    # ── Customer Segmentation Metrics ─────────────────────────────
    st.subheader("👥 Customer Segmentation — K-Means Evaluation")

    sil = load_cluster_silhouette()
    rfm_path = PROC_DIR / "rfm_clusters.parquet"

    col_sil, col_cluster = st.columns([1, 2])

    with col_sil:
        st.metric("📐 Silhouette Score", f"{sil:.4f}")
        st.markdown("""
        **Interpreting Silhouette Score:**
        | Range | Quality |
        |-------|---------|
        | > 0.7 | Strong clusters |
        | 0.5 – 0.7 | Reasonable |
        | 0.3 – 0.5 | Weak but present |
        | < 0.3 | Poor separation |
        """)

    with col_cluster:
        if rfm_path.exists():
            rfm = pd.read_parquet(rfm_path)
            cluster_summary = rfm.groupby("Segment_Label").agg(
                Count=("Customer ID", "count"),
                Avg_Recency=("Recency", "mean"),
                Avg_Frequency=("Frequency", "mean"),
                Avg_Monetary=("Monetary", "mean"),
            ).reset_index().round(2)

            fig_cs = px.scatter(
                rfm.sample(min(500, len(rfm)), random_state=42),
                x="Recency", y="Monetary", color="Segment_Label",
                size="Frequency", size_max=16, opacity=0.7,
                color_discrete_map={
                    "VIP Customers": "#f6e05e",
                    "Regular Customers": "#68d391",
                    "Occasional Shoppers": "#4299e1",
                    "At-Risk Customers": "#e53e3e",
                },
            )
            fig_cs.update_layout(
                title="Customer Segments — Recency vs Monetary",
                height=340, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
                font_color="#a0aec0",
                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
            )
            st.plotly_chart(fig_cs, use_container_width=True)
        else:
            st.info("Run training pipeline to view cluster plot.")

    st.divider()

    # ── Demand Model Metrics ──────────────────────────────────────
    st.subheader("📦 Demand Forecasting Model Metrics")
    demand_model_path = MODELS_DIR / "demand_rf_model.pkl"
    demand_data_path  = PROC_DIR / "monthly_demand.parquet"

    if demand_model_path.exists() and demand_data_path.exists():
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        demand_model = joblib.load(demand_model_path)
        demand_data  = pd.read_parquet(demand_data_path)

        feat_path = MODELS_DIR / "demand_feature_cols.pkl"
        if feat_path.exists():
            feat_cols = joblib.load(feat_path)
            drop_cols = ["Quantity", "Date", "YearMonthStr", "Month", "Year", "Sales"]
            X = demand_data[[c for c in feat_cols if c in demand_data.columns]].select_dtypes(include=[np.number])
            y = demand_data["Quantity"]

            split = int(len(X) * 0.8)
            X_test, y_test = X.iloc[split:], y.iloc[split:]

            if len(X_test) > 0:
                pred = demand_model.predict(X_test)
                mae  = mean_absolute_error(y_test, pred)
                rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                r2   = r2_score(y_test, pred)

                d1, d2, d3 = st.columns(3)
                d1.metric("📉 MAE (units)",  f"{mae:.2f}")
                d2.metric("📉 RMSE (units)", f"{rmse:.2f}")
                d3.metric("📈 R² Score",     f"{r2:.4f}")
            else:
                st.info("Not enough data for demand model test evaluation.")
        else:
            st.info("Feature columns not found for demand model evaluation.")
    else:
        st.info("Demand model not trained yet. Run `uv run python src/train_all.py` first.")

    # ── Pipeline Summary ──────────────────────────────────────────
    st.divider()
    st.subheader("🗂️ Full Data Science Pipeline")
    st.code("""
    Raw CSV (9,994 rows × 21 cols)
         ↓
    Data Cleaning & Validation
    (dedup, date parsing, outlier clipping, derived cols)
         ↓
    Feature Engineering
    (lags t-1,2,3 · rolling avg · cyclical encoding · one-hot)
         ↓
    Train / Test Split (80/20 — time-ordered)
         ↓
    ┌─────────────────────┬──────────────────────┬─────────────────────┐
    │  Sales Forecasting  │  Demand Forecasting  │ Customer Clustering │
    │  LR · RF · XGBoost  │  Random Forest       │ RFM + KMeans (K=4) │
    │  Eval: MAE/RMSE/R²  │  Eval: MAE/RMSE/R²  │ Eval: Silhouette   │
    └─────────────────────┴──────────────────────┴─────────────────────┘
         ↓
    Joblib model serialization → models/*.pkl
         ↓
    Streamlit Multi-Page Dashboard (7 pages)
    """, language="text")


if __name__ == "__main__":
    main()


