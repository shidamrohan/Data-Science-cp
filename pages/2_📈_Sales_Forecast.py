"""
pages/2_📈_Sales_Forecast.py
─────────────────────────────────────────────────────────
Sales Revenue Forecasting page.
  • User selects category, region, discount, prev-month sales
  • Loads the best saved model (XGBoost > RF > LR)
  • Displays predicted sales, growth %, forecast chart
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path as _Path

from preprocessing import load_clean
from formatters import inr, inr_full

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")

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


@st.cache_data(show_spinner=False)
def get_data():
    return load_clean()


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the best available model: XGBoost > Random Forest > Linear Regression."""
    for fname, name in [
        ("sales_xgb_model.pkl", "XGBoost"),
        ("sales_rf_model.pkl", "Random Forest"),
        ("sales_lr_model.pkl", "Linear Regression"),
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            return joblib.load(p), name
    return None, None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    scaler = joblib.load(MODELS_DIR / "sales_scaler.pkl") if (MODELS_DIR / "sales_scaler.pkl").exists() else None
    feat_cols = joblib.load(MODELS_DIR / "sales_feature_cols.pkl") if (MODELS_DIR / "sales_feature_cols.pkl").exists() else None
    return scaler, feat_cols


def build_input_row(category, region, month, quarter, year, discount, prev_sales,
                    roll3, roll6, lag2, lag3, feat_cols):
    """Construct a single-row DataFrame matching the training feature matrix."""
    row = {
        "Quarter":       quarter,
        "Month_Sin":     np.sin(2 * np.pi * month / 12),
        "Month_Cos":     np.cos(2 * np.pi * month / 12),
        "Sales_Lag1":    prev_sales,
        "Sales_Lag2":    lag2,
        "Sales_Lag3":    lag3,
        "Sales_Roll3":   roll3,
        "Sales_Roll6":   roll6,
        "Discount":      discount / 100,
        "Quantity":      0,
        "Orders":        0,
    }
    # One-hot columns for Category
    for col in feat_cols:
        if col.startswith("Category_"):
            row[col] = 1 if col == f"Category_{category}" else 0
        if col.startswith("Region_"):
            row[col] = 1 if col == f"Region_{region}" else 0

    df_row = pd.DataFrame([row])
    # Align columns to training feature set
    for c in feat_cols:
        if c not in df_row.columns:
            df_row[c] = 0
    return df_row[feat_cols]


def main():
    st.markdown("## 📈 Sales Revenue Forecast")
    st.caption("Predict future monthly sales based on historical patterns, category, region and discount level.")
    st.divider()

    df = get_data()
    model, model_name = load_model()

    if model is None:
        st.error("⚠️ No trained model found. Please run `uv run python src/train_all.py` first.")
        return

    scaler, feat_cols = load_artifacts()

    # ── Sidebar input panel ───────────────────────────────────────
    with st.sidebar:
        st.header("🎛️ Forecast Settings")
        st.caption(f"Active Model: **{model_name}**")
        st.divider()

        categories = sorted(df["Category"].unique())
        regions    = sorted(df["Region"].unique())

        sel_cat    = st.selectbox("📦 Category",  categories)
        sel_region = st.selectbox("🗺️ Region",   regions)
        sel_month  = st.slider("📅 Forecast Month (1=Jan…12=Dec)", 1, 12, value=int(df["Month"].max() % 12) + 1)
        sel_year   = st.selectbox("📅 Forecast Year", [2017, 2018, 2019])
        sel_disc   = st.slider("🏷️ Discount (%)", 0, 40, value=10)

        # Historical context from data
        hist = df[(df["Category"] == sel_cat) & (df["Region"] == sel_region)]
        monthly_hist = hist.groupby("YearMonthStr")["Sales"].sum().sort_index()
        last_vals = monthly_hist.values[-6:] if len(monthly_hist) >= 6 else monthly_hist.values

        prev_sales = float(last_vals[-1]) if len(last_vals) > 0 else 10000.0
        roll3 = float(np.mean(last_vals[-3:])) if len(last_vals) >= 3 else prev_sales
        roll6 = float(np.mean(last_vals[-6:])) if len(last_vals) >= 6 else prev_sales
        lag2  = float(last_vals[-2]) if len(last_vals) >= 2 else prev_sales
        lag3  = float(last_vals[-3]) if len(last_vals) >= 3 else prev_sales

        prev_sales_input = st.number_input(
            "💲 Previous Month Sales (INR)", min_value=0.0,
            value=round(prev_sales, 2), step=100.0
        )

    # ── Prediction ────────────────────────────────────────────────
    try:
        quarter = (sel_month - 1) // 3 + 1
        X_input = build_input_row(
            sel_cat, sel_region, sel_month, quarter, sel_year,
            sel_disc, prev_sales_input, roll3, roll6, lag2, lag3, feat_cols
        )
        if model_name == "Linear Regression" and scaler:
            prediction = float(model.predict(scaler.transform(X_input))[0])
        else:
            prediction = float(model.predict(X_input)[0])
        prediction = max(0.0, prediction)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    growth = ((prediction - prev_sales_input) / prev_sales_input * 100) if prev_sales_input else 0

    # ── KPI Cards ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Category",           sel_cat)
    c2.metric("🗺️ Region",             sel_region)
    c3.metric("💰 Current Month Sales", inr(prev_sales_input))
    c4.metric(
        "📈 Predicted Sales",
        inr(prediction),
        delta=f"{growth:+.2f}%",
        delta_color="normal",
    )

    st.divider()

    # ── Forecast chart ────────────────────────────────────────────
    st.subheader("📊 Historical vs Forecast")
    months_labels = list(monthly_hist.index[-18:]) if len(monthly_hist) >= 18 else list(monthly_hist.index)
    hist_values   = list(monthly_hist.values[-18:]) if len(monthly_hist) >= 18 else list(monthly_hist.values)
    import calendar
    forecast_label = f"{sel_year}-{sel_month:02d}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months_labels, y=hist_values,
        mode="lines+markers", name="Historical Sales",
        line=dict(color="#4299e1", width=2.5),
        marker=dict(size=6),
    ))
    # Connect last historical point to forecast
    connect_x = [months_labels[-1], forecast_label] if months_labels else [forecast_label]
    connect_y = [hist_values[-1], prediction] if hist_values else [prediction]
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        mode="lines", name="",
        line=dict(color="#f6ad55", width=1.5, dash="dot"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[forecast_label], y=[prediction],
        mode="markers+text", name="Forecast",
        marker=dict(color="#f6ad55", size=14, symbol="star"),
        text=[inr(prediction)],
        textposition="top center",
        textfont=dict(color="#f6ad55", size=13),
    ))
    fig.update_layout(
        height=400, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
        font_color="#a0aec0", xaxis=dict(tickangle=-45, gridcolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", title="Sales (INR)"),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Multi-period projection ────────────────────────────────────
    st.subheader("📆 6-Month Forward Projection (rolling estimate)")
    projections = []
    curr = prev_sales_input
    base_month = sel_month
    base_year  = sel_year
    for i in range(6):
        m = ((base_month + i - 1) % 12) + 1
        y = base_year + (base_month + i - 1) // 12
        q = (m - 1) // 3 + 1
        X_p = build_input_row(sel_cat, sel_region, m, q, y, sel_disc,
                               curr, roll3, roll6, lag2, lag3, feat_cols)
        if model_name == "Linear Regression" and scaler:
            p = max(0.0, float(model.predict(scaler.transform(X_p))[0]))
        else:
            p = max(0.0, float(model.predict(X_p)[0]))
        projections.append({"Month": f"{y}-{m:02d}", "Projected Sales (INR)": round(p, 2)})
        lag3, lag2, curr = lag2, curr, p
        roll3 = np.mean([roll3, curr, p])

    proj_df = pd.DataFrame(projections)
    fig_proj = go.Figure(go.Scatter(
        x=proj_df["Month"], y=proj_df["Projected Sales (INR)"],
        mode="lines+markers+text",
        text=[inr(v) for v in proj_df["Projected Sales (INR)"]],
        textposition="top center", textfont=dict(size=11),
        line=dict(color="#f6ad55", width=2.5, dash="dot"),
        marker=dict(size=10, color="#f6ad55"),
    ))
    fig_proj.update_layout(
        height=320, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
        font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", title="Projected Sales (INR)"),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_proj, use_container_width=True)
    st.dataframe(proj_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()


