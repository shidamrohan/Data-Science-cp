"""
pages/2_📈_Sales_Forecast.py
─────────────────────────────────────────────────────────
Sales Revenue Forecasting in Clean Minimalist FinTech Light Theme.
Features:
  • Interactive parameter inputs (Category, Region, Month, Discount)
  • Top ML model predictions with delta growth indicator
  • Historical vs Forecast chart with confidence trajectories
  • 6-Month forward rolling projections table with CSV/Excel export
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
from formatters import inr, inr_full, inr_compact
from ui_theme import apply_fintech_theme, update_plotly_theme, render_export_buttons, FINTECH_PALETTE

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")
apply_fintech_theme()

BASE_DIR   = _Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
PROC_DIR   = BASE_DIR / "data" / "processed"


@st.cache_data(show_spinner=False)
def get_data():
    return load_clean()


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the best available model: Random Forest > XGBoost > Linear Regression."""
    for fname, name in [
        ("sales_rf_model.pkl", "Random Forest Regressor (Best Model)"),
        ("sales_xgb_model.pkl", "XGBoost Regressor"),
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
    for col in feat_cols:
        if col.startswith("Category_"):
            row[col] = 1 if col == f"Category_{category}" else 0
        if col.startswith("Region_"):
            row[col] = 1 if col == f"Region_{region}" else 0

    df_row = pd.DataFrame([row])
    for c in feat_cols:
        if c not in df_row.columns:
            df_row[c] = 0
    return df_row[feat_cols]


def main():
    st.markdown("## 📈 Revenue Predictive Modeling")
    st.caption("Forecast future monthly sales based on historical seasonality, category dynamics, and promotional pricing.")
    st.divider()

    df = get_data()
    model, model_name = load_model()

    if model is None:
        st.error("⚠️ No pre-trained model detected in models/. Please run python src/train_all.py first.")
        return

    scaler, feat_cols = load_artifacts()

    # ── Sidebar Input Controls ────────────────────────────────────
    with st.sidebar:
        st.markdown("<p style='font-weight:700; font-size:0.85rem; color:#0f172a; margin-bottom:4px;'>SCENARIO BUILDER</p>", unsafe_allow_html=True)
        st.info(f"Active Model: **{model_name}**")

        categories = sorted(df["Category"].unique())
        regions    = sorted(df["Region"].unique())

        sel_cat    = st.selectbox("📦 Target Category", categories)
        sel_region = st.selectbox("🗺️ Target Region", regions)
        sel_month  = st.slider("📅 Forecast Month (1=Jan … 12=Dec)", 1, 12, value=int(df["Month"].max() % 12) + 1)
        sel_year   = st.selectbox("📅 Forecast Year", [2017, 2018, 2019])
        sel_disc   = st.slider("🏷️ Planned Promotional Discount (%)", 0, 40, value=10)

        # Baseline context
        hist = df[(df["Category"] == sel_cat) & (df["Region"] == sel_region)]
        monthly_hist = hist.groupby("YearMonthStr")["Sales"].sum().sort_index()
        last_vals = monthly_hist.values[-6:] if len(monthly_hist) >= 6 else monthly_hist.values

        prev_sales = float(last_vals[-1]) if len(last_vals) > 0 else 100000.0
        roll3 = float(np.mean(last_vals[-3:])) if len(last_vals) >= 3 else prev_sales
        roll6 = float(np.mean(last_vals[-6:])) if len(last_vals) >= 6 else prev_sales
        lag2  = float(last_vals[-2]) if len(last_vals) >= 2 else prev_sales
        lag3  = float(last_vals[-3]) if len(last_vals) >= 3 else prev_sales

        prev_sales_input = st.number_input(
            "💰 Prior Month Actual Sales (₹)", min_value=0.0,
            value=round(prev_sales, 2), step=5000.0
        )

    # ── Inference Execution ───────────────────────────────────────
    try:
        quarter = (sel_month - 1) // 3 + 1
        X_input = build_input_row(
            sel_cat, sel_region, sel_month, quarter, sel_year,
            sel_disc, prev_sales_input, roll3, roll6, lag2, lag3, feat_cols
        )
        if "Linear Regression" in model_name and scaler:
            prediction = float(model.predict(scaler.transform(X_input))[0])
        else:
            prediction = float(model.predict(X_input)[0])
        prediction = max(0.0, prediction)
    except Exception as e:
        st.error(f"Prediction inference error: {e}")
        return

    growth = ((prediction - prev_sales_input) / prev_sales_input * 100) if prev_sales_input else 0

    # ── KPI Cards ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Category & Region", f"{sel_cat} · {sel_region}")
    c2.metric("🏷️ Planned Discount", f"{sel_disc}%")
    c3.metric("💰 Baseline Month Sales", inr(prev_sales_input))
    c4.metric(
        "📈 Predicted Sales",
        inr(prediction),
        delta=f"{growth:+.2f}% expected growth",
        delta_color="normal",
    )

    st.write("")

    # ── Historical vs Forecast Trajectory ─────────────────────────
    st.markdown("#### 📊 Historical Sales Trajectory vs Forecast")
    months_labels = list(monthly_hist.index[-18:]) if len(monthly_hist) >= 18 else list(monthly_hist.index)
    hist_values   = list(monthly_hist.values[-18:]) if len(monthly_hist) >= 18 else list(monthly_hist.values)
    forecast_label = f"{sel_year}-{sel_month:02d}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months_labels, y=hist_values,
        mode="lines+markers", name="Historical Actuals",
        line=dict(color="#2563eb", width=2.5),
        marker=dict(size=6, color="#2563eb"),
        hovertemplate="<b>%{x}</b><br>Actual: ₹%{y:,.0f}<extra></extra>",
    ))
    connect_x = [months_labels[-1], forecast_label] if months_labels else [forecast_label]
    connect_y = [hist_values[-1], prediction] if hist_values else [prediction]
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        mode="lines", name="",
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[forecast_label], y=[prediction],
        mode="markers+text", name="Predicted Horizon",
        marker=dict(color="#10b981", size=14, symbol="diamond"),
        text=[inr(prediction)],
        textposition="top center",
        textfont=dict(color="#0f172a", size=12, family="Plus Jakarta Sans"),
        hovertemplate="<b>%{x} Forecast</b><br>Predicted: ₹%{y:,.0f}<extra></extra>",
    ))
    update_plotly_theme(fig, height=380)
    fig.update_yaxes(title_text="Sales Revenue (₹)")
    st.plotly_chart(fig, use_container_width=True)

    # ── 6-Month Forward Projection Simulation ──────────────────────
    st.divider()
    c_p_title, c_p_exp = st.columns([3, 1.5])
    with c_p_title:
        st.markdown("#### 📆 6-Month Rolling Horizon Simulation")
        st.caption("Autoregressive simulation projecting dynamic monthly revenue for the next half-year.")
    
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
        if "Linear Regression" in model_name and scaler:
            p = max(0.0, float(model.predict(scaler.transform(X_p))[0]))
        else:
            p = max(0.0, float(model.predict(X_p)[0]))
        projections.append({
            "Period": f"{y}-{m:02d}",
            "Projected Sales (₹)": inr(p),
            "Estimated Value (Float)": round(p, 2),
        })
        lag3, lag2, curr = lag2, curr, p
        roll3 = np.mean([roll3, curr, p])

    proj_df = pd.DataFrame(projections)
    
    with c_p_exp:
        render_export_buttons(proj_df, filename_prefix=f"forecast_6months_{sel_cat}_{sel_region}")

    col_sim_chart, col_sim_tbl = st.columns([2, 1.2])

    with col_sim_chart:
        fig_proj = go.Figure(go.Scatter(
            x=proj_df["Period"], y=proj_df["Estimated Value (Float)"],
            mode="lines+markers+text",
            text=proj_df["Projected Sales (₹)"],
            textposition="top center",
            textfont=dict(size=11, color="#0f172a"),
            line=dict(color="#2563eb", width=2.5, dash="dot"),
            marker=dict(size=8, color="#2563eb"),
            hovertemplate="<b>%{x}</b><br>Projected: ₹%{y:,.0f}<extra></extra>",
        ))
        update_plotly_theme(fig_proj, height=300)
        fig_proj.update_yaxes(title_text="Projected Sales (₹)")
        st.plotly_chart(fig_proj, use_container_width=True)

    with col_sim_tbl:
        st.dataframe(proj_df[["Period", "Projected Sales (₹)"]], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
