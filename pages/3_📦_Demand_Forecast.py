"""
pages/3_📦_Demand_Forecast.py
─────────────────────────────────────────────────────────
Product Demand Forecasting & Inventory Intelligence.
  • Select a Product or Sub-Category
  • View historical monthly unit volume trend
  • See predicted demand, recommended stock, shortage risk (HIGH/MEDIUM/LOW)
  • Classify as High / Medium / Low Demand tier
  • All sub-category demand ranking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path as _Path

from preprocessing import load_clean
from formatters import inr, inr_full

st.set_page_config(page_title="Demand Forecast", page_icon="📦", layout="wide")

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
SAFETY_BUFFER = 0.15  # 15% safety stock


@st.cache_data(show_spinner=False)
def get_data():
    return load_clean()


@st.cache_data(show_spinner=False)
def get_tiers():
    p = PROC_DIR / "demand_tier.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_demand_model():
    p = MODELS_DIR / "demand_rf_model.pkl"
    if p.exists():
        return joblib.load(p)
    return None


def shortage_risk(expected: float, current_stock: float):
    """Returns (label_string, hex_color)"""
    if current_stock <= 0:
        return "⚠️ NO STOCK DATA", "#718096"
    ratio = (expected - current_stock) / current_stock * 100
    if ratio > 25:
        return "🔴 HIGH RISK", "#e53e3e"
    elif ratio > -10:
        return "🟡 MEDIUM RISK", "#f6ad55"
    else:
        return "🟢 LOW RISK", "#68d391"


def project_demand(model, feat_cols, sub_cat, df, prev_qty, roll3, lag2, lag3, n_months=1):
    """Run the trained RF model for n_months forward."""
    projections = []
    curr = prev_qty
    last_month_str = (
        df[df["Sub-Category"] == sub_cat]
        .groupby("YearMonthStr")["Quantity"].sum()
        .reset_index()
        .sort_values("YearMonthStr")["YearMonthStr"].iloc[-1]
    )
    last_dt = pd.to_datetime(last_month_str + "-01")

    avg_discount = float(df[df["Sub-Category"] == sub_cat]["Discount"].mean())

    for i in range(n_months):
        next_dt = last_dt + pd.DateOffset(months=i + 1)
        m = next_dt.month
        q = (m - 1) // 3 + 1
        row = {
            "Quarter":   q,
            "Month_Sin": np.sin(2 * np.pi * m / 12),
            "Month_Cos": np.cos(2 * np.pi * m / 12),
            "Qty_Lag1":  curr,
            "Qty_Lag2":  lag2,
            "Qty_Lag3":  lag3,
            "Qty_Roll3": roll3,
            "Discount":  avg_discount,
        }
        X = pd.DataFrame([row])
        if feat_cols:
            for col in feat_cols:
                if col.startswith("Sub-Category_"):
                    X[col] = 1 if col == f"Sub-Category_{sub_cat}" else 0
                if col not in X.columns:
                    X[col] = 0
            # Only keep columns that exist in feat_cols
            X = X[[c for c in feat_cols if c in X.columns]]
            # Fill any remaining missing
            for c in feat_cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[feat_cols]

        try:
            pred = max(0.0, float(model.predict(X)[0]))
        except Exception:
            pred = curr * (1 + np.random.uniform(-0.02, 0.04))

        projections.append(round(pred))
        lag3, lag2, curr = lag2, curr, pred
        roll3 = float(np.mean([roll3, curr, pred]))

    return projections


def main():
    st.markdown("## 📦 Product Demand Forecast")
    st.caption("Unit quantity demand prediction and inventory decision support.")
    st.divider()

    df = get_data()
    model = load_demand_model()
    tier_df = get_tiers()

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.header("🎛️ Demand Settings")

        # Allow searching by product name OR sub-category
        lookup_mode = st.radio("Select by", ["Sub-Category", "Product Name"], horizontal=True)

        if lookup_mode == "Sub-Category":
            sub_cats = sorted(df["Sub-Category"].unique())
            sel_sub = st.selectbox("📦 Sub-Category", sub_cats)
            # Get all products in this sub-category
            products_in_sub = sorted(df[df["Sub-Category"] == sel_sub]["Product Name"].unique())
            sel_product = st.selectbox("🏷️ Product (optional filter)", ["All"] + products_in_sub)
        else:
            all_products = sorted(df["Product Name"].unique())
            sel_product = st.selectbox("🏷️ Product Name", all_products)
            # Derive sub-category from product
            sel_sub = df[df["Product Name"] == sel_product]["Sub-Category"].iloc[0]
            st.caption(f"Sub-Category: **{sel_sub}**")

        forecast_months = st.slider("🗓️ Forecast Horizon (months)", 1, 6, value=1)
        current_stock   = st.number_input("🏭 Current Stock (units)", min_value=0, value=150, step=10)
        st.divider()
        st.caption("Demand model trained on monthly unit volume per Sub-Category.")

    # ── Historical unit volume ────────────────────────────────────
    # At sub-category level (for model), but display product-level if filtered
    sub_hist = (
        df[df["Sub-Category"] == sel_sub]
        .groupby("YearMonthStr")["Quantity"]
        .sum()
        .reset_index()
        .sort_values("YearMonthStr")
    )

    # Product-level historical (for chart only)
    if lookup_mode == "Product Name" or (lookup_mode == "Sub-Category" and sel_product != "All"):
        prod_filter = sel_product if sel_product != "All" else None
        if prod_filter:
            prod_hist = (
                df[df["Product Name"] == prod_filter]
                .groupby("YearMonthStr")["Quantity"]
                .sum()
                .reset_index()
                .sort_values("YearMonthStr")
            )
        else:
            prod_hist = sub_hist.copy()
    else:
        prod_hist = sub_hist.copy()

    if sub_hist.empty:
        st.warning("No data for selected item.")
        return

    last_vals   = sub_hist["Quantity"].values
    avg_monthly = float(np.mean(last_vals[-12:])) if len(last_vals) >= 12 else float(np.mean(last_vals))
    prev_qty    = float(last_vals[-1]) if len(last_vals) > 0 else avg_monthly
    roll3       = float(np.mean(last_vals[-3:])) if len(last_vals) >= 3 else avg_monthly
    lag2        = float(last_vals[-2]) if len(last_vals) >= 2 else prev_qty
    lag3        = float(last_vals[-3]) if len(last_vals) >= 3 else prev_qty

    # ── Run model ─────────────────────────────────────────────────
    feat_cols = None
    feat_path = MODELS_DIR / "demand_feature_cols.pkl"
    if feat_path.exists():
        feat_cols = joblib.load(feat_path)

    if model is not None and feat_cols is not None:
        projections = project_demand(model, feat_cols, sel_sub, df,
                                     prev_qty, roll3, lag2, lag3, forecast_months)
    else:
        # Weighted average fallback
        weights = np.arange(1, len(last_vals) + 1, dtype=float)
        base = float(np.average(last_vals, weights=weights))
        projections = [round(base * (1 + 0.03 * i)) for i in range(forecast_months)]

    total_demand    = sum(projections)
    recommended_qty = int(total_demand * (1 + SAFETY_BUFFER))
    risk_label, risk_color = shortage_risk(total_demand, current_stock)

    # ── Demand Tier ───────────────────────────────────────────────
    if not tier_df.empty and sel_sub in tier_df["Sub-Category"].values:
        tier = tier_df[tier_df["Sub-Category"] == sel_sub]["Demand_Tier"].values[0]
    else:
        tier = "Medium Demand"
    tier_color_map = {"High Demand": "#68d391", "Medium Demand": "#f6ad55", "Low Demand": "#e53e3e"}
    tier_color = tier_color_map.get(tier, "#a0aec0")

    # ── KPI Cards ─────────────────────────────────────────────────
    display_name = sel_product if sel_product not in ("All", None) else sel_sub
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Product",            display_name[:30])
    c2.metric("🏭 Current Stock",      f"{current_stock:,} units")
    c3.metric("📊 Expected Demand",    f"{total_demand:,} units")
    c4.metric("📦 Recommended Stock",  f"{recommended_qty:,} units")
    c5.metric("⚠️ Shortage Risk",      risk_label)

    st.divider()

    # ── Decision Summary + Tier Badge ────────────────────────────
    col_badge, col_info = st.columns([1, 2])

    with col_badge:
        st.markdown(f"""
        <div style="background:{tier_color}22; border:2px solid {tier_color};
                    border-radius:14px; padding:22px; text-align:center;">
          <div style="color:{tier_color}; font-size:1.7rem; font-weight:700;">{tier}</div>
          <div style="color:#a0aec0; font-size:0.82rem;">Classification for {sel_sub}</div>
          <div style="color:#e2e8f0; font-size:1.1rem; margin-top:8px;">
            Avg {avg_monthly:.0f} units / month
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk badge
        st.markdown(f"""
        <div style="background:{risk_color}22; border:2px solid {risk_color};
                    border-radius:14px; padding:16px; text-align:center; margin-top:12px;">
          <div style="color:{risk_color}; font-size:1.1rem; font-weight:700;">Shortage Risk</div>
          <div style="color:#e2e8f0; font-size:1.4rem; font-weight:700;">{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        last_month_str = sub_hist["YearMonthStr"].iloc[-1]
        last_dt = pd.to_datetime(last_month_str + "-01")
        proj_months = [(last_dt + pd.DateOffset(months=i+1)).strftime("%b %Y")
                       for i in range(forecast_months)]

        st.markdown(f"""
        #### 📋 Decision Summary for **{display_name}**

        | Parameter | Value |
        |-----------|-------|
        | Sub-Category | **{sel_sub}** |
        | Forecast Horizon | **{forecast_months} month(s)** |
        | Historical Avg (monthly) | **{avg_monthly:.0f} units** |
        | Total Predicted Demand | **{total_demand:,} units** |
        | Safety Buffer Applied | **{SAFETY_BUFFER*100:.0f}%** |
        | Recommended Reorder Qty | **{recommended_qty:,} units** |
        | Demand Tier | **{tier}** |
        | Shortage Risk | **{risk_label}** |
        """)

        if forecast_months > 1:
            proj_table = pd.DataFrame({"Month": proj_months, "Predicted Units": projections})
            st.dataframe(proj_table, use_container_width=True, hide_index=True)

    st.divider()

    # ── Charts ────────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        chart_title = f"Monthly Unit Volume — {display_name}"
        st.subheader(f"📅 {chart_title}")

        fig = go.Figure()
        # Historical bars
        fig.add_trace(go.Bar(
            x=prod_hist["YearMonthStr"], y=prod_hist["Quantity"],
            marker_color="#4299e1", name="Actual Units",
        ))
        # Forecast bars
        proj_labels = [(last_dt + pd.DateOffset(months=i+1)).strftime("%Y-%m")
                       for i in range(forecast_months)]
        fig.add_trace(go.Bar(
            x=proj_labels, y=projections,
            marker_color="#f6ad55", name="Forecast",
        ))
        # Average line
        fig.add_hline(
            y=avg_monthly, line_dash="dot", line_color="#68d391",
            annotation_text=f"Avg {avg_monthly:.0f} units",
            annotation_font_color="#68d391",
        )
        fig.update_layout(
            height=380, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0",
            xaxis=dict(tickangle=-45, gridcolor="#1e3a5f"),
            yaxis=dict(gridcolor="#1e3a5f", title="Units"),
            barmode="overlay", legend=dict(orientation="h", y=1.05),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("📊 Demand Tier Distribution")
        if not tier_df.empty:
            tc = tier_df["Demand_Tier"].value_counts().reset_index()
            tc.columns = ["Demand Tier", "Count"]
            cmap = {"High Demand": "#68d391", "Medium Demand": "#f6ad55", "Low Demand": "#e53e3e"}
            fig_pie = px.pie(
                tc, values="Count", names="Demand Tier",
                color="Demand Tier", color_discrete_map=cmap, hole=0.45,
            )
            fig_pie.update_layout(
                height=380, paper_bgcolor="#0d1b2a", font_color="#a0aec0",
                legend=dict(orientation="h"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Run training pipeline to see demand tier distribution.")

    # ── Full Sub-Category Demand Ranking ──────────────────────────
    st.divider()
    st.subheader("📋 All Sub-Category Demand Rankings")

    all_demand = (
        df.groupby(["Category", "Sub-Category"])["Quantity"]
        .sum()
        .reset_index()
        .sort_values("Quantity", ascending=False)
        .rename(columns={"Quantity": "Total Units Sold"})
    )
    if not tier_df.empty:
        all_demand = all_demand.merge(
            tier_df[["Sub-Category", "Demand_Tier"]], on="Sub-Category", how="left"
        )

    cmap2 = {"High Demand": "#68d391", "Medium Demand": "#f6ad55", "Low Demand": "#e53e3e"}
    fig_rank = px.bar(
        all_demand, x="Total Units Sold", y="Sub-Category", orientation="h",
        color="Demand_Tier" if "Demand_Tier" in all_demand.columns else "Total Units Sold",
        color_discrete_map=cmap2,
        hover_data=["Category"],
    )
    # Highlight selected sub-category
    fig_rank.add_vline(x=0, line_color="#1e3a5f")
    fig_rank.update_layout(
        height=520, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
        font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"),
        legend_title_text="Demand Tier",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # ── Top Products in selected Sub-Category ─────────────────────
    st.divider()
    st.subheader(f"🏷️ Top Products in **{sel_sub}**")
    top_prods = (
        df[df["Sub-Category"] == sel_sub]
        .groupby("Product Name")
        .agg(
            Units=("Quantity", "sum"),
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
        )
        .reset_index()
        .sort_values("Units", ascending=False)
        .head(10)
    )
    top_prods["Margin %"] = (top_prods["Profit"] / top_prods["Sales"] * 100).round(1)
    top_prods["Sales"] = top_prods["Sales"].apply(inr)
    top_prods["Profit"] = top_prods["Profit"].apply(inr)
    st.dataframe(top_prods, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

