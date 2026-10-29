"""
pages/3_📦_Demand_Forecast.py
─────────────────────────────────────────────────────────
Product Demand Forecasting & Inventory Intelligence in Clean FinTech Light Theme.
Features:
  • Item selection by Product Name or Sub-Category
  • Historical vs Predicted monthly unit demand
  • Automated Reorder Stock recommendation (15% safety stock)
  • Shortage risk status badge (HIGH, MEDIUM, LOW)
  • Full catalog demand ranking and product ledger export
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
from formatters import inr, inr_full, inr_compact
from ui_theme import apply_fintech_theme, update_plotly_theme, render_export_buttons, FINTECH_PALETTE

st.set_page_config(page_title="Demand Forecast", page_icon="📦", layout="wide")
apply_fintech_theme()

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
    if current_stock <= 0:
        return "⚠️ NO STOCK DATA", "#64748b", "rgba(100, 116, 139, 0.1)"
    ratio = (expected - current_stock) / current_stock * 100
    if ratio > 25:
        return "🔴 HIGH SHORTAGE RISK", "#ef4444", "rgba(239, 68, 68, 0.08)"
    elif ratio > -10:
        return "🟡 MEDIUM STOCK RISK", "#f59e0b", "rgba(245, 158, 11, 0.08)"
    else:
        return "🟢 HEALTHY / LOW RISK", "#10b981", "rgba(16, 185, 129, 0.08)"


def project_demand(model, feat_cols, sub_cat, df, prev_qty, roll3, lag2, lag3, n_months=1):
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
            X = X[[c for c in feat_cols if c in X.columns]]
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
    st.markdown("## 📦 Product Demand & Inventory Forecasting")
    st.caption("Unit quantity demand intelligence to prevent stockouts and size warehouse safety buffers.")
    st.divider()

    df = get_data()
    model = load_demand_model()
    tier_df = get_tiers()

    # ── Sidebar Selection ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("<p style='font-weight:700; font-size:0.85rem; color:#0f172a; margin-bottom:4px;'>CATALOG SELECTION</p>", unsafe_allow_html=True)
        lookup_mode = st.radio("Search Catalog By", ["Sub-Category", "Product Name"], horizontal=True)

        if lookup_mode == "Sub-Category":
            sub_cats = sorted(df["Sub-Category"].unique())
            sel_sub = st.selectbox("📦 Target Sub-Category", sub_cats)
            products_in_sub = sorted(df[df["Sub-Category"] == sel_sub]["Product Name"].unique())
            sel_product = st.selectbox("🏷️ Filter Product (Optional)", ["All Items"] + products_in_sub)
        else:
            all_products = sorted(df["Product Name"].unique())
            sel_product = st.selectbox("🏷️ Specific Product Name", all_products)
            sel_sub = df[df["Product Name"] == sel_product]["Sub-Category"].iloc[0]
            st.caption(f"Mapped Sub-Category: **{sel_sub}**")

        st.markdown("<p style='font-weight:700; font-size:0.85rem; color:#0f172a; margin-top:12px; margin-bottom:4px;'>INVENTORY ASSUMPTIONS</p>", unsafe_allow_html=True)
        forecast_months = st.slider("🗓️ Forward Horizon (Months)", 1, 6, value=1)
        current_stock   = st.number_input("🏭 Current Warehouse On-Hand Stock (Units)", min_value=0, value=150, step=10)

    # ── Historical Data Slicing ───────────────────────────────────
    sub_hist = (
        df[df["Sub-Category"] == sel_sub]
        .groupby("YearMonthStr")["Quantity"]
        .sum()
        .reset_index()
        .sort_values("YearMonthStr")
    )

    if lookup_mode == "Product Name" or (lookup_mode == "Sub-Category" and sel_product != "All Items"):
        prod_filter = sel_product if sel_product != "All Items" else None
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
        st.warning("No transactional history found for this item.")
        return

    last_vals   = sub_hist["Quantity"].values
    avg_monthly = float(np.mean(last_vals[-12:])) if len(last_vals) >= 12 else float(np.mean(last_vals))
    prev_qty    = float(last_vals[-1]) if len(last_vals) > 0 else avg_monthly
    roll3       = float(np.mean(last_vals[-3:])) if len(last_vals) >= 3 else avg_monthly
    lag2        = float(last_vals[-2]) if len(last_vals) >= 2 else prev_qty
    lag3        = float(last_vals[-3]) if len(last_vals) >= 3 else prev_qty

    # ── Inference Execution ───────────────────────────────────────
    feat_cols = None
    feat_path = MODELS_DIR / "demand_feature_cols.pkl"
    if feat_path.exists():
        feat_cols = joblib.load(feat_path)

    if model is not None and feat_cols is not None:
        projections = project_demand(model, feat_cols, sel_sub, df,
                                     prev_qty, roll3, lag2, lag3, forecast_months)
    else:
        weights = np.arange(1, len(last_vals) + 1, dtype=float)
        base = float(np.average(last_vals, weights=weights))
        projections = [round(base * (1 + 0.03 * i)) for i in range(forecast_months)]

    total_demand    = sum(projections)
    recommended_qty = int(total_demand * (1 + SAFETY_BUFFER))
    risk_label, risk_color, risk_bg = shortage_risk(total_demand, current_stock)

    # ── Demand Tier ───────────────────────────────────────────────
    if not tier_df.empty and sel_sub in tier_df["Sub-Category"].values:
        tier = tier_df[tier_df["Sub-Category"] == sel_sub]["Demand_Tier"].values[0]
    else:
        tier = "Medium Demand"

    tier_color_map = {
        "High Demand": ("#10b981", "rgba(16, 185, 129, 0.08)"),
        "Medium Demand": ("#f59e0b", "rgba(245, 158, 11, 0.08)"),
        "Low Demand": ("#ef4444", "rgba(239, 68, 68, 0.08)")
    }
    t_color, t_bg = tier_color_map.get(tier, ("#64748b", "rgba(100, 116, 139, 0.08)"))

    # ── KPI Cards ─────────────────────────────────────────────────
    display_name = sel_product if sel_product not in ("All Items", None) else sel_sub
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Item Selected",       display_name[:28])
    c2.metric("🏭 Current On-Hand",     f"{current_stock:,} units")
    c3.metric("📊 Expected Demand",     f"{total_demand:,} units")
    c4.metric("📦 Recommended Reorder", f"{recommended_qty:,} units")
    c5.metric("⚠️ Inventory Status",    risk_label.split()[1] if len(risk_label.split()) > 1 else risk_label)

    st.write("")

    # ── Decision Cards ────────────────────────────────────────────
    col_badge, col_info = st.columns([1.2, 2.2])

    with col_badge:
        st.markdown(f"""
        <div style="background:{t_bg}; border:1.5px solid {t_color}; border-radius:14px; padding:18px; text-align:center; margin-bottom:12px;">
          <div style="color:{t_color}; font-size:1.45rem; font-weight:800;">{tier}</div>
          <div style="color:#64748b; font-size:0.80rem; font-weight:600; text-transform:uppercase; margin-top:2px;">Historical Velocity Tier</div>
          <div style="color:#0f172a; font-size:1.05rem; font-weight:700; margin-top:6px;">
            Avg {avg_monthly:.0f} units / month
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{risk_bg}; border:1.5px solid {risk_color}; border-radius:14px; padding:16px; text-align:center;">
          <div style="color:{risk_color}; font-size:0.82rem; font-weight:700; text-transform:uppercase;">Shortage Risk Score</div>
          <div style="color:{risk_color}; font-size:1.35rem; font-weight:800; margin-top:2px;">{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div class="fintech-card" style="padding:16px 20px;">
          <h4 style="margin:0 0 10px 0; font-size:1.05rem;">📋 Inventory Replenishment Plan</h4>
          <table style="width:100%; font-size:0.88rem; color:#334155;">
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:4px 0; color:#64748b;">Sub-Category / Item</td><td style="font-weight:700; text-align:right;">{sel_sub}</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:4px 0; color:#64748b;">Forecast Planning Horizon</td><td style="font-weight:700; text-align:right;">{forecast_months} month(s)</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:4px 0; color:#64748b;">Total Predicted Demand</td><td style="font-weight:700; text-align:right;">{total_demand:,} units</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:4px 0; color:#64748b;">Safety Buffer Factor</td><td style="font-weight:700; text-align:right;">+{SAFETY_BUFFER*100:.0f}% buffer</td></tr>
            <tr><td style="padding:4px 0; color:#0f172a; font-weight:700;">Recommended Purchase Order</td><td style="font-weight:800; color:#2563eb; font-size:1rem; text-align:right;">{recommended_qty:,} units</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Monthly Unit Chart ────────────────────────────────────────
    col_l, col_r = st.columns([2.5, 1.5])

    with col_l:
        st.markdown(f"#### 📅 Historical Monthly Unit Demand vs Forecast — {display_name}")
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            x=prod_hist["YearMonthStr"], y=prod_hist["Quantity"],
            marker_color="#2563eb", name="Actual Units", opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Actual: %{y:,} units<extra></extra>",
        ))
        last_month_str = sub_hist["YearMonthStr"].iloc[-1]
        last_dt = pd.to_datetime(last_month_str + "-01")
        proj_labels = [(last_dt + pd.DateOffset(months=i+1)).strftime("%Y-%m") for i in range(forecast_months)]
        fig_bars.add_trace(go.Bar(
            x=proj_labels, y=projections,
            marker_color="#10b981", name="Forecast Units",
            hovertemplate="<b>%{x}</b><br>Forecast: %{y:,} units<extra></extra>",
        ))
        fig_bars.add_hline(
            y=avg_monthly, line_dash="dot", line_color="#f59e0b",
            annotation_text=f"Avg {avg_monthly:.0f} units", annotation_font_color="#f59e0b"
        )
        update_plotly_theme(fig_bars, height=360)
        fig_bars.update_yaxes(title_text="Units Sold")
        st.plotly_chart(fig_bars, use_container_width=True)

    with col_r:
        st.markdown("#### 📊 Demand Tier Breakdown")
        if not tier_df.empty:
            tc = tier_df["Demand_Tier"].value_counts().reset_index()
            tc.columns = ["Demand Tier", "Count"]
            cmap = {"High Demand": "#10b981", "Medium Demand": "#f59e0b", "Low Demand": "#ef4444"}
            fig_pie = px.pie(
                tc, values="Count", names="Demand Tier",
                color="Demand Tier", color_discrete_map=cmap, hole=0.55,
            )
            update_plotly_theme(fig_pie, height=360)
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Top Products & Export ─────────────────────────────────────
    st.divider()
    c_p_title, c_p_exp = st.columns([3, 2])
    with c_p_title:
        st.markdown(f"#### 🏷️ Top Selling Products in **{sel_sub}**", unsafe_allow_html=True)
        st.caption("Review individual product volumes or download the catalog breakdown:")
    
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
        .head(15)
    )
    with c_p_exp:
        render_export_buttons(top_prods, filename_prefix=f"demand_top_products_{sel_sub}")

    top_prods["Margin %"] = (top_prods["Profit"] / top_prods["Sales"] * 100).round(1)
    display_top = top_prods.copy()
    display_top["Sales"] = display_top["Sales"].apply(inr)
    display_top["Profit"] = display_top["Profit"].apply(inr)
    st.dataframe(display_top, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
