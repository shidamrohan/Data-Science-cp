"""
app.py — Smart Retail Analytics: Executive Overview Dashboard
─────────────────────────────────────────────────────────────
Executive Dashboard in Clean Minimalist FinTech Light Theme.
Features:
  • Executive Hero Banner with automated business insights
  • Full Indian Rupee metrics (₹) with zero ellipsis truncation
  • Dual-axis Monthly Trend & Category breakdown
  • Regional performance & top sub-categories
  • Quick Export Buttons (CSV / Excel)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from preprocessing import load_clean
from formatters import inr, inr_full, inr_compact
from ui_theme import apply_fintech_theme, update_plotly_theme, render_export_buttons, FINTECH_PALETTE

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Retail Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apply Clean FinTech Light Theme ───────────────────────────────
apply_fintech_theme()


@st.cache_data(show_spinner=False)
def get_data():
    return load_clean()


def main():
    with st.spinner("Loading enterprise dataset…"):
        df = get_data()

    # ── Sidebar Filters ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
          <div style="background:#2563eb; color:#ffffff; font-size:1.4rem; padding:8px 12px; border-radius:10px; font-weight:800;">
            🏪
          </div>
          <div>
            <div style="font-weight:800; font-size:1.05rem; color:#0f172a; line-height:1.2;">Smart Retail</div>
            <div style="font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase;">Analytics Suite</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Retail Decision-Support System")
        st.divider()

        st.markdown("<p style='font-weight:700; font-size:0.85rem; color:#0f172a; margin-bottom:4px;'>FILTER CRITERIA</p>", unsafe_allow_html=True)
        years = sorted(df["Year"].dropna().unique())
        sel_years = st.multiselect("📅 Order Year", years, default=years)

        regions = sorted(df["Region"].dropna().unique())
        sel_regions = st.multiselect("🗺️ Geographic Region", regions, default=regions)

        cats = sorted(df["Category"].dropna().unique())
        sel_cats = st.multiselect("📦 Product Category", cats, default=cats)

        st.divider()
        st.caption("Active Module: **Executive Overview**")
        st.caption("Navigate across 6 modules using the pages menu ↑")

    # ── Apply Filters ─────────────────────────────────────────────
    mask = (
        df["Year"].isin(sel_years) &
        df["Region"].isin(sel_regions) &
        df["Category"].isin(sel_cats)
    )
    fdf = df[mask].copy()

    if fdf.empty:
        st.warning("⚠️ No records match the active filter criteria. Please broaden your selection in the sidebar.")
        return

    # ── KPI Calculations ──────────────────────────────────────────
    total_sales     = fdf["Sales"].sum()
    total_profit    = fdf["Profit"].sum()
    total_orders    = fdf["Order ID"].nunique()
    total_customers = fdf["Customer ID"].nunique()
    aov             = total_sales / total_orders if total_orders else 0
    profit_margin   = (total_profit / total_sales * 100) if total_sales else 0

    # Dynamic Insights for Hero Banner
    cat_summary = fdf.groupby("Category").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    cat_summary["Margin"] = (cat_summary["Profit"] / cat_summary["Sales"] * 100).round(1)
    top_cat_row = cat_summary.loc[cat_summary["Sales"].idxmax()]
    top_cat_name = top_cat_row["Category"]
    top_cat_margin = top_cat_row["Margin"]

    reg_summary = fdf.groupby("Region")["Sales"].sum().reset_index()
    top_reg_name = reg_summary.loc[reg_summary["Sales"].idxmax()]["Region"]

    loss_orders_cnt = int((fdf["Profit"] < 0).sum())
    loss_orders_pct = (loss_orders_cnt / len(fdf) * 100) if len(fdf) else 0

    # ── 1. Executive Summary Hero Banner ──────────────────────────
    st.markdown(f"""
    <div class="fintech-hero">
      <div class="fintech-badge">Enterprise Intelligence · Sample Superstore</div>
      <h1>Smart Retail Sales & Customer Analytics</h1>
      <p>
        Unified commercial intelligence platform answering revenue predictions, product demand, and customer targeting.
        Active scope encompasses <b>{len(fdf):,} transactions</b> across {len(sel_years)} years and {len(sel_regions)} regions.
      </p>
      <div class="fintech-chip-container">
        <div class="fintech-chip">
          <span>🏆</span> <b>Top Category:</b> {top_cat_name} ({top_cat_margin}% margin)
        </div>
        <div class="fintech-chip">
          <span>📍</span> <b>Leading Region:</b> {top_reg_name}
        </div>
        <div class="fintech-chip">
          <span>⚠️</span> <b>Discount Watch:</b> {loss_orders_cnt:,} loss orders ({loss_orders_pct:.1f}% of lines)
        </div>
        <div class="fintech-chip">
          <span>👥</span> <b>Customer Base:</b> {total_customers:,} active buyers (AOV: {inr(aov)})
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. KPI Scorecard Strip ────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Total Revenue",    inr(total_sales), delta=f"{inr_compact(total_sales)}", delta_color="off")
    c2.metric("📈 Net Profit",       inr(total_profit), delta=f"{inr_compact(total_profit)}", delta_color="off")
    c3.metric("🛒 Orders Placed",    f"{total_orders:,}")
    c4.metric("👥 Active Customers", f"{total_customers:,}")
    c5.metric("🧾 Avg Order Value",  inr(aov))
    c6.metric("📊 Operating Margin", f"{profit_margin:.1f}%")

    st.write("")

    # ── 3. Row 1: Revenue Trends & Category Breakdown ─────────────
    col_left, col_right = st.columns([2.2, 1.2])

    with col_left:
        st.markdown("<h4 style='margin-bottom:8px;'>📅 Monthly Sales & Profit Velocity</h4>", unsafe_allow_html=True)
        monthly = (
            fdf.groupby("YearMonthStr")
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
            .reset_index()
            .sort_values("YearMonthStr")
        )

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Bar(
                x=monthly["YearMonthStr"], y=monthly["Sales"],
                name="Sales (₹)", marker_color="#2563eb", opacity=0.88,
                hovertemplate="<b>%{x}</b><br>Sales: ₹%{y:,.0f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig_trend.add_trace(
            go.Scatter(
                x=monthly["YearMonthStr"], y=monthly["Profit"],
                name="Net Profit (₹)", line=dict(color="#10b981", width=2.5),
                mode="lines+markers", marker=dict(size=5, color="#10b981"),
                hovertemplate="<b>%{x}</b><br>Profit: ₹%{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig_trend.update_layout(
            height=340,
            xaxis=dict(tickangle=-45),
        )
        update_plotly_theme(fig_trend, height=340)
        fig_trend.update_yaxes(title_text="Revenue (₹)", secondary_y=False)
        fig_trend.update_yaxes(title_text="Profit (₹)", secondary_y=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.markdown("<h4 style='margin-bottom:8px;'>📦 Revenue by Category</h4>", unsafe_allow_html=True)
        fig_pie = px.pie(
            cat_summary, values="Sales", names="Category",
            color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b"],
            hole=0.55,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Sales: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>",
        )
        update_plotly_theme(fig_pie, height=340)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── 4. Row 2: Regional Performance & Top Sub-Categories ───────
    col_reg, col_sub = st.columns([1.2, 1.8])

    with col_reg:
        st.markdown("<h4 style='margin-bottom:8px;'>🗺️ Regional Revenue & Profit</h4>", unsafe_allow_html=True)
        reg_df = (
            fdf.groupby("Region")
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
            .reset_index()
            .sort_values("Sales", ascending=True)
        )
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Bar(
            y=reg_df["Region"], x=reg_df["Sales"], orientation="h",
            name="Sales (₹)", marker_color="#2563eb",
            hovertemplate="<b>%{y}</b><br>Sales: ₹%{x:,.0f}<extra></extra>",
        ))
        fig_reg.add_trace(go.Bar(
            y=reg_df["Region"], x=reg_df["Profit"], orientation="h",
            name="Profit (₹)", marker_color="#10b981",
            hovertemplate="<b>%{y}</b><br>Profit: ₹%{x:,.0f}<extra></extra>",
        ))
        fig_reg.update_layout(barmode="group", height=320)
        update_plotly_theme(fig_reg, height=320)
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_sub:
        st.markdown("<h4 style='margin-bottom:8px;'>🏆 Top 10 Sub-Categories by Revenue</h4>", unsafe_allow_html=True)
        top_sub = (
            fdf.groupby("Sub-Category")["Sales"].sum()
            .sort_values(ascending=True).tail(10).reset_index()
        )
        fig_sub = px.bar(
            top_sub, x="Sales", y="Sub-Category", orientation="h",
            color="Sales", color_continuous_scale=["#bfdbfe", "#2563eb"],
        )
        fig_sub.update_traces(
            hovertemplate="<b>%{y}</b><br>Sales: ₹%{x:,.0f}<extra></extra>",
        )
        fig_sub.update_layout(coloraxis_showscale=False, yaxis_title="")
        update_plotly_theme(fig_sub, height=320)
        st.plotly_chart(fig_sub, use_container_width=True)

    # ── 5. Row 3: Segment Breakdown & Quick Export ────────────────
    st.divider()
    col_t_title, col_t_export = st.columns([2, 1.5])
    with col_t_title:
        st.markdown("<h4 style='margin:0;'>📋 Customer Segment Performance & Quick Export</h4>", unsafe_allow_html=True)
        st.caption("Review aggregated figures or download the filtered dataset for presentations.")
    with col_t_export:
        render_export_buttons(fdf, filename_prefix="superstore_executive_filtered")

    seg_table = (
        fdf.groupby("Segment")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "nunique"),
            Customers=("Customer ID", "nunique"),
        )
        .reset_index()
    )
    seg_table["Profit Margin %"] = (seg_table["Profit"] / seg_table["Sales"] * 100).round(1)
    seg_table["AOV (₹)"] = (seg_table["Sales"] / seg_table["Orders"]).apply(inr)
    seg_table["Total Sales (₹)"] = seg_table["Sales"].apply(inr)
    seg_table["Total Profit (₹)"] = seg_table["Profit"].apply(inr)

    display_seg = seg_table[["Segment", "Total Sales (₹)", "Total Profit (₹)", "Orders", "Customers", "AOV (₹)", "Profit Margin %"]]
    st.dataframe(display_seg, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
