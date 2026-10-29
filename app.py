"""
app.py — Smart Retail Analytics: Executive Overview Dashboard
─────────────────────────────────────────────────────────────
Main landing page showing global KPIs, monthly revenue trends,
category performance, regional breakdown, and top products.
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
from formatters import inr, inr_full

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Retail Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  div[data-testid="stMetric"], [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 12px 14px;
    min-width: 0 !important;
  }
  div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] p, [data-testid="metric-container"] label {
    color: #a0aec0 !important;
    font-size: 0.82rem !important;
    white-space: nowrap !important;
  }
  div[data-testid="stMetricValue"],
  div[data-testid="stMetricValue"] > div,
  div[data-testid="stMetricValue"] *,
  [data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e2e8f0 !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: unset !important;
    letter-spacing: -0.3px !important;
  }
  div[data-testid="stMetricDelta"], [data-testid="metric-container"] [data-testid="metric-delta"] {
    font-size: 0.82rem;
  }
  .block-container { padding-top: 1.5rem; }
  h1 { color: #e2e8f0; }
  h2, h3 { color: #cbd5e0; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    return load_clean()


def main():
    # ── Header ────────────────────────────────────────────────────
    st.markdown("## 🏪 Smart Retail Analytics")
    st.markdown("**Executive Overview** — Sample Superstore Dataset")
    st.divider()

    with st.spinner("Loading data…"):
        df = get_data()

    # ── Sidebar filters ───────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/store.png", width=60)
        st.title("📊 Retail Analytics")
        st.caption("Smart Retail Analytics System")
        st.divider()

        years = sorted(df["Year"].dropna().unique())
        sel_years = st.multiselect("📅 Year", years, default=years)

        regions = sorted(df["Region"].dropna().unique())
        sel_regions = st.multiselect("🗺️ Region", regions, default=regions)

        cats = sorted(df["Category"].dropna().unique())
        sel_cats = st.multiselect("📦 Category", cats, default=cats)

        st.divider()
        st.caption("Navigate using the sidebar pages ↑")

    # ── Apply filters ─────────────────────────────────────────────
    mask = (
        df["Year"].isin(sel_years) &
        df["Region"].isin(sel_regions) &
        df["Category"].isin(sel_cats)
    )
    fdf = df[mask].copy()

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    # ── KPI calculations ──────────────────────────────────────────
    total_sales    = fdf["Sales"].sum()
    total_profit   = fdf["Profit"].sum()
    total_orders   = fdf["Order ID"].nunique()
    total_customers = fdf["Customer ID"].nunique()
    aov            = total_sales / total_orders if total_orders else 0
    profit_margin  = (total_profit / total_sales * 100) if total_sales else 0

    # ── KPI Cards ────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Total Sales",      inr(total_sales))
    c2.metric("📈 Total Profit",     inr(total_profit))
    c3.metric("🛒 Total Orders",     f"{total_orders:,}")
    c4.metric("👥 Customers",        f"{total_customers:,}")
    c5.metric("🧾 Avg Order Value",  inr(aov))
    c6.metric("📊 Profit Margin",    f"{profit_margin:.1f}%")

    st.divider()

    # ── Row 1: Monthly Revenue Trend + Category Breakdown ─────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📅 Monthly Sales & Profit Trend")
        monthly = (
            fdf.groupby("YearMonthStr")
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
            .reset_index()
            .sort_values("YearMonthStr")
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=monthly["YearMonthStr"], y=monthly["Sales"],
                   name="Sales", marker_color="#4299e1", opacity=0.85),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=monthly["YearMonthStr"], y=monthly["Profit"],
                       name="Profit", line=dict(color="#68d391", width=2.5), mode="lines+markers"),
            secondary_y=True,
        )
        fig.update_layout(
            height=340, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", legend=dict(orientation="h", y=1.05),
            xaxis=dict(tickangle=-45),
        )
        fig.update_yaxes(title_text="Sales ($)", secondary_y=False, gridcolor="#1e3a5f")
        fig.update_yaxes(title_text="Profit ($)", secondary_y=True, gridcolor="#1e3a5f")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📦 Sales by Category")
        cat_data = fdf.groupby("Category")["Sales"].sum().reset_index()
        fig2 = px.pie(
            cat_data, values="Sales", names="Category",
            color_discrete_sequence=["#4299e1", "#68d391", "#f6ad55"],
            hole=0.45,
        )
        fig2.update_layout(
            height=340, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", legend=dict(orientation="h"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Regional + Top Products ────────────────────────────
    col_l2, col_r2 = st.columns([1, 1])

    with col_l2:
        st.subheader("🗺️ Sales & Profit by Region")
        reg = (
            fdf.groupby("Region")
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
            .reset_index()
            .sort_values("Sales", ascending=True)
        )
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=reg["Sales"], y=reg["Region"], orientation="h",
            name="Sales", marker_color="#4299e1",
        ))
        fig3.add_trace(go.Bar(
            x=reg["Profit"], y=reg["Region"], orientation="h",
            name="Profit", marker_color="#68d391",
        ))
        fig3.update_layout(
            barmode="group", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.subheader("🏆 Top 10 Sub-Categories by Sales")
        top_sub = (
            fdf.groupby("Sub-Category")["Sales"].sum()
            .sort_values(ascending=True).tail(10).reset_index()
        )
        fig4 = px.bar(
            top_sub, x="Sales", y="Sub-Category", orientation="h",
            color="Sales", color_continuous_scale="Blues",
        )
        fig4.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Segment summary table ──────────────────────────────
    st.divider()
    st.subheader("📋 Customer Segment Summary")
    seg = (
        fdf.groupby("Segment")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "nunique"),
            Customers=("Customer ID", "nunique"),
        )
        .reset_index()
    )
    seg["Profit Margin %"] = (seg["Profit"] / seg["Sales"] * 100).round(1)
    seg["Sales"] = seg["Sales"].apply(inr)
    seg["Profit"] = seg["Profit"].apply(inr)
    st.dataframe(seg, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()


