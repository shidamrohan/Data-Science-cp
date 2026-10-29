"""
pages/1_📊_Sales_Analytics.py
─────────────────────────────────────────────────────────
Deep-dive sales analytics with:
  • Multi-dimensional filters (date, region, category, segment)
  • Monthly & weekly revenue/profit trends
  • YoY quarterly comparison
  • Discount impact scatter (the "Discount Trap")
  • Sub-category profitability heatmap
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from preprocessing import load_clean
from formatters import inr, inr_full

st.set_page_config(page_title="Sales Analytics", page_icon="📊", layout="wide")

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


@st.cache_data(show_spinner=False)
def get_data():
    return load_clean()


def main():
    st.markdown("## 📊 Sales Analytics")
    st.caption("Revenue, profit, discount, and trend analysis")
    st.divider()

    df = get_data()

    # ── Sidebar filters ───────────────────────────────────────────
    with st.sidebar:
        st.header("🔍 Filters")

        min_d = df["Order Date"].min().date()
        max_d = df["Order Date"].max().date()
        date_range = st.date_input("📅 Date Range", value=(min_d, max_d),
                                    min_value=min_d, max_value=max_d)

        regions   = st.multiselect("🗺️ Region",   sorted(df["Region"].unique()),   default=sorted(df["Region"].unique()))
        cats      = st.multiselect("📦 Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
        segments  = st.multiselect("👥 Segment",  sorted(df["Segment"].unique()),  default=sorted(df["Segment"].unique()))
        ship_modes = st.multiselect("🚚 Ship Mode", sorted(df["Ship Mode"].unique()), default=sorted(df["Ship Mode"].unique()))

    # ── Apply filters ─────────────────────────────────────────────
    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start, end = df["Order Date"].min(), df["Order Date"].max()

    fdf = df[
        (df["Order Date"] >= start) &
        (df["Order Date"] <= end) &
        (df["Region"].isin(regions)) &
        (df["Category"].isin(cats)) &
        (df["Segment"].isin(segments)) &
        (df["Ship Mode"].isin(ship_modes))
    ].copy()

    if fdf.empty:
        st.warning("No data for selected filters.")
        return

    # ── KPI Strip ─────────────────────────────────────────────────
    total_sales   = fdf["Sales"].sum()
    total_profit  = fdf["Profit"].sum()
    total_orders  = fdf["Order ID"].nunique()
    avg_discount  = fdf["Discount"].mean() * 100
    profit_margin = total_profit / total_sales * 100 if total_sales else 0
    loss_orders   = (fdf["Profit"] < 0).sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Sales",         inr(total_sales))
    c2.metric("📈 Profit",        inr(total_profit))
    c3.metric("🛒 Orders",        f"{total_orders:,}")
    c4.metric("🏷️ Avg Discount",  f"{avg_discount:.1f}%")
    c5.metric("📊 Profit Margin", f"{profit_margin:.1f}%")
    c6.metric("⚠️ Loss Orders",   f"{loss_orders:,}")

    st.divider()

    # ── Tab layout ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Time Trends", "🗺️ Regional", "📦 Category", "🏷️ Discount Impact", "📋 Data Table"
    ])

    # ════════════════════════════════════════
    # TAB 1 — TIME TRENDS
    # ════════════════════════════════════════
    with tab1:
        granularity = st.radio("Granularity", ["Monthly", "Weekly"], horizontal=True)

        if granularity == "Monthly":
            time_grp = fdf.groupby("YearMonthStr").agg(
                Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                Orders=("Order ID", "nunique")
            ).reset_index().sort_values("YearMonthStr")
            x_col = "YearMonthStr"
            x_label = "Month"
        else:
            fdf["Week"] = fdf["Order Date"].dt.strftime("%G-W%V")
            time_grp = fdf.groupby("Week").agg(
                Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                Orders=("Order ID", "nunique")
            ).reset_index().sort_values("Week")
            x_col = "Week"
            x_label = "Week"

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=time_grp[x_col], y=time_grp["Sales"],
                             name="Sales", marker_color="#4299e1", opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=time_grp[x_col], y=time_grp["Profit"],
                                 name="Profit", line=dict(color="#68d391", width=2.5),
                                 mode="lines+markers"), secondary_y=True)
        fig.update_layout(
            title=f"{granularity} Sales & Profit",
            height=360, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", xaxis=dict(tickangle=-45, gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # YoY comparison (quarterly)
        st.subheader("📆 Year-over-Year Quarterly Comparison")
        yoy = fdf.groupby(["Year", "Quarter"])["Sales"].sum().reset_index()
        yoy["Period"] = "Q" + yoy["Quarter"].astype(str)
        fig_yoy = px.bar(
            yoy, x="Period", y="Sales", color=yoy["Year"].astype(str),
            barmode="group", color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_yoy.update_layout(
            height=320, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 2 — REGIONAL ANALYSIS
    # ════════════════════════════════════════
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            reg = fdf.groupby("Region").agg(
                Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                Orders=("Order ID", "nunique")
            ).reset_index().sort_values("Sales", ascending=False)
            reg["Margin%"] = (reg["Profit"] / reg["Sales"] * 100).round(1)

            fig_r = px.bar(reg, x="Region", y=["Sales", "Profit"],
                           barmode="group",
                           color_discrete_sequence=["#4299e1", "#68d391"])
            fig_r.update_layout(
                title="Sales & Profit by Region", height=340,
                plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a", font_color="#a0aec0"
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with col_b:
            state_sales = fdf.groupby("State")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(15)
            fig_s = px.bar(state_sales, x="Sales", y="State", orientation="h",
                           color="Sales", color_continuous_scale="Blues")
            fig_s.update_layout(
                title="Top 15 States by Sales", height=340,
                plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a", font_color="#a0aec0",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_s, use_container_width=True)

        # Region trend over time
        reg_trend = fdf.groupby(["YearMonthStr", "Region"])["Sales"].sum().reset_index()
        fig_rt = px.line(reg_trend, x="YearMonthStr", y="Sales", color="Region",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_rt.update_layout(
            title="Regional Sales Trends Over Time", height=340,
            plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a", font_color="#a0aec0",
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_rt, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 3 — CATEGORY ANALYSIS
    # ════════════════════════════════════════
    with tab3:
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            cat_perf = fdf.groupby(["Category", "Sub-Category"]).agg(
                Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                Qty=("Quantity", "sum")
            ).reset_index()
            cat_perf["Margin%"] = (cat_perf["Profit"] / cat_perf["Sales"] * 100).round(1)

            fig_tree = px.treemap(
                cat_perf, path=[px.Constant("All"), "Category", "Sub-Category"],
                values="Sales", color="Margin%",
                color_continuous_scale=["#e53e3e", "#f6e05e", "#48bb78"],
                color_continuous_midpoint=0,
            )
            fig_tree.update_layout(
                title="Sales Treemap by Category (colored by Margin %)",
                height=420, paper_bgcolor="#0d1b2a", font_color="#a0aec0",
            )
            st.plotly_chart(fig_tree, use_container_width=True)

        with col_c2:
            sub_cat = fdf.groupby("Sub-Category").agg(
                Sales=("Sales", "sum"), Profit=("Profit", "sum")
            ).reset_index().sort_values("Profit")
            sub_cat["Color"] = sub_cat["Profit"].apply(lambda x: "#e53e3e" if x < 0 else "#68d391")

            fig_sc = go.Figure(go.Bar(
                x=sub_cat["Profit"], y=sub_cat["Sub-Category"],
                orientation="h", marker_color=sub_cat["Color"],
            ))
            fig_sc.update_layout(
                title="Profit by Sub-Category (red = loss makers)", height=420,
                plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a", font_color="#a0aec0",
                xaxis=dict(gridcolor="#1e3a5f"),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 4 — DISCOUNT IMPACT
    # ════════════════════════════════════════
    with tab4:
        st.subheader("🏷️ The Discount Trap — Discount % vs Profit Margin")
        st.caption("Transactions with high discount rates that result in negative profit margins.")

        sample = fdf.sample(min(3000, len(fdf)), random_state=42) if len(fdf) > 3000 else fdf
        sample["Discount%"] = sample["Discount"] * 100

        fig_disc = px.scatter(
            sample, x="Discount%", y="Profit_Margin",
            color="Category", size="Sales",
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.7,
            hover_data=["Sub-Category", "Sales", "Profit"],
        )
        fig_disc.add_hline(y=0, line_color="#e53e3e", line_dash="dash",
                            annotation_text="Break-even", annotation_font_color="#e53e3e")
        fig_disc.update_layout(
            height=420, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig_disc, use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            disc_bins = pd.cut(fdf["Discount"] * 100,
                                bins=[0, 5, 10, 20, 30, 40, 100],
                                labels=["0–5%", "5–10%", "10–20%", "20–30%", "30–40%", ">40%"],
                                include_lowest=True)
            disc_grp = fdf.groupby(disc_bins, observed=True).agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
                Orders=("Order ID", "nunique"),
            ).reset_index().rename(columns={"Discount": "Discount Range"})
            fig_dg = px.bar(disc_grp, x="Discount Range", y=["Sales", "Profit"],
                             barmode="group", color_discrete_sequence=["#4299e1", "#68d391"])
            fig_dg.update_layout(
                title="Revenue & Profit by Discount Band", height=320,
                plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a", font_color="#a0aec0",
            )
            st.plotly_chart(fig_dg, use_container_width=True)

        with col_d2:
            disc_sub = fdf.groupby("Sub-Category").agg(
                Avg_Discount=("Discount", "mean"),
                Avg_Margin=("Profit_Margin", "mean"),
            ).reset_index()
            disc_sub["Avg_Discount%"] = disc_sub["Avg_Discount"] * 100
            fig_ds = px.scatter(
                disc_sub, x="Avg_Discount%", y="Avg_Margin",
                text="Sub-Category", size_max=12,
                color="Avg_Margin",
                color_continuous_scale=["#e53e3e", "#f6e05e", "#48bb78"],
            )
            fig_ds.add_hline(y=0, line_color="#e53e3e", line_dash="dash")
            fig_ds.update_traces(textposition="top center", marker_size=10)
            fig_ds.update_layout(
                title="Sub-Category: Avg Discount vs Avg Profit Margin",
                height=320, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
                font_color="#a0aec0", coloraxis_showscale=False,
            )
            st.plotly_chart(fig_ds, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 5 — RAW DATA TABLE
    # ════════════════════════════════════════
    with tab5:
        st.subheader(f"📋 Filtered Data — {len(fdf):,} rows")
        display_cols = ["Order Date", "Customer ID", "Segment", "Category",
                        "Sub-Category", "Product Name", "Region", "State",
                        "Sales", "Quantity", "Discount", "Profit", "Profit_Margin"]
        display_cols = [c for c in display_cols if c in fdf.columns]
        st.dataframe(
            fdf[display_cols].sort_values("Order Date", ascending=False),
            use_container_width=True, hide_index=True,
        )


if __name__ == "__main__":
    main()


