"""
pages/1_📊_Sales_Analytics.py
─────────────────────────────────────────────────────────
Deep-dive Sales Analytics in Clean Minimalist FinTech Light Theme.
Features:
  • Date range, region, category, customer segment filters
  • Monthly & weekly revenue and profit velocity
  • YoY quarterly comparison
  • Sub-category profit & loss leader diagnosis
  • The "Discount Trap" analysis with scatter and discount band breakdowns
  • Export to CSV & Excel
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
from formatters import inr, inr_full, inr_compact
from ui_theme import apply_fintech_theme, update_plotly_theme, render_export_buttons, FINTECH_PALETTE

st.set_page_config(page_title="Sales Analytics", page_icon="📊", layout="wide")
apply_fintech_theme()


@st.cache_data(show_spinner=False)
def get_data():
    return load_clean()


def main():
    st.markdown("## 📊 Sales & Commercial Analytics")
    st.caption("Granular diagnostic engine for revenue, operating profit, and discount margin impact.")
    st.divider()

    df = get_data()

    # ── Sidebar Filters ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("<p style='font-weight:700; font-size:0.85rem; color:#0f172a; margin-bottom:4px;'>FILTER PARAMETERS</p>", unsafe_allow_html=True)

        min_d = df["Order Date"].min().date()
        max_d = df["Order Date"].max().date()
        date_range = st.date_input("📅 Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

        regions = st.multiselect("🗺️ Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
        cats = st.multiselect("📦 Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
        segments = st.multiselect("👥 Customer Segment", sorted(df["Segment"].unique()), default=sorted(df["Segment"].unique()))
        ship_modes = st.multiselect("🚚 Shipping Mode", sorted(df["Ship Mode"].unique()), default=sorted(df["Ship Mode"].unique()))

    # ── Apply Filters ─────────────────────────────────────────────
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
        st.warning("⚠️ No records match the active filter criteria. Please broaden your selection.")
        return

    # ── KPI Strip ─────────────────────────────────────────────────
    total_sales   = fdf["Sales"].sum()
    total_profit  = fdf["Profit"].sum()
    total_orders  = fdf["Order ID"].nunique()
    avg_discount  = fdf["Discount"].mean() * 100
    profit_margin = (total_profit / total_sales * 100) if total_sales else 0
    loss_orders   = (fdf["Profit"] < 0).sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Net Revenue",    inr(total_sales), delta=inr_compact(total_sales), delta_color="off")
    c2.metric("📈 Net Profit",     inr(total_profit), delta=inr_compact(total_profit), delta_color="off")
    c3.metric("🛒 Orders",         f"{total_orders:,}")
    c4.metric("🏷️ Avg Discount",   f"{avg_discount:.1f}%")
    c5.metric("📊 Profit Margin",  f"{profit_margin:.1f}%")
    c6.metric("⚠️ Loss Orders",    f"{loss_orders:,}")

    st.write("")

    # ── Quick Export Bar ──────────────────────────────────────────
    c_hdr, c_exp = st.columns([3, 2])
    with c_hdr:
        st.markdown("<p style='font-size:0.9rem; color:#64748b; font-weight:600; margin-top:6px;'>Explore analytical tabs below or export the filtered transactional dataset:</p>", unsafe_allow_html=True)
    with c_exp:
        render_export_buttons(fdf, filename_prefix="sales_analytics_export")

    st.divider()

    # ── Analytical Tabs ───────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Time Velocity", "🗺️ Geographic Performance", "📦 Category Matrix", "🏷️ The Discount Trap", "📋 Raw Data Explorer"
    ])

    # ════════════════════════════════════════
    # TAB 1 — TIME TRENDS
    # ════════════════════════════════════════
    with tab1:
        granularity = st.radio("Time Aggregation Level", ["Monthly", "Weekly"], horizontal=True)

        if granularity == "Monthly":
            time_grp = fdf.groupby("YearMonthStr").agg(
                Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                Orders=("Order ID", "nunique")
            ).reset_index().sort_values("YearMonthStr")
            x_col = "YearMonthStr"
        else:
            fdf["Week"] = fdf["Order Date"].dt.strftime("%G-W%V")
            time_grp = fdf.groupby("Week").agg(
                Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                Orders=("Order ID", "nunique")
            ).reset_index().sort_values("Week")
            x_col = "Week"

        fig_time = make_subplots(specs=[[{"secondary_y": True}]])
        fig_time.add_trace(
            go.Bar(
                x=time_grp[x_col], y=time_grp["Sales"],
                name="Sales (₹)", marker_color="#2563eb", opacity=0.88,
                hovertemplate="<b>%{x}</b><br>Sales: ₹%{y:,.0f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig_time.add_trace(
            go.Scatter(
                x=time_grp[x_col], y=time_grp["Profit"],
                name="Profit (₹)", line=dict(color="#10b981", width=2.5),
                mode="lines+markers", marker=dict(size=4),
                hovertemplate="<b>%{x}</b><br>Profit: ₹%{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig_time.update_layout(xaxis=dict(tickangle=-45))
        update_plotly_theme(fig_time, height=350, title=f"{granularity} Revenue & Net Profit Velocity")
        fig_time.update_yaxes(title_text="Sales (₹)", secondary_y=False)
        fig_time.update_yaxes(title_text="Profit (₹)", secondary_y=True)
        st.plotly_chart(fig_time, use_container_width=True)

        # YoY Quarterly Breakdown
        st.markdown("#### 📆 Year-over-Year (YoY) Quarterly Performance")
        yoy = fdf.groupby(["Year", "Quarter"])["Sales"].sum().reset_index()
        yoy["Period"] = "Q" + yoy["Quarter"].astype(str)
        fig_yoy = px.bar(
            yoy, x="Period", y="Sales", color=yoy["Year"].astype(str),
            barmode="group", color_discrete_sequence=["#93c5fd", "#60a5fa", "#2563eb", "#1d4ed8"],
        )
        fig_yoy.update_traces(hovertemplate="<b>%{x}</b> (%{data.name})<br>Sales: ₹%{y:,.0f}<extra></extra>")
        update_plotly_theme(fig_yoy, height=300)
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

            fig_r = px.bar(
                reg, x="Region", y=["Sales", "Profit"],
                barmode="group", color_discrete_sequence=["#2563eb", "#10b981"],
            )
            fig_r.update_traces(hovertemplate="<b>%{x}</b><br>Amount: ₹%{y:,.0f}<extra></extra>")
            update_plotly_theme(fig_r, height=340, title="Sales & Profit by Region")
            st.plotly_chart(fig_r, use_container_width=True)

        with col_b:
            state_sales = fdf.groupby("State")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(15)
            fig_s = px.bar(
                state_sales, x="Sales", y="State", orientation="h",
                color="Sales", color_continuous_scale=["#bfdbfe", "#1d4ed8"],
            )
            fig_s.update_traces(hovertemplate="<b>%{y}</b><br>Sales: ₹%{x:,.0f}<extra></extra>")
            fig_s.update_layout(coloraxis_showscale=False, yaxis_title="")
            update_plotly_theme(fig_s, height=340, title="Top 15 States by Revenue")
            st.plotly_chart(fig_s, use_container_width=True)

        # Regional trends over time
        reg_trend = fdf.groupby(["YearMonthStr", "Region"])["Sales"].sum().reset_index()
        fig_rt = px.line(
            reg_trend, x="YearMonthStr", y="Sales", color="Region",
            color_discrete_sequence=FINTECH_PALETTE,
        )
        fig_rt.update_layout(xaxis=dict(tickangle=-45))
        update_plotly_theme(fig_rt, height=340, title="Regional Revenue Trajectories")
        st.plotly_chart(fig_rt, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 3 — CATEGORY MATRIX
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
                cat_perf, path=[px.Constant("All Categories"), "Category", "Sub-Category"],
                values="Sales", color="Margin%",
                color_continuous_scale=["#ef4444", "#fbbf24", "#10b981"],
                color_continuous_midpoint=0,
            )
            update_plotly_theme(fig_tree, height=420, title="Category Treemap (Color = Profit Margin %)")
            st.plotly_chart(fig_tree, use_container_width=True)

        with col_c2:
            sub_cat = fdf.groupby("Sub-Category").agg(
                Sales=("Sales", "sum"), Profit=("Profit", "sum")
            ).reset_index().sort_values("Profit")
            sub_cat["Color"] = sub_cat["Profit"].apply(lambda x: "#ef4444" if x < 0 else "#10b981")

            fig_sc = go.Figure(go.Bar(
                x=sub_cat["Profit"], y=sub_cat["Sub-Category"],
                orientation="h", marker_color=sub_cat["Color"],
                hovertemplate="<b>%{y}</b><br>Profit: ₹%{x:,.0f}<extra></extra>",
            ))
            fig_sc.update_layout(yaxis_title="")
            update_plotly_theme(fig_sc, height=420, title="Profitability by Sub-Category (Red = Loss)")
            st.plotly_chart(fig_sc, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 4 — THE DISCOUNT TRAP
    # ════════════════════════════════════════
    with tab4:
        st.markdown("#### 🏷️ Empirical Diagnosis: The Discount Trap")
        st.caption("Transactions where heavy promotional discounts (>20%) push operating profit into negative territory.")

        sample = fdf.sample(min(3000, len(fdf)), random_state=42) if len(fdf) > 3000 else fdf
        sample["Discount%"] = sample["Discount"] * 100

        fig_disc = px.scatter(
            sample, x="Discount%", y="Profit_Margin",
            color="Category", size="Sales",
            color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b"],
            opacity=0.65,
            hover_data=["Sub-Category", "Sales", "Profit"],
        )
        fig_disc.add_hline(y=0, line_color="#ef4444", line_dash="dash",
                           annotation_text="Break-even (0% Margin)", annotation_font_color="#ef4444")
        update_plotly_theme(fig_disc, height=420, title="Discount % vs Profit Margin per Order")
        st.plotly_chart(fig_disc, use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            disc_bins = pd.cut(
                fdf["Discount"] * 100,
                bins=[0, 5, 10, 20, 30, 40, 100],
                labels=["0–5%", "5–10%", "10–20%", "20–30%", "30–40%", ">40%"],
                include_lowest=True
            )
            disc_grp = fdf.groupby(disc_bins, observed=True).agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
                Orders=("Order ID", "nunique"),
            ).reset_index().rename(columns={"Discount": "Discount Band"})
            fig_dg = px.bar(
                disc_grp, x="Discount Band", y=["Sales", "Profit"],
                barmode="group", color_discrete_sequence=["#2563eb", "#10b981"]
            )
            update_plotly_theme(fig_dg, height=320, title="Sales & Profit by Discount Band")
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
                color_continuous_scale=["#ef4444", "#fbbf24", "#10b981"],
            )
            fig_ds.add_hline(y=0, line_color="#ef4444", line_dash="dash")
            fig_ds.update_traces(textposition="top center", marker_size=10)
            fig_ds.update_layout(coloraxis_showscale=False)
            update_plotly_theme(fig_ds, height=320, title="Sub-Category: Average Discount vs Margin")
            st.plotly_chart(fig_ds, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 5 — RAW DATA TABLE
    # ════════════════════════════════════════
    with tab5:
        st.markdown(f"#### 📋 Transaction Record Ledger ({len(fdf):,} rows)")
        display_cols = ["Order Date", "Customer ID", "Segment", "Category",
                        "Sub-Category", "Product Name", "Region", "State",
                        "Sales", "Quantity", "Discount", "Profit", "Profit_Margin"]
        display_cols = [c for c in display_cols if c in fdf.columns]
        
        display_df = fdf[display_cols].sort_values("Order Date", ascending=False).copy()
        display_df["Sales"] = display_df["Sales"].apply(inr)
        display_df["Profit"] = display_df["Profit"].apply(inr)
        display_df["Discount"] = (display_df["Discount"] * 100).round(1).astype(str) + "%"
        display_df["Profit_Margin"] = display_df["Profit_Margin"].round(1).astype(str) + "%"
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
