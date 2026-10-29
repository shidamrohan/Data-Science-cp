"""
pages/5_🏷️_Product_Analytics.py
─────────────────────────────────────────────────────────
Product Intelligence & Margin Diagnosis in Clean Minimalist FinTech Light Theme.
Features:
  • Top & Bottom 10 by Revenue and Net Profit
  • Comprehensive Loss Leaders Ledger
  • Category & Sub-Category Profitability Treemap & Quadrants
  • Automated Decision-Support Recommendation Cards
  • Export Catalog Intelligence to CSV & Excel
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import load_clean
from formatters import inr, inr_full, inr_compact
from ui_theme import apply_fintech_theme, update_plotly_theme, render_export_buttons, FINTECH_PALETTE

st.set_page_config(page_title="Product Analytics", page_icon="🏷️", layout="wide")
apply_fintech_theme()


@st.cache_data(show_spinner=False)
def get_data():
    return load_clean()


def growth_rate(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    first, last = series.iloc[0], series.iloc[-1]
    return ((last - first) / first * 100) if first != 0 else 0.0


def generate_recommendations(row: pd.Series) -> dict:
    margin = row["Profit_Margin"]
    growth = row.get("Sales_Growth%", 0)
    discount = row.get("Avg_Discount%", 0)

    if margin > 20 and growth > 10:
        color = "#10b981"
        action = f"📈 Star Performer: Expand warehouse replenishment buffer by ~15–20% to meet accelerating organic demand."
        badge = "⭐ Star Product"
    elif margin < 0:
        color = "#ef4444"
        action = f"🚨 Loss Leader: Operating margin is negative ({margin:.1f}%). Cap promotional discounts (currently {discount:.1f}%) or renegotiate supplier COGS."
        badge = "🚨 Margin Drain"
    elif margin > 15 and growth < 0:
        color = "#f59e0b"
        action = f"⚠️ Declining Cash Cow: Healthy margin ({margin:.1f}%) but revenue velocity is slowing ({growth:.1f}%). Deploy targeted email bundle campaign."
        badge = "⚠️ Declining Velocity"
    elif discount > 25 and margin < 10:
        color = "#f97316"
        action = f"🏷️ Over-Discounted Item: High promotional markdown ({discount:.1f}%) is cannibalizing margin. Enforce a 15% discount cap."
        badge = "🏷️ Over-Discounted"
    elif growth > 15:
        color = "#2563eb"
        action = f"🚀 Fast Mover: Sales growing at +{growth:.1f}% YoY. Increase promotional visibility and feature on homepage."
        badge = "🚀 High Growth"
    else:
        color = "#64748b"
        action = "✅ Stable catalog item. Monitor quarterly for seasonal velocity shifts."
        badge = "✅ Stable Core"

    return {"color": color, "action": action, "badge": badge}


def main():
    st.markdown("## 🏷️ Product Intelligence & Profitability Analytics")
    st.caption("Deep-dive catalog performance, loss-leader diagnosis, and automated commercial decision support.")
    st.divider()

    df = get_data()

    # ── Sidebar Filters ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("<p style='font-weight:700; font-size:0.85rem; color:#0f172a; margin-bottom:4px;'>CATALOG FILTERS</p>", unsafe_allow_html=True)
        cats = st.multiselect("📦 Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
        regions = st.multiselect("🗺️ Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
        top_n = st.slider("Top N Display Limit", 5, 20, value=10)

    fdf = df[df["Category"].isin(cats) & df["Region"].isin(regions)].copy()

    # ── Product Aggregations ──────────────────────────────────────
    prod = (
        fdf.groupby(["Product Name", "Category", "Sub-Category"])
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Total_Qty=("Quantity", "sum"),
            Avg_Discount=("Discount", "mean"),
            Orders=("Order ID", "nunique"),
        )
        .reset_index()
    )
    prod["Profit_Margin"] = (prod["Total_Profit"] / prod["Total_Sales"] * 100).round(2)
    prod["Avg_Discount%"] = (prod["Avg_Discount"] * 100).round(1)
    prod["Revenue_Per_Unit"] = (prod["Total_Sales"] / prod["Total_Qty"]).round(2)

    # YoY Growth Calculation per Product
    growth_data = fdf.groupby(["Product Name", "Year"])["Sales"].sum().reset_index()
    growth_rates = {}
    for name, grp in growth_data.groupby("Product Name"):
        grp_sorted = grp.sort_values("Year")
        growth_rates[name] = growth_rate(grp_sorted["Sales"])
    prod["Sales_Growth%"] = prod["Product Name"].map(growth_rates).round(2)

    # ── KPI Strip ─────────────────────────────────────────────────
    total_prods = len(prod)
    profitable  = (prod["Total_Profit"] > 0).sum()
    loss_making = (prod["Total_Profit"] < 0).sum()
    avg_margin  = prod["Profit_Margin"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Active Catalog Items", f"{total_prods:,}")
    c2.metric("✅ Profitable SKUs",       f"{profitable:,}", delta=f"{profitable/total_prods*100:.1f}% share", delta_color="normal")
    c3.metric("🚨 Loss-Making SKUs",      f"{loss_making:,}", delta=f"-{loss_making/total_prods*100:.1f}% share", delta_color="inverse")
    c4.metric("📊 Catalog Mean Margin",   f"{avg_margin:.1f}%")

    st.write("")

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top & Bottom Rankings", "📉 Loss Leaders Diagnosis", "🗂️ Category Treemap & Quadrant", "💡 Automated Action Advisories"
    ])

    # ════════════════════════════════════════
    # TAB 1 — RANKINGS
    # ════════════════════════════════════════
    with tab1:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown(f"#### 🏆 Top {top_n} SKUs by Revenue")
            top_sales = prod.nlargest(top_n, "Total_Sales")
            fig_ts = px.bar(
                top_sales, x="Total_Sales", y="Product Name", orientation="h",
                color="Category", color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b"],
            )
            fig_ts.update_traces(hovertemplate="<b>%{y}</b><br>Sales: ₹%{x:,.0f}<extra></extra>")
            update_plotly_theme(fig_ts, height=400)
            fig_ts.update_layout(yaxis_title="", xaxis_title="Total Revenue (₹)")
            st.plotly_chart(fig_ts, use_container_width=True)

            st.markdown(f"#### 📉 Bottom {top_n} SKUs by Revenue")
            bot_sales = prod.nsmallest(top_n, "Total_Sales")
            fig_bs = px.bar(
                bot_sales, x="Total_Sales", y="Product Name", orientation="h",
                color_discrete_sequence=["#94a3b8"],
            )
            fig_bs.update_traces(hovertemplate="<b>%{y}</b><br>Sales: ₹%{x:,.0f}<extra></extra>")
            update_plotly_theme(fig_bs, height=400)
            fig_bs.update_layout(yaxis_title="", xaxis_title="Total Revenue (₹)")
            st.plotly_chart(fig_bs, use_container_width=True)

        with col_r:
            st.markdown(f"#### 💰 Top {top_n} Most Profitable SKUs")
            top_profit = prod.nlargest(top_n, "Total_Profit")
            fig_tp = px.bar(
                top_profit, x="Total_Profit", y="Product Name", orientation="h",
                color="Total_Profit", color_continuous_scale=["#a7f3d0", "#059669"],
            )
            fig_tp.update_traces(hovertemplate="<b>%{y}</b><br>Profit: ₹%{x:,.0f}<extra></extra>")
            update_plotly_theme(fig_tp, height=400)
            fig_tp.update_layout(yaxis_title="", coloraxis_showscale=False, xaxis_title="Net Profit (₹)")
            st.plotly_chart(fig_tp, use_container_width=True)

            st.markdown(f"#### 🚨 Top {top_n} Loss-Making SKUs")
            bot_profit = prod.nsmallest(top_n, "Total_Profit")
            fig_bp = px.bar(
                bot_profit, x="Total_Profit", y="Product Name", orientation="h",
                color_discrete_sequence=["#ef4444"],
            )
            fig_bp.update_traces(hovertemplate="<b>%{y}</b><br>Loss: ₹%{x:,.0f}<extra></extra>")
            update_plotly_theme(fig_bp, height=400)
            fig_bp.update_layout(yaxis_title="", xaxis_title="Net Operating Loss (₹)")
            st.plotly_chart(fig_bp, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 2 — LOSS LEADERS
    # ════════════════════════════════════════
    with tab2:
        loss = prod[prod["Total_Profit"] < 0].sort_values("Total_Profit")
        st.markdown(f"#### 🚨 Catalog Loss Leaders ({len(loss):,} items operating at a loss)")
        
        c_loss_hdr, c_loss_exp = st.columns([3, 2])
        with c_loss_hdr:
            st.caption("These items destroy profit margins due to aggressive markdowns or high returns. Export to take supplier action:")
        with c_loss_exp:
            render_export_buttons(loss, filename_prefix="loss_leader_products")

        loss_display = loss[[
            "Product Name", "Category", "Sub-Category", "Total_Sales",
            "Total_Profit", "Profit_Margin", "Avg_Discount%", "Total_Qty"
        ]].copy()
        loss_display["Total_Sales"] = loss_display["Total_Sales"].apply(inr)
        loss_display["Total_Profit"] = loss_display["Total_Profit"].apply(inr)
        loss_display.columns = ["Product Name", "Category", "Sub-Category", "Total Revenue (₹)", "Net Profit / Loss (₹)", "Margin %", "Avg Discount %", "Units Sold"]
        st.dataframe(loss_display, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════
    # TAB 3 — CATEGORY MATRIX & QUADRANT
    # ════════════════════════════════════════
    with tab3:
        sub_perf = (
            fdf.groupby(["Category", "Sub-Category"])
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Qty=("Quantity", "sum"))
            .reset_index()
        )
        sub_perf["Margin%"] = (sub_perf["Profit"] / sub_perf["Sales"] * 100).round(1)

        fig_tree = px.treemap(
            sub_perf, path=[px.Constant("All Products"), "Category", "Sub-Category"],
            values="Sales", color="Margin%",
            color_continuous_scale=["#ef4444", "#fbbf24", "#10b981"],
            color_continuous_midpoint=0,
        )
        update_plotly_theme(fig_tree, height=450, title="Category Sales Treemap (Color = Profit Margin %)")
        st.plotly_chart(fig_tree, use_container_width=True)

        fig_quad = px.scatter(
            sub_perf, x="Sales", y="Profit",
            size="Qty", color="Category", text="Sub-Category",
            color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b"],
        )
        fig_quad.add_hline(y=0, line_dash="dash", line_color="#ef4444",
                           annotation_text="Break-even (0 Profit)", annotation_font_color="#ef4444")
        fig_quad.update_traces(textposition="top center")
        update_plotly_theme(fig_quad, height=450, title="Sales vs Operating Profit Quadrant (Bubble Size = Volume)")
        st.plotly_chart(fig_quad, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 4 — AUTOMATED ACTION ADVISORIES
    # ════════════════════════════════════════
    with tab4:
        st.markdown("#### 💡 Algorithmic Decision-Support Advisories")
        st.caption("Actionable rules engine analyzing margin thresholds, discount elasticity, and growth trajectories.")

        search = st.text_input("🔍 Search specific product name…", "")
        filtered = prod[prod["Product Name"].str.contains(search, case=False)] if search else prod
        filtered = filtered.sort_values("Total_Profit")

        for _, row in filtered.head(25).iterrows():
            rec = generate_recommendations(row)
            with st.expander(f"{rec['badge']} · {row['Product Name'][:75]}"):
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("💰 Total Revenue",   inr(row['Total_Sales']))
                r2.metric("📈 Profit Margin",   f"{row['Profit_Margin']:.1f}%")
                r3.metric("🏷️ Avg Discount",   f"{row['Avg_Discount%']:.1f}%")
                r4.metric("🚀 YoY Growth",      f"{row.get('Sales_Growth%', 0):.1f}%")

                st.markdown(f"""
                <div style="background:#f8fafc; border-left:4px solid {rec['color']}; border-radius:8px; padding:12px 16px; margin-top:8px;">
                  <span style="color:#0f172a; font-weight:700; font-size:0.88rem;">Strategic Action:</span><br>
                  <span style="color:#334155; font-size:0.86rem; line-height:1.5;">{rec['action']}</span>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
