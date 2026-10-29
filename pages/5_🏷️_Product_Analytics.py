"""
pages/5_🏷️_Product_Analytics.py
─────────────────────────────────────────────────────────
Product Intelligence module.
  • Top / Bottom 10 by Sales & Profit
  • Loss-making products (negative profit)
  • Category treemap
  • Automated decision-support recommendation cards
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
from formatters import inr, inr_full

st.set_page_config(page_title="Product Analytics", page_icon="🏷️", layout="wide")

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


def growth_rate(series: pd.Series) -> float:
    """Year-over-year growth rate using first vs last year totals."""
    if len(series) < 2:
        return 0.0
    first, last = series.iloc[0], series.iloc[-1]
    return ((last - first) / first * 100) if first != 0 else 0.0


def generate_recommendations(row: pd.Series) -> dict:
    """Generate an automated business recommendation for a product."""
    margin = row["Profit_Margin"]
    growth = row.get("Sales_Growth%", 0)
    discount = row.get("Avg_Discount%", 0)

    if margin > 20 and growth > 10:
        color = "#68d391"
        action = f"📈 Star Product: Consider increasing inventory by ~15–20% to meet growing demand."
        badge = "⭐ Star"
    elif margin < 0:
        color = "#e53e3e"
        action = f"🚨 Loss Leader: Profit margin is negative ({margin:.1f}%). Reduce discount (currently {discount:.1f}%) or renegotiate supplier cost."
        badge = "🚨 Loss Maker"
    elif margin > 15 and growth < 0:
        color = "#f6ad55"
        action = f"⚠️ Declining Product: Good margin ({margin:.1f}%) but sales declining ({growth:.1f}%). Launch promotional campaign."
        badge = "⚠️ Declining"
    elif discount > 25 and margin < 10:
        color = "#fc8181"
        action = f"🏷️ Over-Discounted: {discount:.1f}% average discount is eroding profitability. Cap discounts at 15%."
        badge = "🏷️ Over-Discount"
    elif growth > 15:
        color = "#4299e1"
        action = f"🚀 Fast Growing: Sales up {growth:.1f}%. Increase marketing spend and stock buffer."
        badge = "🚀 Fast Growing"
    else:
        color = "#a0aec0"
        action = "✅ Stable performer. Monitor quarterly for trend changes."
        badge = "✅ Stable"

    return {"color": color, "action": action, "badge": badge}


def main():
    st.markdown("## 🏷️ Product Analytics")
    st.caption("Product performance, profitability analysis, and AI-assisted business recommendations.")
    st.divider()

    df = get_data()

    with st.sidebar:
        st.header("🔍 Filters")
        cats = st.multiselect("📦 Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
        regions = st.multiselect("🗺️ Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
        top_n = st.slider("Top N Products", 5, 20, value=10)

    fdf = df[df["Category"].isin(cats) & df["Region"].isin(regions)].copy()

    # ── Build product-level summary ───────────────────────────────
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

    # Growth: compare first vs last year of sales for each product
    growth_data = (
        fdf.groupby(["Product Name", "Year"])["Sales"]
        .sum()
        .reset_index()
    )
    growth_rates = {}
    for name, grp in growth_data.groupby("Product Name"):
        grp_sorted = grp.sort_values("Year")
        growth_rates[name] = growth_rate(grp_sorted["Sales"])
    prod["Sales_Growth%"] = prod["Product Name"].map(growth_rates).round(2)

    # ── KPIs ──────────────────────────────────────────────────────
    total_prods   = len(prod)
    profitable    = (prod["Total_Profit"] > 0).sum()
    loss_making   = (prod["Total_Profit"] < 0).sum()
    avg_margin    = prod["Profit_Margin"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Products",  f"{total_prods:,}")
    c2.metric("✅ Profitable",      f"{profitable:,}")
    c3.metric("🚨 Loss Making",     f"{loss_making:,}")
    c4.metric("📊 Avg Profit Margin", f"{avg_margin:.1f}%")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "📉 Loss Leaders", "🗂️ Category View", "💡 Recommendations"])

    # ════════════════════════════════════════
    # TAB 1 — RANKINGS
    # ════════════════════════════════════════
    with tab1:
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader(f"🏆 Top {top_n} by Sales")
            top_sales = prod.nlargest(top_n, "Total_Sales")
            fig_ts = px.bar(
                top_sales, x="Total_Sales", y="Product Name", orientation="h",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_ts.update_layout(
                height=400, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
                font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"), yaxis_title="",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            st.subheader(f"📉 Bottom {top_n} by Sales")
            bot_sales = prod.nsmallest(top_n, "Total_Sales")
            fig_bs = px.bar(
                bot_sales, x="Total_Sales", y="Product Name", orientation="h",
                color_discrete_sequence=["#4a5568"],
            )
            fig_bs.update_layout(
                height=400, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
                font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"), yaxis_title="",
            )
            st.plotly_chart(fig_bs, use_container_width=True)

        with col_r:
            st.subheader(f"💰 Top {top_n} by Profit")
            top_profit = prod.nlargest(top_n, "Total_Profit")
            fig_tp = px.bar(
                top_profit, x="Total_Profit", y="Product Name", orientation="h",
                color="Total_Profit", color_continuous_scale="Greens",
            )
            fig_tp.update_layout(
                height=400, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
                font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"),
                coloraxis_showscale=False, yaxis_title="",
            )
            st.plotly_chart(fig_tp, use_container_width=True)

            st.subheader(f"🚨 Top {top_n} Loss Makers by Profit")
            bot_profit = prod.nsmallest(top_n, "Total_Profit")
            fig_bp = px.bar(
                bot_profit, x="Total_Profit", y="Product Name", orientation="h",
                color_discrete_sequence=["#e53e3e"],
            )
            fig_bp.update_layout(
                height=400, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
                font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"), yaxis_title="",
            )
            st.plotly_chart(fig_bp, use_container_width=True)

        st.subheader(f"🚀 Top {top_n} Fastest Growing Products")
        fast_growing = prod.dropna(subset=["Sales_Growth%"]).nlargest(top_n, "Sales_Growth%")
        fig_fg = px.bar(
            fast_growing, x="Sales_Growth%", y="Product Name", orientation="h",
            color="Sales_Growth%", color_continuous_scale="Blues",
        )
        fig_fg.update_layout(
            height=360, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"),
            coloraxis_showscale=False, yaxis_title="",
        )
        st.plotly_chart(fig_fg, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 2 — LOSS LEADERS
    # ════════════════════════════════════════
    with tab2:
        loss = prod[prod["Total_Profit"] < 0].sort_values("Total_Profit")
        st.markdown(f"### 🚨 {len(loss):,} Products Generating Losses")

        fig_loss = go.Figure(go.Bar(
            x=loss["Total_Profit"], y=loss["Product Name"],
            orientation="h",
            marker_color="#e53e3e",
            text=loss["Total_Profit"].apply(lambda v: inr(v)),
            textposition="outside",
        ))
        fig_loss.update_layout(
            title="Loss-Making Products (All)",
            height=max(400, len(loss) * 25),
            plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        st.subheader("📋 Loss Leader Detail Table")
        loss_display = loss[["Product Name", "Category", "Sub-Category",
                               "Total_Sales", "Total_Profit", "Profit_Margin",
                               "Avg_Discount%", "Total_Qty"]].copy()
        loss_display.columns = ["Product", "Category", "Sub-Category",
                                  "Sales ($)", "Profit ($)", "Margin %",
                                  "Avg Discount %", "Units Sold"]
        st.dataframe(loss_display, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════
    # TAB 3 — CATEGORY VIEW
    # ════════════════════════════════════════
    with tab3:
        sub_perf = (
            fdf.groupby(["Category", "Sub-Category"])
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Qty=("Quantity", "sum"))
            .reset_index()
        )
        sub_perf["Margin%"] = (sub_perf["Profit"] / sub_perf["Sales"] * 100).round(1)

        fig_tree = px.treemap(
            sub_perf, path=[px.Constant("All"), "Category", "Sub-Category"],
            values="Sales", color="Margin%",
            color_continuous_scale=["#e53e3e", "#f6e05e", "#48bb78"],
            color_continuous_midpoint=0,
            hover_data={"Profit": ":,.0f", "Qty": True},
        )
        fig_tree.update_layout(
            title="Category / Sub-Category Sales Treemap (color = Profit Margin %)",
            height=500, paper_bgcolor="#0d1b2a", font_color="#a0aec0",
        )
        st.plotly_chart(fig_tree, use_container_width=True)

        fig_quad = px.scatter(
            sub_perf, x="Sales", y="Profit",
            size="Qty", color="Category",
            text="Sub-Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_quad.add_hline(y=0, line_dash="dash", line_color="#e53e3e",
                            annotation_text="Break-even", annotation_font_color="#e53e3e")
        fig_quad.update_traces(textposition="top center")
        fig_quad.update_layout(
            title="Sub-Category: Sales vs Profit Quadrant (bubble = units sold)",
            height=440, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 4 — AUTOMATED RECOMMENDATIONS
    # ════════════════════════════════════════
    with tab4:
        st.markdown("### 💡 Automated Product Recommendations")
        st.caption("System-generated insights based on profit margin, growth rate, and discount level.")

        search = st.text_input("🔍 Search product name", "")
        filtered = prod[prod["Product Name"].str.contains(search, case=False)] if search else prod

        # Sort: loss makers first, then fast growing, then rest
        filtered = filtered.sort_values("Total_Profit")

        for _, row in filtered.head(30).iterrows():
            rec = generate_recommendations(row)
            with st.expander(f"{rec['badge']}  {row['Product Name'][:70]}"):
                r1, r2, r3 = st.columns(3)
                r1.metric("💰 Total Sales",    inr(row['Total_Sales']))
                r2.metric("📈 Profit Margin",  f"{row['Profit_Margin']:.1f}%")
                r3.metric("🚀 YoY Growth",     f"{row.get('Sales_Growth%', 0):.1f}%")

                r4, r5, r6 = st.columns(3)
                r4.metric("📦 Units Sold",     f"{int(row['Total_Qty']):,}")
                r5.metric("🏷️ Avg Discount",  f"{row['Avg_Discount%']:.1f}%")
                r6.metric("🛒 Orders",         f"{int(row['Orders']):,}")

                st.markdown(f"""
                <div class="insight-card" style="background:{rec['color']}18; border:1.5px solid {rec['color']};">
                  <strong style="color:{rec['color']};">💡 Recommendation</strong><br>
                  <span style="color:#e2e8f0;">{rec['action']}</span>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


