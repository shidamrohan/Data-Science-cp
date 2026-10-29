"""
pages/4_👥_Customer_Segmentation.py
─────────────────────────────────────────────────────────
RFM Customer Segmentation in Clean Minimalist FinTech Light Theme.
Features:
  • Recency, Frequency, Monetary (RFM) clustering (K=4)
  • 4 Actionable retail personas (VIP, Regular, Occasional, At-Risk)
  • Interactive 2D & 3D rotating scatter plots with Plotly
  • Tactical targeting action matrices
  • Export customer RFM scores to CSV & Excel
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path as _Path

from preprocessing import load_clean
from formatters import inr, inr_full, inr_compact
from ui_theme import apply_fintech_theme, update_plotly_theme, render_export_buttons, FINTECH_PALETTE

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")
apply_fintech_theme()

BASE_DIR   = _Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
PROC_DIR   = BASE_DIR / "data" / "processed"

PERSONA_COLORS = {
    "VIP Customers":       "#2563eb",
    "Regular Customers":   "#10b981",
    "Occasional Shoppers": "#f59e0b",
    "At-Risk Customers":   "#ef4444",
}

TARGETING_ACTIONS = {
    "VIP Customers":       "👑 Priority Concierge · Loyalty rewards · Early product launches · Dedicated support",
    "Regular Customers":   "⭐ Cross-sell bundles · Personalized product recommendations · Tier-upgrade incentives",
    "Occasional Shoppers": "🎟️ Seasonal promo vouchers · Re-engagement campaigns · Limited-time discounts",
    "At-Risk Customers":   "⚠️ Automated win-back discounts · Feedback surveys · Reactivation emails",
}


@st.cache_data(show_spinner=False)
def get_rfm():
    p = PROC_DIR / "rfm_clusters.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_data():
    return load_clean()


def main():
    st.markdown("## 👥 Customer Segmentation & RFM Intelligence")
    st.caption("Behavioral customer grouping via Recency, Frequency, and Monetary (RFM) K-Means clustering.")
    st.divider()

    rfm = get_rfm()
    if rfm.empty:
        st.error("⚠️ Customer RFM cluster data not found. Please run python src/train_all.py first.")
        return

    df = get_data()

    # ── Sidebar Filter Controls ───────────────────────────────────
    with st.sidebar:
        st.markdown("<p style='font-weight:700; font-size:0.85rem; color:#0f172a; margin-bottom:4px;'>SEGMENT SELECTION</p>", unsafe_allow_html=True)
        personas = sorted(rfm["Segment_Label"].unique())
        sel_personas = st.multiselect("👥 Cohorts", personas, default=personas)
        min_spend = st.slider("💰 Minimum Spend (₹)", 0, int(rfm["Monetary"].max()), 0, step=5000)

    frf = rfm[(rfm["Segment_Label"].isin(sel_personas)) & (rfm["Monetary"] >= min_spend)]

    # ── KPI Strip ─────────────────────────────────────────────────
    total_custs = len(rfm)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Customers", f"{total_custs:,}")
    for i, persona in enumerate(["VIP Customers", "Regular Customers", "Occasional Shoppers", "At-Risk Customers"]):
        cnt = len(rfm[rfm["Segment_Label"] == persona])
        pct = cnt / total_custs * 100
        [c2, c3, c4, c5][i].metric(persona.split()[0], f"{cnt:,}", delta=f"{pct:.1f}% base", delta_color="off")

    st.write("")

    # ── Segment Strategy Cards ────────────────────────────────────
    st.markdown("#### 🎯 Behavioral Personas & Marketing Action Matrix")
    seg_stats = rfm.groupby("Segment_Label").agg(
        Count=("Customer ID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Spend=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum"),
    ).reset_index()

    cols = st.columns(4)
    for i, row in seg_stats.iterrows():
        persona = row["Segment_Label"]
        color = PERSONA_COLORS.get(persona, "#64748b")
        action = TARGETING_ACTIONS.get(persona, "—")
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#ffffff; border:1.5px solid {color}; border-radius:14px; padding:18px; box-shadow:0 1px 3px rgba(15,23,42,0.05); margin-bottom:12px;">
              <div style="color:{color}; font-size:1.15rem; font-weight:800;">{persona}</div>
              <div style="color:#0f172a; font-size:1.6rem; font-weight:800; margin:2px 0;">{int(row['Count']):,}</div>
              <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; font-weight:700;">Active Accounts</div>
              <div style="margin:10px 0; border-top:1px solid #f1f5f9; padding-top:8px; font-size:0.83rem; color:#334155; line-height:1.6;">
                📅 Avg Recency: <b>{row['Avg_Recency']:.0f} days</b><br>
                🔄 Avg Frequency: <b>{row['Avg_Frequency']:.1f} orders</b><br>
                💰 Avg Spend: <b>{inr(row['Avg_Spend'])}</b>
              </div>
              <div style="background:#f8fafc; border-radius:8px; padding:8px 10px; font-size:0.75rem; color:#475569; border:1px solid #e2e8f0; font-weight:600;">
                {action}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 2D Cluster Analysis", "🌐 3D Spatial Scatter", "📈 RFM Distributions", "📋 Customer Ledger & Export"
    ])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig2d = px.scatter(
                frf, x="Recency", y="Monetary",
                color="Segment_Label", color_discrete_map=PERSONA_COLORS,
                size="Frequency", size_max=18,
                hover_data=["Customer ID", "Frequency", "Monetary"],
                opacity=0.75,
            )
            update_plotly_theme(fig2d, height=400, title="Recency vs Monetary Spend (Bubble Size = Order Count)")
            fig2d.update_xaxes(title_text="Recency (Days Elapsed)")
            fig2d.update_yaxes(title_text="Total Spend (₹)")
            st.plotly_chart(fig2d, use_container_width=True)

        with col_b:
            fig_pie = px.pie(
                rfm["Segment_Label"].value_counts().reset_index(),
                values="count", names="Segment_Label",
                color="Segment_Label", color_discrete_map=PERSONA_COLORS,
                hole=0.55,
            )
            update_plotly_theme(fig_pie, height=400, title="Customer Share by Cohort")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Total revenue contribution
        fig_rev = px.bar(
            seg_stats.sort_values("Total_Revenue", ascending=False),
            x="Segment_Label", y="Total_Revenue",
            color="Segment_Label", color_discrete_map=PERSONA_COLORS,
            text=seg_stats.sort_values("Total_Revenue", ascending=False)["Total_Revenue"].apply(inr),
        )
        update_plotly_theme(fig_rev, height=320, title="Revenue Contribution by Customer Segment")
        fig_rev.update_yaxes(title_text="Total Revenue (₹)")
        st.plotly_chart(fig_rev, use_container_width=True)

    with tab2:
        st.markdown("#### 🌐 3D Spatial RFM Representation (Rotate & Zoom)")
        sample_3d = frf.sample(min(1000, len(frf)), random_state=42) if len(frf) > 1000 else frf
        fig3d = px.scatter_3d(
            sample_3d, x="Recency", y="Frequency", z="Monetary",
            color="Segment_Label", color_discrete_map=PERSONA_COLORS,
            opacity=0.8, size_max=7, hover_data=["Customer ID"],
        )
        fig3d.update_layout(
            height=600,
            paper_bgcolor="#ffffff",
            font=dict(family="Plus Jakarta Sans", color="#475569"),
            scene=dict(
                xaxis=dict(title="Recency (Days)", backgroundcolor="#f8fafc", gridcolor="#e2e8f0"),
                yaxis=dict(title="Frequency (#)", backgroundcolor="#f8fafc", gridcolor="#e2e8f0"),
                zaxis=dict(title="Monetary (₹)", backgroundcolor="#f8fafc", gridcolor="#e2e8f0"),
            ),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with tab3:
        col_r1, col_r2, col_r3 = st.columns(3)
        metrics_info = [
            (col_r1, "Recency", "Recency (Days since last purchase)", "#2563eb"),
            (col_r2, "Frequency", "Frequency (Distinct orders placed)", "#10b981"),
            (col_r3, "Monetary", "Monetary (Total revenue spend ₹)", "#f59e0b"),
        ]
        for col, m_col, label, col_color in metrics_info:
            fig_box = px.box(
                rfm, x="Segment_Label", y=m_col,
                color="Segment_Label", color_discrete_map=PERSONA_COLORS,
                points="outliers",
            )
            update_plotly_theme(fig_box, height=320, title=label)
            fig_box.update_layout(showlegend=False, xaxis_title="")
            with col:
                st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        st.markdown(f"#### 📋 Customer Account Ledger ({len(frf):,} buyers)")
        c_tbl_hdr, c_tbl_exp = st.columns([3, 2])
        with c_tbl_hdr:
            st.caption("Search individual customers, examine RFM scores, or export segment profiles:")
        
        display = frf[["Customer ID", "Segment_Label", "Recency", "Frequency", "Monetary"]].copy()
        with c_tbl_exp:
            render_export_buttons(display, filename_prefix="customer_rfm_segments")

        display["Monetary"] = display["Monetary"].apply(inr)
        display.columns = ["Customer ID", "Assigned Cohort", "Recency (Days)", "Frequency (# Orders)", "Total Lifetime Spend (₹)"]
        st.dataframe(display.sort_values("Recency (Days)", ascending=True), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
