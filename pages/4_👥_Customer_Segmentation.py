"""
pages/4_👥_Customer_Segmentation.py
─────────────────────────────────────────────────────────
RFM Customer Segmentation viewer.
  • Cluster summary cards (VIP / Regular / Occasional / At-Risk)
  • 2D scatter: Recency vs Monetary
  • 3D scatter: R × F × M
  • Business targeting recommendation matrix
  • Customer-level searchable data table
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
from formatters import inr, inr_full

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

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

PERSONA_COLORS = {
    "VIP Customers":       "#f6e05e",
    "Regular Customers":   "#68d391",
    "Occasional Shoppers": "#4299e1",
    "At-Risk Customers":   "#e53e3e",
}

TARGETING_ACTIONS = {
    "VIP Customers":       "👑 Priority Access · Loyalty rewards · Exclusive early offers · Personal account manager",
    "Regular Customers":   "⭐ Cross-sell bundles · Personalized recommendations · Tier upgrade nudges",
    "Occasional Shoppers": "🎟️ Discount campaigns · Seasonal promotions · Re-engagement emails",
    "At-Risk Customers":   "⚠️ Win-back vouchers · Automated re-activation flow · Survey to understand churn reason",
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
    st.markdown("## 👥 Customer Segmentation")
    st.caption("RFM Analysis + K-Means Clustering (K=4) — Business-Driven Persona Discovery")
    st.divider()

    rfm = get_rfm()
    if rfm.empty:
        st.error("⚠️ Cluster data not found. Please run `uv run python src/train_all.py` first.")
        return

    df = get_data()

    with st.sidebar:
        st.header("🔍 Filters")
        personas = sorted(rfm["Segment_Label"].unique())
        sel_personas = st.multiselect("👥 Segment", personas, default=personas)
        min_spend = st.slider("💰 Min Total Spend (INR)", 0, int(rfm["Monetary"].max()), 0, step=100)
        st.divider()
        st.caption("Based on Recency, Frequency & Monetary (RFM) features.")

    frf = rfm[(rfm["Segment_Label"].isin(sel_personas)) & (rfm["Monetary"] >= min_spend)]

    # ── Summary KPIs ─────────────────────────────────────────────
    total_custs = len(rfm)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Customers", f"{total_custs:,}")
    for i, persona in enumerate(["VIP Customers", "Regular Customers", "Occasional Shoppers", "At-Risk Customers"]):
        cnt = len(rfm[rfm["Segment_Label"] == persona])
        pct = cnt / total_custs * 100
        [c2, c3, c4, c5][i].metric(persona.split()[0], f"{cnt:,}", delta=f"{pct:.1f}%")

    st.divider()

    # ── Segment Cards ─────────────────────────────────────────────
    st.subheader("🎯 Segment Profiles & Targeting Actions")
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
        color = PERSONA_COLORS.get(persona, "#a0aec0")
        action = TARGETING_ACTIONS.get(persona, "—")
        with cols[i % 4]:
            st.markdown(f"""
            <div class="seg-card" style="background:{color}18; border:2px solid {color};">
              <div style="color:{color}; font-size:1.3rem; font-weight:700;">{persona}</div>
              <div style="color:#e2e8f0; font-size:1.6rem; font-weight:700;">{int(row['Count']):,}</div>
              <div style="color:#a0aec0; font-size:0.78rem;">customers</div>
              <hr style="border-color:{color}44; margin:8px 0;">
              <div style="color:#e2e8f0; font-size:0.82rem;">
                📅 Avg Recency: <b>{row['Avg_Recency']:.0f} days</b><br>
                🔄 Avg Orders: <b>{row['Avg_Frequency']:.1f}</b><br>
                💰 Avg Spend: <b>{inr(row['Avg_Spend'])}</b>
              </div>
              <hr style="border-color:{color}44; margin:8px 0;">
              <div style="color:{color}; font-size:0.78rem;">{action}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 2D Cluster View", "🌐 3D Cluster View", "📈 RFM Distributions", "📋 Customer Table"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig2d = px.scatter(
                frf, x="Recency", y="Monetary",
                color="Segment_Label",
                color_discrete_map=PERSONA_COLORS,
                size="Frequency", size_max=20,
                hover_data=["Customer ID", "Frequency", "Monetary"],
                opacity=0.75,
            )
            fig2d.update_layout(
                title="Recency vs Monetary (bubble size = Frequency)",
                height=440, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
                font_color="#a0aec0", xaxis=dict(gridcolor="#1e3a5f"),
                yaxis=dict(gridcolor="#1e3a5f"),
            )
            st.plotly_chart(fig2d, use_container_width=True)

        with col_b:
            fig_pie = px.pie(
                rfm["Segment_Label"].value_counts().reset_index(),
                values="count", names="Segment_Label",
                color="Segment_Label", color_discrete_map=PERSONA_COLORS,
                hole=0.45,
            )
            fig_pie.update_layout(
                title="Customer Distribution by Segment",
                height=440, paper_bgcolor="#0d1b2a", font_color="#a0aec0",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        fig_rev = px.bar(
            seg_stats.sort_values("Total_Revenue", ascending=False),
            x="Segment_Label", y="Total_Revenue",
            color="Segment_Label", color_discrete_map=PERSONA_COLORS,
            text=seg_stats.sort_values("Total_Revenue", ascending=False)["Total_Revenue"].apply(lambda v: inr(v)),
        )
        fig_rev.update_layout(
            title="Total Revenue Contribution by Segment",
            height=320, plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            font_color="#a0aec0", showlegend=False,
            yaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with tab2:
        fig3d = px.scatter_3d(
            frf.sample(min(1000, len(frf)), random_state=42) if len(frf) > 1000 else frf,
            x="Recency", y="Frequency", z="Monetary",
            color="Segment_Label",
            color_discrete_map=PERSONA_COLORS,
            opacity=0.75, size_max=8,
            hover_data=["Customer ID"],
        )
        fig3d.update_layout(
            title="3D RFM Cluster Scatter (Recency × Frequency × Monetary)",
            height=600, paper_bgcolor="#0d1b2a", font_color="#a0aec0",
            scene=dict(
                xaxis=dict(backgroundcolor="#0d1b2a", gridcolor="#1e3a5f"),
                yaxis=dict(backgroundcolor="#0d1b2a", gridcolor="#1e3a5f"),
                zaxis=dict(backgroundcolor="#0d1b2a", gridcolor="#1e3a5f"),
            ),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with tab3:
        col_r1, col_r2, col_r3 = st.columns(3)
        for col, metric, label, color in [
            (col_r1, "Recency",   "Recency (days since last order)", "#4299e1"),
            (col_r2, "Frequency", "Frequency (# orders)",           "#68d391"),
            (col_r3, "Monetary",  "Monetary (total spend $)",        "#f6ad55"),
        ]:
            fig_box = px.box(
                rfm, x="Segment_Label", y=metric,
                color="Segment_Label", color_discrete_map=PERSONA_COLORS,
                points="outliers",
            )
            fig_box.update_layout(
                title=label, height=320,
                plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
                font_color="#a0aec0", showlegend=False,
                yaxis=dict(gridcolor="#1e3a5f"),
                xaxis_title="",
            )
            with col:
                st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        st.subheader(f"📋 Customer RFM Table — {len(frf):,} customers")
        display = frf[["Customer ID", "Segment_Label", "Recency", "Frequency", "Monetary"]].copy()
        display["Monetary"] = display["Monetary"].round(2)
        display.columns = ["Customer ID", "Segment", "Recency (days)", "# Orders", "Total Spend (INR)"]
        st.dataframe(display.sort_values("Total Spend (INR)", ascending=False),
                     use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()


