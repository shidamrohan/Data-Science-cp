"""
src/ui_theme.py
─────────────────────────────────────────────────────────
Clean Minimalist FinTech Light Theme for Smart Retail Analytics.
Provides shared CSS, Plotly layouts, export buttons, and hero components.
"""

import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Modern FinTech Palette
FINTECH_PALETTE = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#ef4444"]
FINTECH_BG = "#ffffff"
FINTECH_GRID = "#f1f5f9"
FINTECH_TEXT = "#475569"
FINTECH_TITLE = "#0f172a"


def apply_fintech_theme():
    """Injects production-grade Clean Minimalist FinTech CSS."""
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #f8fafc !important;
        color: #0f172a;
      }

      .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px;
      }

      /* Metric Card Container */
      div[data-testid="stMetric"], [data-testid="metric-container"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05), 0 1px 2px -1px rgba(15, 23, 42, 0.05) !important;
        transition: all 0.2s ease-in-out !important;
      }

      div[data-testid="stMetric"]:hover, [data-testid="metric-container"]:hover {
        border-color: #cbd5e1 !important;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.08), 0 2px 4px -2px rgba(15, 23, 42, 0.05) !important;
        transform: translateY(-1px);
      }

      div[data-testid="stMetricLabel"] label, 
      div[data-testid="stMetricLabel"] p, 
      [data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 0.80rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
      }

      /* Full Metric Values without Truncation or Ellipsis */
      div[data-testid="stMetricValue"],
      div[data-testid="stMetricValue"] > div,
      div[data-testid="stMetricValue"] *,
      [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #0f172a !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
        letter-spacing: -0.02em !important;
      }

      div[data-testid="stMetricDelta"], 
      [data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
      }

      /* Executive Hero Banner */
      .fintech-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.04);
      }

      .fintech-badge {
        display: inline-block;
        background: rgba(37, 99, 235, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(147, 197, 253, 0.3);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
      }

      .fintech-hero h1 {
        color: #ffffff !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        margin: 4px 0 8px 0 !important;
        letter-spacing: -0.02em;
      }

      .fintech-hero p {
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
        line-height: 1.5;
      }

      .fintech-chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
        padding-top: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
      }

      .fintech-chip {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 9999px;
        padding: 6px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 6px;
      }

      /* Clean White Card for Tables/Charts */
      .fintech-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
        margin-bottom: 16px;
      }

      /* Streamlit Tab Customization */
      .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f1f5f9;
        padding: 5px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
      }

      .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #64748b !important;
        border: none !important;
        background-color: transparent !important;
      }

      .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.08) !important;
      }

      /* Buttons & Downloads */
      .stDownloadButton button, .stButton button {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 9px !important;
        padding: 6px 16px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.05) !important;
        transition: all 0.15s ease !important;
      }

      .stDownloadButton button:hover, .stButton button:hover {
        border-color: #2563eb !important;
        color: #2563eb !important;
        background: #f8fafc !important;
      }

      /* Headings */
      h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
      }
      
      /* Dataframe / Tables */
      [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
      }

      /* Sidebar branding */
      [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
      }
    </style>
    """, unsafe_allow_html=True)


def update_plotly_theme(fig: go.Figure, height: int = 350, title: str = ""):
    """Applies clean minimalist FinTech light styling to any Plotly figure."""
    fig.update_layout(
        height=height,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color=FINTECH_TITLE, family="Plus Jakarta Sans"),
            x=0.01,
            xanchor="left",
        ) if title else None,
        plot_bgcolor=FINTECH_BG,
        paper_bgcolor=FINTECH_BG,
        font=dict(family="Plus Jakarta Sans", color=FINTECH_TEXT, size=11),
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=FINTECH_TEXT),
        ),
        xaxis=dict(
            gridcolor=FINTECH_GRID,
            zerolinecolor=FINTECH_GRID,
            tickfont=dict(size=10, color=FINTECH_TEXT),
        ),
        yaxis=dict(
            gridcolor=FINTECH_GRID,
            zerolinecolor=FINTECH_GRID,
            tickfont=dict(size=10, color=FINTECH_TEXT),
        ),
    )
    return fig


def render_export_buttons(df: pd.DataFrame, filename_prefix: str = "retail_data"):
    """Renders quick-export buttons for CSV and Excel download."""
    c_csv, c_excel, c_space = st.columns([1.2, 1.2, 3.6])
    
    # CSV export
    csv_data = df.to_csv(index=False).encode('utf-8')
    with c_csv:
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv",
            key=f"dl_csv_{filename_prefix}",
        )

    # Excel export
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    excel_data = excel_buffer.getvalue()
    with c_excel:
        st.download_button(
            label="📊 Export Excel",
            data=excel_data,
            file_name=f"{filename_prefix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_excel_{filename_prefix}",
        )

