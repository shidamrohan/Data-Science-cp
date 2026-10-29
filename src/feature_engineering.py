"""
src/feature_engineering.py
──────────────────────────────────────────────────────────
Builds all feature sets required by the three ML tasks:
  1. Monthly time-series features  → Sales Forecasting
  2. Sub-category demand features  → Demand Forecasting
  3. RFM customer features         → Customer Segmentation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import load_clean


# ═══════════════════════════════════════════════════════
# 1.  SALES FORECASTING FEATURES  (monthly aggregates)
# ═══════════════════════════════════════════════════════

def build_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to monthly level and engineer lag + rolling features
    for regression-based sales forecasting.

    Returns one row per (YearMonthStr, Category, Region).
    """
    grp = (
        df.groupby(["YearMonthStr", "Category", "Region"])
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("Order ID", "nunique"),
            Discount=("Discount", "mean"),
        )
        .reset_index()
        .sort_values("YearMonthStr")
    )

    # Date-derived features
    grp["Date"] = pd.to_datetime(grp["YearMonthStr"] + "-01")
    grp["Month"]   = grp["Date"].dt.month
    grp["Quarter"] = grp["Date"].dt.quarter
    grp["Year"]    = grp["Date"].dt.year

    # Cyclical month encoding  (avoids Jan=1 vs Dec=12 discontinuity)
    grp["Month_Sin"] = np.sin(2 * np.pi * grp["Month"] / 12)
    grp["Month_Cos"] = np.cos(2 * np.pi * grp["Month"] / 12)

    # Lag + rolling features per Category-Region group
    grp = grp.sort_values(["Category", "Region", "Date"])
    for lag in (1, 2, 3):
        grp[f"Sales_Lag{lag}"] = grp.groupby(["Category", "Region"])["Sales"].shift(lag)
    grp["Sales_Roll3"]  = grp.groupby(["Category", "Region"])["Sales"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    grp["Sales_Roll6"]  = grp.groupby(["Category", "Region"])["Sales"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean()
    )

    # Category & Region one-hot encoding
    grp = pd.get_dummies(grp, columns=["Category", "Region"], drop_first=False)

    grp = grp.dropna(subset=["Sales_Lag1"])
    return grp.reset_index(drop=True)


# ═══════════════════════════════════════════════════════
# 2.  DEMAND FORECASTING FEATURES  (sub-category / product level)
# ═══════════════════════════════════════════════════════

def build_monthly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate unit quantity sold per Sub-Category per month.
    Returns features suitable for demand regression and tier classification.
    """
    grp = (
        df.groupby(["YearMonthStr", "Sub-Category"])
        .agg(
            Quantity=("Quantity", "sum"),
            Sales=("Sales", "sum"),
            Discount=("Discount", "mean"),
        )
        .reset_index()
        .sort_values("YearMonthStr")
    )

    grp["Date"]    = pd.to_datetime(grp["YearMonthStr"] + "-01")
    grp["Month"]   = grp["Date"].dt.month
    grp["Quarter"] = grp["Date"].dt.quarter
    grp["Year"]    = grp["Date"].dt.year
    grp["Month_Sin"] = np.sin(2 * np.pi * grp["Month"] / 12)
    grp["Month_Cos"] = np.cos(2 * np.pi * grp["Month"] / 12)

    grp = grp.sort_values(["Sub-Category", "Date"])
    for lag in (1, 2, 3):
        grp[f"Qty_Lag{lag}"] = grp.groupby("Sub-Category")["Quantity"].shift(lag)
    grp["Qty_Roll3"] = grp.groupby("Sub-Category")["Quantity"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    grp = pd.get_dummies(grp, columns=["Sub-Category"], drop_first=False)
    grp = grp.dropna(subset=["Qty_Lag1"])
    return grp.reset_index(drop=True)


# ═══════════════════════════════════════════════════════
# 3.  RFM FEATURES  (customer segmentation)
# ═══════════════════════════════════════════════════════

def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary value for every customer.
    Applies log1p to handle right-skew before K-Means.
    """
    snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer ID")
        .agg(
            CustomerName=("Customer Name", "first") if "Customer Name" in df.columns else ("Customer ID", "first"),
            Segment=("Segment", "first") if "Segment" in df.columns else ("Customer ID", "first"),
            Recency=("Order Date", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Order ID", "nunique"),
            Monetary=("Sales", "sum"),
        )
        .reset_index()
    )

    # Log-transform to reduce skewness
    rfm["R_log"] = np.log1p(rfm["Recency"])
    rfm["F_log"] = np.log1p(rfm["Frequency"])
    rfm["M_log"] = np.log1p(rfm["Monetary"])

    return rfm


if __name__ == "__main__":
    df = load_clean()
    ms = build_monthly_sales(df)
    md = build_monthly_demand(df)
    rfm = build_rfm(df)
    print(f"Monthly Sales rows : {len(ms):,}")
    print(f"Monthly Demand rows: {len(md):,}")
    print(f"RFM customers      : {len(rfm):,}")


