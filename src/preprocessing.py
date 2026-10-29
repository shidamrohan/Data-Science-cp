"""
src/preprocessing.py
──────────────────────────────────────────────────────────
Loads raw Superstore CSV, cleans it, validates columns,
and saves a clean Parquet file for downstream modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "SampleSuperstore.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# USD → INR conversion rate (1 USD = 84 INR, Oct 2026)
USD_TO_INR = 95.28  # 1 USD = ₹95.28 (Oct 2026)


def load_raw() -> pd.DataFrame:
    """Load raw CSV with multiple encoding fallbacks."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(RAW_PATH, encoding=enc)
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode {RAW_PATH}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Standardise column names (strip spaces, title-case).
    2. Parse date columns.
    3. Drop true duplicates.
    4. Impute / drop missing values.
    5. Clip extreme outliers in Sales & Profit (IQR × 3).
    6. Derive helper columns used by all modules.
    """
    # ── 1. Standardise column names ──────────────────────────────────
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    # ── 2. Parse dates ────────────────────────────────────────────────
    for col in ("Order Date", "Ship Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── 3. Drop exact duplicate rows ─────────────────────────────────
    df = df.drop_duplicates()

    # ── 4. Missing values ─────────────────────────────────────────────
    # Drop rows where critical business fields are missing
    critical = ["Order Date", "Sales", "Profit", "Quantity", "Customer ID"]
    critical = [c for c in critical if c in df.columns]
    df = df.dropna(subset=critical)

    # Fill Discount NaN with 0
    if "Discount" in df.columns:
        df["Discount"] = df["Discount"].fillna(0)

    # ── 5. Clip outliers using IQR × 3 ───────────────────────────────
    for col in ("Sales", "Profit"):
        if col in df.columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(lower=q1 - 3 * iqr, upper=q3 + 3 * iqr)

    # ── 5b. Convert USD → INR ─────────────────────────────────────────
    for col in ("Sales", "Profit"):
        if col in df.columns:
            df[col] = (df[col] * USD_TO_INR).round(2)

    # ── 6. Derived helper columns ─────────────────────────────────────
    df["Year"]          = df["Order Date"].dt.year
    df["Month"]         = df["Order Date"].dt.month
    df["Quarter"]       = df["Order Date"].dt.quarter
    df["YearMonth"]     = df["Order Date"].dt.to_period("M")
    df["YearMonthStr"]  = df["Order Date"].dt.strftime("%Y-%m")
    df["YearWeek"]      = df["Order Date"].dt.isocalendar().week.astype(int)
    df["Profit_Margin"] = np.where(
        df["Sales"] != 0, (df["Profit"] / df["Sales"]) * 100, 0
    )
    df["Revenue_Per_Unit"] = np.where(
        df["Quantity"] != 0, df["Sales"] / df["Quantity"], 0
    )

    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, filename: str = "clean_superstore.parquet") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_parquet(out, index=False)
    return out


def load_clean(filename: str = "clean_superstore.parquet") -> pd.DataFrame:
    """Load the processed file (run pipeline first if missing)."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        run_pipeline()
    df = pd.read_parquet(path)
    # Restore period column from string if needed
    if "YearMonthStr" in df.columns and "YearMonth" not in df.columns:
        df["YearMonth"] = pd.PeriodIndex(df["YearMonthStr"], freq="M")
    return df


def run_pipeline() -> pd.DataFrame:
    """Full ETL: load → clean → save → return."""
    raw = load_raw()
    clean_df = clean(raw)
    save_processed(clean_df)
    print(f"[preprocessing] Pipeline complete — {len(clean_df):,} rows saved.")
    return clean_df


if __name__ == "__main__":
    run_pipeline()


