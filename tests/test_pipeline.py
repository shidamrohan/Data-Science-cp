"""
tests/test_pipeline.py
──────────────────────────────────────────────────────────
Unit tests for data preprocessing, formatting, and feature engineering.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import pandas as pd
from formatters import inr, inr_full
from preprocessing import load_clean


def test_inr_formatting():
    assert inr_full(15782368) == "₹1,57,82,368"
    assert inr_full(2407188) == "₹24,07,188"
    assert inr_full(74728) == "₹74,728"
    assert inr_full(500) == "₹500"
    assert inr(1000) == "₹1,000"


def test_clean_data_structure():
    df = load_clean()
    assert not df.empty
    expected_cols = ["Order Date", "Sales", "Profit", "Quantity", "Category", "Sub-Category", "Region"]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"


def test_profit_margin_derived():
    df = load_clean()
    assert "Profit_Margin" in df.columns
    assert "YearMonthStr" in df.columns
