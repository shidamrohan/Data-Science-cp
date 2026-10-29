"""
tests/test_pipeline.py
──────────────────────────────────────────────────────────
Unit tests for data preprocessing, formatting, and feature engineering.
Uses Python's built-in unittest framework (zero extra dependencies).
Run with: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formatters import inr, inr_full
from preprocessing import load_clean


class TestPipeline(unittest.TestCase):
    def test_inr_formatting(self):
        self.assertEqual(inr_full(15782368), "₹1,57,82,368")
        self.assertEqual(inr_full(2407188), "₹24,07,188")
        self.assertEqual(inr_full(74728), "₹74,728")
        self.assertEqual(inr_full(500), "₹500")
        self.assertEqual(inr(1000), "₹1,000")

    def test_clean_data_structure(self):
        df = load_clean()
        self.assertFalse(df.empty)
        expected_cols = ["Order Date", "Sales", "Profit", "Quantity", "Category", "Sub-Category", "Region"]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing expected column: {col}")

    def test_profit_margin_derived(self):
        df = load_clean()
        self.assertIn("Profit_Margin", df.columns)
        self.assertIn("YearMonthStr", df.columns)


if __name__ == "__main__":
    unittest.main()
