"""
src/formatters.py
─────────────────────────────────────────────────────────
Indian number formatting utilities used across all pages.
Provides full Indian comma-separated values (no truncation / no ellipsis).
"""


def inr_full(value: float, decimals: int = 0) -> str:
    """
    Full Indian comma-formatted rupee value:
      e.g., 15782368 -> ₹1,57,82,368 (no ellipsis, complete value)
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "₹0"

    sign = "-" if value < 0 else ""
    abs_val = abs(value)

    if decimals > 0:
        int_part = int(abs_val)
        dec_part = f"{abs_val - int_part:.{decimals}f}"[1:]
    else:
        int_part = int(round(abs_val))
        dec_part = ""

    s = str(int_part)
    if len(s) <= 3:
        return f"{sign}₹{s}{dec_part}"

    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]
    result = result.lstrip(",")
    return f"{sign}₹{result}{dec_part}"


def inr(value: float, decimals: int = 0) -> str:
    """
    Default inr formatter returning the full unabbreviated Indian value.
    Guarantees no ellipsis (...) and full precision display.
    """
    return inr_full(value, decimals=decimals)


def inr_compact(value: float, decimals: int = 2) -> str:
    """
    Compact Lakhs / Crores representation (optional for small charts/badges):
      < 1 Lakh  (1,00,000)    -> ₹XX,XXX
      < 1 Crore (1,00,00,000) -> ₹X.XX L
      >= 1 Crore              -> ₹X.XX Cr
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "₹0"

    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    if abs_val >= 1_00_00_000:
        return f"{sign}₹{abs_val / 1_00_00_000:.{decimals}f} Cr"
    elif abs_val >= 1_00_000:
        return f"{sign}₹{abs_val / 1_00_000:.{decimals}f} L"
    elif abs_val >= 1_000:
        return f"{sign}₹{abs_val:,.0f}"
    else:
        return f"{sign}₹{abs_val:.2f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value:+.{decimals}f}%" if value != 0 else f"{value:.{decimals}f}%"
