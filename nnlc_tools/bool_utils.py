"""Consistent parsing for boolean-like values read from CSV files.

CSV readers may return real booleans, numeric flags, or strings depending on
the file and pandas' type inference.  Python's ``bool("False")`` is True, so
all data-processing paths should use these helpers instead of ``astype(bool)``.
"""

from __future__ import annotations

from numbers import Number

import pandas as pd


TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})
FALSE_VALUES = frozenset({"", "0", "false", "f", "no", "n", "off", "none", "null", "nan"})


def parse_bool(value, default: bool = False) -> bool:
    """Return a conservative boolean interpretation of a scalar value.

    Unknown or missing values use ``default``.  Numeric values follow the
    normal flag convention (zero is false, any non-zero value is true).
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        return bool(value)
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def parse_bool_series(values, default: bool = False) -> pd.Series:
    """Parse a pandas Series while preserving its index and boolean dtype."""
    return values.map(lambda value: parse_bool(value, default=default)).astype(bool)
