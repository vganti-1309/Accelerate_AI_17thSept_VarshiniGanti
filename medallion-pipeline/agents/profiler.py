"""
agents/profiler.py
-------------------
Profiles raw uploaded CSV files: column names, inferred dtypes, null
coverage, cardinality, sample values. Feeds the Bronze STTM Agent so it
has concrete evidence to base its proposed mapping on, and is shown to the
reviewer as context alongside the Bronze STTM table.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataframe(name: str, df: pd.DataFrame) -> dict[str, Any]:
    columns_profile = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        n_nulls = int(series.isna().sum())
        columns_profile.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "null_count": n_nulls,
                "null_pct": round(100 * n_nulls / n_rows, 2) if n_rows else 0.0,
                "distinct_count": int(series.nunique(dropna=True)),
                "sample_values": [str(v) for v in series.dropna().unique()[:5]],
            }
        )
    return {
        "table_name": name,
        "row_count": n_rows,
        "column_count": len(df.columns),
        "columns": columns_profile,
    }


def profile_uploads(dataframes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """dataframes: mapping of logical table name -> loaded DataFrame."""
    return {name: profile_dataframe(name, df) for name, df in dataframes.items()}
