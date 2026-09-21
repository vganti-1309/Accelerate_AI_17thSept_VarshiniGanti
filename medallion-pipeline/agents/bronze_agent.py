"""
agents/bronze_agent.py
-----------------------
Handles raw ingestion. Analyzes the schema of user-uploaded CSVs (column
names, dtypes, cardinality, coverage — via the Profiler) and proposes a
Source-to-Target Mapping describing how each source field lands in the
Bronze table. Nothing is transformed yet beyond column renaming / type
casting / timestamp standardization.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from agents.profiler import profile_uploads
from agents.sttm_generator import generate_sttm
from core.llm import LLMClient
from core.sttm import make_row

BRONZE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bronze")


def load_uploaded_csvs(file_paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    """file_paths: logical table name -> path on disk."""
    return {name: pd.read_csv(path) for name, path in file_paths.items()}


def _fallback_bronze_sttm(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for table_name, table_profile in profile.items():
        for col in table_profile["columns"]:
            target_type = "string"
            if "date" in col["column"].lower():
                target_type = "timestamp"
            elif col["dtype"].startswith(("int", "float")):
                target_type = "double" if "float" in col["dtype"] else "bigint"
            rows.append(
                make_row(
                    source_field=f"{table_name}.{col['column']}",
                    source_type=col["dtype"],
                    target_field=col["column"].strip().lower().replace(" ", "_"),
                    target_type=target_type,
                    transformation="rename + cast" if target_type != "string" else "rename",
                    null_handling="keep as null" if col["null_pct"] < 30 else "flag for silver-layer review",
                    business_rule=f"landing table: bronze_{table_name}",
                )
            )
    return rows


def generate_bronze_sttm(
    llm: LLMClient, profile: dict[str, Any], feedback: str = ""
) -> tuple[list[dict[str, Any]], bool]:
    return generate_sttm(
        llm,
        layer="bronze",
        context={"profile": profile},
        feedback=feedback,
        fallback_fn=lambda: _fallback_bronze_sttm(profile),
    )


def profile_and_propose(
    llm: LLMClient, file_paths: dict[str, str], feedback: str = ""
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    dataframes = load_uploaded_csvs(file_paths)
    profile = profile_uploads(dataframes)
    rows, used_llm = generate_bronze_sttm(llm, profile, feedback)
    return profile, rows, used_llm


def materialize_bronze(file_paths: dict[str, str], run_id: str) -> str:
    """Writes each uploaded CSV to data/bronze/ as Parquet with light
    standardization (column renaming to snake_case, dedup of exact rows)."""
    os.makedirs(BRONZE_DIR, exist_ok=True)
    run_dir = os.path.join(BRONZE_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    for name, path in file_paths.items():
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        out_path = os.path.join(run_dir, f"{name}.parquet")
        df.to_parquet(out_path, index=False)

    return run_dir
