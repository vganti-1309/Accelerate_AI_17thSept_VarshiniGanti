"""
agents/silver_agent.py
------------------------
Reads Bronze data, proposes the cleaning/transformation plan: null
handling, deduplication, category-name normalization, return-flag /
derived-column computation. Once approved, materializes data/silver/.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from agents.sttm_generator import generate_sttm
from core.llm import LLMClient
from core.sttm import make_row

SILVER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "silver")

# Canonical category master list used for normalization
CATEGORY_MASTER = {
    "electronics": "Electronics",
    "electronic": "Electronics",
    "apparel": "Apparel",
    "clothing": "Apparel",
    "outdoor": "Outdoor & Furniture",
    "outdoor furniture": "Outdoor & Furniture",
    "furniture": "Outdoor & Furniture",
    "home": "Home & Kitchen",
    "kitchen": "Home & Kitchen",
    "home & kitchen": "Home & Kitchen",
    "beauty": "Beauty & Personal Care",
    "sports": "Sports & Outdoors",
    "toys": "Toys & Games",
}


def normalize_category(raw: str) -> str:
    if not isinstance(raw, str):
        return "Uncategorized"
    key = raw.strip().lower()
    return CATEGORY_MASTER.get(key, raw.strip().title())


def normalize_table_name(raw_name: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in raw_name.lower()).strip("_")
    normalized = "_".join(part for part in normalized.split("_") if part)

    if "order" in normalized:
        return "orders"
    if "return" in normalized:
        return "returns"
    if "product" in normalized:
        return "products"
    return normalized or "unknown"


def load_bronze(bronze_dir: str) -> dict[str, pd.DataFrame]:
    tables = {}
    for fname in os.listdir(bronze_dir):
        if fname.endswith(".parquet"):
            name = fname[: -len(".parquet")]
            canonical_name = normalize_table_name(name)
            tables[canonical_name] = pd.read_parquet(os.path.join(bronze_dir, fname))
    return tables


def _fallback_silver_sttm(bronze_tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows = []
    for table_name, df in bronze_tables.items():
        for col in df.columns:
            transformation = "direct copy"
            business_rule = ""
            null_handling = "drop row if key field, else keep null"
            if "category" in col:
                transformation = "normalize to master category list"
                business_rule = "map raw category text to canonical master category"
            elif "date" in col:
                transformation = "parse to ISO-8601 date"
                null_handling = "drop row (date is required for time-window KPIs)"
            elif col in ("order_id", "return_id", "product_id"):
                null_handling = "drop row (required key field)"
                business_rule = "deduplicate on this key"
            rows.append(
                make_row(
                    source_field=f"bronze.{table_name}.{col}",
                    source_type=str(df[col].dtype),
                    target_field=col,
                    target_type=str(df[col].dtype),
                    transformation=transformation,
                    null_handling=null_handling,
                    business_rule=business_rule,
                )
            )
    # Derived column: is_returned flag added at the order-line grain
    rows.append(
        make_row(
            source_field="derived (orders x returns join)",
            source_type="n/a",
            target_field="is_returned",
            target_type="boolean",
            transformation="true if order_id present in returns table, else false",
            null_handling="default false",
            business_rule="drives return-rate KPI in Gold layer",
        )
    )
    return rows


def generate_silver_sttm(
    llm: LLMClient, bronze_tables: dict[str, pd.DataFrame], feedback: str = ""
) -> tuple[list[dict[str, Any]], bool]:
    context = {name: list(df.columns) for name, df in bronze_tables.items()}
    return generate_sttm(
        llm,
        layer="silver",
        context={"bronze_schema": context},
        feedback=feedback,
        fallback_fn=lambda: _fallback_silver_sttm(bronze_tables),
    )


def materialize_silver(bronze_dir: str, run_id: str) -> str:
    """Applies the cleaning rules deterministically: dedup, null handling,
    category normalization, is_returned derivation, date parsing."""
    tables = load_bronze(bronze_dir)
    os.makedirs(SILVER_DIR, exist_ok=True)
    run_dir = os.path.join(SILVER_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    orders = tables.get("orders")
    returns = tables.get("returns")
    products = tables.get("products")

    if orders is not None:
        orders = orders.drop_duplicates()
        if "order_id" in orders.columns:
            orders = orders.dropna(subset=["order_id"])
        if "category" in orders.columns:
            orders["category"] = orders["category"].apply(normalize_category)
        if "order_date" in orders.columns:
            orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
            orders = orders.dropna(subset=["order_date"])
        if returns is not None and "order_id" in orders.columns and "order_id" in returns.columns:
            returned_ids = set(returns["order_id"].dropna().unique())
            orders["is_returned"] = orders["order_id"].isin(returned_ids)
        else:
            orders["is_returned"] = False
        orders.to_parquet(os.path.join(run_dir, "orders.parquet"), index=False)

    if returns is not None:
        returns = returns.drop_duplicates()
        if "return_date" in returns.columns:
            returns["return_date"] = pd.to_datetime(returns["return_date"], errors="coerce")
        returns.to_parquet(os.path.join(run_dir, "returns.parquet"), index=False)

    if products is not None:
        products = products.drop_duplicates()
        if "category" in products.columns:
            products["category"] = products["category"].apply(normalize_category)
        products.to_parquet(os.path.join(run_dir, "products.parquet"), index=False)

    return run_dir
