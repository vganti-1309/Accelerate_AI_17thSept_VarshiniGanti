"""
agents/gold_agent.py
----------------------
Reads Silver data, proposes the analytical KPI specification: return rate
definition, revenue erosion estimate, return-reason breakdown, 30/60/90-day
trend windows. Once approved, materializes data/gold/ with the actual
aggregates the dashboard reads from.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from agents.sttm_generator import generate_sttm
from core.llm import LLMClient
from core.sttm import make_row

GOLD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "gold")


def load_silver(silver_dir: str) -> dict[str, pd.DataFrame]:
    tables = {}
    for fname in os.listdir(silver_dir):
        if fname.endswith(".parquet"):
            name = fname[: -len(".parquet")]
            normalized = name.lower()
            if "order" in normalized:
                key = "orders"
            elif "return" in normalized:
                key = "returns"
            elif "product" in normalized:
                key = "products"
            else:
                key = name
            tables[key] = pd.read_parquet(os.path.join(silver_dir, fname))
    return tables


def _fallback_gold_sttm(intent_params: dict[str, Any]) -> list[dict[str, Any]]:
    window = intent_params.get("time_window_days", 180)
    rows = [
        make_row(
            source_field="silver.orders (grouped by category)",
            source_type="table",
            target_field="return_rate_by_category",
            target_type="double",
            transformation="returns_count / orders_count, grouped by category",
            null_handling="exclude categories with < 5 orders (statistically unstable)",
            business_rule=f"return rate = COUNT(is_returned=true) / COUNT(orders) over trailing {window} days",
        ),
        make_row(
            source_field="silver.orders x products (cost basis)",
            source_type="table",
            target_field="revenue_erosion_by_category",
            target_type="double",
            transformation="sum(unit_price * quantity) for returned orders, by category",
            null_handling="treat missing unit_price as 0 and flag row",
            business_rule="revenue erosion = gross revenue lost to returned order lines",
        ),
        make_row(
            source_field="silver.orders (grouped by category, date)",
            source_type="table",
            target_field="return_rate_trend_30_60_90",
            target_type="double[]",
            transformation="rolling return rate computed over 30/60/90-day trailing windows",
            null_handling="null if fewer than 10 orders in window",
            business_rule="trend windows anchored to most recent order_date in dataset",
        ),
        make_row(
            source_field="silver.returns.return_reason_code",
            source_type="string",
            target_field="return_reason_breakdown",
            target_type="table",
            transformation="count and pct share of each reason code, by category",
            null_handling="unmapped codes bucketed as 'Other / Unspecified'",
            business_rule="surfaces top return reasons per category for root-cause analysis",
        ),
    ]
    return rows


def generate_gold_sttm(
    llm: LLMClient, intent_params: dict[str, Any], feedback: str = ""
) -> tuple[list[dict[str, Any]], bool]:
    return generate_sttm(
        llm,
        layer="gold",
        context={"intent_params": intent_params},
        feedback=feedback,
        fallback_fn=lambda: _fallback_gold_sttm(intent_params),
    )


def materialize_gold(silver_dir: str, run_id: str, time_window_days: int = 180) -> str:
    tables = load_silver(silver_dir)
    os.makedirs(GOLD_DIR, exist_ok=True)
    run_dir = os.path.join(GOLD_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    orders = tables.get("orders")
    returns = tables.get("returns")
    if orders is None:
        raise ValueError("Silver 'orders' table not found — cannot materialize Gold layer.")

    orders = orders.copy()
    if "order_date" in orders.columns:
        cutoff = orders["order_date"].max() - pd.Timedelta(days=time_window_days)
        window_orders = orders[orders["order_date"] >= cutoff]
    else:
        window_orders = orders

    # 1. Return rate + revenue erosion by category
    grp = window_orders.groupby("category").agg(
        orders_count=("is_returned", "count"),
        returns_count=("is_returned", "sum"),
    )
    grp["return_rate"] = (grp["returns_count"] / grp["orders_count"]).round(4)

    if "unit_price" in window_orders.columns and "quantity" in window_orders.columns:
        window_orders = window_orders.assign(
            revenue=window_orders["unit_price"].fillna(0) * window_orders["quantity"].fillna(0)
        )
        rev = window_orders[window_orders["is_returned"]].groupby("category")["revenue"].sum()
        grp["revenue_erosion"] = rev.reindex(grp.index).fillna(0).round(2)
    else:
        grp["revenue_erosion"] = 0.0

    category_kpis = grp.reset_index().sort_values("return_rate", ascending=False)
    category_kpis.to_parquet(os.path.join(run_dir, "category_kpis.parquet"), index=False)

    # 2. Trend windows: 30/60/90-day return rate per category
    trend_rows = []
    if "order_date" in orders.columns:
        max_date = orders["order_date"].max()
        for days in (30, 60, 90):
            window = orders[orders["order_date"] >= max_date - pd.Timedelta(days=days)]
            if len(window) == 0:
                continue
            tg = window.groupby("category").agg(
                orders_count=("is_returned", "count"), returns_count=("is_returned", "sum")
            )
            tg["return_rate"] = (tg["returns_count"] / tg["orders_count"]).round(4)
            tg["window_days"] = days
            trend_rows.append(tg.reset_index()[["category", "window_days", "return_rate", "orders_count"]])
    trend_df = pd.concat(trend_rows, ignore_index=True) if trend_rows else pd.DataFrame(
        columns=["category", "window_days", "return_rate", "orders_count"]
    )
    trend_df.to_parquet(os.path.join(run_dir, "return_rate_trends.parquet"), index=False)

    # 3. Return reason breakdown, joined back to category via order_id
    if returns is not None and "reason_code" in "".join(returns.columns).lower() or (
        returns is not None and any("reason" in c for c in returns.columns)
    ):
        reason_col = next((c for c in returns.columns if "reason" in c), None)
        if reason_col and "order_id" in returns.columns and "order_id" in orders.columns:
            merged = returns.merge(
                orders[["order_id", "category"]], on="order_id", how="left"
            )
            reason_breakdown = (
                merged.groupby(["category", reason_col]).size().reset_index(name="count")
            )
            reason_breakdown["pct_of_category"] = reason_breakdown.groupby("category")["count"].transform(
                lambda s: (s / s.sum()).round(4)
            )
            reason_breakdown.rename(columns={reason_col: "return_reason"}, inplace=True)
        else:
            reason_breakdown = pd.DataFrame(columns=["category", "return_reason", "count", "pct_of_category"])
    else:
        reason_breakdown = pd.DataFrame(columns=["category", "return_reason", "count", "pct_of_category"])
    reason_breakdown.to_parquet(os.path.join(run_dir, "return_reason_breakdown.parquet"), index=False)

    return run_dir
