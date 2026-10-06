"""
tests/test_pipeline_e2e.py
----------------------------
End-to-end test of Bronze -> Silver -> Gold materialization plus the
Insights agent's callouts, using the bundled sample data and the offline
LLM fallback (no GITHUB_TOKEN required). Run with: `python -m pytest tests/`
or directly with `python tests/test_pipeline_e2e.py`.
"""

import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import bronze_agent, gold_agent, orchestrator, reporter, silver_agent
from core.llm import LLMClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "data", "sample")


def test_custom_uploaded_names_are_normalized():
    llm = LLMClient()
    run_id = "custom_names_run"
    file_paths = {
        "order_export_v1": os.path.join(SAMPLE_DIR, "orders_sample.csv"),
        "returns_export_v2": os.path.join(SAMPLE_DIR, "returns_sample.csv"),
        "catalog_master": os.path.join(SAMPLE_DIR, "products_sample.csv"),
    }

    bronze_dir = bronze_agent.materialize_bronze(file_paths, run_id)
    silver_dir = silver_agent.materialize_silver(bronze_dir, run_id)
    gold_dir = gold_agent.materialize_gold(silver_dir, run_id, 90)

    assert os.path.exists(os.path.join(silver_dir, "orders.parquet"))
    assert os.path.exists(os.path.join(silver_dir, "returns.parquet"))
    assert os.path.exists(os.path.join(gold_dir, "category_kpis.parquet"))

    for base in ("bronze", "silver", "gold"):
        p = os.path.join(ROOT, "data", base, run_id)
        if os.path.exists(p):
            shutil.rmtree(p)


def test_full_pipeline():
    llm = LLMClient()
    assert llm is not None

    # Intent parsing
    params = orchestrator.parse_intent(
        llm, "Find which categories have the highest return rates and revenue loss over the last 90 days."
    )
    assert "time_window_days" in params
    assert params["time_window_days"] == 90

    run_id = "test_run"
    file_paths = {
        "orders_sample": os.path.join(SAMPLE_DIR, "orders_sample.csv"),
        "returns_sample": os.path.join(SAMPLE_DIR, "returns_sample.csv"),
        "products_sample": os.path.join(SAMPLE_DIR, "products_sample.csv"),
    }

    # Bronze
    profile, bronze_rows, _ = bronze_agent.profile_and_propose(llm, file_paths)
    assert len(bronze_rows) > 0
    bronze_dir = bronze_agent.materialize_bronze(file_paths, run_id)
    assert os.path.exists(os.path.join(bronze_dir, "orders_sample.parquet"))

    # Silver
    bronze_tables = silver_agent.load_bronze(bronze_dir)
    silver_rows, _ = silver_agent.generate_silver_sttm(llm, bronze_tables)
    assert len(silver_rows) > 0
    silver_dir = silver_agent.materialize_silver(bronze_dir, run_id)
    assert os.path.exists(os.path.join(silver_dir, "orders.parquet"))

    # Gold
    gold_rows, _ = gold_agent.generate_gold_sttm(llm, params)
    assert len(gold_rows) > 0
    gold_dir = gold_agent.materialize_gold(silver_dir, run_id, params["time_window_days"])
    assert os.path.exists(os.path.join(gold_dir, "category_kpis.parquet"))

    # Insights
    tables = reporter.load_gold(gold_dir)
    callouts = reporter.top_callouts(tables)
    assert "worst_return_rate_category" in callouts
    answer = reporter.answer_question(llm, "Which categories have the worst return rate?", tables)
    assert isinstance(answer, str) and len(answer) > 0

    print("Callouts:", callouts)
    print("Sample answer:", answer)

    # cleanup test run artifacts
    for base in ("bronze", "silver", "gold"):
        p = os.path.join(ROOT, "data", base, run_id)
        if os.path.exists(p):
            shutil.rmtree(p)


def test_offline_fallback_answers_are_prompt_specific():
    tables = {
        "category_kpis": pd.DataFrame(
            [
                {"category": "Apparel", "return_rate": 0.35, "revenue_erosion": 12000.0},
                {"category": "Electronics", "return_rate": 0.18, "revenue_erosion": 25000.0},
                {"category": "Home & Kitchen", "return_rate": 0.12, "revenue_erosion": 8000.0},
            ]
        ),
        "return_reason_breakdown": pd.DataFrame(
            [
                {"return_reason": "size mismatch", "category": "Apparel", "count": 150},
                {"return_reason": "damaged on arrival", "category": "Electronics", "count": 120},
            ]
        ),
        "return_rate_trends": pd.DataFrame(
            [
                {"category": "Apparel", "window_days": 90, "return_rate": 0.35},
                {"category": "Electronics", "window_days": 90, "return_rate": 0.18},
            ]
        ),
    }

    revenue_answer = reporter._fallback_answer("Which category is costing us the most money?", tables)
    assert "Top categories by revenue lost to returns" in revenue_answer

    return_answer = reporter._fallback_answer("Which category is returning the most products?", tables)
    assert "Categories with the highest return rates" in return_answer

    reason_answer = reporter._fallback_answer("Why are customers returning products?", tables)
    assert "Top return reasons by category" in reason_answer

    trend_answer = reporter._fallback_answer("How are returns trending over time?", tables)
    assert "Recent return-rate trend snapshot" in trend_answer


if __name__ == "__main__":
    test_full_pipeline()
    print("\nEnd-to-end pipeline test passed.")
