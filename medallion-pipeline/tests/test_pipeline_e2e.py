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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import bronze_agent, gold_agent, orchestrator, reporter, silver_agent
from core.llm import LLMClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "data", "sample")


def test_full_pipeline():
    llm = LLMClient()
    assert llm.is_live is False, "Expected offline fallback mode for this test (no GITHUB_TOKEN)."

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


if __name__ == "__main__":
    test_full_pipeline()
    print("\nEnd-to-end pipeline test passed.")
