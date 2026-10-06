"""
agents/reporter.py
--------------------
The Insights Agent. Read-only access to Gold data. Answers natural
language questions about the results and provides the data the dashboard
renders. No data is modified at this stage.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd

from core.llm import LLMClient

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

INSIGHTS_SYSTEM_PROMPT = (
    "You are a data analyst answering questions about e-commerce return-rate Gold-layer "
    "data that has already been computed and is provided to you as JSON tables. Answer "
    "concisely and concretely using only the numbers given. If the data doesn't contain "
    "what's needed to answer, say so plainly."
)


def load_gold(gold_dir: str) -> dict[str, pd.DataFrame]:
    tables = {}
    for fname in os.listdir(gold_dir):
        if fname.endswith(".parquet"):
            name = fname[: -len(".parquet")]
            tables[name] = pd.read_parquet(os.path.join(gold_dir, fname))
    return tables


def top_callouts(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    kpis = tables.get("category_kpis")
    if kpis is None or kpis.empty:
        return {}
    worst = kpis.sort_values("return_rate", ascending=False).iloc[0]
    biggest_loss = kpis.sort_values("revenue_erosion", ascending=False).iloc[0]
    return {
        "worst_return_rate_category": worst["category"],
        "worst_return_rate": float(worst["return_rate"]),
        "biggest_revenue_loss_category": biggest_loss["category"],
        "biggest_revenue_loss": float(biggest_loss["revenue_erosion"]),
    }


def answer_question(llm: LLMClient, question: str, tables: dict[str, pd.DataFrame]) -> str:
    context_parts = []
    for name, df in tables.items():
        context_parts.append(f"### {name}\n{df.head(50).to_json(orient='records')}")
    context = "\n\n".join(context_parts)[:12000]

    response = llm.complete(
        INSIGHTS_SYSTEM_PROMPT,
        f"Gold-layer data:\n{context}\n\nQuestion: {question}",
        json_mode=False,
        temperature=0.3,
    )
    if response.mocked:
        return _fallback_answer(question, tables)
    return response.text


def _question_has_any(question: str, *keywords: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower())
    return any(keyword.lower() in normalized for keyword in keywords)


def _fallback_answer(question: str, tables: dict[str, pd.DataFrame]) -> str:
    kpis = tables.get("category_kpis")
    trends = tables.get("return_rate_trends")
    reasons = tables.get("return_reason_breakdown")
    q = re.sub(r"[^a-z0-9]+", " ", question.lower())
    if kpis is None or kpis.empty:
        return "No Gold-layer data is available yet to answer that question."

    revenue_keywords = [
        "revenue", "money", "profit", "cost", "loss", "erosion", "impact", "spent",
        "lost", "most expensive", "costing", "costs", "driving revenue",
    ]
    trend_keywords = ["trend", "over time", "month", "quarter", "year", "window", "trajectory", "historical"]
    reason_keywords = ["reason", "why", "cause", "root cause", "driver", "issue", "most common"]
    return_rate_keywords = [
        "return rate", "return-rate", "returning the most", "returned most", "returned most often",
        "returned most frequently", "most often returned", "most frequent returns",
        "highest return", "worst return", "returns vs orders", "return ratio", "returns per order",
        "most returned", "biggest return"
    ]

    if _question_has_any(q, *revenue_keywords) and (
        _question_has_any(q, "top", "highest", "largest", "biggest", "worst", "loss", "lost", "most")
        or _question_has_any(q, "revenue", "money", "cost", "loss", "erosion", "impact")
    ):
        top = kpis.sort_values("revenue_erosion", ascending=False).head(5)
        lines = [f"{r.category}: ${r.revenue_erosion:,.2f} lost to returns" for r in top.itertuples()]
        return "Top categories by revenue lost to returns:\n" + "\n".join(lines)

    if _question_has_any(q, *return_rate_keywords) or _question_has_any(q, "return", "returns") and _question_has_any(q, "highest", "worst", "most", "top"):
        top = kpis.sort_values("return_rate", ascending=False).head(5)
        lines = [f"{r.category}: {r.return_rate:.1%} return rate" for r in top.itertuples()]
        return "Categories with the highest return rates:\n" + "\n".join(lines)

    if _question_has_any(q, *trend_keywords):
        if trends is not None and not trends.empty:
            latest = trends.sort_values("window_days", ascending=False)
            lines = [
                f"{r.category}: {r.return_rate:.1%} in the {r.window_days}-day window"
                for r in latest.head(5).itertuples()
            ]
            return "Recent return-rate trend snapshot:\n" + "\n".join(lines)
        callouts = top_callouts(tables)
        if callouts:
            return (
                f"The category with the worst return rate is {callouts['worst_return_rate_category']} "
                f"({callouts['worst_return_rate']:.1%})."
            )

    if _question_has_any(q, *reason_keywords):
        if reasons is not None and not reasons.empty:
            top_reasons = reasons.sort_values("count", ascending=False).head(5)
            lines = [
                f"{r.return_reason}: {r.count} returns in {r.category}"
                for r in top_reasons.itertuples()
            ]
            return "Top return reasons by category:\n" + "\n".join(lines)

    if _question_has_any(q, "what is happening", "summary", "overall", "health", "snapshot"):
        callouts = top_callouts(tables)
        if callouts:
            return (
                f"The category with the worst return rate is {callouts['worst_return_rate_category']} "
                f"({callouts['worst_return_rate']:.1%}). The biggest revenue impact is "
                f"{callouts['biggest_revenue_loss_category']} at ${callouts['biggest_revenue_loss']:,.2f}."
            )

    callouts = top_callouts(tables)
    if callouts:
        return (
            f"The category with the worst return rate is {callouts['worst_return_rate_category']} "
            f"({callouts['worst_return_rate']:.1%}). The biggest revenue impact is "
            f"{callouts['biggest_revenue_loss_category']} at ${callouts['biggest_revenue_loss']:,.2f}."
        )
    return "I don't have enough Gold-layer data to answer that precisely yet."
