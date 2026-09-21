"""
agents/orchestrator.py
------------------------
The Supervisor / Orchestrator Agent. Entry point for business intent,
extracts analysis parameters, and drives the Bronze -> Silver -> Gold ->
Insights sequence. Routes rejection feedback back to the owning layer's
agent and waits for a revised mapping before advancing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.llm import LLMClient, extract_json
from core.pipeline import PipelineState, Stage

INTENT_SYSTEM_PROMPT = (
    "You extract structured analysis parameters from a business user's natural-language "
    "request about e-commerce returns. Respond with a single JSON object only: "
    '{"categories": [list of category names or ["all"]], '
    '"kpis": [list from: "return_rate", "revenue_erosion", "return_reasons", "trend"], '
    '"time_window_days": integer, '
    '"summary": "one sentence restating what you understood"}. '
    "No prose outside the JSON."
)


def _fallback_intent_parse(intent_raw: str) -> dict[str, Any]:
    text = intent_raw.lower()
    window = 180
    m = re.search(r"(\d+)\s*(day|week|month)", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        window = n * {"day": 1, "week": 7, "month": 30}[unit]
    elif "quarter" in text:
        window = 90
    elif "year" in text:
        window = 365

    kpis = []
    if "return rate" in text or "return" in text:
        kpis.append("return_rate")
    if "revenue" in text or "losing" in text or "loss" in text:
        kpis.append("revenue_erosion")
    if "reason" in text:
        kpis.append("return_reasons")
    if "trend" in text or "getting worse" in text or "getting better" in text:
        kpis.append("trend")
    if not kpis:
        kpis = ["return_rate", "revenue_erosion", "trend"]

    return {
        "categories": ["all"],
        "kpis": kpis,
        "time_window_days": window,
        "summary": f"Analyze {', '.join(kpis)} across all categories over the trailing {window} days.",
    }


def parse_intent(llm: LLMClient, intent_raw: str) -> dict[str, Any]:
    response = llm.complete(INTENT_SYSTEM_PROMPT, intent_raw, json_mode=True, temperature=0.1)
    parsed = extract_json(response.text)
    if parsed and "kpis" in parsed and "time_window_days" in parsed:
        return parsed
    return _fallback_intent_parse(intent_raw)


def initialize_pipeline(intent_raw: str, intent_params: dict[str, Any]) -> PipelineState:
    state = PipelineState()
    state.intent_raw = intent_raw
    state.intent_params = intent_params
    state.stage = Stage.BRONZE
    return state


def advance_stage(state: PipelineState) -> None:
    order = [Stage.INTENT, Stage.BRONZE, Stage.SILVER, Stage.GOLD, Stage.INSIGHTS]
    idx = order.index(state.stage)
    if idx < len(order) - 1:
        state.stage = order[idx + 1]
