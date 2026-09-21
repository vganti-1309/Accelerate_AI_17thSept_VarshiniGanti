"""
agents/sttm_generator.py
-------------------------
Shared helper that asks gpt-4o (via core.llm.LLMClient) to propose an STTM
given a description of the layer's purpose and the available evidence
(profile stats / prior layer schema). Each of the Bronze / Silver / Gold
agents calls this with a layer-specific prompt, then falls back to a
deterministic rule-based STTM if the LLM is unavailable or returns
something unusable — so the pipeline is never blocked on the LLM.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from core.llm import LLMClient, extract_json
from core.sttm import validate_sttm

SYSTEM_PROMPT = (
    "You are a meticulous data engineering assistant that produces Source-to-Target "
    "Mappings (STTM) for a medallion (Bronze/Silver/Gold) ETL pipeline analyzing "
    "e-commerce returns. Always respond with a single JSON object of the form "
    '{"rows": [{"source_field":..., "source_type":..., "target_field":..., '
    '"target_type":..., "transformation":..., "null_handling":..., "business_rule":...}, ...]}. '
    "No prose, no markdown fences — JSON only."
)


def generate_sttm(
    llm: LLMClient,
    layer: str,
    context: dict[str, Any],
    feedback: str,
    fallback_fn: Callable[[], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], bool]:
    """Returns (rows, used_llm)."""
    user_prompt = (
        f"Layer: {layer}\n"
        f"Context/evidence (JSON): {json.dumps(context, default=str)[:6000]}\n"
    )
    if feedback:
        user_prompt += f"\nA human reviewer rejected the previous version with this feedback, revise accordingly:\n{feedback}\n"

    response = llm.complete(SYSTEM_PROMPT, user_prompt, json_mode=True, temperature=0.2)
    parsed = extract_json(response.text)
    rows = parsed.get("rows") if isinstance(parsed, dict) else None

    if rows and isinstance(rows, list):
        ok, _errors = validate_sttm(rows)
        if ok:
            return rows, not response.mocked

    # Fallback: deterministic rule-based STTM
    return fallback_fn(), False
