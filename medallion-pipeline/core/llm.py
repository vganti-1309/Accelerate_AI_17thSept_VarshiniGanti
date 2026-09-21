"""
core/llm.py
-----------
Thin client around GitHub Models (Azure AI Inference-compatible endpoint) using
gpt-4o, authenticated with a fine-grained GitHub Personal Access Token (models:read).

If no token is configured, the client transparently falls back to a local
"mock" mode so the rest of the pipeline (STTM generation, chat, etc.) can
still be exercised end-to-end without any external API calls. This makes the
sample-data walkthrough in the README work out of the box.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "gpt-4o").strip()
GITHUB_MODELS_ENDPOINT = os.getenv(
    "GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com"
).strip()


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any] | None = None
    mocked: bool = False


class LLMClient:
    """Wraps chat-completion calls to GitHub Models (gpt-4o).

    Falls back to a deterministic mock responder when GITHUB_TOKEN is not
    set, so the app is fully runnable without credentials for local/demo use.
    """

    def __init__(self) -> None:
        self.token = GITHUB_TOKEN
        self.model = GITHUB_MODEL
        self.endpoint = GITHUB_MODELS_ENDPOINT.rstrip("/")
        self._client = None

        if self.token:
            try:
                # openai>=1.0 client works against the Azure-compatible
                # GitHub Models endpoint with base_url override.
                from openai import OpenAI

                self._client = OpenAI(base_url=self.endpoint, api_key=self.token)
            except Exception:
                self._client = None

    @property
    def is_live(self) -> bool:
        return self._client is not None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        if self._client is not None:
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                return LLMResponse(text=text, raw=resp.model_dump(), mocked=False)
            except Exception as exc:  # noqa: BLE001
                # Fall through to mock so the pipeline never hard-crashes
                # on a transient auth/network issue during a demo.
                return LLMResponse(
                    text=self._mock(system_prompt, user_prompt, json_mode, error=str(exc)),
                    mocked=True,
                )

        return LLMResponse(
            text=self._mock(system_prompt, user_prompt, json_mode), mocked=True
        )

    # ------------------------------------------------------------------
    # Mock fallback — heuristic, rule-based "reasoning" used only when no
    # GITHUB_TOKEN is present. Keeps STTM generation & chat usable offline.
    # ------------------------------------------------------------------
    def _mock(
        self, system_prompt: str, user_prompt: str, json_mode: bool, error: str | None = None
    ) -> str:
        note = f" (LLM call failed: {error}; using offline fallback)" if error else " (offline/mock mode — no GITHUB_TOKEN set)"
        if json_mode:
            return json.dumps(
                {
                    "note": "mock-response" + note,
                    "summary": "Generated via rule-based fallback logic, not gpt-4o.",
                }
            )
        return (
            "I'm currently running in offline fallback mode" + note +
            ". I can still walk the pipeline using rule-based logic, but "
            "for richer natural-language reasoning, set GITHUB_TOKEN in your .env file."
        )


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM text response."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}
