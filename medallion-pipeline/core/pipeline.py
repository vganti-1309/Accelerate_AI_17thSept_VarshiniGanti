"""
core/pipeline.py
-----------------
Shared pipeline state passed between all agents. No agent mutates another
agent's section directly — the Orchestrator (Supervisor) is the only thing
that sequences stage transitions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Stage(str, Enum):
    INTENT = "intent"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    INSIGHTS = "insights"


class ReviewDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    APPROVED_WITH_NOTES = "approved_with_notes"
    REJECTED = "rejected"


@dataclass
class ReviewRecord:
    decision: ReviewDecision = ReviewDecision.PENDING
    notes: str = ""
    timestamp: str | None = None
    revision: int = 0


@dataclass
class PipelineState:
    """The single object threaded through Bronze -> Silver -> Gold -> Insights."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Phase 1 — intent captured from the chat UI
    intent_raw: str = ""
    intent_params: dict[str, Any] = field(default_factory=dict)

    # Uploaded raw file paths (as saved to disk before Bronze ingestion)
    uploaded_files: dict[str, str] = field(default_factory=dict)

    # Current stage the pipeline is sitting at
    stage: Stage = Stage.INTENT

    # STTM tables produced by each agent: list[dict] rows
    bronze_sttm: list[dict[str, Any]] = field(default_factory=list)
    silver_sttm: list[dict[str, Any]] = field(default_factory=list)
    gold_sttm: list[dict[str, Any]] = field(default_factory=list)

    # Review state per layer
    bronze_review: ReviewRecord = field(default_factory=ReviewRecord)
    silver_review: ReviewRecord = field(default_factory=ReviewRecord)
    gold_review: ReviewRecord = field(default_factory=ReviewRecord)

    # Paths written once a layer is approved and materialized
    bronze_path: str | None = None
    silver_path: str | None = None
    gold_path: str | None = None

    # Free-form profiler output (schema/quality stats) attached at Bronze time
    profile: dict[str, Any] = field(default_factory=dict)

    def reset_layer(self, layer: str) -> None:
        """Used when a reviewer rejects a layer — bump revision, clear approval."""
        review_attr = f"{layer}_review"
        record: ReviewRecord = getattr(self, review_attr)
        record.decision = ReviewDecision.PENDING
        record.revision += 1

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["stage"] = self.stage.value
        for layer in ("bronze", "silver", "gold"):
            rec: ReviewRecord = d[f"{layer}_review"]
            d[f"{layer}_review"] = {
                "decision": rec.decision.value,
                "notes": rec.notes,
                "timestamp": rec.timestamp,
                "revision": rec.revision,
            }
        return d
