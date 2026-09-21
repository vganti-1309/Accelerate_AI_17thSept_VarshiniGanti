"""
core/hitl.py
------------
Human-in-the-loop review component & audit logging.

Renders an STTM as an editable/reviewable table in Streamlit with three
actions (Approve / Approve with Notes / Reject with Feedback), and appends
a timestamped, structured record to audit_logs/ for every decision.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from core.pipeline import PipelineState, ReviewDecision

AUDIT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audit_logs")


def _audit_path(run_id: str) -> str:
    os.makedirs(AUDIT_DIR, exist_ok=True)
    return os.path.join(AUDIT_DIR, f"{run_id}.jsonl")


def log_audit_event(run_id: str, layer: str, event: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "layer": layer,
        **event,
    }
    path = _audit_path(run_id)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_audit_log(run_id: str) -> list[dict[str, Any]]:
    path = _audit_path(run_id)
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def render_sttm_review(
    state: PipelineState,
    layer: str,
    sttm_rows: list[dict[str, Any]],
    on_reject_regenerate,
) -> str | None:
    """Renders the STTM table + Approve/Approve-with-Notes/Reject-with-Feedback
    controls for a given layer ('bronze' | 'silver' | 'gold').

    Returns the decision string if an action was taken this render, else None.
    `on_reject_regenerate(feedback: str)` is called to ask the owning agent
    to revise the STTM when the reviewer rejects it.
    """
    review = getattr(state, f"{layer}_review")

    st.subheader(f"{layer.capitalize()} Layer — Source-to-Target Mapping")
    st.caption(
        f"Revision {review.revision} · Review each row, then approve or reject with feedback."
    )

    df = pd.DataFrame(sttm_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    notes_key = f"{layer}_notes_input"
    notes = st.text_area(
        "Reviewer notes (optional for Approve, required for Reject)",
        key=notes_key,
        height=80,
    )

    col1, col2, col3 = st.columns(3)
    decision_made: str | None = None

    with col1:
        if st.button(f"✅ Approve {layer.capitalize()}", key=f"{layer}_approve", use_container_width=True):
            review.decision = ReviewDecision.APPROVED
            review.notes = notes
            review.timestamp = datetime.now(timezone.utc).isoformat()
            log_audit_event(
                state.run_id, layer,
                {"decision": "approved", "notes": notes, "revision": review.revision, "sttm_row_count": len(sttm_rows)},
            )
            decision_made = "approved"

    with col2:
        if st.button(f"📝 Approve with Notes", key=f"{layer}_approve_notes", use_container_width=True):
            review.decision = ReviewDecision.APPROVED_WITH_NOTES
            review.notes = notes
            review.timestamp = datetime.now(timezone.utc).isoformat()
            log_audit_event(
                state.run_id, layer,
                {"decision": "approved_with_notes", "notes": notes, "revision": review.revision, "sttm_row_count": len(sttm_rows)},
            )
            decision_made = "approved_with_notes"

    with col3:
        if st.button(f"❌ Reject with Feedback", key=f"{layer}_reject", use_container_width=True):
            if not notes.strip():
                st.warning("Please add feedback notes before rejecting so the agent knows what to revise.")
            else:
                review.notes = notes
                review.timestamp = datetime.now(timezone.utc).isoformat()
                log_audit_event(
                    state.run_id, layer,
                    {"decision": "rejected", "notes": notes, "revision": review.revision, "sttm_row_count": len(sttm_rows)},
                )
                state.reset_layer(layer)
                on_reject_regenerate(notes)
                decision_made = "rejected"

    return decision_made
