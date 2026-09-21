"""
core/sttm.py
------------
Source-to-Target Mapping (STTM) schema, validation, and helper utilities
shared by the Bronze / Silver / Gold STTM agents.

An STTM row describes exactly one field/transformation:
    source_field, source_type, target_field, target_type,
    transformation, null_handling, business_rule
"""

from __future__ import annotations

from typing import Any

STTM_COLUMNS = [
    "source_field",
    "source_type",
    "target_field",
    "target_type",
    "transformation",
    "null_handling",
    "business_rule",
]


def make_row(
    source_field: str,
    source_type: str,
    target_field: str,
    target_type: str,
    transformation: str = "direct copy",
    null_handling: str = "keep as null",
    business_rule: str = "",
) -> dict[str, Any]:
    return {
        "source_field": source_field,
        "source_type": source_type,
        "target_field": target_field,
        "target_type": target_type,
        "transformation": transformation,
        "null_handling": null_handling,
        "business_rule": business_rule,
    }


def validate_sttm(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Structural validation only — does every row have the required columns?"""
    errors: list[str] = []
    if not rows:
        errors.append("STTM has no rows.")
    for i, row in enumerate(rows):
        missing = [c for c in STTM_COLUMNS if c not in row]
        if missing:
            errors.append(f"Row {i}: missing columns {missing}")
        if not row.get("target_field"):
            errors.append(f"Row {i}: target_field is empty")
    return (len(errors) == 0, errors)


def sttm_to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No STTM rows generated._"
    header = "| " + " | ".join(STTM_COLUMNS) + " |"
    sep = "|" + "|".join(["---"] * len(STTM_COLUMNS)) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in STTM_COLUMNS) + " |")
    return "\n".join(lines)
