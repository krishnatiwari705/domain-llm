"""Prompt construction and robust intent extraction."""
from __future__ import annotations

SYSTEM_PROMPT = "You are a banking support intent classifier. Return only the exact intent label, with no explanation."


def prompt(text: str, labels: list[str] | None = None) -> str:
def normalize_prediction(value: str, labels: list[str]) -> str:
    value = value.strip().lower().replace(" ", "_")
    if value in labels:
        return value
    for label in labels:
        if label in value:
            return label
    return "__unknown__"
