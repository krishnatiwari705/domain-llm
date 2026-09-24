"""Prompt construction and robust intent extraction."""
from __future__ import annotations

SYSTEM_PROMPT = "You are a banking support intent classifier. Return only the exact intent label, with no explanation."


def prompt(text: str, labels: list[str] | None = None) -> str:
    choices = f"\nValid labels: {', '.join(labels)}" if labels else ""
    return f"{SYSTEM_PROMPT}{choices}\nCustomer message: {text}\nIntent:"


def chat_messages(text: str, labels: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Return only the exact Banking77 intent label."},
        {"role": "user", "content": prompt(text, labels)},
    ]


def training_text(text: str, label: str, labels: list[str]) -> str:
    messages = chat_messages(text, labels)
    messages.append({"role": "assistant", "content": label})
    return messages


def normalize_prediction(value: str, labels: list[str]) -> str:
    value = value.strip().lower().replace(" ", "_")
    if value in labels:
        return value
    for label in labels:
        if label in value:
            return label
    return "__unknown__"
