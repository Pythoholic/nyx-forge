"""Shared ForgeBAT values that must mean the same thing at every layer."""

from __future__ import annotations


RATING_LABELS = {
    "SAFE": "Safe",
}


def normalize_content_rating(value: str) -> str:
    """Translate UI and engine spellings to the prompt-engine enum value."""
    normalized = str(value).strip().casefold()
    ratings = {"safe": "SAFE"}
    try:
        return ratings[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported content rating: {value}") from exc
