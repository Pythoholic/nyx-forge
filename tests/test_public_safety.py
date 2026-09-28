"""Regression tests for the public repository's Safe-only boundary."""

from __future__ import annotations

import json
import re

import pytest
from fastapi import HTTPException, Request

from backend.batch_common import normalize_content_rating
from backend.keywords import BY_ID, CATEGORIES, DATA_DIR
from backend.main import _require_safe_rating
from backend.prompt_engine import ExposureState, Rating, generate_prompt


UNSAFE_POSITIVE = re.compile(
    r"\b(?:explicit|questionable|objectionable|boudoir|nude|naked|topless|"
    r"bottomless|lingerie|fetish|erotic|sexual|seduct\w*|sensual\w*|"
    r"smolder\w*|languid|fishnet|robe|parted lips|half-lidded|bedroom gaze)\b",
    re.IGNORECASE,
)


def _request() -> Request:
    return Request({"type": "http", "headers": []})


def test_public_api_accepts_only_safe_rating() -> None:
    _require_safe_rating(_request(), "Safe")
    assert normalize_content_rating("Safe") == "SAFE"
    with pytest.raises((HTTPException, ValueError)):
        _require_safe_rating(_request(), "Unsupported")


def test_keyword_bundle_contains_only_safe_nonsexual_records() -> None:
    assert BY_ID
    for record in BY_ID.values():
        searchable = " ".join((
            str(record["id"]),
            str(record["canonical_text"]),
            *(str(value) for value in record.get("aliases", ())),
            *(str(value) for value in record.get("semantic_tags", ())),
            *(str(value) for value in record.get("model_text_override", {}).values()),
        ))
        assert record["rating_floor"] == "SAFE"
        assert record["safety"]["sexual_content"] is False
        assert not UNSAFE_POSITIVE.search(searchable)


def test_public_wardrobe_is_complete_and_clothed() -> None:
    wardrobe = CATEGORIES["wardrobe"]
    assert wardrobe
    assert all(record["subcategory"] == "garment" for record in wardrobe)
    assert all(record.get("exposure_state") == ExposureState.CLOTHED.value for record in wardrobe)


def test_generated_positive_prompt_stays_safe() -> None:
    result = generate_prompt("juggernaut_ragnarok", 42, rating=Rating.SAFE)
    assert "fully clothed" in result.positive.casefold()
    assert "non-sexual scene" in result.positive.casefold()
    assert not UNSAFE_POSITIVE.search(result.positive.replace("non-sexual", "safe"))


def test_model_overlay_exposes_only_safe_rating_tag() -> None:
    overlay = json.loads((DATA_DIR / "wai_illustrious_v17.json").read_text(encoding="utf-8"))
    assert overlay["rating_tags"] == {"SAFE": "general"}
