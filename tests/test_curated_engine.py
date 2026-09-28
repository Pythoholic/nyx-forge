"""Phase 3: history-aware curated selection on Prompt Engine V1."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend import database, engine_bridge
from backend.database import PromptHistoryRecord
from backend.diversity import DiversityStore
from backend.engine_bridge import generate_curated
from backend.main import SurpriseRequest, surprise
from backend.prompt_engine import (
    Decision, MODEL_IMAGE_TYPE_RECIPES, PromptEngine, PromptEngineError, ReasonCode,
)


PONY = r"sd\cyberrealisticPony_v180Coreshift.safetensors [9d0e340f5f]"
PONY_V6 = r"sd\ponyDiffusionV6XL_v6StartWithThisOne.safetensors [67ab2fd8ec]"
RAGNAROK = r"sd\juggernautXL_ragnarok.safetensors [dd08fa32f9]"
LIGHTNING = r"sd\juggernautXL_juggXILightningByRD.safetensors [609fde646e]"
ASPECT = "Landscape (1216x832)"


def _generate(model: str, history=(), *, seed: int | None = 4242):
    return generate_curated(
        model,
        "Safe",
        "front",
        history,
        creativity_level="experimental",
        aspect_ratio=ASPECT,
        seed=seed,
    )


def _history_item(result) -> PromptHistoryRecord:
    return PromptHistoryRecord(
        result.compiled.positive,
        result.novelty.brief.as_json(),
        None,
        (),
    )


def _concept_ids(result) -> tuple[str, ...]:
    return tuple(
        concept.id
        for concept in result.compiled.ast.positive_concepts()
        if concept.category != "control"
    )


@pytest.mark.parametrize("model", (PONY, RAGNAROK))
def test_curated_seed_replays_a_byte_identical_valid_prompt_pair(model: str) -> None:
    first = _generate(model, seed=8181)
    second = _generate(model, seed=8181)

    assert (first.compiled.positive, first.compiled.negative) == (
        second.compiled.positive,
        second.compiled.negative,
    )
    assert first.compiled.ast == second.compiled.ast
    assert first.compiled.validation.decision is Decision.ACCEPT
    assert first.compiled.ast.orientation.value == "front"


def test_curated_request_seed_replays_the_same_prompt_pair(monkeypatch) -> None:
    prompt_run_ids = iter((81, 82))
    monkeypatch.setattr("backend.main.recent_prompt_history", lambda **_: [])
    monkeypatch.setattr(
        "backend.main.save_prompt_run", lambda **_: next(prompt_run_ids)
    )
    request = SurpriseRequest(
        model=PONY,
        style="Photoreal",
        content_rating="Safe",
        aspect_ratio=ASPECT,
        prompt_engine="curated",
        creativity_level="experimental",
        orientation="front",
        seed=6161,
    )

    first = surprise(request)
    second = surprise(request)

    assert (first.prompt, first.negative_prompt) == (
        second.prompt,
        second.negative_prompt,
    )
    assert first.prompt_run_id != second.prompt_run_id


def test_pony_v6_illustration_uses_prompt_engine_v1() -> None:
    first = generate_curated(
        PONY_V6,
        "Safe",
        "front",
        (),
        creativity_level="experimental",
        aspect_ratio=ASPECT,
        style="Illustration",
        seed=6262,
    )
    second = generate_curated(
        PONY_V6,
        "Safe",
        "front",
        (),
        creativity_level="experimental",
        aspect_ratio=ASPECT,
        style="Illustration",
        seed=6262,
    )

    assert engine_bridge.engine_model_for(PONY_V6) == "cyberrealistic_pony"
    assert first.compiled.validation.decision is Decision.ACCEPT
    assert first.compiled.ast.style_parent == "Illustration"
    assert not first.compiled.ast.photographic
    assert first.compiled.ast.style[0].text in first.compiled.positive
    assert "source_photo" not in first.compiled.positive
    assert (first.compiled.positive, first.compiled.negative) == (
        second.compiled.positive,
        second.compiled.negative,
    )


@pytest.mark.parametrize("style", ("Anime", "Cartoon"))
def test_pony_v6_style_vocabulary_uses_prompt_engine_without_fallback(
    monkeypatch, style,
) -> None:
    saved: dict[str, object] = {}
    legacy = Mock(return_value=("legacy positive", "legacy negative"))
    monkeypatch.setattr("backend.main.recent_prompt_history", lambda **_: [])
    monkeypatch.setattr("backend.main.generate_random_prompt_pair", legacy)
    monkeypatch.setattr(
        "backend.main.save_prompt_run",
        lambda **values: saved.update(values) or 91,
    )
    request = SurpriseRequest(
        model=PONY_V6,
        style=style,
        content_rating="Safe",
        aspect_ratio=ASPECT,
        prompt_engine="curated",
        orientation="front",
        seed=7171,
    )

    result = surprise(request)

    assert result.engine == "curated"
    assert result.notice is None
    assert str(saved["engine"]).startswith("curated:engine:")
    assert "fallback_occurrence" not in json.loads(str(saved["novelty_signature"]))
    style_texts = {
        str(engine_bridge.BY_ID[record_id]["canonical_text"])
        for record_id in engine_bridge.STYLE_MEDIUM_PROFILES[style].primary_ids
    }
    assert any(text in result.prompt for text in style_texts)
    legacy.assert_not_called()


def test_unsupported_model_uses_legacy_compatibility_fallback(monkeypatch) -> None:
    saved: dict[str, object] = {}
    legacy = Mock(return_value=("legacy positive", "legacy negative"))
    monkeypatch.setattr("backend.main.recent_prompt_history", lambda **_: [])
    monkeypatch.setattr("backend.main.generate_random_prompt_pair", legacy)
    monkeypatch.setattr(
        "backend.main.save_prompt_run",
        lambda **values: saved.update(values) or 92,
    )

    result = surprise(SurpriseRequest(
        model=LIGHTNING,
        style="Photoreal",
        content_rating="Safe",
        aspect_ratio=ASPECT,
        prompt_engine="curated",
        orientation="front",
        seed=8181,
    ))

    assert result.prompt == "legacy positive"
    assert "MODEL_UNSUPPORTED" in (result.notice or "")
    assert saved["engine"] == "curated:legacy-fallback:MODEL_UNSUPPORTED"
    legacy.assert_called_once()


def test_unrelated_curated_engine_error_does_not_fall_back(monkeypatch) -> None:
    legacy = Mock(return_value=("legacy positive", "legacy negative"))
    monkeypatch.setattr("backend.main.recent_prompt_history", lambda **_: [])
    monkeypatch.setattr("backend.main.generate_random_prompt_pair", legacy)
    monkeypatch.setattr(
        "backend.main.generate_curated",
        Mock(side_effect=PromptEngineError(
            (ReasonCode.BUDGET_UNRECOVERABLE,), "budget failure"
        )),
    )

    with pytest.raises(HTTPException) as raised:
        surprise(SurpriseRequest(
            model=RAGNAROK,
            style="Photoreal",
            content_rating="Safe",
            aspect_ratio=ASPECT,
            prompt_engine="curated",
            orientation="front",
            seed=9191,
        ))

    assert raised.value.status_code == 422
    assert "BUDGET_UNRECOVERABLE" in str(raised.value.detail)
    assert isinstance(raised.value.__cause__, PromptEngineError)
    legacy.assert_not_called()


def test_mixed_compatibility_and_real_defect_error_does_not_fall_back(monkeypatch) -> None:
    legacy = Mock(return_value=("legacy positive", "legacy negative"))
    monkeypatch.setattr("backend.main.recent_prompt_history", lambda **_: [])
    monkeypatch.setattr("backend.main.generate_random_prompt_pair", legacy)
    monkeypatch.setattr(
        "backend.main.generate_curated",
        Mock(side_effect=PromptEngineError(
            (ReasonCode.STYLE_CONFLICT, ReasonCode.BUDGET_UNRECOVERABLE),
            "mixed failure",
        )),
    )

    with pytest.raises(HTTPException) as raised:
        surprise(SurpriseRequest(
            model=RAGNAROK,
            style="Photoreal",
            content_rating="Safe",
            aspect_ratio=ASPECT,
            prompt_engine="curated",
            orientation="front",
            seed=9292,
        ))

    assert raised.value.status_code == 422
    assert "STYLE_CONFLICT" in str(raised.value.detail)
    assert "BUDGET_UNRECOVERABLE" in str(raised.value.detail)
    legacy.assert_not_called()


def test_mixed_compatibility_only_error_still_falls_back(monkeypatch) -> None:
    legacy = Mock(return_value=("legacy positive", "legacy negative"))
    monkeypatch.setattr("backend.main.recent_prompt_history", lambda **_: [])
    monkeypatch.setattr("backend.main.generate_random_prompt_pair", legacy)
    monkeypatch.setattr("backend.main.save_prompt_run", lambda **_: 93)
    monkeypatch.setattr(
        "backend.main.generate_curated",
        Mock(side_effect=PromptEngineError(
            (ReasonCode.MODEL_UNSUPPORTED, ReasonCode.STYLE_CONFLICT),
            "compatibility-only failure",
        )),
    )

    result = surprise(SurpriseRequest(
        model=RAGNAROK,
        style="Photoreal",
        content_rating="Safe",
        aspect_ratio=ASPECT,
        prompt_engine="curated",
        orientation="front",
        seed=9393,
    ))

    assert result.prompt == "legacy positive"
    legacy.assert_called_once()


def test_curated_uses_the_model_compiler_serialization() -> None:
    pony = _generate(PONY, seed=9191).compiled
    ragnarok = _generate(RAGNAROK, seed=9191).compiled

    assert pony.positive.startswith("score_9, score_8_up, score_7_up, source_photo")
    assert ragnarok.positive.startswith(
        MODEL_IMAGE_TYPE_RECIPES[("juggernaut_ragnarok", "Photoreal")].positive.capitalize()
        + ": "
    )
    assert pony.positive_tokens <= 75 and pony.negative_tokens <= 75
    assert ragnarok.positive_tokens <= 75 and ragnarok.negative_tokens <= 75


def test_curated_keeps_sampling_until_a_novel_candidate(monkeypatch) -> None:
    scores = iter((0.81, 0.19))
    monkeypatch.setattr(engine_bridge, "maximum_similarity", lambda *_: next(scores))

    result = _generate(PONY, seed=7171)

    assert result.attempt_count == 2
    assert result.similarity_score == 0.19


def test_history_changes_novelty_and_canonical_concept_selection() -> None:
    baseline = _generate(PONY, seed=4242)
    history: list[PromptHistoryRecord] = []
    for offset in range(5):
        result = _generate(PONY, history, seed=4242 + offset)
        history.insert(0, _history_item(result))

    history_aware = _generate(PONY, history, seed=4242)

    assert len({item.positive_prompt for item in history}) == 5
    assert history_aware.novelty.brief != baseline.novelty.brief
    assert _concept_ids(history_aware) != _concept_ids(baseline)
    assert history_aware.compiled.positive != baseline.compiled.positive


def test_unseeded_curated_records_only_the_selected_engine_fingerprint(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "curated-diversity.db")
    database.initialize_database()
    monkeypatch.setattr(
        engine_bridge,
        "_ENGINE",
        PromptEngine(diversity_store=DiversityStore(), diversity_enabled=True),
    )
    monkeypatch.setattr(engine_bridge.secrets, "randbelow", lambda _: 5151)
    monkeypatch.setattr(engine_bridge, "maximum_similarity", lambda *_: 0.99)

    result = _generate(PONY, seed=None)

    assert result.attempt_count == 6
    with sqlite3.connect(database.DATABASE_PATH) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM diversity_fingerprints"
        ).fetchone()[0] == 1
