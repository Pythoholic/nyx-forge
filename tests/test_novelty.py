"""Tests for history-aware creative direction and similarity scoring."""

from __future__ import annotations

import json
import random
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from backend.main import SurpriseRequest, surprise
from backend.novelty import (
    PRIVATE_SCENE_DIRECTIONS,
    create_novelty_context,
    prompt_similarity,
    provider_novelty_instruction,
)


@dataclass(frozen=True)
class HistoryItem:
    positive_prompt: str
    novelty_signature: str | None = None
    user_rating: int | None = None
    feedback_reasons: tuple[str, ...] = ()


class NoveltyEngineTests(unittest.TestCase):
    """Ensure deterministic wrappers do not disguise repetitive creative scenes."""

    def test_similarity_ignores_common_quality_wrapper(self) -> None:
        wrapper = "score_9, score_8_up, source_photo, rating_safe, 1woman, solo, "
        bedroom = wrapper + "reclining on a velvet bed in a luxury bedroom, warm golden light"
        repeated = wrapper + "lying on a velvet sofa in a luxury bedroom, warm window light"
        different = wrapper + "calibrating a telescope at a desert observatory under cool moonlight"

        self.assertGreater(prompt_similarity(bedroom, repeated), prompt_similarity(bedroom, different))

    def test_context_uses_categories_not_raw_history_in_provider_instruction(self) -> None:
        secret_phrase = "private bedroom with a uniquely named velvet chaise"
        history = [HistoryItem(secret_phrase)]
        context = create_novelty_context(
            history,
            creativity_level="balanced",
            aspect_ratio="Landscape (1216x832)",
            rng=random.Random(7),
        )
        instruction = provider_novelty_instruction(context)

        self.assertIn("bedroom interiors", instruction)
        self.assertNotIn("uniquely named velvet chaise", instruction)
        self.assertEqual(context.creativity_level, "balanced")
        self.assertEqual(json.loads(context.brief.as_json())["theme"], context.brief.theme)

    def test_experimental_mode_uses_stricter_similarity_threshold(self) -> None:
        history = [HistoryItem("a woman walking through a rain-washed tram platform")]
        consistent = create_novelty_context(
            history,
            creativity_level="consistent",
            aspect_ratio="Portrait (832x1216)",
            rng=random.Random(2),
        )
        experimental = create_novelty_context(
            history,
            creativity_level="experimental",
            aspect_ratio="Portrait (832x1216)",
            rng=random.Random(2),
        )

        self.assertLess(experimental.similarity_threshold, consistent.similarity_threshold)

    def test_low_rating_reasons_become_prompt_guidance(self) -> None:
        history = [
            HistoryItem(
                "a generated portrait",
                novelty_signature='{"theme":"urban discovery"}',
                user_rating=1,
                feedback_reasons=("anatomy", "prompt_match"),
            )
        ]
        context = create_novelty_context(
            history,
            creativity_level="balanced",
            aspect_ratio="Portrait (832x1216)",
            rng=random.Random(3),
        )

        guidance = " ".join(context.quality_guidance)
        self.assertIn("anatomically coherent poses", guidance)
        self.assertIn("literal subject and action adherence", guidance)
        instruction = provider_novelty_instruction(context)
        self.assertIn("preferences learned from recent ratings", instruction)

    def test_eye_feedback_becomes_specific_prompt_guidance(self) -> None:
        history = [
            HistoryItem(
                "a generated portrait",
                novelty_signature='{"theme":"editorial portrait"}',
                user_rating=1,
                feedback_reasons=("eyes",),
            )
        ]
        context = create_novelty_context(
            history,
            creativity_level="balanced",
            aspect_ratio="Portrait (832x1216)",
            rng=random.Random(4),
        )

        self.assertIn("symmetrical eyes", " ".join(context.quality_guidance))

    def test_private_curated_pool_has_broad_scene_coverage(self) -> None:
        self.assertGreaterEqual(len(PRIVATE_SCENE_DIRECTIONS), 10)
        self.assertGreaterEqual(
            len({environment for item in PRIVATE_SCENE_DIRECTIONS for environment in item.environments}),
            30,
        )

    @patch("backend.main.save_prompt_run", return_value=41)
    @patch("backend.main.recent_prompt_history", return_value=[])
    def test_curated_endpoint_persists_novelty_metadata(self, _history, save_run) -> None:
        with patch("backend.main.generate_random_prompt_pair") as legacy_formatter:
            result = surprise(
                SurpriseRequest(
                    model="sd\\cyberrealisticPony_v180Coreshift.safetensors",
                    style="Photoreal",
                    content_rating="Safe",
                    aspect_ratio="Landscape (1216x832)",
                    prompt_engine="curated",
                    creativity_level="experimental",
                )
            )

        self.assertEqual(result.prompt_run_id, 41)
        legacy_formatter.assert_not_called()
        self.assertIn("novelty_signature", save_run.call_args.kwargs)
        self.assertEqual(save_run.call_args.kwargs["creativity_level"], "experimental")
        self.assertIsInstance(save_run.call_args.kwargs["similarity_score"], float)
        signature = json.loads(save_run.call_args.kwargs["novelty_signature"])
        self.assertGreaterEqual(
            len(signature["canonical_concept_ids"]),
            10,
        )


if __name__ == "__main__":
    unittest.main()
