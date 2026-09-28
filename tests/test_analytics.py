"""Tests for deterministic dashboard aggregation."""

from __future__ import annotations

import unittest
from dataclasses import replace

from backend.analytics import (
    AnalyticsRun,
    AttemptCounts,
    canonical_model_name,
    summarize_analytics,
)
from backend.versioning import PROMPT_COMPILER_VERSION


def run(
    identifier: int,
    duration: float | None,
    user_rating: int | None,
    *,
    quality: str = "standard",
    prompt_duration: float | None = None,
    compiler_version: str = PROMPT_COMPILER_VERSION,
) -> AnalyticsRun:
    """Build a concise run fixture."""
    return AnalyticsRun(
        id=identifier,
        timestamp=f"2026-08-{identifier:02d} 12:00:00",
        model="checkpoint",
        style="Photoreal",
        content_rating="Safe",
        aspect_ratio="Square (1024x1024)",
        quality_mode=quality,
        width=1024,
        height=1024,
        sampler_name="DPM++ 2M SDE",
        scheduler="Karras",
        steps=28,
        cfg_scale=6.0,
        duration_seconds=duration,
        diffusion_duration_seconds=duration,
        upscale_duration_seconds=0.0 if duration is not None else None,
        prompt_duration_seconds=prompt_duration,
        prompt_engine="curated",
        creativity_level="balanced",
        similarity_score=0.1,
        novelty_retry_count=0,
        user_rating=user_rating,
        compiler_version=compiler_version,
    )


class AnalyticsTests(unittest.TestCase):
    def test_summary_reports_distributions_and_coverage(self) -> None:
        snapshot = summarize_analytics(
            [
                run(1, 10, 5, prompt_duration=1.0),
                run(2, 20, 3, prompt_duration=3.0),
                run(3, 40, None, quality="high"),
                run(4, None, None, quality="high"),
            ],
            AttemptCounts(succeeded=4, failed=1),
            days=30,
        )
        summary = snapshot["summary"]
        self.assertEqual(summary["total_generations"], 4)
        self.assertEqual(summary["measured_generations"], 3)
        self.assertEqual(summary["median_seconds"], 20)
        self.assertEqual(summary["p90_seconds"], 40)
        self.assertEqual(summary["average_prompt_seconds"], 2)
        self.assertEqual(summary["rated_count"], 2)
        self.assertEqual(summary["success_rate"], 80)
        self.assertEqual(len(snapshot["by_quality"]), 2)
        self.assertEqual(len(snapshot["rating_distribution"]), 5)
        recommendation_ids = {
            item["id"] for item in snapshot["recommendations"]
        }
        self.assertIn("quality-recipe", recommendation_ids)
        self.assertIn("speed-recipe", recommendation_ids)
        self.assertIn("prompt-engine", recommendation_ids)
        self.assertIn("rating-coverage", recommendation_ids)
        self.assertIn("reliability", recommendation_ids)
        for recommendation in snapshot["recommendations"]:
            self.assertIn(recommendation["confidence"], {"learning", "medium", "high"})
            self.assertTrue(recommendation["evidence"])

    def test_one_off_rating_does_not_create_a_winner(self) -> None:
        """A single lucky result must not become a recipe recommendation."""
        snapshot = summarize_analytics(
            [run(1, 10, 5, prompt_duration=1.0)],
            AttemptCounts(succeeded=1),
            days=7,
        )
        recommendation_ids = {
            item["id"] for item in snapshot["recommendations"]
        }
        self.assertNotIn("quality-recipe", recommendation_ids)
        self.assertNotIn("speed-recipe", recommendation_ids)
        self.assertNotIn("prompt-engine", recommendation_ids)

    def test_legacy_prompt_runs_are_preserved_but_excluded_from_baseline(self) -> None:
        snapshot = summarize_analytics(
            [
                run(1, 10, 5),
                run(2, 90, 1, compiler_version="legacy"),
            ],
            AttemptCounts(succeeded=1),
            days=30,
        )

        summary = snapshot["summary"]
        self.assertEqual(summary["compiler_version"], PROMPT_COMPILER_VERSION)
        self.assertEqual(summary["total_generations"], 1)
        self.assertEqual(summary["legacy_generations_excluded"], 1)
        self.assertEqual(summary["median_seconds"], 10)

    def test_forge_hashes_and_duplicate_suffixes_share_one_model_identity(self) -> None:
        first = run(1, 10, 4)
        second = run(2, 20, 5)
        first = replace(
            first, model="sd\\juggernautXL_ragnarok.safetensors [dd08fa32f9]"
        )
        second = replace(
            second, model="sd\\juggernautXL_ragnarok (1).safetensors"
        )
        snapshot = summarize_analytics(
            [first, second], AttemptCounts(succeeded=2), days=30
        )
        self.assertEqual(canonical_model_name(first.model), "Juggernaut Ragnarok")
        self.assertEqual(len(snapshot["by_model"]), 1)
        self.assertEqual(snapshot["by_model"][0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
