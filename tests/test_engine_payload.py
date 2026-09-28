"""An engine prompt must reach Forge unchanged.

The engine compiles against a measured token budget. Forge's build_payload
rebuilds a prompt it does not recognise as finished, appending its own
scaffolding, which silently pushed the pilot's prompts from 62 tokens to 116 -
well past the 75-token limit the engine had just fitted them into.
"""

from __future__ import annotations

import unittest

import pytest

from backend import database, engine_bridge
from backend.clip_tokenizer import CLIPBPETokenizer
from backend.diversity import DiversityStore
from backend.forge import GenerationSettings, aspect_ratios_for, build_payload
from backend.main import SurpriseRequest, surprise
from backend.prompt_engine import PromptEngine

PONY = r"sd\cyberrealisticPony_v180Coreshift.safetensors [9d0e340f5f]"
RAGNAROK = r"sd\juggernautXL_ragnarok.safetensors [dd08fa32f9]"
REALVIS = r"sd\realvisxlV50_v50Bakedvae.safetensors"
ASPECT = "Landscape (1216x832)"
TOKENIZER = CLIPBPETokenizer()


@pytest.fixture(autouse=True)
def isolated_engine_diversity(tmp_path, monkeypatch):
    """Keep payload selection independent of suite order and local app history."""
    seeds = iter(range(10_000, 10_100))
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "payload-test.db")
    database.initialize_database()
    monkeypatch.setattr(engine_bridge.secrets, "randbelow", lambda _: next(seeds))
    monkeypatch.setattr(
        engine_bridge,
        "_ENGINE",
        PromptEngine(diversity_store=DiversityStore(), diversity_enabled=True),
    )


def payload_for(model: str, orientation: str):
    body = SurpriseRequest(
        model=model,
        style="Photoreal",
        content_rating="Safe",
        aspect_ratio=ASPECT,
        high_res=True,
        orientation=orientation,
        prompt_engine="engine",
    )
    resolved = surprise(body)
    width, height = aspect_ratios_for(model)[ASPECT]
    settings = GenerationSettings(
        width=width,
        height=height,
        model=model,
        rating="Safe",
        style="Photoreal",
        high_res=True,
    )
    payload = build_payload(
        resolved.prompt, resolved.negative_prompt, settings, prebuilt=True
    )
    return resolved, payload


class EnginePayloadTests(unittest.TestCase):
    def test_forge_does_not_rewrite_an_engine_prompt(self) -> None:
        for model in (PONY, RAGNAROK, REALVIS):
            for orientation in ("front", "side", "back"):
                with self.subTest(model=model, orientation=orientation):
                    resolved, payload = payload_for(model, orientation)
                    self.assertEqual(payload.prompt, resolved.prompt)
                    self.assertEqual(payload.negative_prompt, resolved.negative_prompt)

    def test_the_payload_stays_inside_the_token_budget(self) -> None:
        for model in (PONY, RAGNAROK, REALVIS):
            for orientation in ("front", "side", "back"):
                with self.subTest(model=model, orientation=orientation):
                    _, payload = payload_for(model, orientation)
                    self.assertLessEqual(TOKENIZER.count(payload.prompt), 75)
                    self.assertLessEqual(TOKENIZER.count(payload.negative_prompt), 75)

    def test_a_legacy_prompt_is_still_built_by_forge(self) -> None:
        """prebuilt=False preserves the assembler for scripts and manual callers."""
        width, height = aspect_ratios_for(PONY)[ASPECT]
        settings = GenerationSettings(
            width=width,
            height=height,
            model=PONY,
            rating="Safe",
            style="Photoreal",
            high_res=True,
        )
        payload = build_payload("a woman standing", "blurry", settings)
        self.assertNotEqual(payload.prompt, "a woman standing")
        self.assertIn("score_9", payload.prompt)


if __name__ == "__main__":
    unittest.main()
