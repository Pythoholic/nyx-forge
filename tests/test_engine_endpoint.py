"""The engine prompt source, exercised through the app's own request path."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend import database
from backend.main import SurpriseRequest, surprise

PONY = r"sd\cyberrealisticPony_v180Coreshift.safetensors [9d0e340f5f]"
RAGNAROK = r"sd\juggernautXL_ragnarok.safetensors [dd08fa32f9]"


def request_for(model: str, orientation: str) -> SurpriseRequest:
    return SurpriseRequest(
        model=model,
        style="Photoreal",
        content_rating="Safe",
        aspect_ratio="Landscape (1216x832)",
        high_res=True,
        orientation=orientation,
        prompt_engine="engine",
    )


class EngineEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self._database_patch = patch.object(
            database,
            "DATABASE_PATH",
            Path(self._temporary_directory.name) / "engine-endpoint.db",
        )
        self._database_patch.start()
        database.initialize_database()

    def tearDown(self) -> None:
        self._database_patch.stop()
        self._temporary_directory.cleanup()

    def test_every_model_and_orientation_returns_a_prompt(self) -> None:
        """The two wiring bugs here were a wrong attribute and a wrong kwarg,
        neither of which any test would have caught before this one."""
        for model in (PONY, RAGNAROK):
            for orientation in ("front", "side", "back"):
                with self.subTest(model=model, orientation=orientation):
                    response = surprise(request_for(model, orientation))
                    self.assertTrue(response.prompt.strip())
                    self.assertTrue(response.negative_prompt.strip())
                    self.assertEqual(response.engine, "engine")
                    self.assertIsNotNone(response.prompt_run_id)

    def test_the_prompt_run_is_persisted(self) -> None:
        first = surprise(request_for(PONY, "back"))
        second = surprise(request_for(PONY, "back"))
        self.assertNotEqual(first.prompt_run_id, second.prompt_run_id)

    def test_the_curated_path_still_works(self) -> None:
        body = request_for(PONY, "back")
        body.prompt_engine = "curated"
        self.assertEqual(surprise(body).engine, "curated")


if __name__ == "__main__":
    unittest.main()
