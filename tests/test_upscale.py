"""Regression tests for re-rendering an existing draft with the detail pass."""

from __future__ import annotations

import tempfile
import time
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from backend import auth, database, main
from backend.main import app

GENERATE_BODY = {
    "prompt": "a clearly adult woman standing in a garden, portrait photograph",
    "negative_prompt": "",
    "model": "sd\\cyberrealisticPony_v180Coreshift.safetensors",
    "style": "Photoreal",
    "aspect_ratio": "Portrait (832x1216)",
    "content_rating": "Safe",
    "high_res": False,
    "super_res": False,
}


def _png_bytes(size: tuple[int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "gray").save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_forge_result(seed: int = 123456, size: tuple[int, int] = (832, 1216)) -> SimpleNamespace:
    return SimpleNamespace(
        image_bytes=_png_bytes(size),
        seed=seed,
        diffusion_duration_seconds=1.0,
        upscale_duration_seconds=0.0,
    )


class UpscaleEndpointTests(unittest.TestCase):
    """Verify /api/generations/{id}/upscale re-renders the same seed on demand."""

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        db_path = root / "test.db"
        self.patches = (
            patch.object(database, "DATABASE_PATH", db_path),
            patch.object(database, "GENERATED_DIR", root / "generated"),
            # main.py binds GENERATED_DIR by value at import, so the request
            # handlers need their own copy patched too.
            patch.object(main, "GENERATED_DIR", root / "generated"),
            patch.object(database, "THUMBNAILS_DIR", root / "generated" / "thumbnails"),
            patch.object(database, "RECOVERY_DIR", root / "generated" / "recovery"),
            patch.object(auth, "DATABASE_PATH", db_path),
        )
        for active_patch in self.patches:
            active_patch.start()
        database.initialize_database()
        auth.initialize_auth_database()
        account = auth.create_account(
            "upscale@example.com", "Upscale Admin", "long-password"
        )
        session_token = auth.create_session(account.id)
        auth.set_session_workspace(session_token, "forgeai")
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.client.cookies.set("aiqg_session", session_token)

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_directory.cleanup()

    def _wait_for_generation(self, response) -> dict[str, object]:
        self.assertEqual(response.status_code, 202)
        handle = response.json()
        for _ in range(100):
            job = self.client.get(
                f"/api/jobs/{handle['job_id']}",
                params={"access": handle.get("access_token")},
            ).json()
            if job["status"] == "succeeded":
                return job["generation"]
            if job["status"] == "failed":
                self.fail(job.get("error_message"))
            time.sleep(0.01)
        self.fail("Job did not finish in time.")

    @patch("backend.main.ForgeApiClient.generate_image")
    def test_upscale_reuses_the_source_seed_and_prompt(self, forge_generate) -> None:
        forge_generate.return_value = _fake_forge_result(seed=987654)
        draft = self._wait_for_generation(self.client.post("/api/generate", json=GENERATE_BODY))
        self.assertFalse(draft["high_res"])

        forge_generate.return_value = _fake_forge_result(
            seed=987654, size=(1248, 1824)
        )
        response = self.client.post(
            f"/api/generations/{draft['id']}/upscale",
            json={"super_res": False, "access_token": draft.get("access_token")},
        )

        upscaled = self._wait_for_generation(response)
        self.assertTrue(upscaled["high_res"])
        self.assertFalse(upscaled["super_res"])
        self.assertEqual(upscaled["source_generation_id"], draft["id"])
        self.assertEqual(upscaled["seed"], draft["seed"])
        self.assertEqual(upscaled["positive_prompt"], draft["positive_prompt"])
        self.assertGreater(upscaled["width"], draft["width"])
        # The refine pass was actually requested, not silently skipped.
        call_kwargs = forge_generate.call_args.kwargs
        self.assertTrue(call_kwargs["high_res"])

    @patch("backend.main.ForgeApiClient.generate_image")
    def test_upscaling_an_already_high_res_image_is_allowed(self, forge_generate) -> None:
        forge_generate.return_value = _fake_forge_result()
        body = {**GENERATE_BODY, "high_res": True}
        draft = self._wait_for_generation(self.client.post("/api/generate", json=body))
        forge_generate.reset_mock()
        forge_generate.return_value = _fake_forge_result(size=(1664, 2432))

        # Local generation has no per-call cost - re-running the detail pass
        # on an already-high_res image must not be blocked at any stage.
        response = self.client.post(
            f"/api/generations/{draft['id']}/upscale",
            json={"super_res": True, "access_token": draft.get("access_token")},
        )

        self._wait_for_generation(response)
        forge_generate.assert_called_once()

    @patch("backend.main.ForgeApiClient.refine_and_upscale_image_bytes")
    @patch("backend.main.ForgeApiClient.generate_image")
    def test_upscale_renders_from_native_resolution_not_the_sources_current_size(
        self, forge_generate, pixel_upscale
    ) -> None:
        # A source already grown past native (e.g. by a prior pixel upscale)
        # must not have its detail pass anchored to that larger size - doing
        # so compounds resolution every re-render and pushes Forge toward a
        # VRAM-starved tiled-VAE fallback instead of just adding detail.
        forge_generate.return_value = _fake_forge_result()
        draft = self._wait_for_generation(self.client.post("/api/generate", json=GENERATE_BODY))
        self.assertEqual((draft["width"], draft["height"]), (832, 1216))

        pixel_upscale.return_value = _png_bytes((3328, 4864))
        pixel_response = self.client.post(
            f"/api/generations/{draft['id']}/pixel-upscale",
            json={"multiplier": 4.0, "access_token": draft.get("access_token")},
        )
        pixel_upscaled = self._wait_for_generation(pixel_response)
        self.assertEqual((pixel_upscaled["width"], pixel_upscaled["height"]), (3328, 4864))

        forge_generate.reset_mock()
        forge_generate.return_value = _fake_forge_result(size=(1248, 1824))
        response = self.client.post(
            f"/api/generations/{pixel_upscaled['id']}/upscale",
            json={"super_res": False, "access_token": pixel_upscaled.get("access_token")},
        )
        self._wait_for_generation(response)

        sent_payload = forge_generate.call_args.args[0]
        self.assertEqual((sent_payload.width, sent_payload.height), (832, 1216))

    def test_upscaling_a_missing_generation_returns_404(self) -> None:
        response = self.client.post(
            "/api/generations/999999/upscale",
            json={"super_res": False, "access_token": "irrelevant"},
        )
        self.assertEqual(response.status_code, 404)

    @patch("backend.main.ForgeApiClient.generate_image")
    def test_upscaling_with_a_wrong_access_token_is_denied(self, forge_generate) -> None:
        forge_generate.return_value = _fake_forge_result()
        draft = self._wait_for_generation(self.client.post("/api/generate", json=GENERATE_BODY))
        forge_generate.reset_mock()
        self.client.post("/api/auth/logout")

        response = self.client.post(
            f"/api/generations/{draft['id']}/upscale",
            json={"super_res": False, "access_token": "not-the-real-token"},
        )

        self.assertEqual(response.status_code, 403)
        forge_generate.assert_not_called()


class PixelUpscaleEndpointTests(unittest.TestCase):
    """Verify /api/generations/{id}/pixel-upscale works from any source tier."""

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        db_path = root / "test.db"
        self.patches = (
            patch.object(database, "DATABASE_PATH", db_path),
            patch.object(database, "GENERATED_DIR", root / "generated"),
            # main.py binds GENERATED_DIR by value at import, so the request
            # handlers need their own copy patched too.
            patch.object(main, "GENERATED_DIR", root / "generated"),
            patch.object(database, "THUMBNAILS_DIR", root / "generated" / "thumbnails"),
            patch.object(database, "RECOVERY_DIR", root / "generated" / "recovery"),
            patch.object(auth, "DATABASE_PATH", db_path),
        )
        for active_patch in self.patches:
            active_patch.start()
        database.initialize_database()
        auth.initialize_auth_database()
        account = auth.create_account(
            "pixel-upscale@example.com", "Pixel Upscale Admin", "long-password"
        )
        session_token = auth.create_session(account.id)
        auth.set_session_workspace(session_token, "forgeai")
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.client.cookies.set("aiqg_session", session_token)

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_directory.cleanup()

    def _wait_for_generation(self, response) -> dict[str, object]:
        self.assertEqual(response.status_code, 202)
        handle = response.json()
        for _ in range(100):
            job = self.client.get(
                f"/api/jobs/{handle['job_id']}",
                params={"access": handle.get("access_token")},
            ).json()
            if job["status"] == "succeeded":
                return job["generation"]
            if job["status"] == "failed":
                self.fail(job.get("error_message"))
            time.sleep(0.01)
        self.fail("Job did not finish in time.")

    def _upscaled_bytes(self, multiplier: float, source_size=(832, 1216)) -> bytes:
        buffer = BytesIO()
        target = (round(source_size[0] * multiplier), round(source_size[1] * multiplier))
        Image.new("RGB", target, "gray").save(buffer, format="PNG")
        return buffer.getvalue()

    def test_refine_caps_the_diffused_step_and_resizes_the_remainder(self) -> None:
        # Diffusing beyond REFINE_MAX_STEP costs far more time than the detail
        # it returns, so a 4x refines 2x and covers the rest with a resize.
        from backend import forge

        client = forge.ForgeApiClient()
        calls: list[tuple[str, object]] = []

        def fake_request(method, path, **kwargs):
            calls.append((path, kwargs.get("json", {})))
            response = MagicMock()
            response.json.return_value = {"images": ["x"], "image": "x"}
            return response

        with patch.object(forge.ForgeApiClient, "_request", staticmethod(fake_request)),              patch.object(forge.ForgeApiClient, "_decode_image", lambda self, value: b"img"):
            client.refine_and_upscale_image_bytes(
                b"src", prompt="p", negative_prompt="n",
                target_width=4000, target_height=4000,
                sampler_name="DPM++ SDE", scheduler="Karras", steps=30,
                cfg_scale=4.5, seed=1, multiplier=4.0,
            )

        paths = [path for path, _ in calls]
        self.assertEqual(paths, ["/sdapi/v1/img2img", "/sdapi/v1/extra-single-image"])
        # the diffused step is capped, not the full 4x
        self.assertEqual(calls[0][1]["script_args"][3], forge.REFINE_MAX_STEP)
        self.assertEqual(calls[0][1]["script_name"], "sd upscale")

        calls.clear()
        with patch.object(forge.ForgeApiClient, "_request", staticmethod(fake_request)),              patch.object(forge.ForgeApiClient, "_decode_image", lambda self, value: b"img"):
            client.refine_and_upscale_image_bytes(
                b"src", prompt="p", negative_prompt="n",
                target_width=2000, target_height=2000,
                sampler_name="DPM++ SDE", scheduler="Karras", steps=30,
                cfg_scale=4.5, seed=1, multiplier=2.0,
            )
        # a 2x is fully covered by diffusion, so no resize follows
        self.assertEqual([path for path, _ in calls], ["/sdapi/v1/img2img"])

    @patch("backend.main.ForgeApiClient.generate_image")
    @patch("backend.main.ForgeApiClient.refine_and_upscale_image_bytes")
    def test_pixel_upscale_refines_then_resizes_to_the_multiplier(
        self, pixel_upscale, forge_generate
    ) -> None:
        forge_generate.return_value = _fake_forge_result(seed=555)
        draft = self._wait_for_generation(self.client.post("/api/generate", json=GENERATE_BODY))
        pixel_upscale.return_value = self._upscaled_bytes(2.0)

        response = self.client.post(
            f"/api/generations/{draft['id']}/pixel-upscale",
            json={"multiplier": 2.0, "access_token": draft.get("access_token")},
        )

        result = self._wait_for_generation(response)
        self.assertEqual(result["width"], draft["width"] * 2)
        self.assertEqual(result["height"], draft["height"] * 2)
        self.assertEqual(result["seed"], draft["seed"])
        self.assertEqual(result["source_generation_id"], draft["id"])
        # No new diffusion happened - only the pixel upscaler was called.
        forge_generate.assert_called_once()
        pixel_upscale.assert_called_once()

    @patch("backend.main.ForgeApiClient.generate_image")
    @patch("backend.main.ForgeApiClient.refine_and_upscale_image_bytes")
    def test_pixel_upscale_works_on_an_already_high_res_image(
        self, pixel_upscale, forge_generate
    ) -> None:
        forge_generate.return_value = _fake_forge_result(seed=777, size=(1248, 1824))
        body = {**GENERATE_BODY, "high_res": True}
        draft = self._wait_for_generation(self.client.post("/api/generate", json=body))
        self.assertTrue(draft["high_res"])
        pixel_upscale.return_value = self._upscaled_bytes(3.0, (1248, 1824))

        response = self.client.post(
            f"/api/generations/{draft['id']}/pixel-upscale",
            json={"multiplier": 3.0, "access_token": draft.get("access_token")},
        )

        self._wait_for_generation(response)
        pixel_upscale.assert_called_once()

    def test_pixel_upscale_rejects_a_multiplier_above_four(self) -> None:
        response = self.client.post(
            "/api/generations/1/pixel-upscale",
            json={"multiplier": 5.0, "access_token": "irrelevant"},
        )
        self.assertEqual(response.status_code, 422)

    def test_pixel_upscale_of_a_missing_generation_returns_404(self) -> None:
        response = self.client.post(
            "/api/generations/999999/pixel-upscale",
            json={"multiplier": 2.0, "access_token": "irrelevant"},
        )
        self.assertEqual(response.status_code, 404)

    @patch("backend.main.ForgeApiClient.generate_image")
    def test_pixel_upscale_with_a_wrong_access_token_is_denied(self, forge_generate) -> None:
        forge_generate.return_value = _fake_forge_result()
        draft = self._wait_for_generation(self.client.post("/api/generate", json=GENERATE_BODY))
        self.client.post("/api/auth/logout")

        response = self.client.post(
            f"/api/generations/{draft['id']}/pixel-upscale",
            json={"multiplier": 2.0, "access_token": "not-the-real-token"},
        )

        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
