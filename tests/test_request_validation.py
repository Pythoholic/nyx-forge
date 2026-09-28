"""Request-boundary regression tests for manual generation rules."""

from __future__ import annotations

import unittest
from time import sleep
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from backend.main import app
from backend import auth, database


class GenerateRequestValidationTests(unittest.TestCase):
    """Reject contradictory manual prompts before any Forge request is sent."""

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        db_path = root / "test.db"
        self.patches = (
            patch.object(database, "DATABASE_PATH", db_path),
            patch.object(database, "GENERATED_DIR", root / "generated"),
            patch.object(database, "THUMBNAILS_DIR", root / "generated" / "thumbnails"),
            patch.object(database, "RECOVERY_DIR", root / "generated" / "recovery"),
            patch.object(auth, "DATABASE_PATH", db_path),
        )
        for active_patch in self.patches:
            active_patch.start()
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        response = self.client.post(
            "/api/auth/register",
            json={
                "display_name": "Creator",
                "email": "creator@example.com",
                "password": "long-password",
            },
        )
        self.assertEqual(response.status_code, 200)
        workspace = self.client.post(
            "/api/auth/workspace", json={"workspace": "forgeai"}
        )
        self.assertEqual(workspace.status_code, 200)

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_directory.cleanup()

    def _wait_for_terminal_job(self, response) -> dict[str, object]:
        self.assertEqual(response.status_code, 202)
        handle = response.json()
        for _ in range(100):
            job = self.client.get(
                f"/api/jobs/{handle['job_id']}",
                params={"access": handle.get("access_token")},
            ).json()
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                return job
            time.sleep(0.01)
        self.fail("Job did not finish in time.")

    def _register_admin(self) -> dict[str, object]:
        response = self.client.get("/api/auth/session")
        self.assertEqual(response.status_code, 200)
        return response.json()["user"]

    @patch("backend.main.JobRunner.enqueue")
    def test_persisted_account_queue_limit_rejects_extra_jobs(self, _enqueue) -> None:
        self._register_admin()
        updated = self.client.patch(
            "/api/auth/settings", json={"max_active_jobs": 1}
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["max_active_jobs"], 1)

        payload = {
            "prompt": "an adult woman posing in a library",
            "model": "sd\\cyberrealisticPony_v180Coreshift.safetensors",
            "style": "Photoreal",
            "aspect_ratio": "Portrait (832x1216)",
            "content_rating": "Safe",
        }
        first = self.client.post("/api/generate", json=payload)
        second = self.client.post("/api/generate", json=payload)

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Queue limit reached", second.json()["detail"])

    @patch("backend.main.JobRunner.enqueue")
    @patch("backend.main.surprise")
    def test_cloud_prompt_is_resolved_before_gpu_job_is_enqueued(
        self, surprise_prompt, enqueue
    ) -> None:
        from backend.main import GenerateRequest, PromptPairResponse

        self._register_admin()
        surprise_prompt.return_value = PromptPairResponse(
            prompt="a cinematic portrait in a library",
            negative_prompt="blurry",
            engine="cloud",
            prompt_run_id=41,
        )

        response = self.client.post(
            "/api/surprise-generate",
            json={
                "model": "sd\\cyberrealisticPony_v180Coreshift.safetensors",
                "style": "Photoreal",
                "aspect_ratio": "Portrait (832x1216)",
                "content_rating": "Safe",
                "prompt_engine": "cloud",
                "cloud_provider": "deepseek",
                "cloud_model": "deepseek-chat",
                "cloud_api_key": "temporary-secret",
            },
        )

        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        # The prompt now resolves on a background thread so the click returns
        # immediately, rather than the request waiting on the provider.
        for _ in range(50):
            if database.get_job(job_id).request_payload.get("resolved_positive_prompt"):
                break
            sleep(0.02)
        surprise_prompt.assert_called_once()
        job = database.get_job(job_id)
        self.assertEqual(
            job.request_payload["resolved_positive_prompt"],
            "a cinematic portrait in a library",
        )
        self.assertEqual(job.request_payload["resolved_prompt_run_id"], 41)
        self.assertNotIn("cloud_api_key", job.request_payload)

    @patch("backend.main.ForgeApiClient.generate_image")
    def test_manual_male_prompt_returns_unprocessable_entity(self, forge_generate) -> None:
        response = self.client.post(
            "/api/generate",
            json={
                "prompt": "an adult woman and a man posing in a library",
                "negative_prompt": "",
                "model": "sd\\cyberrealisticPony_v180Coreshift.safetensors",
                "style": "Photoreal",
                "aspect_ratio": "Portrait (832x1216)",
                "content_rating": "Safe",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("male subjects", response.json()["detail"])
        forge_generate.assert_not_called()

    @patch("backend.main.ForgeApiClient.generate_image")
    @patch("backend.main.create_job", side_effect=sqlite3.OperationalError("locked"))
    def test_history_failure_stops_before_forge(self, _create_job, forge_generate) -> None:
        response = self.client.post(
            "/api/generate",
            json={
                "prompt": "an adult woman posing in a library",
                "model": "sd\\cyberrealisticPony_v180Coreshift.safetensors",
                "style": "Photoreal",
                "aspect_ratio": "Portrait (832x1216)",
                "content_rating": "Safe",
            },
        )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("locked", response.json()["detail"])
        forge_generate.assert_not_called()

    @patch("backend.main.ForgeApiClient.generate_image")
    @patch("backend.main.JobRunner.enqueue", side_effect=RuntimeError("stopping"))
    def test_stopping_worker_fails_the_persisted_job_cleanly(
        self, _enqueue, forge_generate
    ) -> None:
        response = self.client.post(
            "/api/generate",
            json={
                "prompt": "an adult woman posing in a library",
                "model": "sd\\cyberrealisticPony_v180Coreshift.safetensors",
                "style": "Photoreal",
                "aspect_ratio": "Portrait (832x1216)",
                "content_rating": "Safe",
            },
        )

        self.assertEqual(response.status_code, 503)
        jobs = database.recent_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, "failed")
        self.assertEqual(jobs[0].error_message, "The local job worker is unavailable.")
        forge_generate.assert_not_called()

    def test_forge_timeout_is_mapped_and_finalizes_the_prompt_attempt(self) -> None:
        with (
            patch("backend.main.save_prompt_run", return_value=41),
            patch("backend.main.average_generation_duration", return_value=10.0),
            patch(
                "backend.main.ForgeApiClient.generate_image",
                side_effect=requests.Timeout("slow"),
            ),
            patch("backend.main.finish_prompt_run") as finish_prompt,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "prompt": "an adult woman posing in a library",
                    "model": "sd\\cyberrealisticPony_v180Coreshift.safetensors",
                    "style": "Photoreal",
                    "aspect_ratio": "Portrait (832x1216)",
                    "content_rating": "Safe",
                },
            )
            job = self._wait_for_terminal_job(response)
        self.assertEqual(job["status"], "failed")
        self.assertIn("timed out", job["error_message"].lower())
        finish_prompt.assert_called_once_with(
            41,
            status="failed",
            generation_id=None,
            error_message="Forge timed out",
        )

    def test_disk_failure_is_sanitized_and_finalizes_the_prompt_attempt(self) -> None:
        forge_result = SimpleNamespace(
            image_bytes=b"png",
            seed=1,
            diffusion_duration_seconds=1.0,
            upscale_duration_seconds=0.0,
        )
        with (
            patch("backend.main.save_prompt_run", return_value=42),
            patch("backend.main.average_generation_duration", return_value=10.0),
            patch(
                "backend.main.ForgeApiClient.generate_image",
                return_value=forge_result,
            ),
            patch(
                "backend.main.save_generation",
                side_effect=OSError("secret disk path is full"),
            ),
            patch("backend.main.finish_prompt_run") as finish_prompt,
        ):
            response = self.client.post(
                "/api/generate",
                json={
                    "prompt": "an adult woman posing in a library",
                    "model": "sd\\cyberrealisticPony_v180Coreshift.safetensors",
                    "style": "Photoreal",
                    "aspect_ratio": "Portrait (832x1216)",
                    "content_rating": "Safe",
                },
            )
            job = self._wait_for_terminal_job(response)
        self.assertEqual(job["status"], "failed")
        self.assertNotIn("secret disk path", job["error_message"])
        finish_prompt.assert_called_once_with(
            42,
            status="failed",
            generation_id=None,
            error_message="Local image storage failed",
        )


if __name__ == "__main__":
    unittest.main()
