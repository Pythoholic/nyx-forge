"""Coverage for durable job actions and complete generation lineage."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from threading import Event
import tempfile
import unittest
from time import sleep
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from backend import auth, database, main
from backend.forge import GenerationSettings, Txt2ImgPayload
from backend.jobs import JobRunner


class JobActionEndpointTests(unittest.TestCase):
    """Exercise authorization, payload replay, cancellation, and ancestry."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        database_path = root / "jobs.db"
        self.patches = (
            patch.object(database, "DATABASE_PATH", database_path),
            patch.object(database, "GENERATED_DIR", root / "generated"),
            patch.object(database, "THUMBNAILS_DIR", root / "generated" / "thumbnails"),
            patch.object(database, "RECOVERY_DIR", root / "generated" / "recovery"),
            patch.object(auth, "DATABASE_PATH", database_path),
            patch.object(auth, "PASSWORD_ITERATIONS", 10_000),
            patch.object(main, "GENERATED_DIR", root / "generated"),
        )
        for active_patch in self.patches:
            active_patch.start()
        database.initialize_database()
        auth.initialize_auth_database()
        self.account = auth.create_account(
            "jobs@example.com", "Jobs Admin", "long-password"
        )
        self.session_token = auth.create_session(self.account.id)
        auth.set_session_workspace(self.session_token, "forgeai")
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()
        self.client.cookies.set("aiqg_session", self.session_token)

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temporary_directory.cleanup()

    def test_rerun_replays_image_settings_without_transient_ids(self) -> None:
        original_id = database.create_job(
            "generate",
            {
                "prompt": "studio portrait",
                "negative_prompt": "blur",
                "model": "Forge default",
                "style": "Photoreal",
                "content_rating": "Safe",
                "aspect_ratio": "Portrait (832x1216)",
                "prompt_run_id": 41,
                "request_id": "request-original",
            },
            self.account.id,
            None,
        )
        database.finish_job(original_id, status="failed", error_message="retry me")
        fake_runner = MagicMock()

        with patch.object(main, "_job_runner", fake_runner):
            response = self.client.post(f"/api/jobs/{original_id}/rerun")

        self.assertEqual(response.status_code, 202)
        rerun = database.get_job(response.json()["job_id"])
        self.assertIsNotNone(rerun)
        self.assertEqual(rerun.request_payload["prompt"], "studio portrait")
        self.assertIsNone(rerun.request_payload["prompt_run_id"])
        self.assertIsNone(rerun.request_payload["request_id"])
        fake_runner.enqueue.assert_called_once()

    def _cloud_surprise_job(self, **extra: object) -> int:
        payload = {
            "model": "Forge default",
            "style": "Photoreal",
            "content_rating": "Safe",
            "aspect_ratio": "Portrait (832x1216)",
            "prompt_engine": "cloud",
            "cloud_provider": "anthropic",
            "cloud_model": "claude-sonnet-5",
            "request_id": "request-original",
        }
        payload.update(extra)
        job_id = database.create_job("surprise_generate", payload, self.account.id, None)
        database.finish_job(job_id, status="failed", error_message="restarted")
        return job_id

    def test_cloud_rerun_reuses_a_prompt_that_was_already_generated(self) -> None:
        # The API key is deliberately never stored, but once the prompt exists
        # the worker reuses it without calling the provider, so the missing key
        # must not block the re-run.
        original_id = self._cloud_surprise_job(
            resolved_positive_prompt="a lantern-lit alley",
            resolved_negative_prompt="blur",
            resolved_prompt_run_id=7,
            resolved_prompt_engine="cloud",
        )
        fake_runner = MagicMock()

        with patch.object(main, "_job_runner", fake_runner):
            response = self.client.post(f"/api/jobs/{original_id}/rerun")

        self.assertEqual(response.status_code, 202)
        rerun = database.get_job(response.json()["job_id"])
        self.assertEqual(
            rerun.request_payload["resolved_positive_prompt"], "a lantern-lit alley"
        )
        self.assertNotIn("cloud_api_key", rerun.request_payload)
        fake_runner.enqueue.assert_called_once()

    def test_cloud_rerun_recovers_a_prompt_from_its_generation(self) -> None:
        # Jobs created before the resolved_* fields existed still produced a
        # generation, and that record holds the prompt the UI displays. A
        # completed job must stay re-runnable rather than being refused over a
        # key it no longer needs.
        original_id = self._cloud_surprise_job()
        with database._connection() as connection:  # noqa: SLF001 - fixture row
            generation_id = connection.execute(
                "INSERT INTO generations (positive_prompt, negative_prompt, timestamp,"
                " width, height, rating, style, high_res, super_res, model, filename)"
                " VALUES ('a lantern-lit alley','blur','2026-01-01 00:00:00',832,1216,"
                "'Safe','Photoreal',0,0,'Forge default','generation-1.png')"
            ).lastrowid
        database.finish_job(
            original_id, status="succeeded", generation_id=int(generation_id)
        )
        fake_runner = MagicMock()

        with patch.object(main, "get_generation") as fetch, patch.object(
            main, "_job_runner", fake_runner
        ):
            fetch.return_value = SimpleNamespace(
                positive_prompt="a lantern-lit alley", negative_prompt="blur"
            )
            response = self.client.post(f"/api/jobs/{original_id}/rerun")

        self.assertEqual(response.status_code, 202)
        rerun = database.get_job(response.json()["job_id"])
        self.assertEqual(
            rerun.request_payload["resolved_positive_prompt"], "a lantern-lit alley"
        )
        fake_runner.enqueue.assert_called_once()

    def test_cloud_rerun_without_a_prompt_is_still_refused(self) -> None:
        original_id = self._cloud_surprise_job()
        fake_runner = MagicMock()

        with patch.object(main, "_job_runner", fake_runner):
            response = self.client.post(f"/api/jobs/{original_id}/rerun")

        self.assertEqual(response.status_code, 409)
        fake_runner.enqueue.assert_not_called()

    def test_cancelling_one_queued_job_leaves_the_others_runnable(self) -> None:
        # Cancelling must not stall the FIFO worker: the cancelled job is
        # skipped and the queue continues with the next item.
        executed = []
        runner = JobRunner(lambda job_id, payload: executed.append(job_id) or 1)
        first = database.create_job("generate", {"prompt": "a"}, None, None)
        second = database.create_job("generate", {"prompt": "b"}, None, None)
        runner.enqueue(first)
        runner.enqueue(second)

        self.assertEqual(runner.request_cancel(first), "queued")
        runner.start()
        for _ in range(200):
            if database.get_job(second).status in {"succeeded", "failed"}:
                break
            sleep(0.02)
        runner.stop()

        self.assertNotIn(first, executed)
        self.assertIn(second, executed)
        self.assertIn(database.get_job(second).status, {"succeeded", "failed"})

    def test_delete_removes_terminal_records_and_keeps_active_ones(self) -> None:
        finished = database.create_job("generate", {"prompt": "a"}, self.account.id, None)
        database.finish_job(finished, status="failed", error_message="x")
        queued = database.create_job("generate", {"prompt": "b"}, self.account.id, None)

        response = self.client.post(
            "/api/jobs/delete", json={"job_ids": [finished, queued]}
        )

        self.assertEqual(response.status_code, 200)
        # Only the terminal row goes: deleting a queued job would orphan work
        # the runner still references.
        self.assertEqual(response.json()["deleted"], 1)
        self.assertIsNone(database.get_job(finished))
        self.assertIsNotNone(database.get_job(queued))

    def test_cancel_queued_job_persists_terminal_state(self) -> None:
        job_id = database.create_job(
            "generate", {"prompt": "portrait"}, self.account.id, None
        )
        fake_runner = MagicMock()
        fake_runner.request_cancel.return_value = "queued"

        with patch.object(main, "_job_runner", fake_runner):
            response = self.client.post(f"/api/jobs/{job_id}/cancel")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cancelled")
        self.assertEqual(database.get_job(job_id).status, "cancelled")

    def test_jobs_history_filters_product_pipeline_kinds(self) -> None:
        created = {
            kind: database.create_job(kind, {}, self.account.id, None)
            for kind in ("generate", "upscale", "img2img", "character")
        }

        response = self.client.get(
            "/api/jobs?kind=img2img&kind=character&limit=100"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {item["id"] for item in response.json()["items"]},
            {created["img2img"], created["character"]},
        )
        invalid = self.client.get("/api/jobs?kind=unknown")
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["detail"], "Unsupported job kind filter.")

    def test_jobs_history_mine_scope_keeps_workspace_monitor_account_scoped(self) -> None:
        own = database.create_job(
            "batch_generate", {"batch_id": 17}, self.account.id, None
        )
        # The local product has one self-service admin, but historical/imported
        # rows may still belong to another account id. Admin-wide Deploy sees
        # them; the shell's personal operation monitor deliberately does not.
        database.create_job("generate", {}, self.account.id + 1, None)

        all_visible = self.client.get("/api/jobs?limit=100")
        mine = self.client.get("/api/jobs?limit=100&mine=true")

        self.assertEqual(all_visible.status_code, 200)
        self.assertEqual(len(all_visible.json()["items"]), 2)
        self.assertEqual(mine.status_code, 200)
        self.assertEqual([item["id"] for item in mine.json()["items"]], [own])
        self.assertEqual(mine.json()["items"][0]["request_payload"]["batch_id"], 17)

    def test_cancel_running_diffusion_uses_forge_interrupt(self) -> None:
        job_id = database.create_job(
            "generate", {"prompt": "portrait"}, self.account.id, None
        )
        database.update_job_progress(
            job_id, status="running", percent=30, stage="sampling"
        )
        fake_runner = MagicMock()

        def request_cancel(_job_id, *, interrupt_active, allow_active):
            self.assertTrue(allow_active)
            interrupt_active()
            return "running"

        fake_runner.request_cancel.side_effect = request_cancel
        with (
            patch.object(main, "_job_runner", fake_runner),
            patch.object(main.ForgeApiClient, "interrupt_generation") as interrupt,
        ):
            response = self.client.post(f"/api/jobs/{job_id}/cancel")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "running")
        interrupt.assert_called_once_with()

    def _save_generation(
        self,
        *,
        source_generation_id: int | None,
        high_res: bool,
        super_res: bool,
    ):
        image = BytesIO()
        Image.new("RGB", (24, 32), "gray").save(image, format="PNG")
        return database.save_generation(
            image.getvalue(),
            Txt2ImgPayload("portrait", "blur", 24, 32, seed=123),
            GenerationSettings(
                24,
                32,
                "Forge default",
                "Safe",
                "Photoreal",
                high_res,
                super_res,
            ),
            actual_seed=123,
            source_generation_id=source_generation_id,
        )

    def test_jobs_history_projects_list_fields_but_detail_stays_complete(self) -> None:
        generation = self._save_generation(
            source_generation_id=None, high_res=True, super_res=False
        )
        payload = {
            "prompt": "studio portrait",
            "negative_prompt": "blur",
            "resolved_positive_prompt": "resolved studio portrait",
            "internal_state": {"large": "private worker state"},
            "generation_id": generation.id,
            "model": "Forge default",
            "style": "Photoreal",
            "style_variant": "Editorial",
            "aspect_ratio": "Portrait (832x1216)",
            "quality_mode": "8k",
            "high_res": True,
            "super_res": False,
        }
        job_id = database.create_job("upscale", payload, self.account.id, None)
        database.finish_job(job_id, status="succeeded", generation_id=generation.id)

        list_response = self.client.get("/api/jobs?limit=20")

        self.assertEqual(list_response.status_code, 200)
        listed = list_response.json()["items"][0]
        self.assertEqual(
            listed["request_payload"],
            {
                "model": "Forge default",
                "style": "Photoreal",
                "style_variant": "Editorial",
                "aspect_ratio": "Portrait (832x1216)",
                "quality_mode": "8k",
                "high_res": True,
                "super_res": False,
                "source_generation_id": generation.id,
            },
        )
        self.assertNotIn("positive_prompt", listed)
        self.assertNotIn("negative_prompt", listed)
        self.assertEqual(
            set(listed["generation"]),
            {
                "id",
                "thumbnail_url",
                "full_url",
                "width",
                "height",
                "rating",
                "style_variant",
                "seed",
                "source_generation_id",
                "favorite",
            },
        )

        detail_response = self.client.get(f"/api/jobs/{job_id}")

        self.assertEqual(detail_response.status_code, 200)
        detail = detail_response.json()
        self.assertEqual(detail["request_payload"], payload)
        self.assertEqual(detail["positive_prompt"], generation.positive_prompt)
        self.assertIn("sampler_name", detail["generation"])

    def test_failed_derived_job_exposes_its_source_prompts(self) -> None:
        source = self._save_generation(
            source_generation_id=None, high_res=False, super_res=False
        )
        job_id = database.create_job(
            "upscale",
            {"generation_id": source.id, "super_res": True},
            self.account.id,
            None,
        )
        database.finish_job(
            job_id, status="failed", error_message="Interrupted by server restart"
        )

        response = self.client.get(f"/api/jobs/{job_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["positive_prompt"], "portrait")
        self.assertEqual(response.json()["negative_prompt"], "blur")

    def test_restart_interrupted_upscale_can_be_rerun(self) -> None:
        source = self._save_generation(
            source_generation_id=None, high_res=False, super_res=False
        )
        job_id = database.create_job(
            "upscale",
            {"generation_id": source.id, "super_res": True},
            self.account.id,
            None,
        )
        database.finish_job(
            job_id, status="failed", error_message="Interrupted by server restart"
        )
        fake_runner = MagicMock()

        with patch.object(main, "_job_runner", fake_runner):
            response = self.client.post(f"/api/jobs/{job_id}/rerun")

        self.assertEqual(response.status_code, 202)
        rerun = database.get_job(response.json()["job_id"])
        self.assertEqual(rerun.kind, "upscale")
        self.assertEqual(rerun.request_payload["generation_id"], source.id)
        fake_runner.enqueue.assert_called_once()

    def test_lineage_returns_the_complete_oldest_first_chain(self) -> None:
        root = self._save_generation(
            source_generation_id=None, high_res=False, super_res=False
        )
        high = self._save_generation(
            source_generation_id=root.id, high_res=True, super_res=False
        )
        super_result = self._save_generation(
            source_generation_id=high.id, high_res=True, super_res=True
        )
        for kind, generation_id in (
            ("generate", root.id),
            ("upscale", high.id),
            ("upscale", super_result.id),
        ):
            job_id = database.create_job(kind, {}, self.account.id, None)
            database.finish_job(job_id, status="succeeded", generation_id=generation_id)

        response = self.client.get(
            f"/api/generations/{super_result.id}/lineage"
        )

        self.assertEqual(response.status_code, 200)
        items = response.json()["items"]
        self.assertEqual([item["id"] for item in items], [root.id, high.id, super_result.id])
        self.assertEqual(
            [item["pipeline_kind"] for item in items],
            ["generate", "upscale", "upscale"],
        )


class JobRunnerCancellationTests(unittest.TestCase):
    """Prove an interrupt cannot race into the next FIFO job."""

    def test_active_cancel_finishes_as_cancelled_not_succeeded(self) -> None:
        started = Event()
        release = Event()
        finished = Event()

        def executor(_job_id, _payload):
            started.set()
            release.wait(timeout=2)
            return 99

        runner = JobRunner(executor)
        finish_calls = []

        def record_finish(*args, **kwargs):
            finish_calls.append((args, kwargs))
            finished.set()

        with (
            patch("backend.jobs.update_job_progress"),
            patch("backend.jobs.finish_job", side_effect=record_finish),
        ):
            runner.start()
            runner.enqueue(7)
            self.assertTrue(started.wait(timeout=1))
            outcome = runner.request_cancel(7, interrupt_active=release.set)
            self.assertEqual(outcome, "running")
            self.assertTrue(finished.wait(timeout=1))
            runner.stop()

        self.assertEqual(finish_calls[-1][1]["status"], "cancelled")
        self.assertNotIn("generation_id", finish_calls[-1][1])


if __name__ == "__main__":
    unittest.main()
