"""Persistence regression tests for prompt runs and generation feedback."""

from __future__ import annotations

import tempfile
import unittest
import sqlite3
from contextlib import closing
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend import database
from backend.forge import GenerationSettings, Txt2ImgPayload
from backend.versioning import PROMPT_COMPILER_VERSION


class DatabaseTests(unittest.TestCase):
    """Verify telemetry and ratings survive separate database connections."""

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        self.patches = (
            patch.object(database, "DATABASE_PATH", root / "test.db"),
            patch.object(database, "GENERATED_DIR", root / "generated"),
            patch.object(database, "THUMBNAILS_DIR", root / "generated" / "thumbnails"),
            patch.object(database, "RECOVERY_DIR", root / "generated" / "recovery"),
        )
        for active_patch in self.patches:
            active_patch.start()
        database.initialize_database()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_directory.cleanup()

    def test_prompt_outcome_and_rating_are_persistent(self) -> None:
        prompt_id = database.save_prompt_run(
            positive_prompt="a detailed portrait",
            negative_prompt="blurry",
            source="auto_generated",
            engine="curated",
            model="test-model",
            style="Photoreal",
            aspect_ratio="Square (1024x1024)",
            content_rating="Safe",
            high_res=False,
            super_res=False,
            creativity_level="experimental",
            novelty_signature='{"theme":"scientific exploration"}',
            similarity_score=0.12,
            novelty_retry_count=1,
            prompt_duration_seconds=2.25,
        )
        image_buffer = BytesIO()
        Image.new("RGB", (32, 24), "red").save(image_buffer, "PNG")
        settings = GenerationSettings(1024, 1024, "test-model", "Safe", "Photoreal", False)
        record = database.save_generation(
            image_buffer.getvalue(),
            Txt2ImgPayload("a detailed portrait", "blurry", 1024, 1024),
            settings,
            duration_seconds=12.5,
            diffusion_duration_seconds=10.0,
            upscale_duration_seconds=2.5,
        )
        database.finish_prompt_run(prompt_id, status="succeeded", generation_id=record.id)

        self.assertEqual(
            database.rate_generation(record.id, 2, ("eyes", "anatomy")),
            (2, ("eyes", "anatomy")),
        )
        self.assertEqual(
            database.rate_generation(
                record.id, 5, ("prompt_match", "composition")
            ),
            (5, ("prompt_match", "composition")),
        )
        persisted = database.recent_generations(1)[0]
        self.assertEqual((persisted.width, persisted.height), (32, 24))
        self.assertEqual(persisted.user_rating, 5)
        self.assertEqual(
            persisted.feedback_reasons, ("prompt_match", "composition")
        )
        self.assertEqual(persisted.duration_seconds, 12.5)
        self.assertEqual(persisted.diffusion_duration_seconds, 10.0)
        self.assertEqual(persisted.upscale_duration_seconds, 2.5)
        self.assertEqual(
            database.average_generation_duration(
                model="test-model", style="Photoreal", high_res=False, super_res=False
            ),
            12.5,
        )

        with database._connection() as connection:
            prompt = connection.execute(
                "SELECT status, generation_id, compiler_version FROM prompt_runs WHERE id = ?", (prompt_id,)
            ).fetchone()
            stored_blob = connection.execute(
                "SELECT image_bytes FROM generations WHERE id = ?", (record.id,)
            ).fetchone()[0]
            feedback_count = connection.execute(
                "SELECT COUNT(*) FROM generation_feedback WHERE generation_id = ?", (record.id,)
            ).fetchone()[0]
        self.assertEqual((prompt["status"], prompt["generation_id"]), ("succeeded", record.id))
        self.assertEqual(prompt["compiler_version"], PROMPT_COMPILER_VERSION)
        self.assertIsNone(stored_blob)

        self.assertEqual(feedback_count, 1)

        history = database.recent_prompt_history(
            model="test-model",
            style="Photoreal",
            content_rating="Safe",
            limit=5,
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].novelty_signature, '{"theme":"scientific exploration"}')
        self.assertEqual(history[0].user_rating, 5)
        self.assertEqual(
            history[0].feedback_reasons, ("prompt_match", "composition")
        )
        analytics, attempts = database.analytics_records(30)
        self.assertEqual(len(analytics), 1)
        self.assertEqual(analytics[0].prompt_duration_seconds, 2.25)
        self.assertEqual(analytics[0].quality_mode, "normal")
        self.assertEqual((attempts.succeeded, attempts.failed), (1, 0))

    def test_generation_records_selected_style_variant(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (32, 24), "blue").save(image_buffer, "PNG")
        settings = GenerationSettings(
            832, 1216, "anime-model", "Safe", "Anime", False
        )
        payload = Txt2ImgPayload(
            "manga, lineart, screentones",
            "colorized, 3d",
            832,
            1216,
            style_variant="manga",
        )

        saved = database.save_generation(image_buffer.getvalue(), payload, settings)
        self.assertEqual(saved.style_variant, "manga")
        self.assertEqual(database.get_generation(saved.id).style_variant, "manga")
        self.assertEqual(database.recent_generations(1)[0].style_variant, "manga")

    def test_invalid_rating_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            database.rate_generation(1, 0)
        with self.assertRaises(ValueError):
            database.rate_generation(1, 4, ("unknown",))
        with self.assertRaisesRegex(ValueError, "at least one reason"):
            database.rate_generation(1, 2)

    def test_favorite_flag_round_trips_and_defaults_off(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (32, 24), "blue").save(image_buffer, "PNG")
        settings = GenerationSettings(832, 1216, "anime-model", "Safe", "Anime", False)
        payload = Txt2ImgPayload("a", "b", 832, 1216)
        record = database.save_generation(image_buffer.getvalue(), payload, settings)
        self.assertFalse(record.favorite)

        self.assertTrue(database.set_generation_favorite(record.id, True))
        self.assertTrue(database.get_generation(record.id).favorite)

        self.assertTrue(database.set_generation_favorite(record.id, False))
        self.assertFalse(database.get_generation(record.id).favorite)

    def test_favoriting_a_missing_generation_reports_failure(self) -> None:
        self.assertFalse(database.set_generation_favorite(999999, True))

    def test_stalled_running_job_is_released_so_the_queue_can_advance(self) -> None:
        # A job whose worker stops reporting holds the single FIFO worker and
        # blocks everything behind it. Startup recovery only covers a restart,
        # so a live process needs this too.
        job_id = database.create_job("generate", {"prompt": "x"}, None, None)
        with database._connection() as connection:  # noqa: SLF001 - fixture time
            connection.execute(
                "UPDATE jobs SET status = 'running',"
                " started_at = datetime('now','localtime','-9 hours') WHERE id = ?",
                (job_id,),
            )

        self.assertEqual(database.fail_stalled_running_jobs(3600), [job_id])
        released = database.get_job(job_id)
        self.assertEqual(released.status, "failed")
        self.assertIn("stopped responding", released.error_message)

    def test_a_recently_started_job_is_left_running(self) -> None:
        job_id = database.create_job("generate", {"prompt": "x"}, None, None)
        with database._connection() as connection:  # noqa: SLF001 - fixture time
            connection.execute(
                "UPDATE jobs SET status = 'running',"
                " started_at = datetime('now','localtime') WHERE id = ?",
                (job_id,),
            )

        self.assertEqual(database.fail_stalled_running_jobs(3600), [])
        self.assertEqual(database.get_job(job_id).status, "running")

    def test_interrupted_prompt_runs_are_recovered_without_touching_recent_runs(self) -> None:
        values = {
            "positive_prompt": "portrait",
            "negative_prompt": "blurry",
            "source": "auto_generated",
            "engine": "curated",
            "model": "test-model",
            "style": "Photoreal",
            "aspect_ratio": "Square (1024x1024)",
            "content_rating": "Safe",
            "high_res": False,
            "super_res": False,
        }
        old_id = database.save_prompt_run(**values)
        recent_id = database.save_prompt_run(**values)
        with database._connection() as connection:
            connection.execute(
                "UPDATE prompt_runs SET created_at = datetime('now', '-1 hour') WHERE id = ?",
                (old_id,),
            )

        self.assertEqual(database.recover_interrupted_prompt_runs(10), 1)
        with database._connection() as connection:
            rows = connection.execute(
                "SELECT id, status, error_message FROM prompt_runs ORDER BY id"
            ).fetchall()
        statuses = {int(row["id"]): str(row["status"]) for row in rows}
        self.assertEqual(statuses[old_id], "failed")
        self.assertEqual(statuses[recent_id], "created")
        self.assertEqual(database.recover_interrupted_prompt_runs(0), 1)

    def test_generation_write_failure_rolls_back_database_and_files(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (32, 24), "blue").save(image_buffer, "PNG")
        settings = GenerationSettings(32, 24, "test-model", "Safe", "Photoreal", False)
        with patch.object(database, "_thumbnail_bytes", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                database.save_generation(
                    image_buffer.getvalue(),
                    Txt2ImgPayload("portrait", "blurry", 32, 24),
                    settings,
                )

        with database._connection() as connection:
            count = connection.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
        self.assertEqual(count, 0)
        self.assertEqual(
            [path for path in database.GENERATED_DIR.iterdir() if path.is_file()], []
        )

    def test_startup_preserves_unindexed_generation_artifacts(self) -> None:
        """A restart must never infer that an image file is disposable."""
        image = database.GENERATED_DIR / "unindexed-user-image"
        thumbnail = database.THUMBNAILS_DIR / "unindexed-user-thumbnail"
        image.write_bytes(b"durable image")
        thumbnail.write_bytes(b"durable thumbnail")

        database.initialize_database()

        self.assertEqual(image.read_bytes(), b"durable image")
        self.assertEqual(thumbnail.read_bytes(), b"durable thumbnail")

    def test_busy_database_preserves_and_recovers_completed_image(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (32, 24), "purple").save(image_buffer, "PNG")
        settings = GenerationSettings(32, 24, "test-model", "Safe", "Photoreal", False)
        prompt_run_id = database.save_prompt_run(
            positive_prompt="portrait",
            negative_prompt="blurry",
            source="auto_generated",
            engine="curated",
            model="test-model",
            style="Photoreal",
            aspect_ratio="Landscape (32x24)",
            content_rating="Safe",
            high_res=False,
            super_res=False,
        )
        locked = sqlite3.OperationalError("database is locked")
        with (
            patch.object(database, "_commit_pending_generation", side_effect=locked),
            patch.object(database, "sleep"),
            self.assertRaises(sqlite3.OperationalError),
        ):
            database.save_generation(
                image_buffer.getvalue(),
                Txt2ImgPayload("portrait", "blurry", 32, 24),
                settings,
                prompt_run_id=prompt_run_id,
            )

        self.assertEqual(len(list(database.RECOVERY_DIR.glob("*.json"))), 1)
        self.assertEqual(len(list(database.RECOVERY_DIR.glob("*.png"))), 1)
        database.finish_prompt_run(
            prompt_run_id,
            status="failed",
            error_message="Local history database was busy; image queued for recovery",
        )
        self.assertEqual(database.recover_pending_generations(), 1)
        self.assertEqual(len(list(database.RECOVERY_DIR.glob("*.json"))), 0)
        recovered_filename = database.generation_filename(1)
        self.assertIsNotNone(recovered_filename)
        self.assertEqual(Path(recovered_filename).suffix, "")
        self.assertTrue((database.GENERATED_DIR / recovered_filename).is_file())
        with database._connection() as connection:
            prompt = connection.execute(
                "SELECT status, generation_id FROM prompt_runs WHERE id = ?",
                (prompt_run_id,),
            ).fetchone()
        self.assertEqual((prompt["status"], prompt["generation_id"]), ("succeeded", 1))

    def test_transient_busy_database_retries_without_leaving_recovery_files(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (24, 24), "orange").save(image_buffer, "PNG")
        settings = GenerationSettings(24, 24, "test-model", "Safe", "Photoreal", False)
        original_commit = database._commit_pending_generation
        attempts = 0

        def flaky_commit(pending: database.PendingGeneration):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise sqlite3.OperationalError("database is busy")
            return original_commit(pending)

        with (
            patch.object(database, "_commit_pending_generation", side_effect=flaky_commit),
            patch.object(database, "sleep") as retry_sleep,
        ):
            record = database.save_generation(
                image_buffer.getvalue(),
                Txt2ImgPayload("portrait", "blurry", 24, 24),
                settings,
            )

        self.assertEqual(record.id, 1)
        self.assertEqual(attempts, 3)
        self.assertEqual(retry_sleep.call_count, 2)
        self.assertEqual(list(database.RECOVERY_DIR.iterdir()), [])

    def test_legacy_generation_files_migrate_once_to_a_shared_opaque_name(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (24, 24), "gold").save(image_buffer, "PNG")
        settings = GenerationSettings(24, 24, "test-model", "Safe", "Photoreal", False)
        record = database.save_generation(
            image_buffer.getvalue(),
            Txt2ImgPayload("portrait", "blurry", 24, 24),
            settings,
        )
        legacy_image = database.GENERATED_DIR / f"generation-{record.id}.png"
        legacy_thumbnail = database.THUMBNAILS_DIR / f"thumb-{record.id}.jpg"
        (database.GENERATED_DIR / record.filename).replace(legacy_image)
        (database.THUMBNAILS_DIR / record.filename).replace(legacy_thumbnail)
        with database._connection() as connection:
            connection.execute(
                "UPDATE generations SET filename = ? WHERE id = ?",
                (legacy_image.name, record.id),
            )

        database.initialize_database()
        migrated = database.get_generation(record.id)
        self.assertIsNotNone(migrated)
        self.assertRegex(migrated.filename, r"^[0-9a-f]{32}$")
        self.assertTrue((database.GENERATED_DIR / migrated.filename).is_file())
        self.assertTrue((database.THUMBNAILS_DIR / migrated.filename).is_file())
        self.assertFalse(legacy_image.exists())
        self.assertFalse(legacy_thumbnail.exists())

        database.initialize_database()
        self.assertEqual(database.get_generation(record.id).filename, migrated.filename)

    def test_prefixed_legacy_generation_migrates_its_nested_thumbnail(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (24, 24), "indigo").save(image_buffer, "PNG")
        settings = GenerationSettings(24, 24, "test-model", "Safe", "Photoreal", False)
        record = database.save_generation(
            image_buffer.getvalue(),
            Txt2ImgPayload("portrait", "blurry", 24, 24),
            settings,
        )
        stale = database.save_generation(
            image_buffer.getvalue(),
            Txt2ImgPayload("missing portrait", "blurry", 24, 24),
            settings,
        )

        private_directory = database.GENERATED_DIR / ".nyx-private"
        legacy_image = private_directory / f"generation-{record.id}.png"
        legacy_thumbnail = (
            private_directory / "thumbnails" / f"thumb-{record.id}.jpg"
        )
        legacy_thumbnail.parent.mkdir(parents=True)
        (database.GENERATED_DIR / record.filename).replace(legacy_image)
        (database.THUMBNAILS_DIR / record.filename).replace(legacy_thumbnail)
        stale_filename = f"generation-{stale.id}.png"
        (database.GENERATED_DIR / stale.filename).unlink()
        (database.THUMBNAILS_DIR / stale.filename).unlink()
        with database._connection() as connection:
            connection.execute(
                "UPDATE generations SET filename = ? WHERE id = ?",
                (legacy_image.relative_to(database.GENERATED_DIR).as_posix(), record.id),
            )
            connection.execute(
                "UPDATE generations SET filename = ? WHERE id = ?",
                (stale_filename, stale.id),
            )

        self.assertEqual(database.sweep_orphan_generation_files(), 0)
        self.assertTrue(legacy_image.is_file())
        self.assertTrue(legacy_thumbnail.is_file())

        with self.assertLogs(database.LOGGER, level="WARNING") as migration_logs:
            self.assertEqual(database.migrate_legacy_generation_filenames(), 1)
        self.assertTrue(
            any(
                f"Generation {stale.id} still uses legacy filename {stale_filename}"
                in message
                for message in migration_logs.output
            )
        )
        migrated = database.get_generation(record.id)
        self.assertIsNotNone(migrated)
        self.assertRegex(migrated.filename, r"^[0-9a-f]{32}$")
        self.assertTrue((database.GENERATED_DIR / migrated.filename).is_file())
        self.assertTrue((database.THUMBNAILS_DIR / migrated.filename).is_file())
        self.assertFalse(legacy_image.exists())
        self.assertFalse(legacy_thumbnail.exists())
        self.assertFalse(private_directory.exists())
        self.assertEqual(database.get_generation(stale.id).filename, stale_filename)

        self.assertEqual(database.migrate_legacy_generation_filenames(), 0)
        self.assertEqual(database.sweep_orphan_generation_files(), 0)
        self.assertEqual(database.get_generation(record.id).filename, migrated.filename)
        self.assertEqual(database.get_generation(stale.id).filename, stale_filename)

    def test_job_lifecycle_and_restart_recovery_are_durable(self) -> None:
        job_id = database.create_job(
            "generate",
            {"prompt": "portrait", "model": "test-model"},
            account_id=7,
            access_token_hash="hashed-capability",
        )
        database.update_job_progress(
            job_id,
            status="running",
            percent=42.5,
            stage="sampling",
            current_step=12,
            total_steps=28,
            eta_seconds=6.4,
            current_image="data:image/png;base64,preview",
        )
        running = database.get_job(job_id)
        self.assertIsNotNone(running)
        self.assertEqual(running.status, "running")
        self.assertEqual(running.current_step, 12)
        self.assertEqual(running.request_payload["prompt"], "portrait")

        database.finish_job(job_id, status="succeeded", generation_id=None)
        completed = database.get_job(job_id)
        self.assertEqual(completed.status, "succeeded")
        self.assertEqual(completed.progress_percent, 100)
        self.assertIsNone(completed.current_image)

        # A failed job must keep the real stage it died in (e.g. "sampling")
        # rather than have progress_stage overwritten to the literal string
        # "failed" - the UI needs to know which pipeline stage to mark failed,
        # not just that the run as a whole did not succeed.
        failing_id = database.create_job("generate", {"prompt": "x"}, None, "hash3")
        database.update_job_progress(failing_id, status="running", stage="sampling")
        database.finish_job(failing_id, status="failed", error_message="Forge timed out")
        failed = database.get_job(failing_id)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.progress_stage, "sampling")

        queued_id = database.create_job("upscale", {"generation_id": 1}, None, "hash")
        running_id = database.create_job("pixel_upscale", {"generation_id": 1}, None, "hash2")
        database.update_job_progress(running_id, status="running", stage="upscaling")
        self.assertEqual(database.recover_stalled_jobs(), 2)
        # A job interrupted mid-render failed; one that never started is put
        # back on the queue rather than blamed for work it never attempted.
        interrupted = database.get_job(running_id)
        self.assertEqual(interrupted.status, "failed")
        self.assertEqual(interrupted.error_message, "Interrupted by server restart")
        never_started = database.get_job(queued_id)
        self.assertEqual(never_started.status, "queued")
        self.assertIsNone(never_started.error_message)
        self.assertIn(queued_id, database.resumable_job_ids())
        # recover_stalled_jobs must not clobber the real stage either.
        self.assertEqual(database.get_job(running_id).progress_stage, "upscaling")
        self.assertEqual(database.get_job(queued_id).progress_stage, "queued")

    def test_account_queue_limit_counts_queued_and_running_jobs(self) -> None:
        first_id = database.create_job(
            "generate", {"prompt": "first"}, 7, None, max_outstanding=2
        )
        second_id = database.create_job(
            "generate", {"prompt": "second"}, 7, None, max_outstanding=2
        )
        database.update_job_progress(first_id, status="running", stage="diffusion")

        with self.assertRaises(database.JobQueueLimitError) as raised:
            database.create_job(
                "generate", {"prompt": "third"}, 7, None, max_outstanding=2
            )
        self.assertEqual(raised.exception.limit, 2)

        database.finish_job(first_id, status="succeeded")
        third_id = database.create_job(
            "generate", {"prompt": "third"}, 7, None, max_outstanding=2
        )
        self.assertIsNotNone(database.get_job(third_id))
        self.assertEqual(database.get_job(second_id).status, "queued")

    def test_authenticated_request_id_is_idempotent_before_queue_limit(self) -> None:
        payload = {"prompt": "one image", "request_id": "same-user-action"}
        first_id = database.create_job(
            "generate", payload, 7, None, max_outstanding=50
        )

        with self.assertRaises(database.DuplicateJobSubmission) as raised:
            database.create_job(
                "generate", payload, 7, None, max_outstanding=50
            )

        self.assertEqual(raised.exception.job_id, first_id)
        with database._connection() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE account_id = 7"
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_character_job_is_requeued_after_restart(self) -> None:
        character_id = database.create_job(
            "character",
            {"preset": "variation", "character_stage": "CANDIDATE_A_GENERATED"},
            None,
            "character-hash",
        )
        ordinary_id = database.create_job(
            "img2img", {"preset": "outfit"}, None, "img2img-hash"
        )
        database.update_job_progress(
            character_id,
            status="running",
            percent=42,
            stage="identity_lock",
        )
        database.update_job_progress(
            ordinary_id,
            status="running",
            percent=42,
            stage="diffusion",
        )

        self.assertEqual(database.recover_stalled_jobs(), 2)
        character = database.get_job(character_id)
        ordinary = database.get_job(ordinary_id)
        self.assertEqual(character.status, "queued")
        self.assertEqual(character.progress_stage, "identity_lock")
        self.assertIsNone(character.error_message)
        self.assertEqual(database.resumable_character_job_ids(), [character_id])
        self.assertEqual(ordinary.status, "failed")
        self.assertEqual(ordinary.error_message, "Interrupted by server restart")

    def test_generation_history_uses_a_stable_id_cursor(self) -> None:
        image_buffer = BytesIO()
        Image.new("RGB", (16, 16), "green").save(image_buffer, "PNG")
        settings = GenerationSettings(16, 16, "test-model", "Safe", "Photoreal", False)
        for index in range(3):
            database.save_generation(
                image_buffer.getvalue(),
                Txt2ImgPayload(f"portrait {index}", "blurry", 16, 16),
                settings,
            )
        first_page = database.recent_generations(2)
        second_page = database.recent_generations(2, before_id=first_page[-1].id)
        self.assertEqual([record.id for record in first_page], [3, 2])
        self.assertEqual([record.id for record in second_page], [1])

    def test_populated_legacy_schema_migrates_idempotently(self) -> None:
        database.DATABASE_PATH.unlink()
        with closing(sqlite3.connect(database.DATABASE_PATH)) as connection:
            connection.execute(
                """
                CREATE TABLE generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, image_bytes BLOB,
                    positive_prompt TEXT NOT NULL, negative_prompt TEXT NOT NULL,
                    timestamp TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                    rating TEXT NOT NULL, style TEXT NOT NULL, high_res INTEGER NOT NULL,
                    model TEXT NOT NULL
                )
                """
            )
            connection.commit()
            connection.execute(
                """
                CREATE TABLE prompt_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    positive_prompt TEXT NOT NULL, negative_prompt TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    source TEXT NOT NULL, engine TEXT NOT NULL, model TEXT NOT NULL,
                    style TEXT NOT NULL, aspect_ratio TEXT NOT NULL,
                    content_rating TEXT NOT NULL, high_res INTEGER NOT NULL DEFAULT 0,
                    super_res INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'created', generation_id INTEGER,
                    error_message TEXT
                )
                """
            )
            connection.execute(
                """
                INSERT INTO prompt_runs (
                    positive_prompt, negative_prompt, source, engine, model,
                    style, aspect_ratio, content_rating
                ) VALUES ('portrait', 'blurry', 'auto_generated', 'curated',
                          'legacy-model', 'Photoreal', 'Square', 'Safe')
                """
            )
            connection.commit()

        database.initialize_database()
        database.initialize_database()
        with database._connection() as connection:
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(prompt_runs)")
            }
            row = connection.execute(
                "SELECT status, compiler_version, positive_prompt FROM prompt_runs"
            ).fetchone()
            count = connection.execute("SELECT COUNT(*) FROM prompt_runs").fetchone()[0]
        self.assertTrue(
            {"creativity_level", "similarity_score", "prompt_duration_seconds", "compiler_version"}
            <= columns
        )
        self.assertEqual(count, 1)
        self.assertEqual(
            (row["status"], row["compiler_version"], row["positive_prompt"]),
            ("failed", "legacy", "portrait"),
        )

    def test_legacy_not_null_image_bytes_is_rebuilt_and_accepts_null(self) -> None:
        database.DATABASE_PATH.unlink()
        buffer = BytesIO()
        Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
        tiny_png = buffer.getvalue()
        with closing(sqlite3.connect(database.DATABASE_PATH)) as connection:
            connection.execute(
                """
                CREATE TABLE generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, image_bytes BLOB NOT NULL,
                    positive_prompt TEXT NOT NULL, negative_prompt TEXT NOT NULL,
                    timestamp TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                    rating TEXT NOT NULL, style TEXT NOT NULL, high_res INTEGER NOT NULL,
                    model TEXT NOT NULL, filename TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE prompt_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    positive_prompt TEXT NOT NULL, negative_prompt TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    source TEXT NOT NULL, engine TEXT NOT NULL, model TEXT NOT NULL,
                    style TEXT NOT NULL, aspect_ratio TEXT NOT NULL,
                    content_rating TEXT NOT NULL, high_res INTEGER NOT NULL DEFAULT 0,
                    super_res INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'created',
                    generation_id INTEGER REFERENCES generations(id) ON DELETE SET NULL,
                    error_message TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE generation_feedback (
                    generation_id INTEGER PRIMARY KEY REFERENCES generations(id) ON DELETE CASCADE,
                    score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 5),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                INSERT INTO generations (
                    image_bytes, positive_prompt, negative_prompt, timestamp,
                    width, height, rating, style, high_res, model, filename
                ) VALUES (?, 'a portrait', 'blurry', '2026-01-01 00:00:00',
                          512, 512, 'Safe', 'Photoreal', 0, 'legacy-model', 'generation-1.png')
                """,
                (tiny_png,),
            )
            connection.execute(
                """
                INSERT INTO prompt_runs (
                    positive_prompt, negative_prompt, source, engine, model,
                    style, aspect_ratio, content_rating, status, generation_id
                ) VALUES ('a portrait', 'blurry', 'submitted', 'manual',
                          'legacy-model', 'Photoreal', 'Square', 'Safe', 'succeeded', 1)
                """
            )
            connection.execute(
                "INSERT INTO generation_feedback (generation_id, score) VALUES (1, 5)"
            )
            connection.commit()

        database.initialize_database()
        database.initialize_database()

        with database._connection() as connection:
            schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='generations'"
            ).fetchone()["sql"]
            self.assertNotIn("image_bytes BLOB NOT NULL", schema)

            # A previously-orphaned row referencing generation 1 must still
            # resolve after the rebuild, proving foreign keys were not left
            # pointing at a scratch table name.
            connection.execute("PRAGMA foreign_keys=ON")
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            self.assertEqual(violations, [])
            prompt_run = connection.execute(
                "SELECT status, generation_id FROM prompt_runs WHERE id=1"
            ).fetchone()
            self.assertEqual((prompt_run["status"], prompt_run["generation_id"]), ("succeeded", 1))
            feedback = connection.execute(
                "SELECT score FROM generation_feedback WHERE generation_id=1"
            ).fetchone()
            self.assertEqual(feedback["score"], 5)

            # The whole point of the rebuild: new rows can now omit the blob
            # that save_generation() intentionally leaves NULL.
            connection.execute(
                """
                INSERT INTO generations (
                    image_bytes, positive_prompt, negative_prompt, timestamp,
                    width, height, rating, style, high_res, model, filename
                ) VALUES (NULL, 'another portrait', 'blurry', '2026-01-02 00:00:00',
                          512, 512, 'Safe', 'Photoreal', 0, 'legacy-model', 'generation-2.png')
                """
            )


if __name__ == "__main__":
    unittest.main()
