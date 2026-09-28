"""ForgeBAT feeds the sole durable runner gradually and fairly."""

from types import SimpleNamespace
import sqlite3

import pytest

from backend import database, main
from backend.batch_coordinator import BatchCoordinator


class RecordingRunner:
    def __init__(self) -> None:
        self.enqueued: list[int] = []

    def enqueue(self, job_id: int, runtime_payload=None) -> None:
        self.enqueued.append(job_id)


@pytest.fixture()
def coordinator_db(tmp_path, monkeypatch):
    path = tmp_path / "coordinator.db"
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY, email TEXT, display_name TEXT,
                password_hash TEXT, is_admin INTEGER, created_at TEXT,
                max_active_jobs INTEGER NOT NULL DEFAULT 50
            )
            """
        )
        connection.execute("INSERT INTO users VALUES (1, 'a@b.test', 'A', 'x', 0, 'now', 50)")
    database.initialize_database()
    return path


def _plan_items(count: int) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "request_id": f"batch-child-{index:04d}",
            "engine_seed": index + 100,
            "model": "Forge default",
            "style": "Photoreal",
            "content_rating": "Safe",
            "quality_mode": "normal",
            "aspect_ratio": "Portrait (832x1216)",
            "orientation": "front",
            "positive_prompt": f"adult woman prompt {index}",
            "negative_prompt": "child, teen",
        }
        for index in range(count)
    ]


def test_large_batch_never_prepopulates_the_runner_queue(coordinator_db) -> None:
    batch = database.create_batch_with_items(
        account_id=1, request_id="large-parent-request", requested_count=500,
        planner_seed=5, config={"count": 500}, plan_items=_plan_items(500),
    )
    runner = RecordingRunner()
    coordinator = BatchCoordinator(runner)  # type: ignore[arg-type]

    first_job = coordinator.promote_once()
    assert first_job is not None
    assert len(runner.enqueued) == 1
    assert CounterStatus(batch.id) == {"pending": 499, "queued": 1}

    # A queued batch child does not permit another queued child. Once it is
    # actually running, exactly one successor may sit behind it.
    assert coordinator.promote_once() is None
    database.update_job_progress(first_job, status="running")
    second_job = coordinator.promote_once()
    assert second_job is not None
    assert len(runner.enqueued) == 2
    assert CounterStatus(batch.id) == {"pending": 498, "queued": 1, "running": 1}
    assert coordinator.promote_once() is None


def CounterStatus(batch_id: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in database.get_batch_items(batch_id):
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts


def test_batch_generate_dispatches_through_standard_generate_path(coordinator_db, monkeypatch) -> None:
    payload = {
        "prompt": "34-year-old adult woman",
        "negative_prompt": "child, teen",
        "model": "Forge default",
        "style": "Photoreal",
        "content_rating": "Safe",
        "quality_mode": "normal",
        "aspect_ratio": "Portrait (832x1216)",
        "prompt_engine": "engine",
        "request_id": "batch-child-dispatch",
        "seed": 7781,
    }
    job_id = database.create_job("batch_generate", payload, 1, None)
    captured = {}

    def fake_generate(body, *, job_id=None):
        captured["body"] = body
        captured["job_id"] = job_id
        return SimpleNamespace(id=42)

    monkeypatch.setattr(main, "generate", fake_generate)
    assert main._execute_job(job_id, None) == 42
    assert captured["job_id"] == job_id
    assert captured["body"].seed == 7781
    assert captured["body"].prompt_engine == "engine"
