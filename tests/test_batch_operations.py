"""Batch cancel/retry operations preserve isolation and reproducibility."""

import sqlite3

import pytest

from backend import database
from backend.jobs import JobRunner


@pytest.fixture()
def operations_db(tmp_path, monkeypatch):
    path = tmp_path / "operations.db"
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


def _create(request_id: str, count: int = 3):
    plans = [
        {
            "index": index,
            "request_id": f"{request_id}-child-{index}",
            "engine_seed": 901 + index,
            "positive_prompt": f"planned prompt {index}",
            "negative_prompt": "planned negative",
        }
        for index in range(count)
    ]
    return database.create_batch_with_items(
        account_id=1, request_id=request_id, requested_count=count,
        planner_seed=77, config={"count": count, "batch_seed": 77}, plan_items=plans,
    )


def test_cancel_marks_only_the_target_batch_pending_items(operations_db) -> None:
    target = _create("cancel-target")
    unrelated = _create("cancel-unrelated")

    assert database.cancel_pending_batch_items(target.id) == 3
    assert {item.status for item in database.get_batch_items(target.id)} == {"cancelled"}
    assert {item.status for item in database.get_batch_items(unrelated.id)} == {"pending"}


def test_runner_cancellation_for_one_batch_child_leaves_unrelated_job_alone(operations_db) -> None:
    target_job = database.create_job("batch_generate", {"batch_id": 1}, 1, None)
    unrelated_job = database.create_job("generate", {"prompt": "unrelated"}, 1, None)
    runner = JobRunner(lambda *_: None)
    runner.enqueue(target_job)
    runner.enqueue(unrelated_job)

    assert runner.request_cancel(target_job) == "queued"
    assert runner.cancellation_requested(target_job)
    assert not runner.cancellation_requested(unrelated_job)


def test_retry_failed_preserves_item_plan_request_id_and_seed(operations_db) -> None:
    batch = _create("retry-parent", count=1)
    before = database.get_batch_items(batch.id)[0]
    database.update_batch_item(before.id, status="failed", error="temporary failure")

    assert database.retry_failed_batch_items(batch.id) == 1
    after = database.get_batch_items(batch.id)[0]
    assert after.status == "pending"
    assert after.error is None
    assert after.request_id == before.request_id
    assert after.plan == before.plan
    assert after.plan["engine_seed"] == 901
    assert database.get_batch(batch.id).planner_seed == 77
