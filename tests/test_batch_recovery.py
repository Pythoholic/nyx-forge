"""Restart reconciliation resumes ForgeBAT without a browser connection."""

import sqlite3

import pytest

from backend import database


@pytest.fixture()
def recovery_db(tmp_path, monkeypatch):
    path = tmp_path / "recovery.db"
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


def _batch(request_id: str):
    plans = [
        {
            "index": index,
            "request_id": f"{request_id}-child-{index}",
            "engine_seed": 500 + index,
            "positive_prompt": f"prompt {index}",
            "negative_prompt": "negative",
        }
        for index in range(2)
    ]
    return database.create_batch_with_items(
        account_id=1, request_id=request_id, requested_count=2,
        planner_seed=44, config={"batch_seed": 44}, plan_items=plans,
    )


def test_interrupted_running_batch_child_requeues_with_the_same_plan(recovery_db) -> None:
    batch = _batch("restart-parent")
    item = database.get_batch_items(batch.id)[0]
    job_id = database.create_job(
        "batch_generate",
        {**item.plan, "batch_id": batch.id, "batch_item_id": item.id},
        1,
        None,
    )
    database.update_batch_item(item.id, status="running", job_id=job_id)
    database.update_job_progress(job_id, status="running")

    database.recover_stalled_jobs()
    database.recover_nonterminal_batches()

    recovered = database.get_batch_items(batch.id)[0]
    assert database.get_job(job_id).status == "queued"
    assert recovered.status == "queued"
    assert recovered.plan == item.plan
    assert recovered.plan["engine_seed"] == 500
    assert job_id in database.resumable_job_ids()
    assert database.get_batch_items(batch.id)[1].status == "pending"


def test_recovery_repairs_job_insert_to_item_link_crash_window(recovery_db) -> None:
    batch = _batch("crash-window-parent")
    item = database.get_batch_items(batch.id)[0]
    job_id = database.create_job(
        "batch_generate",
        {**item.plan, "batch_id": batch.id, "batch_item_id": item.id},
        1,
        None,
    )

    assert item.job_id is None
    assert database.recover_nonterminal_batches() >= 1
    recovered = database.get_batch_items(batch.id)[0]
    assert recovered.job_id == job_id
    assert recovered.status == "queued"
    assert recovered.request_id == item.request_id
