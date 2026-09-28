"""Additive, atomic persistence for ForgeBAT parents and items."""

import sqlite3

import pytest

from backend import database


@pytest.fixture()
def batch_db(tmp_path, monkeypatch):
    path = tmp_path / "batch.db"
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY, email TEXT, display_name TEXT,
                password_hash TEXT, is_admin INTEGER, created_at TEXT
            )
            """
        )
        connection.execute("INSERT INTO users VALUES (1, 'a@b.test', 'A', 'x', 0, 'now')")
    database.initialize_database()
    return path


def _items(count: int) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "request_id": f"child-request-{index}",
            "engine_seed": 100 + index,
            "positive_prompt": f"prompt {index}",
            "negative_prompt": "negative",
        }
        for index in range(count)
    ]


def test_batch_schema_is_additive_indexed_and_idempotent(batch_db) -> None:
    database.initialize_database()
    with sqlite3.connect(batch_db) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        indexes = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )}
    assert {"batches", "batch_items", "generations", "jobs"} <= tables
    assert {
        "idx_batches_account_created", "idx_batches_status",
        "idx_batch_items_batch_status", "idx_batch_items_job",
        "idx_batch_items_generation",
    } <= indexes


def test_parent_and_exact_item_count_are_persisted_atomically(batch_db) -> None:
    batch = database.create_batch_with_items(
        account_id=1,
        request_id="parent-request-1",
        requested_count=4,
        planner_seed=812,
        config={"count": 4, "content_rating": "Safe"},
        plan_items=_items(4),
    )
    items = database.get_batch_items(batch.id)

    assert batch.status == "pending"
    assert batch.requested_count == 4
    assert batch.config["content_rating"] == "Safe"
    assert len(items) == 4
    assert [item.item_index for item in items] == [0, 1, 2, 3]
    assert len({item.request_id for item in items}) == 4
    assert all(item.plan["positive_prompt"].startswith("prompt") for item in items)


def test_bad_count_rolls_back_before_a_parent_exists(batch_db) -> None:
    with pytest.raises(ValueError, match="exactly"):
        database.create_batch_with_items(
            account_id=1,
            request_id="parent-request-bad",
            requested_count=3,
            planner_seed=9,
            config={"count": 3},
            plan_items=_items(2),
        )
    assert database.get_batch_by_request_id("parent-request-bad") is None


def test_duplicate_parent_request_is_idempotent_and_secrets_are_rejected(batch_db) -> None:
    first = database.create_batch_with_items(
        account_id=1, request_id="parent-request-replay", requested_count=2,
        planner_seed=11, config={"count": 2}, plan_items=_items(2),
    )
    replay = database.create_batch_with_items(
        account_id=1, request_id="parent-request-replay", requested_count=2,
        planner_seed=11, config={"count": 2}, plan_items=_items(2),
    )
    assert replay.id == first.id
    assert len(database.get_batch_items(first.id)) == 2

    with pytest.raises(ValueError, match="credentials"):
        database.create_batch_with_items(
            account_id=1, request_id="parent-request-secret", requested_count=1,
            planner_seed=1, config={"cloud_api_key": "secret"}, plan_items=_items(1),
        )
