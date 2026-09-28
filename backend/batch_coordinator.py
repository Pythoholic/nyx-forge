"""Fair, gradual promotion of persisted ForgeBAT items into JobRunner."""

from __future__ import annotations

import logging
from threading import Event, Lock, Thread

from .database import (
    JobQueueLimitError,
    account_job_limit,
    active_batch_child_counts,
    create_job,
    finish_job,
    get_batch,
    next_pending_batch_item,
    reconcile_batch_jobs,
    update_batch_item,
)
from .jobs import JobRunner


LOGGER = logging.getLogger(__name__)


class BatchCoordinator:
    """Keep at most one running and one queued batch child globally."""

    def __init__(self, runner: JobRunner, *, poll_seconds: float = 0.5) -> None:
        self.runner = runner
        self.poll_seconds = poll_seconds
        self._wake = Event()
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._last_batch_id: int | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = Thread(target=self._run, name="aiqg-batch-coordinator", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 0.5) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def wake(self) -> None:
        self._wake.set()

    def promote_once(self) -> int | None:
        """Reconcile and enqueue at most one child, preserving FIFO room."""
        reconcile_batch_jobs()
        running, queued = active_batch_child_counts()
        if queued >= 1 or running > 1:
            return None
        item = next_pending_batch_item(after_batch_id=self._last_batch_id)
        if item is None:
            return None
        parent = get_batch(item.batch_id)
        if parent is None:
            return None
        payload = {
            **item.plan,
            "prompt": item.plan["positive_prompt"],
            "negative_prompt": item.plan.get("negative_prompt", ""),
            "seed": item.plan["engine_seed"],
            "prompt_engine": "engine",
            "batch_id": parent.id,
            "batch_item_id": item.id,
        }
        try:
            job_id = create_job(
                "batch_generate",
                payload,
                parent.account_id,
                None,
                max_outstanding=account_job_limit(parent.account_id),
            )
        except JobQueueLimitError:
            return None
        update_batch_item(item.id, status="queued", job_id=job_id)
        try:
            self.runner.enqueue(job_id)
        except RuntimeError:
            finish_job(job_id, status="failed", error_message="The local job worker is unavailable.")
            update_batch_item(
                item.id, status="failed", error="The local job worker is unavailable.",
            )
            return None
        self._last_batch_id = item.batch_id
        return job_id

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.promote_once()
            except Exception:  # noqa: BLE001 - a coordinator must retry safely
                LOGGER.exception("Could not promote the next ForgeBAT item")
            self._wake.wait(self.poll_seconds)
            self._wake.clear()
