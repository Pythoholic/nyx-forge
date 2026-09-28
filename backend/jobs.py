"""Single-worker durable queue for Forge-bound operations."""

from __future__ import annotations

import logging
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable, Literal

from .database import fail_stalled_running_jobs, finish_job, update_job_progress

LOGGER = logging.getLogger(__name__)


class JobExecutionError(RuntimeError):
    """A sanitized failure safe to expose through the jobs API."""


JobExecutor = Callable[[int, object | None], int | None]


class JobRunner:
    """Run Forge jobs FIFO on one daemon thread so GPU work never overlaps."""

    def __init__(self, executor: JobExecutor) -> None:
        self._executor = executor
        self._queue: Queue[tuple[int, object | None] | None] = Queue()
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._active_job_id: int | None = None
        self._queued_job_ids: set[int] = set()
        self._cancel_requested: set[int] = set()

    def start(self) -> None:
        """Start the sole worker exactly once."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = Thread(
                target=self._run,
                name="aiqg-forge-worker",
                daemon=True,
            )
            self._thread.start()

    def enqueue(self, job_id: int, runtime_payload: object | None = None) -> None:
        """Append one already-persisted job to the FIFO queue."""
        if self._stop.is_set():
            raise RuntimeError("The Forge job worker is stopping.")
        with self._lock:
            self._queued_job_ids.add(job_id)
        self._queue.put((job_id, runtime_payload))

    def stop(self, timeout: float = 0.25) -> None:
        """Stop accepting work without attempting to cancel an active Forge call."""
        self._stop.set()
        self._queue.put(None)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def request_cancel(
        self,
        job_id: int,
        *,
        interrupt_active: Callable[[], None] | None = None,
        allow_active: bool = True,
    ) -> Literal["queued", "running", "not_active", "not_interruptible"]:
        """Cancel a queued job or request cancellation of the active job.

        The runner lock remains held while the optional Forge interrupt is
        sent. This prevents the worker from completing one job and starting
        another between the active-job check and the interrupt request.
        """
        with self._lock:
            if self._active_job_id == job_id:
                if not allow_active:
                    return "not_interruptible"
                if interrupt_active is not None:
                    interrupt_active()
                self._cancel_requested.add(job_id)
                return "running"
            if job_id in self._queued_job_ids:
                self._cancel_requested.add(job_id)
                return "queued"
            return "not_active"

    def cancellation_requested(self, job_id: int) -> bool:
        """Return whether cancellation has been requested for one job."""
        with self._lock:
            return job_id in self._cancel_requested

    def _begin_job(self, job_id: int) -> bool:
        """Claim a queued job unless it was cancelled before execution."""
        with self._lock:
            self._queued_job_ids.discard(job_id)
            if job_id in self._cancel_requested:
                self._cancel_requested.discard(job_id)
                return False
            self._active_job_id = job_id
            return True

    def _complete_job(self, job_id: int) -> bool:
        """Release the active slot and return whether cancellation won."""
        with self._lock:
            cancelled = job_id in self._cancel_requested
            self._cancel_requested.discard(job_id)
            if self._active_job_id == job_id:
                self._active_job_id = None
            return cancelled

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                released = fail_stalled_running_jobs()
                if released:
                    LOGGER.warning("Released stalled Forge jobs: %s", released)
            except Exception:  # noqa: BLE001 - never block the queue on cleanup
                LOGGER.exception("Could not release stalled jobs")
            job_id, runtime_payload = item
            if not self._begin_job(job_id):
                self._queue.task_done()
                continue
            try:
                update_job_progress(
                    job_id,
                    status="running",
                    percent=1,
                    stage="starting",
                )
                generation_id = self._executor(job_id, runtime_payload)
                if self._complete_job(job_id):
                    finish_job(
                        job_id,
                        status="cancelled",
                        error_message="Cancelled by user.",
                    )
                else:
                    finish_job(job_id, status="succeeded", generation_id=generation_id)
            except JobExecutionError as exc:
                cancelled = self._complete_job(job_id)
                LOGGER.warning("Forge job %s %s: %s", job_id, "cancelled" if cancelled else "failed", exc)
                try:
                    finish_job(
                        job_id,
                        status="cancelled" if cancelled else "failed",
                        error_message="Cancelled by user." if cancelled else str(exc),
                    )
                except Exception:
                    LOGGER.exception("Could not persist failure for job %s", job_id)
            except Exception:
                cancelled = self._complete_job(job_id)
                if not cancelled:
                    LOGGER.exception("Unexpected failure in Forge job %s", job_id)
                try:
                    finish_job(
                        job_id,
                        status="cancelled" if cancelled else "failed",
                        error_message=(
                            "Cancelled by user."
                            if cancelled
                            else "The job failed unexpectedly. Check the local server log."
                        ),
                    )
                except Exception:
                    LOGGER.exception("Could not persist failure for job %s", job_id)
            finally:
                # Idempotent cleanup also covers persistence failures after a
                # successful executor return.
                self._complete_job(job_id)
                self._queue.task_done()
