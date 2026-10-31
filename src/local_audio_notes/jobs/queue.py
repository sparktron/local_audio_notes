"""Durable SQLite job queue with flock-based liveness for restart recovery.

A worker takes the meeting's processing lock before marking a job running and releases
it only after recording the outcome. The kernel releases that lock when the process dies
(crash, SIGKILL, power loss), so recovery treats a running job whose lock it can acquire
as abandoned: it is requeued until max_attempts, then failed. Failed jobs are never
retried automatically; a new job resumes from the verified stage checkpoints.
"""

import contextlib
import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from local_audio_notes import pipeline
from local_audio_notes.config import Settings
from local_audio_notes.errors import Cancelled, Interrupted, NotesError
from local_audio_notes.ingestion.importer import adopt
from local_audio_notes.storage.db import connect, now, transaction
from local_audio_notes.storage.files import meeting_dir
from local_audio_notes.storage.locking import try_lock
from local_audio_notes.transcription.whisperx_backend import is_oom

ACTIVE = ("queued", "running")
LOCK = ".processing.lock"


def get_job(connection: sqlite3.Connection, job_id: int) -> dict[str, Any]:
    row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise NotesError("Job not found")
    return dict(row)


def enqueue(
    connection: sqlite3.Connection,
    settings: Settings,
    meeting_id: str,
    kind: str = "full",
    mode: str = "transcribe",
    force: bool = False,
) -> int:
    """Idempotent: an identical active job is returned instead of queuing a duplicate."""
    meeting_dir(settings.data_dir, meeting_id)  # validates the identifier shape
    pipeline.get_meeting(connection, meeting_id)
    with transaction(connection):
        active = connection.execute(
            "SELECT id, kind, mode, force FROM jobs WHERE meeting_id = ? AND state IN (?, ?)",
            (meeting_id, *ACTIVE),
        ).fetchone()
        if active:
            if (active["kind"], active["mode"], bool(active["force"])) == (kind, mode, force):
                return active["id"]
            raise NotesError("A different job is already active for this meeting; cancel it first")
        return connection.execute(
            "INSERT INTO jobs (meeting_id, kind, mode, force, state, max_attempts, created_at) "
            "VALUES (?, ?, ?, ?, 'queued', ?, ?)",
            (meeting_id, kind, mode, int(force), settings.job_max_attempts, now()),
        ).lastrowid


def cancel(connection: sqlite3.Connection, job_id: int) -> str:
    """Queued jobs cancel immediately; running jobs stop at their next safe checkpoint."""
    with transaction(connection):
        job = get_job(connection, job_id)
        if job["state"] == "queued":
            connection.execute(
                "UPDATE jobs SET state = 'cancelled', finished_at = ?, error_class = 'Cancelled' "
                "WHERE id = ?",
                (now(), job_id),
            )
            return "cancelled"
        if job["state"] == "running":
            connection.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
            return "cancel requested"
    raise NotesError("Job is not active")


def _finish(
    connection: sqlite3.Connection,
    job_id: int,
    state: str,
    error_class: str | None = None,
    message: str | None = None,
) -> None:
    connection.execute(
        "UPDATE jobs SET state = ?, finished_at = CASE WHEN ? = 'queued' THEN NULL ELSE ? END, "
        "error_stage = CASE WHEN ? IS NULL THEN NULL ELSE stage END, error_class = ?, "
        "error_message = ?, worker = NULL WHERE id = ?",
        (state, state, now(), error_class, error_class, message, job_id),
    )


def execute(settings: Settings, connection: sqlite3.Connection, job: dict[str, Any]) -> str:
    """Run a claimed job and durably record its outcome. Messages are NotesError text only,
    which is written to be free of recording content; other exceptions record a class name."""

    def check() -> None:
        row = connection.execute(
            "SELECT cancel_requested FROM jobs WHERE id = ?", (job["id"],)
        ).fetchone()
        if row and row[0]:
            raise Cancelled

    try:
        pipeline.run_job(settings, connection, job, check)
    except Cancelled as error:
        _finish(connection, job["id"], "cancelled", "Cancelled", str(error))
        return "cancelled"
    except Interrupted:
        retry = job["attempts"] < job["max_attempts"]
        _finish(
            connection,
            job["id"],
            "queued" if retry else "failed",
            "Interrupted",
            "Stopped by a termination signal" + ("" if retry else "; retry limit reached"),
        )
        raise
    except KeyboardInterrupt:
        _finish(connection, job["id"], "cancelled", "KeyboardInterrupt", "Interrupted by user")
        raise
    except NotesError as error:
        _finish(connection, job["id"], "failed", type(error).__name__, str(error))
        return "failed"
    except Exception as error:
        if is_oom(error):
            _finish(
                connection,
                job["id"],
                "failed",
                "GpuOutOfMemory",
                "GPU out of memory; lower batch_size or compute_type, or free VRAM",
            )
        else:
            # Native library messages may contain transcript text or paths: class only.
            _finish(connection, job["id"], "failed", type(error).__name__)
        return "failed"
    _finish(connection, job["id"], "succeeded")
    return "succeeded"


@contextmanager
def claim(
    connection: sqlite3.Connection, settings: Settings, worker: str, job_id: int | None = None
) -> Iterator[dict[str, Any] | None]:
    """Yield the next queued job (or the given one) with its meeting lock held, else None."""
    query = "SELECT id, meeting_id FROM jobs WHERE state = 'queued'"
    rows = connection.execute(
        query + (" AND id = ?" if job_id is not None else "") + " ORDER BY id",
        (job_id,) if job_id is not None else (),
    ).fetchall()
    for row in rows:
        with try_lock(meeting_dir(settings.data_dir, row["meeting_id"]) / LOCK) as acquired:
            if not acquired:
                continue
            with transaction(connection):
                claimed = connection.execute(
                    "UPDATE jobs SET state = 'running', attempts = attempts + 1, worker = ?, "
                    "started_at = ?, stage = NULL, error_stage = NULL, error_class = NULL, "
                    "error_message = NULL WHERE id = ? AND state = 'queued'",
                    (worker, now(), row["id"]),
                ).rowcount
            if claimed:
                yield get_job(connection, row["id"])
                return
    yield None


def adopt_orphans(connection: sqlite3.Connection, settings: Settings) -> dict[str, int]:
    with try_lock(settings.data_dir / ".ingestion.lock") as acquired:
        if not acquired:  # an import is in progress; its directory may not be registered yet
            return {"adopted": 0, "unadoptable": 0}
        return adopt(connection, settings.data_dir)


def recover(connection: sqlite3.Connection, settings: Settings) -> dict[str, int]:
    counts = {"requeued": 0, "failed": 0, "alive": 0}
    running = connection.execute("SELECT id, meeting_id FROM jobs WHERE state = 'running'")
    for row in running.fetchall():
        with try_lock(meeting_dir(settings.data_dir, row["meeting_id"]) / LOCK) as acquired:
            if not acquired:
                counts["alive"] += 1
                continue
            with transaction(connection):
                job = get_job(connection, row["id"])
                if job["state"] != "running":  # finished between the query and the lock
                    continue
                connection.execute(
                    "UPDATE stage_runs SET state = 'interrupted', finished_at = ?, "
                    "error_class = 'Interrupted' WHERE job_id = ? AND state = 'running'",
                    (now(), row["id"]),
                )
                retry = job["attempts"] < job["max_attempts"]
                _finish(
                    connection,
                    row["id"],
                    "queued" if retry else "failed",
                    "Interrupted",
                    "Worker stopped unexpectedly" + ("" if retry else "; retry limit reached"),
                )
            counts["requeued" if retry else "failed"] += 1
    return {**counts, **adopt_orphans(connection, settings)}


def work(
    settings: Settings,
    connection: sqlite3.Connection,
    *,
    job_id: int | None = None,
    once: bool = False,
    on_claim: Callable[[dict[str, Any]], None] = lambda job: None,
) -> list[tuple[int, str]]:
    """Recover abandoned jobs, then run queued jobs in order. Returns (job id, outcome)."""
    recover(connection, settings)
    worker = f"pid:{os.getpid()}"
    results = []
    while True:
        with claim(connection, settings, worker, job_id) as job:
            if job is None:
                break
            on_claim(job)
            results.append((job["id"], execute(settings, connection, job)))
        if once or job_id is not None:
            break
    return results


def job_error(job: dict[str, Any]) -> str:
    detail = job["error_message"] or job["error_class"] or "unknown error"
    return f"Job {job['id']} {job['state']} at {job['error_stage'] or 'preflight'}: {detail}"


def process(
    settings: Settings,
    meeting_id: str,
    mode: str = "transcribe",
    notes_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Queue one meeting and run it in this process; raise NotesError unless it succeeds."""
    with contextlib.closing(connect(settings.data_dir)) as connection:
        kind = "notes" if notes_only else "full"
        job_id = enqueue(connection, settings, meeting_id, kind, mode, force)
        results = work(settings, connection, job_id=job_id)
        job = get_job(connection, job_id)
    if job["state"] == "succeeded":
        return job
    if not results:
        raise NotesError(f"Job {job_id} is {job['state']} in another process")
    raise NotesError(job_error(job))


def describe(connection: sqlite3.Connection, meeting_id: str) -> dict[str, Any]:
    """Non-sensitive processing state: no transcript text, titles or speaker names."""
    meeting = pipeline.get_meeting(connection, meeting_id)
    job = connection.execute(
        "SELECT * FROM jobs WHERE meeting_id = ? ORDER BY id DESC LIMIT 1", (meeting_id,)
    ).fetchone()
    status: dict[str, Any] = {
        "id": meeting["id"],
        "duration_seconds": meeting["duration"],
        "imported_at": meeting["imported_at"],
        "state": job["state"] if job else "imported",
        "checkpoints": [
            r[0]
            for r in connection.execute(
                "SELECT stage FROM checkpoints WHERE meeting_id = ?", (meeting_id,)
            )
        ],
    }
    if job:
        status["job"] = {
            k: job[k]
            for k in (
                "id",
                "kind",
                "mode",
                "stage",
                "attempts",
                "max_attempts",
                "cancel_requested",
                "error_stage",
                "error_class",
                "error_message",
            )
        }
        status["stages"] = [
            {
                "stage": r["stage"],
                "state": r["state"],
                "seconds": r["seconds"],
                **({"detail": json.loads(r["detail"])} if r["detail"] else {}),
            }
            for r in connection.execute(
                "SELECT * FROM stage_runs WHERE job_id = ? ORDER BY id", (job["id"],)
            )
        ]
    return status


def lock_path(settings: Settings, meeting_id: str) -> Path:
    return meeting_dir(settings.data_dir, meeting_id) / LOCK
