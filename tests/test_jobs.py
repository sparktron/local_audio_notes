import contextlib
import json
import shutil
import sqlite3
import stat
import subprocess
import sys
import textwrap

import pytest

import local_audio_notes.pipeline as pipeline
from local_audio_notes.errors import Interrupted, NotesError
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.jobs import queue
from local_audio_notes.storage import db
from local_audio_notes.storage.locking import try_lock
from local_audio_notes.storage.migrations import MIGRATIONS


@pytest.fixture
def connection(settings):
    with contextlib.closing(db.connect(settings.data_dir)) as value:
        yield value


def test_database_private_migrated_and_refuses_downgrade(settings, connection):
    path = settings.data_dir / db.DB_NAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert connection.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db.migrate(connection) == len(MIGRATIONS)  # idempotent
    connection.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")
    with pytest.raises(NotesError, match="newer"):
        db.migrate(connection)


def test_database_symlink_rejected(settings, tmp_path):
    settings.data_dir.mkdir(mode=0o700)
    (settings.data_dir / db.DB_NAME).symlink_to(tmp_path / "elsewhere")
    with pytest.raises(NotesError, match="symlink"):
        db.connect(settings.data_dir)
    assert not (tmp_path / "elsewhere").exists()


def test_schema_rejects_invalid_rows(connection):
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO meetings (id, audio_sha256, duration, container, imported_at) "
            "VALUES ('short', 'x', -1, '[]', 'now')"
        )


def test_ingest_registers_and_is_idempotent(audio, settings, connection):
    identifier = ingest(audio, settings)
    assert ingest(audio, settings) == identifier
    rows = connection.execute("SELECT id FROM meetings").fetchall()
    assert [r[0] for r in rows] == [identifier]


def test_enqueue_idempotent_and_conflicting_job_rejected(audio, settings, connection):
    identifier = ingest(audio, settings)
    first = queue.enqueue(connection, settings, identifier)
    assert queue.enqueue(connection, settings, identifier) == first
    with pytest.raises(NotesError, match="different job"):
        queue.enqueue(connection, settings, identifier, kind="notes")
    with pytest.raises(NotesError, match="Invalid meeting ID"):
        queue.enqueue(connection, settings, "../escape")


def test_cancel_queued_and_unknown(audio, settings, connection):
    identifier = ingest(audio, settings)
    job_id = queue.enqueue(connection, settings, identifier)
    assert queue.cancel(connection, job_id) == "cancelled"
    assert queue.get_job(connection, job_id)["state"] == "cancelled"
    with pytest.raises(NotesError, match="not active"):
        queue.cancel(connection, job_id)
    with pytest.raises(NotesError, match="not found"):
        queue.cancel(connection, 999)
    assert queue.work(settings, connection) == []


def test_cancel_running_job_stops_at_checkpoint_and_resumes(audio, settings, mocked):
    calls, ollama = mocked
    identifier = ingest(audio, settings)
    folder = settings.data_dir / identifier
    with contextlib.closing(db.connect(settings.data_dir)) as connection:
        job_id = queue.enqueue(connection, settings, identifier)
        original = ollama.generate

        def cancel_during_llm(*args):
            queue.cancel(connection, job_id)
            return original(*args)

        ollama.generate = cancel_during_llm
        assert queue.work(settings, connection) == [(job_id, "cancelled")]
        job = queue.get_job(connection, job_id)
        last = queue.describe(connection, identifier)["stages"][-1]
        assert (job["error_stage"], job["error_class"]) == ("consolidating", "Cancelled")
        assert (last["stage"], last["state"]) == ("consolidating", "cancelled")
        assert (folder / "extraction.json").exists()
        assert not (folder / "evidence.json").exists() and not (folder / "notes.md").exists()
        ollama.generate = original
        sent = ollama.extraction_calls
        resumed = queue.enqueue(connection, settings, identifier)
        assert queue.work(settings, connection) == [(resumed, "succeeded")]
    assert calls.transcribe == 1 and ollama.extraction_calls == sent
    assert (folder / "notes.md").exists()


def test_recovery_requeues_dead_worker_then_fails_after_max_attempts(audio, settings, connection):
    identifier = ingest(audio, settings)
    job_id = queue.enqueue(connection, settings, identifier)
    for attempt in range(1, settings.job_max_attempts + 1):
        # Simulate a worker killed mid-stage: running row, no lock holder.
        connection.execute(
            "UPDATE jobs SET state = 'running', attempts = ? WHERE id = ?", (attempt, job_id)
        )
        connection.execute(
            "INSERT INTO stage_runs (job_id, stage, state, started_at) "
            "VALUES (?, 'transcribing', 'running', 'then')",
            (job_id,),
        )
        counts = queue.recover(connection, settings)
        expected = "queued" if attempt < settings.job_max_attempts else "failed"
        assert queue.get_job(connection, job_id)["state"] == expected
        assert counts["requeued" if expected == "queued" else "failed"] == 1
    states = {r[0] for r in connection.execute("SELECT state FROM stage_runs")}
    assert states == {"interrupted"}
    assert queue.get_job(connection, job_id)["error_message"].endswith("retry limit reached")


def test_recovery_leaves_live_worker_alone(audio, settings, connection):
    identifier = ingest(audio, settings)
    job_id = queue.enqueue(connection, settings, identifier)
    connection.execute("UPDATE jobs SET state = 'running', attempts = 1 WHERE id = ?", (job_id,))
    with try_lock(queue.lock_path(settings, identifier)) as acquired:
        assert acquired
        assert queue.recover(connection, settings)["alive"] == 1
    assert queue.get_job(connection, job_id)["state"] == "running"


def test_killed_worker_process_is_recovered(audio, settings, connection, tmp_path):
    """A real process holding the lock is SIGKILLed; the kernel releases the lock."""
    identifier = ingest(audio, settings)
    job_id = queue.enqueue(connection, settings, identifier)
    connection.execute("UPDATE jobs SET state = 'running', attempts = 1 WHERE id = ?", (job_id,))
    script = textwrap.dedent(
        f"""
        import sys, time
        from pathlib import Path
        from local_audio_notes.storage.locking import try_lock
        with try_lock(Path({str(queue.lock_path(settings, identifier))!r})) as ok:
            print(ok, flush=True)
            time.sleep(60)
        """
    )
    worker = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert worker.stdout.readline().strip() == "True"
        assert queue.recover(connection, settings)["alive"] == 1
    finally:
        worker.kill()
        worker.wait()
    assert queue.recover(connection, settings)["requeued"] == 1
    assert queue.get_job(connection, job_id)["state"] == "queued"


def test_sigterm_requeues_and_keyboard_interrupt_cancels(audio, settings, connection, monkeypatch):
    identifier = ingest(audio, settings)

    def stop(*args):
        raise Interrupted

    monkeypatch.setattr(pipeline, "run_job", stop)
    job_id = queue.enqueue(connection, settings, identifier)
    with pytest.raises(Interrupted):
        queue.work(settings, connection)
    job = queue.get_job(connection, job_id)
    assert (job["state"], job["error_class"]) == ("queued", "Interrupted")

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "run_job", interrupt)
    with pytest.raises(KeyboardInterrupt):
        queue.work(settings, connection)
    assert queue.get_job(connection, job_id)["state"] == "cancelled"


@pytest.mark.parametrize(
    "error, error_class, message",
    [
        (RuntimeError("CUDA out of memory. PRIVATE"), "GpuOutOfMemory", "GPU out of memory"),
        (ValueError("transcript excerpt PRIVATE"), "ValueError", None),
        (NotesError("Safe message"), "NotesError", "Safe message"),
    ],
)
def test_failure_recorded_without_library_messages(
    audio, settings, connection, monkeypatch, error, error_class, message
):
    identifier = ingest(audio, settings)

    def fail(*args):
        raise error

    monkeypatch.setattr(pipeline, "run_job", fail)
    job_id = queue.enqueue(connection, settings, identifier)
    assert queue.work(settings, connection) == [(job_id, "failed")]
    job = queue.get_job(connection, job_id)
    assert job["error_class"] == error_class
    assert "PRIVATE" not in json.dumps(job)
    assert (job["error_message"] or "").startswith(message or "")
    # Failed jobs are not retried automatically; a new job can be queued.
    assert queue.work(settings, connection) == []
    assert queue.enqueue(connection, settings, identifier) != job_id


def test_worker_runs_queue_in_order(audio, settings, connection, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(pipeline, "run_job", lambda s, c, job, check: seen.append(job["id"]))
    other = tmp_path / "other.wav"
    shutil.copy(audio, other)
    with open(other, "ab") as handle:
        handle.write(b"\0\0")  # different bytes, still valid audio
    ids = [ingest(path, settings) for path in (audio, other)]
    jobs = [queue.enqueue(connection, settings, i) for i in ids]
    assert queue.work(settings, connection, once=True) == [(jobs[0], "succeeded")]
    assert queue.work(settings, connection) == [(jobs[1], "succeeded")]
    assert seen == jobs


def test_orphan_import_directory_adopted(audio, settings, connection):
    identifier = ingest(audio, settings)
    # Simulate a crash between moving the import into place and registering it.
    connection.execute("DELETE FROM meetings")
    assert queue.recover(connection, settings)["adopted"] == 1
    assert ingest(audio, settings) == identifier
    (settings.data_dir / identifier / "original.audio").write_bytes(b"tampered")
    connection.execute("DELETE FROM meetings")
    assert queue.recover(connection, settings)["unadoptable"] == 1


def test_phase1_meeting_directory_adopted(audio, settings, connection):
    identifier = ingest(audio, settings)
    folder = settings.data_dir / identifier
    record = json.loads((folder / "meeting.json").read_text())
    legacy = {**record, "version": 1, "state": "complete", "meeting_date": None}
    (folder / "meeting.json").write_text(json.dumps(legacy))
    connection.execute("DELETE FROM meetings")
    assert queue.recover(connection, settings)["adopted"] == 1


def test_status_for_unknown_meeting(settings, connection):
    with pytest.raises(NotesError, match="not found"):
        queue.describe(connection, "00000000-0000-4000-8000-000000000000")


def test_worker_reloads_settings_between_jobs(audio, settings, connection, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        pipeline, "run_job", lambda current, c, job, check: seen.append(current.batch_size)
    )
    other = tmp_path / "other.wav"
    shutil.copy(audio, other)
    with open(other, "ab") as handle:
        handle.write(b"\0\0")
    for path in (audio, other):
        queue.enqueue(connection, settings, ingest(path, settings))
    sizes = iter([8, 2, 2])
    reload = lambda: settings.model_copy(update={"batch_size": next(sizes)})  # noqa: E731
    queue.work(settings, connection, reload=reload)
    assert seen == [8, 2]
