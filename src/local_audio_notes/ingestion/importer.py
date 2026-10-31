import json
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from local_audio_notes.audio.media import inspect_audio
from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError
from local_audio_notes.storage.db import connect, transaction
from local_audio_notes.storage.files import (
    copy_bounded,
    meeting_dir,
    private_dir,
    sha256,
    write_json,
)
from local_audio_notes.storage.locking import exclusive_lock

# meeting.json is the immutable import record: written once, used to adopt a directory whose
# database registration was interrupted. Processing state lives only in SQLite.
MANIFEST_VERSION = 2


def ingest(audio: Path, settings: Settings, transcript: Path | None = None) -> str:
    root = private_dir(settings.data_dir)
    connection = connect(root)
    try:
        with exclusive_lock(root / ".ingestion.lock"):
            adopt(connection, root)
            return _ingest(connection, audio, settings, transcript)
    finally:
        connection.close()


def register(connection: sqlite3.Connection, record: dict[str, Any]) -> None:
    attachment = record.get("attachment") or {}
    connection.execute(
        "INSERT INTO meetings (id, audio_sha256, attachment_kind, attachment_sha256, duration, "
        "container, imported_at, meeting_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record["id"],
            record["sha256"],
            attachment.get("kind"),
            attachment.get("sha256"),
            float(record["duration"]),
            json.dumps(record["format"]),
            record["imported_at"],
            record.get("meeting_date"),
        ),
    )


def adopt(connection: sqlite3.Connection, root: Path) -> dict[str, int]:
    """Register verified meeting directories missing from the database (an import that
    crashed between move and insert, or a Phase 1 data directory). Never deletes anything.

    The caller holds the ingestion lock.
    """
    counts = {"adopted": 0, "unadoptable": 0}
    known = {r[0] for r in connection.execute("SELECT id FROM meetings")}
    for manifest in sorted(root.glob("*/meeting.json")):
        directory = manifest.parent
        if directory.name in known or directory.name.startswith("."):
            continue
        try:
            meeting_dir(root, directory.name)
            record = json.loads(manifest.read_text())
            attachment = record.get("attachment") or {}
            if record["id"] != directory.name:
                raise ValueError
            if sha256(directory / "original.audio") != record["sha256"]:
                raise ValueError
            if attachment and sha256(directory / "attached.transcript") != attachment["sha256"]:
                raise ValueError
            with transaction(connection):
                register(connection, record)
            counts["adopted"] += 1
        except (OSError, KeyError, TypeError, ValueError, NotesError, sqlite3.Error):
            counts["unadoptable"] += 1
    return counts


def _ingest(
    connection: sqlite3.Connection, audio: Path, settings: Settings, transcript: Path | None
) -> str:
    root = settings.data_dir
    identifier = str(uuid4())
    # Only scratch data is removed on failed validation. Caller-owned originals are untouched.
    with tempfile.TemporaryDirectory(dir=root, prefix=".import-") as scratch:
        staging = Path(scratch)
        digest = copy_bounded(audio, staging / "original.audio", settings.max_audio_bytes)
        info = inspect_audio(staging / "original.audio", settings)
        attached = None
        if transcript is not None:
            kind = transcript.suffix.lower()
            if kind not in {".txt", ".srt", ".json"}:
                raise NotesError("Transcript must be UTF-8 TXT, SRT or JSON")
            attached = {
                "kind": kind,
                "sha256": copy_bounded(
                    transcript, staging / "attached.transcript", settings.max_transcript_bytes
                ),
            }
            try:
                text = (staging / "attached.transcript").read_text(encoding="utf-8-sig")
                if "\0" in text or not text.strip():
                    raise ValueError
            except (UnicodeError, ValueError):
                raise NotesError("Transcript must contain nonempty UTF-8 text") from None
        # Idempotent audio + attachment import. No caller filenames enter storage paths.
        existing = connection.execute(
            "SELECT id, attachment_kind FROM meetings "
            "WHERE audio_sha256 = ? AND ifnull(attachment_sha256, '') = ?",
            (digest, attached["sha256"] if attached else ""),
        ).fetchone()
        if existing:
            directory = root / existing["id"]
            if sha256(directory / "original.audio") != digest:
                raise NotesError("Existing recording failed integrity verification")
            if attached and sha256(directory / "attached.transcript") != attached["sha256"]:
                raise NotesError("Existing attached transcript failed integrity verification")
            return existing["id"]
        record = {
            "version": MANIFEST_VERSION,
            "id": identifier,
            "sha256": digest,
            **info,
            "attachment": attached,
            "imported_at": datetime.now(UTC).isoformat(),
        }
        write_json(staging / "meeting.json", record)
        shutil.move(str(staging), str(root / identifier))
    # A crash here leaves a verified directory that the next ingest/recover adopts.
    with transaction(connection):
        register(connection, record)
    return identifier
