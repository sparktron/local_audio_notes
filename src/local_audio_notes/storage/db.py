"""Private SQLite store for metadata and processing state. No audio is stored here."""

import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from local_audio_notes.errors import NotesError
from local_audio_notes.storage.files import private_dir
from local_audio_notes.storage.migrations import MIGRATIONS

DB_NAME = "notes.sqlite3"


def now() -> str:
    return datetime.now(UTC).isoformat()


def _private_file(path: Path) -> None:
    # Create with 0600 before SQLite opens it, and never follow a planted symlink.
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError:
        raise NotesError("Database path must be a regular file, not a symlink") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise NotesError("Database must be a regular file owned by the current user")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Write transaction taken up front so concurrent claimers serialize, not deadlock."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def migrate(connection: sqlite3.Connection) -> int:
    with transaction(connection):
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > len(MIGRATIONS):
            raise NotesError("Database schema is newer than this application; refusing to run")
        for number, statements in enumerate(MIGRATIONS[version:], start=version + 1):
            for statement in statements:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {number:d}")
    return len(MIGRATIONS)


def connect(data_dir: Path) -> sqlite3.Connection:
    path = private_dir(data_dir) / DB_NAME
    _private_file(path)
    # Autocommit mode; every multi-statement write uses an explicit transaction.
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    for pragma in (
        "foreign_keys = ON",
        "journal_mode = WAL",
        "synchronous = FULL",
        "trusted_schema = OFF",
        "secure_delete = ON",
    ):
        connection.execute(f"PRAGMA {pragma}")
    migrate(connection)
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.exists() and not sidecar.is_symlink():
            sidecar.chmod(0o600)
    return connection
