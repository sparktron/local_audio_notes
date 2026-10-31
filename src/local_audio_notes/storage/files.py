import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from local_audio_notes.errors import NotesError


def private_dir(path: Path) -> Path:
    if path.is_symlink():
        raise NotesError("Storage directory must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.geteuid():
        raise NotesError("Storage must be owned by the current user")
    path.chmod(0o700)
    return path


def meeting_dir(root: Path, identifier: str) -> Path:
    try:
        if str(UUID(identifier)) != identifier:
            raise ValueError
    except ValueError:
        raise NotesError("Invalid meeting ID") from None
    return private_dir(private_dir(root) / identifier)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    private_dir(path.parent)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def copy_bounded(source: Path, target: Path, limit: int) -> str:
    """Reject symlinks/devices/FIFOs; enforce the limit while copying, not just at stat."""
    try:
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise NotesError("Input must be a readable regular file, not a symlink") from None
    digest = hashlib.sha256()
    created = False
    try:
        with os.fdopen(fd, "rb") as incoming:
            metadata = os.fstat(incoming.fileno())
            if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= limit:
                raise NotesError("Input is empty, non-regular, or exceeds the size limit")
            with target.open("xb") as outgoing:
                created = True
                target.chmod(0o600)
                size = 0
                for block in iter(lambda: incoming.read(1024**2), b""):
                    size += len(block)
                    if size > limit:
                        raise NotesError("Input exceeds the size limit")
                    digest.update(block)
                    outgoing.write(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            after = os.fstat(incoming.fileno())
            if size != metadata.st_size or after.st_mtime_ns != metadata.st_mtime_ns:
                raise NotesError("Input changed during import; retry with a stable file")
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return digest.hexdigest()
