import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from local_audio_notes.errors import NotesError


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Advisory single-writer protection; not a durable queue or a security boundary."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise NotesError(
                "Another process is writing this storage scope; retry after it finishes"
            ) from None
        yield
    finally:
        os.close(descriptor)
