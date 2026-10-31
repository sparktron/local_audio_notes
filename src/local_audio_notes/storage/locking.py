import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from local_audio_notes.errors import NotesError


@contextmanager
def try_lock(path: Path) -> Iterator[bool]:
    """Non-blocking advisory lock; yields whether it was acquired.

    The kernel drops the lock when its holder exits (including SIGKILL and power loss),
    so holding it is the liveness signal restart recovery relies on. Not a security boundary.
    """
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            yield True
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    with try_lock(path) as acquired:
        if not acquired:
            raise NotesError(
                "Another process is writing this storage scope; retry after it finishes"
            )
        yield
