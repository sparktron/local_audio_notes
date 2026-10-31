class NotesError(Exception):
    """A safe diagnostic that does not include recording or transcript content."""


class Cancelled(NotesError):
    """A job observed a cancellation request at a safe checkpoint."""

    def __init__(self) -> None:
        super().__init__("Processing cancelled by request; completed stages are retained")


class Interrupted(BaseException):
    """SIGTERM/service stop: the job returns to the queue for restart recovery.

    BaseException so library code catching Exception cannot swallow a shutdown.
    """
