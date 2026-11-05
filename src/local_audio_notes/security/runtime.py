"""Defense in depth for Python networking; not a native-code network sandbox."""

import logging
import os
import socket
import sys
from typing import Any

from local_audio_notes.errors import NotesError

_installed = False
_llm_port = 11434


def isolated_network() -> bool:
    """Require a network namespace with no interfaces other than loopback.

    Ordinary 'disconnected Wi-Fi' does not satisfy this check. A namespace with
    no interfaces cannot be reconnected by another process in the host namespace.
    """
    try:
        return sys.platform == "linux" and {n for _, n in socket.if_nameindex()} <= {"lo"}
    except OSError:
        return False


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if event == "socket.getaddrinfo" and (
        args[0] not in ("127.0.0.1", b"127.0.0.1") or args[1] != _llm_port
    ):
        raise NotesError("Outbound network lookup blocked")
    if event in {"socket.connect", "socket.sendto"}:
        sock, address = args[0], args[-1]
        if sock.family == socket.AF_UNIX:
            raise NotesError("Unix socket access blocked during inference")
        if (
            sock.family != socket.AF_INET
            or sock.type & socket.SOCK_STREAM == 0
            or address != ("127.0.0.1", _llm_port)
        ):
            raise NotesError("Outbound network connection blocked")


def enable_runtime(*, enforced: bool = False, llm_port: int = 11434) -> None:
    """Install the guard. `llm_port` is the configured local LLM server's loopback port."""
    global _installed, _llm_port
    if os.geteuid() == 0:
        raise NotesError("Run the application as an unprivileged user")
    if enforced and not isolated_network():
        raise NotesError("Offline enforcement requires a network namespace with only loopback")
    os.umask(0o077)
    # Deliberately overwrite permissive environment values BEFORE importing ML libraries.
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "PYANNOTE_METRICS_ENABLED": "0",
            "OTEL_SDK_DISABLED": "true",
            "FASTAPI_OTEL_AUTO_CONFIGURE": "false",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
        }
    )
    logging.disable(logging.CRITICAL)  # libraries may log transcript text and file paths
    _llm_port = llm_port
    if not _installed:
        sys.addaudithook(_audit)
        _installed = True
