import json
import math
from typing import Any

import httpx

from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError


def local_json(
    settings: Settings, label: str, method: str, endpoint: str, payload: dict[str, Any] | None
) -> Any:
    """One request to the configured loopback LLM server: no proxies, no redirects, bounded."""
    try:
        with (
            httpx.Client(
                trust_env=False, follow_redirects=False, timeout=settings.ollama_timeout
            ) as client,
            client.stream(method, settings.ollama_url + endpoint, json=payload) as response,
        ):
            response.raise_for_status()
            chunks = bytearray()
            for block in response.iter_bytes():
                chunks.extend(block)
                if len(chunks) > 2 * 1024**2:
                    raise NotesError(f"{label} response exceeds limit")
            return json.loads(chunks)
    except (httpx.HTTPError, ValueError):
        raise NotesError(f"Local {label} API unavailable or returned invalid data") from None


def valid_vectors(vectors: Any, count: int) -> bool:
    return (
        isinstance(vectors, list)
        and len(vectors) == count
        and all(isinstance(v, list) and v for v in vectors)
        and len({len(v) for v in vectors}) == 1
        and all(isinstance(x, int | float) and math.isfinite(x) for v in vectors for x in v)
    )
