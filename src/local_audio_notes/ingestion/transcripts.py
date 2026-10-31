"""Explicit supported interchange schemas, not guesses at WhisperBoard export formats."""

import json
import re
from pathlib import Path

from pydantic import ValidationError

from local_audio_notes.errors import NotesError
from local_audio_notes.schemas import Segment, Transcript


def _seconds(value: str) -> float:
    match = re.fullmatch(r"(\d{2,}):([0-5]\d):([0-5]\d)[,.](\d{3})", value)
    if match is None:
        raise NotesError("SRT timestamps must use HH:MM:SS,mmm")
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def parse_transcript(path: Path, kind: str, duration: float, language: str) -> Transcript:
    try:
        text = path.read_text(encoding="utf-8-sig")
        rows = []
        if kind == ".json":
            value = json.loads(text)
            # Documented JSON schema: seconds (numbers), no ms or string coercion.
            if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
                raise ValueError
            for item in value["segments"]:
                if not isinstance(item, dict) or set(item) - {"start", "end", "text"}:
                    raise ValueError
                if any(type(item.get(k)) not in (int, float) for k in ("start", "end")):
                    raise ValueError
                rows.append(item)
        elif kind == ".srt":
            blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
            for block in blocks:
                lines = block.splitlines()
                if len(lines) < 3 or not lines[0].isdigit():
                    raise ValueError
                times = lines[1].split(" --> ")
                if len(times) != 2:
                    raise ValueError
                rows.append(
                    {
                        "start": _seconds(times[0]),
                        "end": _seconds(times[1]),
                        "text": "\n".join(lines[2:]),
                    }
                )
        else:
            raise NotesError(
                "Untimed TXT cannot be force-aligned; use compare mode or timed SRT/JSON"
            )
        if not rows:
            raise ValueError
        segments = []
        for i, row in enumerate(rows):
            segment = Segment(id=f"s{i:06d}", **row)
            if segment.end > duration + 0.05 or segment.end - segment.start > 300:
                raise ValueError
            segments.append(segment)
        return Transcript(language=language, segments=segments)
    except (UnicodeError, ValueError, TypeError, KeyError, ValidationError):
        raise NotesError(
            "Invalid transcript schema, timestamps, ordering or segment length"
        ) from None
