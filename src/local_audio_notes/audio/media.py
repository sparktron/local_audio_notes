import json
import math
import subprocess
from pathlib import Path
from typing import Any

from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError


def run_media(args: list[str], timeout: float) -> bytes:
    try:
        result = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        raise NotesError("Audio tool failed; check ffmpeg/ffprobe and the input format") from None
    return result.stdout


def inspect_audio(path: Path, settings: Settings) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic = handle.read(64)
    expected: set[str]
    if magic[:4] == b"RIFF" and magic[8:12] == b"WAVE":
        expected = {"wav"}
    elif magic[:4] == b"fLaC":
        expected = {"flac"}
    elif magic[4:8] == b"ftyp":
        expected = {"mov", "mp4", "m4a"}
    elif magic[:3] == b"ID3" or (len(magic) > 1 and magic[0] == 255 and magic[1] & 0xE0 == 0xE0):
        expected = {"mp3"}
    else:
        raise NotesError("Unsupported audio content; expected WAV, M4A, MP3 or FLAC")
    raw = run_media(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-show_entries",
            "format=format_name,duration:stream=codec_type,codec_name",
            "-of",
            "json",
            str(path.resolve()),
        ],
        30,
    )
    try:
        info = json.loads(raw)
        duration = float(info["format"]["duration"])
        formats = set(info["format"]["format_name"].split(","))
        streams = info["streams"]
        if not formats & expected or not any(s["codec_type"] == "audio" for s in streams):
            raise ValueError
        if any(s["codec_type"] not in {"audio", "video"} for s in streams):
            raise ValueError
        if not math.isfinite(duration) or not 0 < duration <= settings.max_audio_seconds:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise NotesError(
            "Invalid audio container or recording exceeds the duration limit"
        ) from None
    return {"duration": duration, "format": sorted(formats)}


def normalize(source: Path, target: Path, settings: Settings) -> None:
    run_media(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-threads",
            str(settings.cpu_threads),
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(source.resolve()),
            "-map",
            "0:a:0",
            "-vn",
            "-map_metadata",
            "-1",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-t",
            str(settings.max_audio_seconds + 1),
            "-f",
            "wav",
            str(target.resolve()),
        ],
        max(60, settings.max_audio_seconds),
    )
    target.chmod(0o600)
    # Probe the decoded stream too: container metadata alone cannot enforce the length limit.
    inspect_audio(target, settings)
