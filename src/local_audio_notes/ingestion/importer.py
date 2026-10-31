import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from local_audio_notes.audio.media import inspect_audio
from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError
from local_audio_notes.storage.files import copy_bounded, private_dir, sha256, write_json
from local_audio_notes.storage.locking import exclusive_lock


def ingest(audio: Path, settings: Settings, transcript: Path | None = None) -> str:
    root = private_dir(settings.data_dir)
    with exclusive_lock(root / ".ingestion.lock"):
        return _ingest(audio, settings, transcript)


def _ingest(audio: Path, settings: Settings, transcript: Path | None = None) -> str:
    root = private_dir(settings.data_dir)
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
        for metadata in root.glob("*/meeting.json"):
            if metadata.parent.name.startswith("."):
                continue
            value = json.loads(metadata.read_text())
            if value["sha256"] == digest and value.get("attachment") == attached:
                if sha256(metadata.parent / "original.audio") != digest:
                    raise NotesError("Existing recording failed integrity verification")
                if (
                    attached
                    and sha256(metadata.parent / "attached.transcript") != attached["sha256"]
                ):
                    raise NotesError("Existing attached transcript failed integrity verification")
                return value["id"]
        write_json(
            staging / "meeting.json",
            {
                "version": 1,
                "id": identifier,
                "sha256": digest,
                **info,
                "attachment": attached,
                "imported_at": datetime.now(UTC).isoformat(),
                "meeting_date": None,
                "state": "imported",
            },
        )
        shutil.move(str(staging), str(root / identifier))
    return identifier
