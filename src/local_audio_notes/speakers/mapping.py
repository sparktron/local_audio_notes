"""Per-meeting, user-confirmed speaker names and segment corrections.

Nothing here infers identity. Names are only what the user explicitly confirms, scoped to
one meeting's attributed transcript. Cross-meeting voice identification is not implemented;
a future consent-based enrollment would add its own table and never write to these.
"""

import re
import sqlite3
from dataclasses import dataclass, field

from local_audio_notes.errors import NotesError
from local_audio_notes.extraction.grounding import normalize
from local_audio_notes.schemas import Transcript
from local_audio_notes.storage.db import now, transaction

LABEL = re.compile(r"Speaker [1-9]\d*")
UNRESOLVED = ("Unknown", "Uncertain")


@dataclass
class SpeakerView:
    names: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, str] = field(default_factory=dict)
    stale_entries: int = 0

    def speaker(self, segment_id: str, original: str) -> str:
        return self.overrides.get(segment_id, original)

    def display(self, label: str) -> str:
        name = self.names.get(label)
        return f"{name} [{label}]" if name else label


def transcript_labels(transcript: Transcript) -> list[str]:
    """Speaker labels in order of first appearance, from segments and their words.

    A speaker heard only inside mixed segments has no segment-level label (those segments
    are "Uncertain") but keeps word-level labels; they must still be nameable.
    """
    found = (
        label
        for segment in transcript.segments
        for label in (segment.speaker, *(w.speaker for w in segment.words))
        if LABEL.fullmatch(label)
    )
    return list(dict.fromkeys(found))


def labels(connection: sqlite3.Connection, meeting_id: str) -> set[str]:
    rows = connection.execute(
        "SELECT label FROM speaker_labels WHERE meeting_id = ?", (meeting_id,)
    )
    return {r[0] for r in rows}


def load_view(connection: sqlite3.Connection, meeting_id: str, transcript_sha: str) -> SpeakerView:
    view = SpeakerView()
    for table, target, column in (
        ("speaker_names", view.names, "label, name"),
        ("speaker_corrections", view.overrides, "segment_id, speaker"),
    ):
        for key, value in connection.execute(
            f"SELECT {column} FROM {table} WHERE meeting_id = ? AND transcript_sha256 = ?",
            (meeting_id, transcript_sha),
        ):
            target[key] = value
        view.stale_entries += connection.execute(
            f"SELECT count(*) FROM {table} WHERE meeting_id = ? AND transcript_sha256 != ?",
            (meeting_id, transcript_sha),
        ).fetchone()[0]
    return view


def clean_name(name: str) -> str:
    value = normalize(name)
    if not 1 <= len(value) <= 100 or any(not c.isprintable() for c in value):
        raise NotesError("Speaker name must be 1-100 printable characters")
    if LABEL.fullmatch(value) or value in UNRESOLVED:
        raise NotesError("Speaker name must be a person's name, not a speaker label")
    return value


def set_name(
    connection: sqlite3.Connection, meeting_id: str, transcript_sha: str, label: str, name: str
) -> None:
    if label not in labels(connection, meeting_id):
        raise NotesError("Unknown speaker label for this meeting")
    with transaction(connection):
        connection.execute(
            "INSERT OR REPLACE INTO speaker_names VALUES (?, ?, ?, ?, ?)",
            (meeting_id, transcript_sha, label, clean_name(name), now()),
        )


def clear_name(
    connection: sqlite3.Connection, meeting_id: str, transcript_sha: str, label: str
) -> None:
    with transaction(connection):
        connection.execute(
            "DELETE FROM speaker_names "
            "WHERE meeting_id = ? AND transcript_sha256 = ? AND label = ?",
            (meeting_id, transcript_sha, label),
        )


def correct_segment(
    connection: sqlite3.Connection,
    meeting_id: str,
    transcript_sha: str,
    segment_id: str,
    speaker: str,
) -> None:
    exists = connection.execute(
        "SELECT 1 FROM segments WHERE meeting_id = ? AND segment_id = ?", (meeting_id, segment_id)
    ).fetchone()
    if not exists:
        raise NotesError("Unknown segment ID for this meeting")
    if speaker not in labels(connection, meeting_id) and speaker not in UNRESOLVED:
        raise NotesError("Corrections must use an existing speaker label, Unknown or Uncertain")
    with transaction(connection):
        connection.execute(
            "INSERT OR REPLACE INTO speaker_corrections VALUES (?, ?, ?, ?, ?)",
            (meeting_id, transcript_sha, segment_id, speaker, now()),
        )


def clear_correction(
    connection: sqlite3.Connection, meeting_id: str, transcript_sha: str, segment_id: str
) -> None:
    with transaction(connection):
        connection.execute(
            "DELETE FROM speaker_corrections "
            "WHERE meeting_id = ? AND transcript_sha256 = ? AND segment_id = ?",
            (meeting_id, transcript_sha, segment_id),
        )
