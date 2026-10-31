import json

import pytest

from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.transcripts import parse_transcript


def test_srt(tmp_path):
    path = tmp_path / "meeting.srt"
    path.write_text("1\n00:00:01,250 --> 00:00:02,500\nHello\nworld\n")
    transcript = parse_transcript(path, ".srt", 10, "en")
    assert transcript.segments[0].start == 1.25
    assert transcript.segments[0].text == "Hello\nworld"


@pytest.mark.parametrize(
    "rows",
    [
        [{"start": 0, "end": 2, "text": "Valid"}],
    ],
)
def test_json(tmp_path, rows):
    path = tmp_path / "meeting.json"
    path.write_text(json.dumps({"segments": rows}))
    assert parse_transcript(path, ".json", 3, "en").segments[0].id == "s000000"


@pytest.mark.parametrize(
    "rows",
    [
        [{"start": "00:00:01", "end": "00:00:02", "text": "No guessing"}],
        [{"start": 1000, "end": 2000, "text": "Not silently treated as ms"}],
        [{"start": -1, "end": 2, "text": "Invalid"}],
        [{"start": 2, "end": 1, "text": "Invalid"}],
        [{"start": 0, "end": float("nan"), "text": "Invalid"}],
        [{"start": True, "end": 2, "text": "Invalid"}],
        [{"start": 1, "end": 3, "text": "One"}, {"start": 0, "end": 2, "text": "Two"}],
        [],
    ],
)
def test_invalid_json(tmp_path, rows):
    path = tmp_path / "meeting.json"
    path.write_text(json.dumps({"segments": rows}))
    with pytest.raises(NotesError):
        parse_transcript(path, ".json", 10, "en")


def test_untimed_txt_rejected(tmp_path):
    path = tmp_path / "meeting.txt"
    path.write_text("No timestamps")
    with pytest.raises(NotesError, match="Untimed TXT"):
        parse_transcript(path, ".txt", 10, "en")
