import contextlib

import pytest

from local_audio_notes.errors import NotesError
from local_audio_notes.extraction.grounding import grounded
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.jobs.queue import process
from local_audio_notes.pipeline import export_notes
from local_audio_notes.speakers import mapping
from local_audio_notes.storage.db import connect


@pytest.fixture
def processed(audio, settings, mocked):
    identifier = ingest(audio, settings)
    process(settings, identifier)
    with contextlib.closing(connect(settings.data_dir)) as connection:
        sha = connection.execute(
            "SELECT artifacts FROM checkpoints WHERE meeting_id = ? AND stage = 'attributing'",
            (identifier,),
        ).fetchone()[0]
        yield identifier, connection, sha.split('"')[3]


def test_confirmed_names_apply_to_this_meeting_only(processed, settings):
    identifier, connection, sha = processed
    assert mapping.labels(connection, identifier) == {"Speaker 1", "Speaker 2", "Speaker 3"}
    mapping.set_name(connection, identifier, sha, "Speaker 3", "  Alice  ")
    mapping.correct_segment(connection, identifier, sha, "s000006", "Speaker 1")
    view = mapping.load_view(connection, identifier, sha)
    assert view.names == {"Speaker 3": "Alice"} and view.overrides == {"s000006": "Speaker 1"}
    export_notes(settings, connection, identifier)
    notes = (settings.data_dir / identifier / "notes.md").read_text()
    assert "Alice \\[Speaker 3\\] (confirmed)" in notes
    assert "s000006 [00:00:18–00:00:20] Speaker 1:" in notes
    # Names are display mappings, never owners: owners still come from quoted words only.
    assert "| Bob | November 6 | Speaker 1 |" in notes
    mapping.clear_name(connection, identifier, sha, "Speaker 3")
    mapping.clear_correction(connection, identifier, sha, "s000006")
    assert mapping.load_view(connection, identifier, sha) == mapping.SpeakerView()


@pytest.mark.parametrize(
    "label, name, error",
    [
        ("Speaker 9", "Alice", "Unknown speaker label"),
        ("Uncertain", "Alice", "Unknown speaker label"),
        ("Speaker 1", "Speaker 2", "not a speaker label"),
        ("Speaker 1", "", "printable"),
        ("Speaker 1", "x" * 101, "printable"),
        ("Speaker 1", "Al\x1bice", "printable"),
    ],
)
def test_invalid_names_rejected(processed, label, name, error):
    identifier, connection, sha = processed
    with pytest.raises(NotesError, match=error):
        mapping.set_name(connection, identifier, sha, label, name)


def test_corrections_validated(processed):
    identifier, connection, sha = processed
    with pytest.raises(NotesError, match="Unknown segment"):
        mapping.correct_segment(connection, identifier, sha, "s999999", "Speaker 1")
    with pytest.raises(NotesError, match="existing speaker label"):
        mapping.correct_segment(connection, identifier, sha, "s000001", "Alice")
    mapping.correct_segment(connection, identifier, sha, "s000001", "Uncertain")


def test_mappings_for_an_older_transcript_are_reported_not_applied(processed, settings):
    identifier, connection, sha = processed
    mapping.set_name(connection, identifier, sha, "Speaker 1", "Bob")
    view = mapping.load_view(connection, identifier, "0" * 64)
    assert view.names == {} and view.stale_entries == 1


def test_sentence_start_function_words_are_not_entities():
    assert grounded("Is there consent for that?", "do we have consent for that?")
    assert not grounded("Carol has consent.", "do we have consent for that?")
    assert not grounded("The Board has consent.", "do we have consent for that?")
