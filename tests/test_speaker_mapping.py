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


def test_speakers_heard_only_inside_mixed_segments_are_nameable(settings):
    from local_audio_notes.ingestion.importer import register
    from local_audio_notes.notes.render import participants
    from local_audio_notes.pipeline import store_segments
    from local_audio_notes.schemas import Segment, Transcript, Word

    transcript = Transcript(
        language="en",
        segments=[
            Segment(
                id=f"s{i:06d}",
                start=i * 2,
                end=i * 2 + 2,
                text="Yes no",
                speaker="Uncertain",
                words=[
                    Word(text="Yes", start=i * 2, end=i * 2 + 1, speaker="Speaker 1"),
                    Word(text="no", start=i * 2 + 1, end=i * 2 + 2, speaker="Speaker 2"),
                ],
            )
            for i in range(2)
        ],
    )
    identifier = "00000000-0000-4000-8000-000000000001"
    with contextlib.closing(connect(settings.data_dir)) as connection:
        register(
            connection,
            {
                "id": identifier,
                "sha256": "a" * 64,
                "duration": 4,
                "format": ["wav"],
                "imported_at": "now",
            },
        )
        store_segments(connection, identifier, transcript)
        assert mapping.labels(connection, identifier) == {"Speaker 1", "Speaker 2"}
        mapping.set_name(connection, identifier, "f" * 64, "Speaker 2", "Dana")
        mapping.correct_segment(connection, identifier, "f" * 64, "s000000", "Speaker 1")
        view = mapping.load_view(connection, identifier, "f" * 64)
    assert participants(transcript, view) == [
        {"label": "Speaker 1", "name": None},
        {"label": "Speaker 2", "name": "Dana"},
    ]
