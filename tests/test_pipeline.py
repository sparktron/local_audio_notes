import contextlib
import json
import stat

import pytest

import local_audio_notes.pipeline as pipeline
from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.jobs.queue import describe, process
from local_audio_notes.schemas import Transcript
from local_audio_notes.storage.db import connect


def stages(settings, identifier):
    with contextlib.closing(connect(settings.data_dir)) as connection:
        return {s["stage"]: s["state"] for s in describe(connection, identifier)["stages"]}


def test_end_to_end_real_media_mocked_inference(audio, settings, mocked):
    calls, ollama = mocked
    identifier = ingest(audio, settings)
    job = process(settings, identifier)
    assert job["state"] == "succeeded" and job["attempts"] == 1
    folder = settings.data_dir / identifier
    transcript = Transcript.model_validate_json((folder / "transcript.json").read_text())
    assert [s.speaker for s in transcript.segments][:7] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 3",
        "Speaker 1",
        "Speaker 2",
        "Uncertain",
        "Unknown",
    ]
    assert transcript.segments[5].overlap
    notes = (folder / "notes.md").read_text()
    assert "| Alice | Friday |" in notes and "| Bob | November 6 |" in notes
    assert "Carol" not in notes and "2026-11-06" not in notes
    exported = json.loads((folder / "notes.json").read_text())
    assert exported["generated_by"]["model"] == "qwen2.5:14b@sha256:fixture"
    for name in ("notes.md", "notes.json", "evidence.json", "summary.json", "transcript.json"):
        assert stat.S_IMODE((folder / name).stat().st_mode) == 0o600
    with contextlib.closing(connect(settings.data_dir)) as connection:
        status = describe(connection, identifier)
        stored = connection.execute(
            "SELECT category, owner, deadline FROM items WHERE meeting_id = ? "
            "AND category = 'action' ORDER BY item_id",
            (identifier,),
        ).fetchall()
        refs = connection.execute(
            "SELECT count(*) FROM item_evidence WHERE meeting_id = ?", (identifier,)
        ).fetchone()[0]
        segments = connection.execute(
            "SELECT count(*) FROM segments WHERE meeting_id = ?", (identifier,)
        ).fetchone()[0]
    assert [tuple(r) for r in stored] == [
        ("action", "Alice", "Friday"),
        ("action", "Bob", "November 6"),
    ]
    assert refs > len(stored) and segments == len(transcript.segments)
    assert status["state"] == "succeeded"
    assert all(s["state"] == "succeeded" and s["seconds"] >= 0 for s in status["stages"])
    # Status never carries transcript content.
    assert "SQLite" not in json.dumps(status) and "Alice" not in json.dumps(status)


def test_rerun_reuses_verified_checkpoints(audio, settings, mocked):
    calls, ollama = mocked
    identifier = ingest(audio, settings)
    process(settings, identifier)
    llm_calls = len(ollama.payloads)
    process(settings, identifier)
    assert (calls.transcribe, calls.align, calls.diarize) == (1, 1, 1)
    assert len(ollama.payloads) == llm_calls
    assert set(stages(settings, identifier).values()) == {"skipped", "succeeded"}
    assert stages(settings, identifier)["exporting"] == "succeeded"


def test_changed_artifact_reruns_that_stage_and_dependents_only(audio, settings, mocked):
    calls, ollama = mocked
    identifier = ingest(audio, settings)
    process(settings, identifier)
    (settings.data_dir / identifier / "aligned.json").write_text("{}")
    process(settings, identifier)
    assert (calls.transcribe, calls.align, calls.diarize) == (1, 2, 1)
    result = stages(settings, identifier)
    assert result["transcribing"] == result["diarizing"] == "skipped"
    assert result["aligning"] == "succeeded"
    # Fingerprints use input artifact hashes: an identical re-aligned output reuses dependents.
    assert result["attributing"] == result["extracting"] == "skipped"


def test_force_recomputes_everything(audio, settings, mocked):
    calls, ollama = mocked
    identifier = ingest(audio, settings)
    process(settings, identifier)
    calls_before = ollama.extraction_calls
    process(settings, identifier, force=True)
    assert calls.transcribe == 2 and ollama.extraction_calls == 2 * calls_before


def test_failed_llm_resumes_without_audio_or_completed_chunks(audio, settings, mocked):
    calls, ollama = mocked
    settings = settings.model_copy(update={"chunk_chars": 500})
    identifier = ingest(audio, settings)
    ollama.fail_on_call = 2
    with pytest.raises(NotesError, match="failed at extracting: Ollama output failed"):
        process(settings, identifier)
    folder = settings.data_dir / identifier
    assert (folder / "original.audio").exists() and (folder / "transcript.json").exists()
    assert not (folder / "notes.md").exists()
    ollama.fail_on_call = None
    sent = len(ollama.payloads)
    process(settings, identifier)
    assert (calls.transcribe, calls.align, calls.diarize) == (1, 1, 1)
    extraction = json.loads((folder / "extraction.json").read_text())
    assert extraction["chunks"] >= 2 and extraction["cached_chunks"] == 1
    assert ollama.extraction_calls - 2 == extraction["chunks"] - 1
    assert len(ollama.payloads) > sent


def test_notes_only_requires_verified_transcript(audio, settings, mocked):
    calls, ollama = mocked
    identifier = ingest(audio, settings)
    with pytest.raises(NotesError, match="Attributed transcript missing"):
        process(settings, identifier, notes_only=True)
    process(settings, identifier)
    folder = settings.data_dir / identifier
    (folder / "transcript.json").write_text('{"language": "en", "segments": []}')
    with pytest.raises(NotesError, match="integrity verification"):
        process(settings, identifier, notes_only=True)
    assert calls.transcribe == 1


def test_integrity_failure_before_models(audio, settings, mocked):
    calls, ollama = mocked
    identifier = ingest(audio, settings)
    (settings.data_dir / identifier / "original.audio").write_text("corrupted")
    with pytest.raises(NotesError, match="Original recording integrity"):
        process(settings, identifier)
    assert calls.transcribe == 0 and ollama.payloads == []


def test_align_mode_requires_attachment(audio, settings, mocked):
    identifier = ingest(audio, settings)
    with pytest.raises(NotesError, match="requires an attached transcript"):
        process(settings, identifier, mode="align")


def test_changed_transcript_invalidates_notes_until_regenerated(
    audio, settings, mocked, monkeypatch
):
    """A rerun that changes the attributed transcript but fails during extraction must not
    leave the old evidence/summary exportable alongside the new transcript."""
    from conftest import TURNS

    from local_audio_notes.pipeline import export_notes, verified_checkpoint
    from local_audio_notes.schemas import SpeakerTurn

    calls, ollama = mocked
    identifier = ingest(audio, settings)
    process(settings, identifier)
    folder = settings.data_dir / identifier
    swapped = [SpeakerTurn(start=t.start, end=t.end, speaker=f"other_{t.speaker}") for t in TURNS]
    swapped.reverse()  # different first-appearance order -> different labels
    monkeypatch.setattr(pipeline, "diarize", lambda *args: swapped)
    ollama.fail_on_call = len(ollama.payloads) + 1
    with pytest.raises(NotesError, match="extracting"):
        process(settings, identifier, force=True)
    with contextlib.closing(connect(settings.data_dir)) as connection:
        for stage in ("extracting", "consolidating", "summarizing", "exporting"):
            assert verified_checkpoint(connection, identifier, folder, stage) is None, stage
        assert verified_checkpoint(connection, identifier, folder, "attributing")
        with pytest.raises(NotesError, match="Notes inputs missing"):
            export_notes(settings, connection, identifier)
    ollama.fail_on_call = None
    process(settings, identifier)
    with contextlib.closing(connect(settings.data_dir)) as connection:
        export_notes(settings, connection, identifier)
