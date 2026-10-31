import json
from types import SimpleNamespace

import pytest

import local_audio_notes.pipeline as pipeline
from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.schemas import Segment, SpeakerTurn, Transcript, Word


def install_mock_backends(monkeypatch):
    class Backend:
        device = "cpu"

        def __init__(self, *args):
            pass

        def transcribe(self, path):
            return Transcript(
                language="en",
                segments=[
                    Segment(
                        id="s000000", start=0, end=1.5, text="Alice will write tests by Friday."
                    )
                ],
            )

        def align(self, path, transcript):
            transcript.segments[0].words = [Word(text="Alice", start=0.1, end=0.4)]
            return transcript

    monkeypatch.setattr(pipeline, "verify_models", lambda *args: {"diarization": "local-model"})
    monkeypatch.setattr(pipeline, "WhisperXBackend", Backend)
    monkeypatch.setattr(
        pipeline, "diarize", lambda *args: [SpeakerTurn(start=0, end=1.5, speaker="local-A")]
    )
    client = SimpleNamespace(
        require_local_model=lambda: None,
        generate=lambda *args: json.dumps(
            {
                "items": [
                    {
                        "category": "action",
                        "quote": "Alice will write tests by Friday.",
                        "segment_id": "s000000",
                        "owner": "Alice",
                        "deadline": "Friday",
                    }
                ]
            }
        ),
    )
    monkeypatch.setattr(pipeline, "OllamaClient", lambda *args: client)
    return client


def test_pipeline_real_media_mocked_inference(audio, settings, monkeypatch):
    install_mock_backends(monkeypatch)
    identifier = ingest(audio, settings)
    pipeline.process(settings, identifier)
    folder = settings.data_dir / identifier
    assert (folder / "notes.md").exists()
    assert "Friday" in (folder / "notes.md").read_text()
    transcript = Transcript.model_validate_json((folder / "transcript.json").read_text())
    assert transcript.segments[0].speaker == "Speaker 1"
    assert transcript.segments[0].words[0].speaker == "Speaker 1"
    metadata = json.loads((folder / "meeting.json").read_text())
    assert metadata["state"] == "complete"
    assert metadata["timings"]["transcribing"] >= 0


def test_failed_notes_retry_does_not_retranscribe(audio, settings, monkeypatch):
    client = install_mock_backends(monkeypatch)
    identifier = ingest(audio, settings)
    generate = client.generate
    client.generate = lambda *args: "invalid JSON"
    with pytest.raises(NotesError):
        pipeline.process(settings, identifier)
    folder = settings.data_dir / identifier
    assert json.loads((folder / "meeting.json").read_text())["failed_stage"] == "notes"
    assert (folder / "original.audio").exists()
    assert (folder / "transcript.json").exists()
    monkeypatch.setattr(
        pipeline, "WhisperXBackend", lambda *args: pytest.fail("must not transcribe")
    )
    client.generate = generate
    pipeline.process(settings, identifier, notes_only=True)
    assert json.loads((folder / "meeting.json").read_text())["state"] == "complete"


def test_integrity_failure_before_models(audio, settings, monkeypatch):
    identifier = ingest(audio, settings)
    (settings.data_dir / identifier / "original.audio").write_text("corrupted")
    with pytest.raises(NotesError, match="integrity"):
        pipeline.process(settings, identifier)
