import json
from types import SimpleNamespace

import pytest

import local_audio_notes.pipeline as pipeline
from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.security import runtime


def test_notes_retry_rejects_changed_transcript(audio, settings, monkeypatch):
    identifier = ingest(audio, settings)
    folder = settings.data_dir / identifier
    (folder / "transcript.json").write_text('{"language": "en", "segments": []}')
    monkeypatch.setattr(
        pipeline, "OllamaClient", lambda *args: SimpleNamespace(require_local_model=lambda: None)
    )
    with pytest.raises(NotesError, match="transcript integrity"):
        pipeline.process(settings, identifier, notes_only=True)
    assert json.loads((folder / "meeting.json").read_text())["state"] == "failed"


def test_isolation_probe_failure_is_closed(monkeypatch):
    def denied():
        raise PermissionError

    monkeypatch.setattr(runtime.socket, "if_nameindex", denied)
    assert runtime.isolated_network() is False
