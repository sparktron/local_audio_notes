import pytest

from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.jobs.queue import process
from local_audio_notes.security import runtime


def test_notes_retry_rejects_changed_transcript(audio, settings, mocked):
    calls, ollama = mocked
    identifier = ingest(audio, settings)
    process(settings, identifier)
    folder = settings.data_dir / identifier
    (folder / "transcript.json").write_text('{"language": "en", "segments": []}')
    with pytest.raises(NotesError, match="integrity verification"):
        process(settings, identifier, notes_only=True)
    assert calls.transcribe == 1


def test_isolation_probe_failure_is_closed(monkeypatch):
    def denied():
        raise PermissionError

    monkeypatch.setattr(runtime.socket, "if_nameindex", denied)
    assert runtime.isolated_network() is False
