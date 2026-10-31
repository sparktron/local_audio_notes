import json

import pytest

from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.storage.files import copy_bounded
from local_audio_notes.storage.locking import exclusive_lock


def test_copy_never_deletes_preexisting_destination(audio, tmp_path):
    destination = tmp_path / "existing"
    destination.write_text("keep this")
    with pytest.raises(FileExistsError):
        copy_bounded(audio, destination, 100000)
    assert destination.read_text() == "keep this"


def test_lock_rejects_concurrent_writer_and_releases(tmp_path):
    with (
        exclusive_lock(tmp_path / "lock"),
        pytest.raises(NotesError, match="Another process"),
        exclusive_lock(tmp_path / "lock"),
    ):
        pytest.fail("lock must be exclusive")
    with exclusive_lock(tmp_path / "lock"):
        pass


def test_duplicate_import_verifies_original(audio, settings):
    identifier = ingest(audio, settings)
    directory = settings.data_dir / identifier
    (directory / "original.audio").write_text("corrupted")
    with pytest.raises(NotesError, match="integrity"):
        ingest(audio, settings)
    assert json.loads((directory / "meeting.json").read_text())["id"] == identifier
