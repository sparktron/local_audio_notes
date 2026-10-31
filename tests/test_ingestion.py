import json
import os
import stat
import subprocess

import pytest

from local_audio_notes.audio.media import normalize
from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.storage.files import copy_bounded, meeting_dir, sha256


def test_import_idempotent_and_private(audio, settings):
    identifier = ingest(audio, settings)
    assert ingest(audio, settings) == identifier
    folder = settings.data_dir / identifier
    assert sha256(folder / "original.audio") == sha256(audio)
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE((folder / "original.audio").stat().st_mode) == 0o600
    assert audio.exists()
    assert not list(settings.data_dir.glob(".import-*"))


@pytest.mark.parametrize(
    "codec,suffix", [("flac", ".flac"), ("libmp3lame", ".mp3"), ("aac", ".m4a")]
)
def test_real_ffmpeg_formats(audio, settings, tmp_path, codec, suffix):
    path = tmp_path / ("encoded" + suffix)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio), "-c:a", codec, str(path)],
        check=True,
    )
    identifier = ingest(path, settings)
    normalize(settings.data_dir / identifier / "original.audio", tmp_path / "normal.wav", settings)


def test_extension_not_trusted(audio, settings, tmp_path):
    renamed = tmp_path / "audio.txt"
    renamed.write_bytes(audio.read_bytes())
    assert ingest(renamed, settings)
    forged = tmp_path / "fake.wav"
    forged.write_text("file:///private\nhttps://example.com")
    with pytest.raises(NotesError, match="Unsupported"):
        ingest(forged, settings)


def test_attachment_and_bounded_copy(audio, settings, tmp_path):
    text = tmp_path / "export.txt"
    text.write_text("Local fixture only")
    identifier = ingest(audio, settings, text)
    metadata = json.loads((settings.data_dir / identifier / "meeting.json").read_text())
    assert metadata["attachment"]["sha256"] == sha256(text)
    with pytest.raises(NotesError):
        copy_bounded(audio, tmp_path / "too-large", 5)
    assert not (tmp_path / "too-large").exists()


def test_duration_and_transcript_limits(audio, settings, tmp_path):
    with pytest.raises(NotesError, match="duration"):
        ingest(audio, settings.model_copy(update={"max_audio_seconds": 1}))
    text = tmp_path / "bad.txt"
    text.write_bytes(b"\xff\x00\xfe")
    with pytest.raises(NotesError, match="UTF-8"):
        ingest(audio, settings, text)
    assert not list(settings.data_dir.glob(".import-*"))


def test_symlink_and_fifo_rejected(audio, settings, tmp_path):
    symlink = tmp_path / "link.wav"
    symlink.symlink_to(audio)
    with pytest.raises(NotesError):
        ingest(symlink, settings)
    fifo = tmp_path / "fifo.wav"
    os.mkfifo(fifo)
    with pytest.raises(NotesError):
        ingest(fifo, settings)


@pytest.mark.parametrize("identifier", ["../outside", "/etc/passwd", "", "..", "a/../b"])
def test_path_traversal(settings, identifier):
    with pytest.raises(NotesError, match="Invalid meeting ID"):
        meeting_dir(settings.data_dir, identifier)
