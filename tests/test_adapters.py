import sys
from types import SimpleNamespace

import numpy as np
import pytest

import local_audio_notes.transcription.whisperx_backend as backend_module
from local_audio_notes.diarization.community import diarize
from local_audio_notes.errors import NotesError
from local_audio_notes.schemas import Segment, Transcript


@pytest.fixture
def backend(monkeypatch, settings, tmp_path):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), set_num_threads=lambda count: None
    )
    nltk = SimpleNamespace(data=SimpleNamespace(path=[]), download=lambda *args: pytest.fail())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    monkeypatch.setattr(backend_module, "release_gpu", lambda: None)
    paths = {kind: tmp_path / kind for kind in ["whisper", "alignment", "vad", "tokenizer"]}
    return backend_module.WhisperXBackend(settings, paths)


def test_audio_windows_bounded(audio):
    windows = list(backend_module.audio_windows(audio, 1))
    assert [offset for offset, _ in windows] == [0, 1]
    assert all(len(samples) == 16000 and samples.dtype == np.float32 for _, samples in windows)


def test_transcribe_uses_local_model_and_oom_retry(backend, audio, monkeypatch):
    batches = []

    def transcribe(samples, **kwargs):
        batches.append(kwargs["batch_size"])
        if kwargs["batch_size"] > 2:
            raise RuntimeError("CUDA out of memory")
        return {"language": "en", "segments": [{"start": 0, "end": 1, "text": "Local fixture"}]}

    def load_model(path, device, **kwargs):
        assert path == str(backend.paths["whisper"])
        assert device == "cpu" and kwargs["compute_type"] == "int8"
        assert kwargs["local_files_only"] is True
        assert kwargs["vad_model"] == "local-vad"
        return SimpleNamespace(transcribe=transcribe)

    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(backend_module, "local_vad", lambda *args: "local-vad")
    result = backend.transcribe(audio)
    assert batches == [8, 4, 2]
    assert result.segments[0].id == "s000000"


def test_alignment_preserves_ids_verbatim_and_untimed_words(backend, audio, monkeypatch):
    def load(language, device, **kwargs):
        assert language == "en" and kwargs["model_cache_only"] is True
        assert kwargs["model_name"] == str(backend.paths["alignment"])
        return "model", "metadata"

    def align(rows, model, metadata, waveform, device, **kwargs):
        assert rows[0]["start"] == 0
        assert waveform.dtype == np.float32
        return {
            "segments": [
                {
                    "words": [
                        {"word": "hello", "start": 0.1, "end": 0.2, "score": 0.8},
                        {"word": "unclear"},
                    ]
                }
            ]
        }

    monkeypatch.setitem(
        sys.modules, "whisperx", SimpleNamespace(load_align_model=load, align=align)
    )
    original = Transcript(
        language="en", segments=[Segment(id="s000005", start=0.5, end=1.5, text="hello unclear")]
    )
    result = backend.align(audio, original)
    assert result.segments[0].id == "s000005" and result.segments[0].text == "hello unclear"
    assert result.segments[0].words[0].start == pytest.approx(0.6)
    assert result.segments[0].words[1].start is None
    assert original.segments[0].words == []
    with pytest.raises(NotesError, match="language"):
        backend.align(audio, Transcript(language="fr", segments=[]))
    with pytest.raises(NotesError, match="tokenizer"):
        sys.modules["nltk"].download("anything")


def test_community_regular_output_preserves_overlaps(settings, audio, tmp_path, monkeypatch):
    calls = []
    pipeline = SimpleNamespace(
        to=lambda device: calls.append(device),
    )

    class Pipeline:
        @staticmethod
        def from_pretrained(path):
            assert path == str(tmp_path)
            return Pipeline()

        def to(self, device):
            calls.append(device)

        def __call__(self, path, **kwargs):
            assert kwargs == {"min_speakers": 1, "max_speakers": 8}
            annotation = SimpleNamespace(
                itertracks=lambda **kwargs: [
                    (SimpleNamespace(start=0, end=1), None, "A"),
                    (SimpleNamespace(start=0.5, end=1.5), None, "B"),
                ]
            )
            return SimpleNamespace(speaker_diarization=annotation)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda value: value))
    monkeypatch.setitem(sys.modules, "pyannote.audio", SimpleNamespace(Pipeline=Pipeline))
    monkeypatch.setattr("local_audio_notes.diarization.community.release_gpu", lambda: None)
    result = diarize(audio, tmp_path, settings, "cpu")
    assert len(result) == 2 and result[1].start < result[0].end
    assert calls == ["cpu"]
    del pipeline
