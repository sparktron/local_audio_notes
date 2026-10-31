import struct
import wave
from pathlib import Path

import pytest

from local_audio_notes.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data", models_dir=tmp_path / "models", ollama_model="qwen2.5:14b"
    )


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack("<h", 0) * 32000)
    return path
