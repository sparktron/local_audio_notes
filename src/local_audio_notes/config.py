import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    whisper_model: str = "large-v3"
    language: str = "en"
    alignment_language: str = "en"
    batch_size: int = Field(default=8, ge=1, le=64)
    compute_type: Literal["float16", "int8_float16", "int8", "float32"] = "float16"
    cpu_threads: int = Field(default=8, ge=1, le=32)
    min_speakers: int = Field(default=1, ge=1, le=32)
    max_speakers: int = Field(default=8, ge=1, le=32)
    max_audio_bytes: int = Field(default=2 * 1024**3, ge=1024, le=16 * 1024**3)
    max_audio_seconds: float = Field(default=8 * 3600, gt=0, le=24 * 3600)
    max_transcript_bytes: int = Field(default=10 * 1024**2, ge=1, le=100 * 1024**2)
    audio_window_seconds: int = Field(default=600, ge=30, le=1800)
    chunk_chars: int = Field(default=5000, ge=500, le=6000)
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    ollama_context: int = Field(default=8192, ge=8192, le=16384)
    ollama_predict: int = Field(default=2048, ge=256, le=4096)
    ollama_timeout: float = Field(default=600, ge=1, le=3600)
    offline_enforced: bool = False

    @field_validator("ollama_url")
    @classmethod
    def local_ollama(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port != 11434
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Ollama must use http://127.0.0.1:11434")
        return value.rstrip("/")

    @model_validator(mode="after")
    def speaker_bounds(self) -> "Settings":
        if self.min_speakers > self.max_speakers:
            raise ValueError("min_speakers exceeds max_speakers")
        return self


def load_settings(path: Path | None) -> Settings:
    if path is None:
        return Settings()
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    for key in ("data_dir", "models_dir"):
        if key in values:
            values[key] = path.resolve().parent / values[key]
    return Settings.model_validate(values)
