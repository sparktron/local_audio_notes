import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError
from local_audio_notes.models import install_model, register_model, verify_models


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0:11434",
        "http://localhost:11434",
        "https://127.0.0.1:11434",
        "http://127.0.0.1:11434@evil.test",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?x=1",
        "http://127.0.0.1:9999",
    ],
)
def test_remote_ollama_config_rejected(url):
    with pytest.raises(ValidationError):
        Settings(ollama_url=url)


def test_download_never_implicit(tmp_path):
    with pytest.raises(NotesError, match="allow-download"):
        install_model(tmp_path, "whisper", "example/model", "a" * 40, False)
    assert not list(tmp_path.iterdir())


def test_required_models_and_tamper_detection(tmp_path):
    with pytest.raises(NotesError, match="manifest"):
        verify_models(tmp_path)
    kinds = {
        "whisper": ["model.bin", "config.json", "tokenizer.json"],
        "alignment": ["config.json", "vocab.json", "preprocessor_config.json"],
        "diarization": ["config.yaml"],
        "tokenizer": ["tokenizers/punkt_tab/english/ortho_context.tab"],
        "vad": ["silero_vad.jit"],
    }
    for kind, names in kinds.items():
        path = tmp_path / kind
        for name in names:
            item = path / name
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_text("fixture — not model weights")
        register_model(tmp_path, kind, path, "test", "synthetic")
    assert set(verify_models(tmp_path)) == set(kinds)
    (tmp_path / "whisper" / "model.bin").write_text("tampered")
    with pytest.raises(NotesError, match="changed"):
        verify_models(tmp_path)


def test_python_network_guard_and_environment():
    script = """
import os, socket
from local_audio_notes.security.runtime import enable_runtime
from local_audio_notes.errors import NotesError
enable_runtime()
assert os.environ['PYANNOTE_METRICS_ENABLED'] == '0'
assert os.environ['HF_HUB_OFFLINE'] == '1'
assert os.environ['FASTAPI_OTEL_AUTO_CONFIGURE'] == 'false'
for host, port in [('203.0.113.1', 443), ('127.0.0.1', 80), ('::1', 11434)]:
    family = socket.AF_INET6 if ':' in host else socket.AF_INET
    with socket.socket(family) as s:
        try:
            s.connect((host, port))
        except NotesError:
            pass
        else:
            raise AssertionError('network guard did not reject')
try:
    socket.getaddrinfo('example.com', 443)
except NotesError:
    pass
else:
    raise AssertionError('DNS guard did not reject')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYANNOTE_METRICS_ENABLED": "1", "FASTAPI_OTEL_AUTO_CONFIGURE": "true"},
    )
    assert result.returncode == 0, result.stderr


def test_strict_offline_fails_closed(monkeypatch):
    from local_audio_notes.security import runtime

    monkeypatch.setattr(runtime, "isolated_network", lambda: False)
    with pytest.raises(NotesError, match="namespace"):
        runtime.enable_runtime(enforced=True)
