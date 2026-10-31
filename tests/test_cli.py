import os
import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "local_audio_notes", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Phase 1 CLI" in result.stdout


def test_cli_ingest_and_status(audio, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(f'data_dir = "{tmp_path / "private"}"\n')
    result = subprocess.run(
        [sys.executable, "-m", "local_audio_notes", "--config", str(config), "ingest", str(audio)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    identifier = result.stdout.strip()
    result = subprocess.run(
        [sys.executable, "-m", "local_audio_notes", "--config", str(config), "status", identifier],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert '"state": "imported"' in result.stdout


def test_cli_error_does_not_log_private_path(tmp_path):
    secret = tmp_path / "CONFIDENTIAL-title.wav"
    result = subprocess.run(
        [sys.executable, "-m", "local_audio_notes", "ingest", str(secret)],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert result.returncode == 1
    assert "CONFIDENTIAL" not in result.stdout + result.stderr
