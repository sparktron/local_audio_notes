import json

import pytest

from local_audio_notes.cli import main as cli
from local_audio_notes.ingestion.importer import ingest


@pytest.fixture
def run_cli(settings, tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.toml"
    config.write_text(
        f'data_dir = "{settings.data_dir}"\nmodels_dir = "{settings.models_dir}"\n'
        'ollama_model = "qwen2.5:14b"\n'
    )
    monkeypatch.setattr(cli, "enable_runtime", lambda **kwargs: None)

    def invoke(*args):
        code = cli.main(["--config", str(config), *args])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return invoke


def test_queue_worker_status_and_jobs(audio, settings, mocked, run_cli):
    identifier = ingest(audio, settings)
    code, out, _ = run_cli("enqueue", identifier)
    assert code == 0 and out.strip() == "1"
    code, out, _ = run_cli("worker")
    assert code == 0 and json.loads(out) == {"job": 1, "outcome": "succeeded"}
    code, out, _ = run_cli("status", identifier)
    status = json.loads(out)
    assert status["state"] == "succeeded" and "exporting" in status["checkpoints"]
    code, out, _ = run_cli("jobs")
    assert json.loads(out.splitlines()[0])["state"] == "succeeded"
    assert "SQLite" not in out


def test_run_reports_safe_stage_error(audio, settings, mocked, run_cli):
    calls, ollama = mocked
    ollama.fail_on_call = 1
    identifier = ingest(audio, settings)
    code, out, err = run_cli("run", identifier)
    assert code == 1
    assert "failed at extracting: Ollama output failed evidence schema validation" in err
    code, _, _ = run_cli("run", identifier, "--notes-only")
    assert code == 0 and calls.transcribe == 1


def test_cancel_and_recover_commands(audio, settings, mocked, run_cli):
    identifier = ingest(audio, settings)
    run_cli("enqueue", identifier)
    code, out, _ = run_cli("cancel", "1")
    assert (code, out.strip()) == (0, "cancelled")
    code, out, _ = run_cli("recover")
    assert json.loads(out) == {
        "requeued": 0,
        "failed": 0,
        "alive": 0,
        "adopted": 0,
        "unadoptable": 0,
    }
    code, _, err = run_cli("cancel", "1")
    assert code == 1 and "not active" in err


def test_meeting_metadata_speakers_and_export(audio, settings, mocked, run_cli, tmp_path):
    identifier = ingest(audio, settings)
    code, _, err = run_cli("speakers", identifier)
    assert code == 1 and "only after processing" in err
    assert run_cli("run", identifier)[0] == 0
    code, out, _ = run_cli(
        "meeting-set", identifier, "--title", "Storage review", "--date", "2026-10-30"
    )
    assert (code, out.strip()) == (0, "Notes re-rendered")
    code, out, _ = run_cli(
        "speakers", identifier, "--name", "Speaker 2=Dana", "--assign", "s000006=Speaker 2"
    )
    result = json.loads(out)
    assert result["confirmed_names"] == {"Speaker 2": "Dana"} and result["notes_rerendered"]
    notes = (settings.data_dir / identifier / "notes.md").read_text()
    assert notes.startswith("# Storage review\n") and "Date: 2026\\-10\\-30" in notes
    assert "Dana \\[Speaker 2\\] (confirmed)" in notes
    output = tmp_path / "exported.json"
    code, _, _ = run_cli("export", identifier, "--format", "json", "--output", str(output))
    assert code == 0 and json.loads(output.read_text())["meeting"]["title"] == "Storage review"
    code, _, err = run_cli("export", identifier, "--output", str(output))
    assert code == 1 and "never overwritten" in err
    code, _, err = run_cli("meeting-set", identifier, "--date", "30/10/2026")
    assert code == 1 and "YYYY-MM-DD" in err
