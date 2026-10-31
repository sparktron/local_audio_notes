"""Real quality gate requires human-annotated audio; synthetic text is not a benchmark."""

import json
import os
import time
from pathlib import Path

import pytest

from local_audio_notes.config import load_settings
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.jobs.queue import process
from local_audio_notes.schemas import Transcript
from local_audio_notes.security.runtime import enable_runtime


@pytest.mark.integration
def test_real_local_audio_quality():
    config, audio = os.getenv("LAN_TEST_CONFIG"), os.getenv("LAN_TEST_AUDIO")
    if not config or not audio:
        pytest.skip("Set LAN_TEST_CONFIG and LAN_TEST_AUDIO to opt into actual inference")
    settings = load_settings(Path(config))
    enable_runtime(enforced=settings.offline_enforced)
    identifier = ingest(Path(audio), settings)
    started = time.monotonic()
    process(settings, identifier)
    folder = settings.data_dir / identifier
    result = Transcript.model_validate_json((folder / "transcript.json").read_text())
    assert result.segments and any(s.words for s in result.segments)
    assert (folder / "notes.md").is_file()
    (folder / "integration-result.json").write_text(
        json.dumps(
            {
                "wall_seconds": time.monotonic() - started,
                "quality": "Requires human comparison with time-aligned reference annotations",
            }
        )
    )
