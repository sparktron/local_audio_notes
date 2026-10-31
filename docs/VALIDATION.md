# Validation — 2026-10-30

## Demonstrated

- Isolated Python 3.12.13 package/development installation succeeded. No system Python,
  driver, CUDA, Ollama model or system service was changed.
- `.venv/bin/python -m pytest -q`: **68 passed, 1 skipped** (about 3 seconds, not an
  inference performance result). Run outside the tool sandbox so socket creation is
  allowed and application audit blocking is actually exercised.
- `.venv/bin/ruff check .`: passed.
- `.venv/bin/ruff format --check .`: passed.
- `.venv/bin/python -m pip check`: passed for installed lightweight dependencies.
- `.venv/bin/python -m pip wheel --no-build-isolation --no-deps --no-index
  --disable-pip-version-check --wheel-dir /tmp/local-audio-notes-wheel .`: passed;
  built local_audio_notes-0.1.0-py3-none-any.whl without network access.
- `bash -n scripts/setup.sh scripts/verify_offline.sh`: passed.
- `GIT_OPTIONAL_LOCKS=0 git diff --check`: passed.
- `.venv/bin/local-audio-notes doctor`: passed; Python 3.12.13, ffmpeg/ffprobe,
  RTX 3090 / 24576 MiB / driver 595.99.02 detected. No torch/WhisperX/pyannote/Silero or
  Ollama executable found in this environment.
- `.venv/bin/local-audio-notes models-check`: expected exit 1, missing model manifest.
- `bash scripts/verify_offline.sh`: expected exit 1 on the host, requiring a namespace
  with only loopback. This demonstrates rejection, not successful enforced inference.

Coverage includes real WAV/FLAC/MP3/M4A encoding/import/normalization; content inspection;
byte/duration bounds; path traversal, symlink/FIFO rejection; deduplication and preservation;
transcript parsing and timestamp/schema rejection; three-speaker attribution, overlap and
unknowns; local-only adapter arguments and mocked OOM retry; schema/quote/owner/deadline
validation; long Unicode transcript chunks; Markdown escaping; HTTP response checks;
mocked full pipeline with real media stages; notes-only retries; tamper checks; CLI exit
codes; suppression of sensitive third-party prints; concurrent writer rejection;
Python egress/DNS guard and strict offline prerequisite rejection.

## Not demonstrated

- Real torch/CTranslate2 CUDA inference, VAD, language detection, Whisper recognition,
  forced alignment, Community-1 model loading, speaker count/quality or genuine Ollama
  structured response. Heavyweight inference stack compatibility is not validated.
- Model installation/auth/license workflows against actual gated repositories; full
  local cache sufficiency across languages. Public upstream APIs were checked, but mocked
  adapter tests do not establish model compatibility on this host.
- Real transcription word error rate, diarization error rate or a Qwen/model benchmark.
  `tests/fixtures/meeting.json` is synthetic text, not an acoustic benchmark.
- Zero network traffic, native-library egress, packet capture, OS-isolated inference,
  successful namespace/GPU/Ollama integration or network isolation against a privileged user.
- Durable job restart recovery, SQLite/web/search behavior (not implemented).

The separate `tests/integration/test_real_audio.py` is skipped unless LAN_TEST_CONFIG and
LAN_TEST_AUDIO explicitly opt in. It checks actual stage completion and records wall time;
it still requires human comparison with a time-aligned acoustic reference for quality.

The original failing CLI exit-code test was fixed and passed. A sandbox-denied socket
test was rerun on the host and passed, rather than being counted as an application guard
success. An offline wheel build initially lacked setuptools in the development environment;
the dev extra now includes build tooling so packaging can be validated offline.

## Concurrent edits during closeout

After the successful full lint/format checks and wheel build, separate edits appeared in
`storage/db.py`, `storage/migrations.py` and the shared `storage/locking.py`. They were not
written as part of this Phase 1 implementation and were left intact. The final pytest
gate still returned **68 passed, 1 skipped**. The current full-checkout Ruff gate then
reported line-length errors in the concurrently edited storage files; formatting also
reported locking.py after it changed. The 47 unchanged Phase 1 Python files pass both
Ruff checks when the concurrently modified locking.py is excluded. Earlier full-gate
passes are historical results, not a claim that the changing checkout is currently lint-clean.
