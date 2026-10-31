# Validation

## Phase 2 — 2026-10-31

Run on the inspected workstation in this session's shell, Python 3.12.13 venv.

- `.venv/bin/python -m pytest -q`: **126 passed, 1 skipped** (the opt-in real-audio test),
  about 16 seconds. Not an inference performance result.
- `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .`: passed (60 files).
- `.venv/bin/python -m pip check`: passed.
- Offline wheel build (`pip wheel --no-build-isolation --no-deps --no-index`): passed after
  removing a stale, gitignored `build/` cache that had packaged the removed Phase 1
  `notes/generate.py`; the rebuilt wheel contains 47 files and no stale module.
- `config.example.toml` loads and validates, including the new keys.
- Manual CLI run without mocks (temporary data directory, application network guard on,
  no Ollama running): `ingest` twice returned the same ID; `enqueue`, `jobs`, `worker`
  recorded a durable failed job at `preflight` with the safe message "Local Ollama API
  unavailable"; `run --notes-only` created a second job and failed safely; `export` and
  `speakers` refused before processing; `recover` reported nothing to do; database and
  artifacts were mode 0600 in 0700 directories.

New coverage: migrations (idempotent, downgrade refused, symlinked DB refused, CHECK
constraints); ingest registration/idempotency; orphan and Phase 1 directory adoption, with
tampered originals refused; enqueue idempotency and conflicts; cancel of queued and running
jobs (stops at the next stage, resumes without repeating audio or LLM work); recovery of a
running job with no lock holder, with a live holder, with a real SIGKILLed subprocess
holding the lock, and after `job_max_attempts`; SIGTERM requeue; Ctrl-C cancel; failure
recording without library exception text; GPU OOM classification; queue ordering;
checkpoint reuse, tamper-driven recomputation of exactly the affected stage, `--force`;
LLM failure resumed from cached chunks without retranscribing; notes-only integrity
refusal; grounding (quote location, entity/number checks, sentence-start exemptions);
rejection of fabricated quotes, wrong-segment citations, unknown segments and inferred
decisions; stripping of invented deadlines, absent or pronoun owners and owners on
non-commitments; chunk overlap and bounds; chunk cache invalidation by model;
consolidation of overlap duplicates and restatements, distinct owners kept apart, short
repeats not merged, disputes flagged; summary rejection of ungrounded points and titles,
unknown citations and unsupported decision wording; rendering of all sections, markers,
speaker names, corrections and escaping; JSON export; speaker name/correction validation
and staleness; diarization OOM batch halving and CPU fallback (fake pipeline); and the
new CLI commands end to end with mocked inference.

Not demonstrated in Phase 2: any real model output (all inference mocked), real rejection
or dispute rates, extraction/summary quality, stage timings or GPU memory on real audio,
pyannote batch attribute names on an installed pyannote, multi-host or network-filesystem
recovery. The synthetic fixture is text, not an acoustic or quality benchmark.

## Phase 1 — 2026-10-30 (historical)

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

The storage files that appeared during Phase 1 closeout (`storage/db.py`,
`storage/migrations.py`, `storage/locking.py`) were the start of the Phase 2 implementation in
a parallel session. They are now complete, lint-clean and covered by the Phase 2 results above.
