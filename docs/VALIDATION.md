# Validation

## Phase 4 — 2026-10-31

- `pytest -q`: **220 passed, 1 skipped**; Ruff check and format clean.
- New tests: FTS search over transcripts, items and titles; diacritics/case; category, date
  and participant filters (labels, confirmed names, corrections); FTS operators and column
  syntax handled literally; empty queries rejected; highlight markers cannot be forged by
  content; `reindex` rebuild and busy-meeting skip; semantic search with a deterministic fake
  embedder (stored vectors, ranking, checkpoint reuse), pure-Python similarity fallback; CLI
  and web search (escaping, highlighting, timestamp links, validation); static checks over
  every source file (no shell, eval/exec, os.system, pickle; subprocess argument arrays) and
  web assets (no external URLs, inline script or innerHTML); Python audit-hook recording of
  a full mocked run plus web pages and search (only 127.0.0.1:11434 attempted); real
  `OllamaClient` targets only that endpoint; symlinked meeting directories refused; private
  database sidecars; hash cache hits, invalidation on content and metadata change, corrupt
  cache tolerance, cached model verification still detecting tampering.
- Network observation with `scripts/observe_network.sh` (strace 5.16, this host, no Ollama
  running, no models installed):
  - `ingest tone.m4a`: no network system calls at all.
  - `run <id>`: one connect to `127.0.0.1:11434`, then the expected preflight failure.
  - `serve --port 8798` for 10 s while answering 3 curl requests: only bound 127.0.0.1:8798;
    no outbound endpoints.
  - Positive control `curl https://example.com` (one deliberate outbound request): flagged,
    including `127.0.0.53:53` (DNS via systemd-resolved) and remote IPv4/IPv6 addresses.
- Benchmarks (`scripts/benchmark_local.py --dir data`, synthetic data, 20 CPUs, SQLite 3.53.1):
  SHA-256 of 1 GiB 2.62 s (391 MiB/s, warm cache), hash-cache hit about 3 µs; FTS5 build of
  200,000 segments 0.82 s, median query 38 ms; numpy cosine search 100,000 × 768 in 0.20 s.

Not demonstrated in Phase 4: a real embedding model, search relevance on real meetings,
observation of real inference runs, cold-disk hashing speed.

## Phase 3 — 2026-10-31

- `pytest -q`: **149 passed, 1 skipped**. Ruff check and format clean.
- Dependencies installed explicitly into the venv: FastAPI 0.162.0, Starlette 1.10.0,
  uvicorn 0.54.0, Jinja2 3.1.6 (`requirements-dev.lock` regenerated). FastAPI now depends on
  `opentelemetry-api` (API only, no SDK or exporter installed); its instrumentation is
  disabled in the app and its environment auto-configuration forced off.
- Real server smoke test (`serve --port 8799`, network guard active, no Ollama): listened on
  127.0.0.1 only (`ss`); unauthenticated 401; Host `evil.example` 400; sign-in 303 with
  cookie; CSP, X-Frame-Options present, no Server header; raw upload accepted; import with a
  cross-site Origin 403; import 303 to the meeting; the real `worker` subprocess ran the job,
  which failed safely at preflight (Ollama unavailable); staged uploads removed.
- Tests: sign-in and cookie flags, security headers, no external resources, docs/OpenAPI
  absent, telemetry switches off, DNS-rebinding Host, cross-site Origin and Sec-Fetch-Site,
  missing/forged CSRF, upload size and kind limits, non-audio upload, forged upload IDs,
  traversal and unknown meetings, full import → worker → notes flow (mocked inference),
  escaped script in titles, speaker names, corrections (including invalid targets), Markdown
  and JSON downloads, audio range requests, process/cancel, settings validation and fixed
  fields, read-only settings without a config file, error redaction, stale upload cleanup,
  launcher argument array, port validation.

Not demonstrated: visual/browser rendering, accessibility, large uploads near the 2 GiB
limit, long transcripts in the page, behaviour behind proxies.

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
