# Implementation plan and scope

User requested scaffold and Phase 1 only. Stop after its tests and handoff.

1. Inspect existing checkout, interpreter, driver, CUDA, ffmpeg, Ollama. Complete.
2. Scaffold a typed Python package and security-first local configuration.
3. Implement bounded content-validated CLI ingestion, immutable original copies and hashes.
4. Implement offline local WhisperX transcription, language-specific forced alignment,
   Community-1 diarization and conservative word-to-speaker attribution.
5. Implement localhost Ollama schema-constrained draft notes and deterministic Markdown/JSON
   exports. Preserve verbatim transcripts and intermediate artifacts.
6. Provide explicit model preparation/checking and installation/offline documentation.
7. Run automated security, ingestion, parser, attribution, adapter, note and CLI tests.
   Keep real model/audio quality integration separate and opt-in.

Phase 1 uses a synchronous CLI and atomic JSON stage artifacts. They permit manual reruns
and notes-only retries, but are **not** a durable job queue or restart recovery system.
SQLite/migrations, durable jobs, full evidence consolidation, web UI, speaker editing,
search, embeddings, cancellation service and performance benchmarking are deferred.
Some basic grounding checks and security tests are necessary now to avoid unsafe notes.

## Inspection, 2026-10-30 (user timezone)

- Repository: local_audio_notes; master; clean; single initial README commit; no tests,
  instructions on disk, architecture/status docs or cross-repository contracts found.
  jCodemunch could not index the empty repository (no source files).
- Ubuntu 22.04.5; default Python 3.10.12; existing uv-managed Python 3.12.13 selected
  for a new repository-local .venv. No system Python changes.
- Host read-only check: NVIDIA RTX 3090, 24576 MiB, driver 595.99.02.
  CUDA toolkit 12.8; ffmpeg/ffprobe 4.4.2 available.
- PyTorch/WhisperX absent in default interpreter. No Ollama executable on current PATH;
  host 127.0.0.1:11434 refused connection. Existing models therefore undiscovered.
- No models downloaded, no system services changed. GPU inference unverified.

Invariants: private local data, no implicit downloads, no remote inference, no automatic
identity assignment, no source content in logs, no deletion of originals. Each later phase
requires a separate implementation request.

## Phase 2 — reliability and evidence (requested 2026-10-30, implemented 2026-10-31)

Scope: SQLite persistence, recovery, structured evidence extraction, source references,
validation tests. Design and trade-offs: [ADR-0002](decisions/0002-phase2-evidence-and-durable-jobs.md).

1. SQLite store (`storage/db.py`, `storage/migrations.py`): private file, WAL, STRICT tables,
   forward-only migrations, downgrade refusal. Meetings registered at import; orphaned or
   Phase 1 import directories adopted after hash verification.
2. Durable jobs (`jobs/queue.py`): one active job per meeting, claim under the meeting flock,
   cancellation at safe points, SIGTERM requeue, crash recovery bounded by `job_max_attempts`,
   safe error recording (fixed messages or class names only), per-stage timings and counts.
3. Checkpointed pipeline (`pipeline.py`): input-fingerprinted, hash-verified stage reuse so a
   failed LLM stage or interruption never repeats completed audio work; Stage 1 chunk cache.
4. Evidence (`extraction/`): chunked schema-constrained extraction, per-item grounding
   validation, cross-chunk consolidation with source attribution and dispute marking.
5. Notes (`notes/`): Stage 2 cited, grounded summary; deterministic Markdown/JSON with the ten
   required sections and verbatim source references.
6. Speakers (`speakers/mapping.py`): confirmed per-meeting names and segment corrections,
   bound to the transcript hash; CLI only (UI is Phase 3). GPU OOM handling for diarization.
7. Tests: storage/migrations, queue/recovery (including a SIGKILLed lock holder), extraction
   hallucination cases, consolidation, summary checks, rendering, speakers and CLI commands.

Not in Phase 2: web UI (Phase 3), search/embeddings/OS sandbox validation (Phase 4),
retention policies and deletion (no request yet), real-model validation (needs models and
consented, annotated audio).

## Phase 3 — web interface (requested and implemented 2026-10-31)

Scope: dashboard, import, transcript viewer, speaker correction, notes and export, settings.
Security model: [ADR-0003](decisions/0003-phase3-loopback-web-interface.md).

1. `serve`: FastAPI on a self-bound 127.0.0.1 socket; per-launch sign-in token; Host,
   Origin/Sec-Fetch-Site and CSRF checks; strict CSP; docs, OpenAPI and telemetry disabled.
2. Pages: dashboard (meetings, status, title/date filter, job counts, stage timings), import
   (streamed uploads, transcript attachment, mode, start processing), meeting (audio,
   process/cancel, details, notes, action items, speaker names, segment corrections,
   Markdown/JSON download), settings (bounded editable subset, written to the config file).
3. Processing via a `worker` subprocess launched with fixed arguments.
4. Tests for authentication, rebinding, cross-site and CSRF refusal, upload limits,
   traversal, escaping, the full import → process → notes flow with mocked inference,
   corrections, settings validation, error redaction and the launcher.

Not in Phase 3: full-text and semantic search (Phase 4).

