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
identity assignment, no source content in logs, no deletion of originals. All phases
after Phase 1 require a separate implementation request.
