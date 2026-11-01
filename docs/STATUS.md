# Status — 2026-10-31

Branches (stacked PRs): `phases-1-2` (Phase 1 + Phase 2, PR #1 → master), `phase-3`
(PR → `phases-1-2`). Phase 1 was written in a separate Codex session.

## Current state

- Phase 1 (unchanged in substance): typed package and CLI; validated, hash-preserving import;
  ffmpeg handling; offline WhisperX/Community-1 adapters; conservative speaker attribution;
  explicit model preparation/checks; Python network guard; strict namespace prerequisite.
- Phase 2: private SQLite store with migrations; durable job queue (enqueue, worker, cancel,
  recover, status, jobs) with flock-based crash recovery and SIGTERM requeue; input-fingerprinted,
  hash-verified stage checkpoints and a Stage 1 chunk cache; two-stage grounded evidence
  extraction, consolidation and cited summary; Markdown/JSON exports with verbatim source
  references; confirmed per-meeting speaker names/corrections and meeting title/date via CLI;
  diarization GPU-OOM batch reduction with opt-in CPU fallback.
- Phase 3: loopback-only web interface (`serve`): dashboard, import with streamed uploads,
  meeting page with audio, notes, action items, speaker names and corrections, exports,
  settings. Sign-in token, Host/Origin/CSRF checks, strict CSP; processing in a `worker`
  subprocess.
- 149 tests pass, 1 opt-in integration test skipped; Ruff check/format clean. See VALIDATION.md.

## Active work

Phase 4 (search, optional local embeddings, security tests, offline validation, performance).

## Known limitations and unverified

- Still no real inference: inference stack and models are not installed and Ollama was not
  reachable. All model behavior in tests is mocked. Real extraction quality, rejection rates,
  consolidation on paraphrased repeats and summary usefulness with a Qwen 14B-class model are
  unmeasured. Performance (stage timings, GPU memory) has not been measured.
- Grounding checks catch fabricated quotes, names, numbers and dates, not misclassification of
  real quotes or lowercase paraphrase. Sentence-start exemptions are English-only.
- Diarization OOM recovery is tested with a fake pipeline only; pyannote 4 batch-size attribute
  names are taken from its API, not verified against an installed version.
- Recovery assumes a single host and a local filesystem with working `flock`.
- Cancellation cannot interrupt a single native call (one transcription window, one LLM request,
  whole-file diarization).
- No retention policy or deletion; no encryption at rest; search not implemented.
- The web UI was exercised with a real server and curl, and with TestClient; it has not been
  checked visually in a browser or for accessibility.
- Earlier unverified items remain: GPU inference, offline namespace operation, native egress.

## Next (only on explicit request)

Review and merge or push `phases-1-2` when ready. Install the inference stack and
approved models, select an installed local model, and run the opt-in integration test on
consented audio to obtain real stage timings and review rejection counts. Then Phase 3.
