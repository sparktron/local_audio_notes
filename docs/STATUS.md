# Status — 2026-11-05

Branches: PR #1 merged `phases-1-2` into master. PRs #2 and #3 merged `phase-3` into `phases-1-2`
and `phase-4` into `phase-3`, so Phases 3 and 4 reach master through PR #4 (`phase-3` → master).
Phase 1 was written in a separate Codex session.

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
- Phase 4: SQLite FTS5 search (CLI, web) with participant/category/date filters and
  timestamp links; optional local-embedding semantic search; `reindex`; stat-keyed hash cache;
  strace network observation script; synthetic benchmark script; hardening tests.
- LLM backends (2026-11-05): `llm_backend = "lmstudio"` selects an LM Studio/Bionic client
  (`/v1/chat/completions` with `json_schema`, `/v1/embeddings`, `/api/v0/models`) next to
  Ollama. The loopback port in `ollama_url` is configurable (1024–65535) and the network guard
  and `observe_network.sh` (`LLM_PORT`) allow only that port.
- 269 tests pass, 1 opt-in integration test skipped; Ruff check/format clean. See VALIDATION.md.

## Active work

None. All four requested phases are implemented and reviewed through PRs #1–#3; PR #4 lands
Phases 3 and 4 on master.

## Known limitations and unverified

- Still no real inference. The inference stack is installed in `.venv` (2026-11-05: torch
  2.8.0+cu128 with CUDA on the RTX 3090, whisperx 3.8.6, pyannote.audio 4.0.7, silero-vad
  6.2.0, ctranslate2 4.8.5; `pip check` clean), but no models are prepared and Ollama is not
  installed. All model behavior in tests is mocked. Real extraction quality, rejection rates,
  consolidation on paraphrased repeats and summary usefulness with a Qwen 14B-class model are
  unmeasured. Performance (stage timings, GPU memory) has not been measured.
- `uv.lock` resolves all extras, including `inference`. The venv was installed with pip from
  `pyproject.toml`, not from `uv.lock`; `requirements-dev.lock` remains the tested dev set.
- The LM Studio client was checked live against Bionic 1.1.10 on 2026-11-05 (port 1235): model
  listing, local-format filtering, one `json_schema` request to Qwen3.5 27B and the `ttl`
  unload (see VALIDATION.md). Still unverified: a real meeting transcript (only short synthetic
  prompts were sent), embeddings through Bionic, how LM Link remote models appear in
  `/api/v0/models`, and the Qwen3 14B model. The `ttl` applies only to a model the server loads
  on demand; an instance that is already loaded keeps its own timer (3600 s was observed).
  The loaded-context check cannot run before an on-demand load.
- Bionic's server config has `logSensitiveData` and CORS on: prompts (meeting transcripts) are
  written to its logs and local web pages can call it. Both should be turned off in Bionic.
- Grounding checks catch fabricated quotes, names, numbers and dates, not misclassification of
  real quotes or lowercase paraphrase. Sentence-start exemptions are English-only.
- Diarization OOM recovery is tested with a fake pipeline only; pyannote 4 batch-size attribute
  names are taken from its API, not verified against an installed version.
- Recovery assumes a single host and a local filesystem with working `flock`.
- Cancellation cannot interrupt a single native call (one transcription window, one LLM request,
  whole-file diarization).
- No retention policy or deletion; no encryption at rest.
- Semantic search tested only with a deterministic fake embedder; no real embedding model
  has been run. Network observation covers ingest, a preflight-failing run and serve on this
  host, not real inference.
- The web UI was exercised with a real server and curl, and with TestClient; it has not been
  checked visually in a browser or for accessibility.
- Earlier unverified items remain: GPU inference, offline namespace operation, native egress.

## Next (only on explicit request)

Merge PR #4 when ready. Turn off Bionic's prompt logging and CORS, and set
`llm_backend = "lmstudio"`, `ollama_url = "http://127.0.0.1:1235"` and an installed model.
Prepare the approved speech models and run the opt-in integration test on consented audio to
obtain real stage timings and review rejection counts, observing that run with
`LLM_PORT=1235 scripts/observe_network.sh`. Free GPU memory first: a Qwen3.5 27B Q4_K_S
instance holds about 17.7 GB of the 24 GB, and a model loaded by hand stays loaded.
