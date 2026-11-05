# ADR-0004: SQLite search, optional local embeddings, stat-keyed hash cache

- **Date:** 2026-10-31
- **Status:** accepted for Phase 4

## Context

Phase 4 asks for search (title, date, transcript text, participants, action items, semantic
similarity) that links to the meeting and timestamp, without a vector database server unless
SQLite proves inadequate; plus security tests, offline validation and performance work.

## Decisions

1. **FTS5 in the existing database.** `segment_search` and `item_search` (unicode61,
   diacritics removed) are filled whenever segments or consolidated items are stored, and
   rebuilt by `reindex` from hash-verified artifacts. The database therefore now holds
   transcript text. User input never reaches FTS5 syntax: it is reduced to at most 16 words,
   each quoted, so operators and column filters are matched literally.
2. **Participants resolve through confirmed mappings.** Filters match the effective speaker
   label or confirmed name of each segment, including segment corrections, per meeting. The
   filter is applied while streaming ranked matches, before counting toward the limit.
3. **Embeddings are optional, local and off by default.** `embedding_model` must name an
   already installed Ollama model; it is verified like the notes model and called only on
   127.0.0.1. Unit-normalized float32 vectors live in SQLite keyed by model name and digest
   and bound to the transcript digest they were computed from; search uses only vectors of
   the meeting's current verified transcript, so a changed model or a reprocessed transcript
   never mixes with old vectors. Semantic search covers transcript segments only and rejects
   a category filter rather than ignoring it. Similarity is brute-force cosine (numpy
   when installed, pure Python otherwise). The embedding stage runs after notes are exported,
   so its failure never costs the notes.
4. **Hash cache keyed by device, inode, size, mtime and ctime.** Model weights and recordings
   are not reread while those are unchanged. ctime cannot be set by ordinary processes, so
   writes, renames and chmods invalidate entries. `hash_cache = false` and
   `models-check --full` restore full hashing.
5. **Offline evidence by observation.** `scripts/observe_network.sh` records the network
   system calls of a real command and its children with strace and fails on any IP endpoint
   other than loopback Ollama or a port the command bound. Tests add Python audit-hook
   recording across a complete mocked run, web requests and search.

## Measurements (this workstation, synthetic data; `scripts/benchmark_local.py`)

| Measurement | Result |
| --- | --- |
| SHA-256, 1 GiB, warm page cache | 2.6 s (391 MiB/s); cache hit ~3 µs |
| FTS5 build, 200,000 segments | 0.82 s |
| FTS5 query, 200,000 segments | 38 ms median |
| Cosine search, 100,000 × 768 (numpy) | 0.20 s |

At these sizes (roughly 200 hours of meetings) SQLite is adequate; no vector server is needed.

## Alternatives considered

- External search or vector servers: another network service to secure, for no measured need.
- Passing FTS5 syntax through: lets crafted input change query semantics and raise errors.
- mtime-only cache keys: mtime can be set with `touch -d`; ctime cannot.

## Consequences

The database is as sensitive as the transcripts: encrypt the volume. Timestamps have
kernel-tick granularity, so a same-size rewrite within one tick of hashing could be missed
by the cache (shown in tests); full verification remains available. Brute-force similarity
grows linearly; revisit past a few million segments. strace observation covers the runs
performed, not every code path, and not the Ollama server process.
