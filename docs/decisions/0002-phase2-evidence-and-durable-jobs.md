# ADR-0002: Two-stage grounded evidence, fingerprinted checkpoints, flock-based recovery

- **Date:** 2026-10-31
- **Status:** accepted for Phase 2
- **Supersedes:** ADR-0001's single extractive classification pass (its grounding rules remain)

## Context

Phase 2 requires SQLite persistence, restart recovery, structured evidence extraction with
source references, and validation tests. Priorities are unchanged: nothing leaves the host,
no invented owners/deadlines/decisions, and a failed LLM stage must not force retranscription.

## Decisions

1. **SQLite holds metadata and state, not transcript text.** Meetings, jobs, stage runs,
   checkpoints, segment references (IDs, times, speakers), validated extracted items with
   their quotes, and speaker mappings. Full transcripts stay in integrity-checked JSON
   artifacts. Forward-only migrations in code (`PRAGMA user_version`); a newer schema refuses
   to run. `meeting.json` becomes an immutable import record used to adopt orphaned imports.
2. **Stage checkpoints keyed by input fingerprints.** Each stage's fingerprint covers its code
   version, output-relevant settings, model identity (manifest revision + file hashes) and the
   SHA-256 of its input artifacts. Reuse requires a matching fingerprint and unchanged output
   hashes. Changing an output changes downstream fingerprints. Performance knobs are excluded.
3. **Liveness by kernel flock, not heartbeats.** A worker holds the meeting's processing lock
   from claim until it records the outcome. The kernel drops it on any process death, so
   recovery treats a `running` job whose lock it can take as abandoned and requeues it up to
   `job_max_attempts`. Deterministic failures are never retried automatically.
4. **Two LLM stages.** Stage 1 extracts items per bounded, overlapping chunk: category,
   statement, explicit/inferred basis, 1–4 verbatim quotes with segment IDs, optional
   owner/deadline. Validation is per item: unknown segments, absent quotes and inferred
   decisions/actions/follow-ups reject the item; unsupported, pronoun or misplaced owners and
   deadlines are stripped; statements introducing capitalized words or numbers absent from the
   quotes are dropped in favour of the quote. Only reason counts are recorded. Raw chunk
   responses are cached after schema validation and re-validated on reuse. Consolidation merges
   items whose quotes overlap within a segment, or match exactly (4+ words) across segments,
   when owners/deadlines do not conflict; conflicting decision/proposal or action/proposal
   classifications of one excerpt are marked disputed. Stage 2 writes a cited summary from
   evidence only; points with unknown citations, ungrounded entities or decision wording
   without a cited decision are dropped. All other note sections render deterministically.
5. **Speaker names are display mappings bound to one attributed transcript.** They never
   become owners. A re-diarized transcript leaves earlier mappings stale and unapplied.

## Alternatives considered

- Heartbeat leases: need a timer thread and clock tuning, and still guess at liveness while
  native code blocks. The flock is exact on a single host. Multi-host workers are out of scope.
- Free-form Stage 2 Markdown: cannot be checked; only the summary bullets are model-written.
- Failing a whole chunk on one bad item (Phase 1): one misquote from a 14B model would block
  notes for a meeting; per-item rejection with counts keeps the guarantee and the useful items.
- Storing transcript text in SQLite now: duplicates sensitive text before search (Phase 4)
  needs it.

## Consequences

Grounding checks reject fabricated quotes, names, numbers and dates, not wrong meaning: a real
quote can still be misclassified, and lowercase paraphrase is unchecked (the quote is always
shown in source references). The sentence-start word exemption list is English-only.
Consolidation is lexical, so paraphrased repeats of one commitment can remain separate.
Cancellation is checked between stages, transcription windows and LLM chunks, not inside a
single native call. Recovery assumes one host and a local filesystem with working flock.

## Revisit if

Real local-model runs on consented, annotated meetings show high stripped/rejected counts,
missed duplicates, or misclassification rates that the review section does not surface.
