# ADR-0001: Use extractive draft notes in Phase 1

- **Date:** 2026-10-30
- **Status:** accepted for the Phase 1 scaffold; extraction design superseded by ADR-0002 (grounding rules retained)
- **Deciders:** implementation choice under the user's privacy and Phase 1 scope

## Context

The user prioritizes privacy and prohibits invented decisions, owners and deadlines.
Full evidence consolidation belongs to Phase 2. A free-form first-pass summary would
introduce unsupported paraphrases before that validation system exists.

## Decision

Use local Ollama to classify exact transcript excerpts against a Pydantic JSON schema.
Verify each excerpt against its cited segment, and require optional owner/deadline text
to appear in that excerpt. Derive timestamps in code and render Markdown deterministically.
Keep proposals separate, label the notes as drafts and require human review.

## Alternatives considered

- Free-form model-written Markdown: harder to validate source support in this phase.
- Implement the full two-stage consolidation system now: exceeds the explicitly requested
  Phase 1 stopping point.
- Omit notes: fails the requested Phase 1 end-to-end pipeline.

## Consequences

This makes source attribution inspectable and keeps raw transcripts independent. Notes
are more repetitive and less polished. Exact quote checks do not prove semantic category
correctness; a proposal can still be misclassified, so this is not a hallucination-proof
system. Exact deduplication retains distinct commitments but can leave paraphrased duplicates.

## Revisit if

Phase 2 is explicitly authorized and a reviewed evidence/consolidation system has real
local-model validation against consented, annotated recordings.
