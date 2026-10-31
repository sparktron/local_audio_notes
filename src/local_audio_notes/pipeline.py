"""Checkpointed stage pipeline.

A stage fingerprint covers its code version, the settings that change its output, model
identities and the SHA-256 of its input artifacts. A checkpoint is reused only when the
fingerprint matches and every recorded artifact still has its recorded hash. A rerun stage
whose output changes changes downstream fingerprints, so dependent stages rerun too.
Performance knobs (batch size, threads, device) do not invalidate checkpoints; use force.
"""

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_audio_notes.alignment.speakers import attribute
from local_audio_notes.audio.media import normalize
from local_audio_notes.config import Settings
from local_audio_notes.diarization.community import diarize
from local_audio_notes.errors import Cancelled, Interrupted, NotesError
from local_audio_notes.extraction.consolidate import CONSOLIDATION_VERSION, consolidate
from local_audio_notes.extraction.evidence import PROMPT_VERSION, extract
from local_audio_notes.ingestion.transcripts import parse_transcript
from local_audio_notes.llm.ollama import OllamaClient
from local_audio_notes.models import model_fingerprints, verify_models
from local_audio_notes.notes.render import export_json, render_markdown
from local_audio_notes.notes.summary import SUMMARY_VERSION, summarize
from local_audio_notes.schemas import (
    ConsolidatedEvidence,
    EvidenceItem,
    ExtractionOutcome,
    SpeakerTurn,
    SummaryRecord,
    Transcript,
)
from local_audio_notes.search.index import (
    EMBED_VERSION,
    embed_segments,
    index_items,
    index_segments,
)
from local_audio_notes.speakers.mapping import load_view, transcript_labels
from local_audio_notes.storage.db import now, transaction
from local_audio_notes.storage.files import HashCache, atomic_write, meeting_dir, write_json
from local_audio_notes.storage.locking import try_lock
from local_audio_notes.transcription.whisperx_backend import WhisperXBackend

STAGES = (
    "normalizing",
    "transcribing",
    "aligning",
    "diarizing",
    "attributing",
    "extracting",
    "consolidating",
    "summarizing",
    "exporting",
)
# Direct consumers of each stage's artifacts.
CONSUMERS = {
    "normalizing": ("transcribing", "aligning", "diarizing"),
    "transcribing": ("aligning",),
    "aligning": ("attributing",),
    "diarizing": ("attributing",),
    "attributing": ("extracting", "indexing"),
    "extracting": ("consolidating",),
    "consolidating": ("summarizing",),
    "summarizing": ("exporting",),
}


def dependents(stage: str) -> set[str]:
    found: set[str] = set()
    pending = list(CONSUMERS.get(stage, ()))
    while pending:
        current = pending.pop()
        if current not in found:
            found.add(current)
            pending.extend(CONSUMERS.get(current, ()))
    return found


VERSIONS = {
    "normalizing": "ffmpeg-pcm16k-mono-v1",
    "transcribing": "whisperx-windows-v1",
    "aligning": "ctc-per-segment-v1",
    "diarizing": "community1-regular-v1",
    "attributing": "majority-coverage-v1",
}


def digest(*parts: Any) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def get_meeting(connection: sqlite3.Connection, identifier: str) -> dict[str, Any]:
    row = connection.execute("SELECT * FROM meetings WHERE id = ?", (identifier,)).fetchone()
    if row is None:
        raise NotesError("Meeting not found")
    return dict(row)


def verify_originals(
    directory: Path, meeting: Mapping[str, Any], hashes: HashCache | None = None
) -> None:
    hashes = hashes or HashCache(None)
    pairs = [("original.audio", meeting["audio_sha256"], "Original recording")]
    if meeting["attachment_sha256"]:
        pairs.append(("attached.transcript", meeting["attachment_sha256"], "Attached transcript"))
    for name, expected, label in pairs:
        path = directory / name
        if path.is_symlink() or not path.is_file() or hashes.sha256(path) != expected:
            raise NotesError(f"{label} integrity verification failed")


def verified_checkpoint(
    connection: sqlite3.Connection,
    meeting_id: str,
    directory: Path,
    stage: str,
    fingerprint: str | None = None,
    hashes: HashCache | None = None,
) -> dict[str, str] | None:
    hashes = hashes or HashCache(None)
    row = connection.execute(
        "SELECT fingerprint, artifacts FROM checkpoints WHERE meeting_id = ? AND stage = ?",
        (meeting_id, stage),
    ).fetchone()
    if row is None or (fingerprint is not None and row[0] != fingerprint):
        return None
    artifacts: dict[str, str] = json.loads(row[1])
    for name, expected in artifacts.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or hashes.sha256(path) != expected:
            return None
    return artifacts


class ChunkCache:
    """Schema-valid raw LLM responses per chunk, so a failed run resumes mid-extraction."""

    def __init__(self, connection: sqlite3.Connection, meeting_id: str, refresh: bool):
        self.connection, self.meeting_id, self.refresh = connection, meeting_id, refresh

    def get(self, key: str) -> str | None:
        if self.refresh:
            return None
        row = self.connection.execute(
            "SELECT result FROM extraction_chunks WHERE meeting_id = ? AND cache_key = ?",
            (self.meeting_id, key),
        ).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO extraction_chunks VALUES (?, ?, ?, ?)",
            (self.meeting_id, key, value, now()),
        )


@dataclass
class Run:
    settings: Settings
    connection: sqlite3.Connection
    job_id: int
    meeting: dict[str, Any]
    directory: Path
    force: bool = False
    check: Callable[[], None] = lambda: None
    outputs: dict[str, dict[str, str]] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    hashes: HashCache = field(default_factory=lambda: HashCache(None))
    _backend: WhisperXBackend | None = None

    def artifact(self, name: str) -> Path:
        return self.directory / name

    def backend(self) -> WhisperXBackend:
        if self._backend is None:
            self._backend = WhisperXBackend(self.settings, self.paths)
        return self._backend

    def stage(
        self,
        name: str,
        fingerprint: str | None,
        produce: Callable[[], tuple[list[str], dict[str, Any]]],
    ) -> None:
        """Run or reuse one stage, recording a non-sensitive stage_runs row either way."""
        self.connection.execute("UPDATE jobs SET stage = ? WHERE id = ?", (name, self.job_id))
        run_id = self.connection.execute(
            "INSERT INTO stage_runs (job_id, stage, state, started_at) VALUES (?, ?, 'running', ?)",
            (self.job_id, name, now()),
        ).lastrowid
        started = time.monotonic()
        state, detail, error_class = "succeeded", {}, None
        try:
            self.check()
            reused = None
            if not self.force and fingerprint is not None:
                reused = verified_checkpoint(
                    self.connection,
                    self.meeting["id"],
                    self.directory,
                    name,
                    fingerprint,
                    self.hashes,
                )
            if reused is not None:
                state, self.outputs[name] = "skipped", reused
                return
            previous = self.connection.execute(
                "SELECT artifacts FROM checkpoints WHERE meeting_id = ? AND stage = ?",
                (self.meeting["id"], name),
            ).fetchone()
            # Drop the old checkpoint first: a crash mid-stage must not leave it trusted.
            self.connection.execute(
                "DELETE FROM checkpoints WHERE meeting_id = ? AND stage = ?",
                (self.meeting["id"], name),
            )
            names, detail = produce()
            artifacts = {n: self.hashes.sha256(self.artifact(n)) for n in names}
            if previous is None or json.loads(previous[0]) != artifacts:
                # Changed output: every downstream checkpoint was built from the old one and
                # must not be reused or exported alongside it (e.g. new transcript, old notes).
                for stale in sorted(dependents(name)):
                    self.connection.execute(
                        "DELETE FROM checkpoints WHERE meeting_id = ? AND stage = ?",
                        (self.meeting["id"], stale),
                    )
            self.connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                (self.meeting["id"], name, fingerprint or "", json.dumps(artifacts), now()),
            )
            self.outputs[name] = artifacts
        except BaseException as error:
            state = (
                "cancelled"
                if isinstance(error, Cancelled)
                else "interrupted"
                if isinstance(error, KeyboardInterrupt | Interrupted)
                else "failed"
            )
            error_class = type(error).__name__
            raise
        finally:
            self.connection.execute(
                "UPDATE stage_runs SET state = ?, finished_at = ?, seconds = ?, error_class = ?, "
                "detail = ? WHERE id = ?",
                (
                    state,
                    now(),
                    round(time.monotonic() - started, 3),
                    error_class,
                    json.dumps(detail) if detail else None,
                    run_id,
                ),
            )

    def read(self, stage: str, name: str) -> str:
        return self.artifact(name).read_text(encoding="utf-8")


def store_segments(connection: sqlite3.Connection, meeting_id: str, transcript: Transcript) -> None:
    with transaction(connection):
        connection.execute("DELETE FROM segments WHERE meeting_id = ?", (meeting_id,))
        connection.executemany(
            "INSERT INTO segments VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (meeting_id, s.id, i, s.start, s.end, s.speaker, int(s.overlap))
                for i, s in enumerate(transcript.segments)
            ],
        )
        connection.execute("DELETE FROM speaker_labels WHERE meeting_id = ?", (meeting_id,))
        connection.executemany(
            "INSERT INTO speaker_labels VALUES (?, ?)",
            [(meeting_id, label) for label in transcript_labels(transcript)],
        )
        index_segments(connection, meeting_id, transcript)


def store_items(connection: sqlite3.Connection, meeting_id: str, items: list[EvidenceItem]) -> None:
    with transaction(connection):
        connection.execute("DELETE FROM items WHERE meeting_id = ?", (meeting_id,))
        for item in items:
            connection.execute(
                "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    meeting_id,
                    item.id,
                    item.category,
                    item.statement,
                    item.basis,
                    item.owner,
                    item.deadline,
                    int(item.disputed),
                ),
            )
            connection.executemany(
                "INSERT INTO item_evidence VALUES (?, ?, ?, ?, ?)",
                [
                    (meeting_id, item.id, n, e.segment_id, e.quote)
                    for n, e in enumerate(item.evidence)
                ],
            )
        index_items(connection, meeting_id, items)


def export_notes(settings: Settings, connection: sqlite3.Connection, meeting_id: str) -> list[str]:
    """Render notes.md/notes.json from verified artifacts and current speaker mappings.

    Callers outside a job must hold the meeting's processing lock.
    """
    meeting = get_meeting(connection, meeting_id)
    directory = meeting_dir(settings.data_dir, meeting_id)
    found = {
        stage: verified_checkpoint(connection, meeting_id, directory, stage)
        for stage in ("attributing", "consolidating", "summarizing")
    }
    if any(v is None for v in found.values()):
        raise NotesError("Notes inputs missing or failed integrity verification; process first")
    transcript = Transcript.model_validate_json((directory / "transcript.json").read_text())
    evidence = ConsolidatedEvidence.model_validate_json((directory / "evidence.json").read_text())
    summary = SummaryRecord.model_validate_json((directory / "summary.json").read_text())
    view = load_view(connection, meeting_id, found["attributing"]["transcript.json"])
    atomic_write(
        directory / "notes.md",
        render_markdown(meeting, transcript, evidence.items, summary, evidence.rejections, view),
    )
    generated_by = {**evidence.generated_by, "summary_version": SUMMARY_VERSION}
    write_json(
        directory / "notes.json",
        export_json(
            meeting, transcript, evidence.items, summary, evidence.rejections, view, generated_by
        ),
    )
    return ["notes.md", "notes.json"]


def _audio_stages(run: Run, mode: str) -> None:
    meeting, settings, models = run.meeting, run.settings, run.models
    normalized = run.artifact("normalized.wav")

    def normalizing() -> tuple[list[str], dict[str, Any]]:
        normalized.unlink(missing_ok=True)
        normalize(run.artifact("original.audio"), normalized, settings)
        return ["normalized.wav"], {}

    run.stage(
        "normalizing",
        digest(VERSIONS["normalizing"], meeting["audio_sha256"], settings.max_audio_seconds),
        normalizing,
    )

    def transcribing() -> tuple[list[str], dict[str, Any]]:
        provided = None
        if mode == "align" or (mode == "compare" and meeting["attachment_kind"] != ".txt"):
            provided = parse_transcript(
                run.artifact("attached.transcript"),
                meeting["attachment_kind"],
                meeting["duration"],
                settings.alignment_language,
            )
        transcript = (
            provided if mode == "align" else run.backend().transcribe(normalized, run.check)
        )
        if not transcript or not transcript.segments:
            raise NotesError("No speech was transcribed; notes were not generated")
        write_json(run.artifact("verbatim.json"), transcript.model_dump(mode="json"))
        names = ["verbatim.json"]
        if mode == "compare":
            # Exact comparison only; no quality metric or assumed word-error-rate reference.
            original = run.artifact("attached.transcript").read_text(encoding="utf-8-sig")
            if provided:
                original = " ".join(s.text for s in provided.segments)
            generated = " ".join(s.text for s in transcript.segments)
            write_json(
                run.artifact("comparison.json"),
                {
                    "method": "exact-text",
                    "identical": original.strip() == generated.strip(),
                    "provided_text": original,
                    "local_text": generated,
                },
            )
            names.append("comparison.json")
        return names, {"segments": len(transcript.segments), "language": transcript.language}

    run.stage(
        "transcribing",
        digest(
            VERSIONS["transcribing"],
            mode,
            meeting["attachment_sha256"],
            run.outputs["normalizing"],
            models["whisper"],
            models["vad"],
            settings.language,
            settings.alignment_language,
            settings.audio_window_seconds,
        ),
        transcribing,
    )

    def aligning() -> tuple[list[str], dict[str, Any]]:
        verbatim = Transcript.model_validate_json(run.read("transcribing", "verbatim.json"))
        aligned = run.backend().align(normalized, verbatim)
        write_json(run.artifact("aligned.json"), aligned.model_dump(mode="json"))
        timed = sum(w.start is not None for s in aligned.segments for w in s.words)
        return ["aligned.json"], {"timed_words": timed}

    run.stage(
        "aligning",
        digest(
            VERSIONS["aligning"],
            run.outputs["transcribing"]["verbatim.json"],
            run.outputs["normalizing"],
            models["alignment"],
            models["tokenizer"],
            settings.alignment_language,
        ),
        aligning,
    )

    def diarizing() -> tuple[list[str], dict[str, Any]]:
        turns = diarize(normalized, run.paths["diarization"], settings, run.backend().device)
        write_json(run.artifact("diarization.json"), [t.model_dump(mode="json") for t in turns])
        return ["diarization.json"], {"speakers": len({t.speaker for t in turns})}

    run.stage(
        "diarizing",
        digest(
            VERSIONS["diarizing"],
            run.outputs["normalizing"],
            models["diarization"],
            settings.min_speakers,
            settings.max_speakers,
        ),
        diarizing,
    )

    def attributing() -> tuple[list[str], dict[str, Any]]:
        aligned = Transcript.model_validate_json(run.read("aligning", "aligned.json"))
        raw = json.loads(run.read("diarizing", "diarization.json"))
        transcript = attribute(aligned, [SpeakerTurn.model_validate(t) for t in raw])
        write_json(run.artifact("transcript.json"), transcript.model_dump(mode="json"))
        return ["transcript.json"], {}

    run.stage(
        "attributing",
        digest(VERSIONS["attributing"], run.outputs["aligning"], run.outputs["diarizing"]),
        attributing,
    )


def _notes_stages(
    run: Run, client: OllamaClient, model_key: str, embed_key: str | None = None
) -> None:
    settings, meeting_id = run.settings, run.meeting["id"]
    transcript_sha = run.outputs["attributing"]["transcript.json"]
    transcript = Transcript.model_validate_json(run.read("attributing", "transcript.json"))
    store_segments(run.connection, meeting_id, transcript)

    def extracting() -> tuple[list[str], dict[str, Any]]:
        cache = ChunkCache(run.connection, meeting_id, refresh=run.force)
        outcome = extract(transcript, client, settings, model_key, cache, run.check)
        write_json(run.artifact("extraction.json"), outcome.model_dump(mode="json"))
        return ["extraction.json"], {
            "chunks": outcome.chunks,
            "cached_chunks": outcome.cached_chunks,
            "items": len(outcome.items),
            "rejected_or_stripped": sum(outcome.rejections.values()),
        }

    run.stage(
        "extracting",
        digest(
            PROMPT_VERSION,
            transcript_sha,
            model_key,
            settings.chunk_chars,
            settings.chunk_overlap_segments,
            settings.ollama_context,
            settings.ollama_predict,
        ),
        extracting,
    )

    def consolidating() -> tuple[list[str], dict[str, Any]]:
        outcome = ExtractionOutcome.model_validate_json(run.read("extracting", "extraction.json"))
        items = consolidate(outcome.items, transcript.segments)
        evidence = ConsolidatedEvidence(
            items=items,
            rejections=outcome.rejections,
            generated_by={
                "model": model_key,
                "prompt_version": PROMPT_VERSION,
                "consolidation_version": CONSOLIDATION_VERSION,
            },
        )
        write_json(run.artifact("evidence.json"), evidence.model_dump(mode="json"))
        store_items(run.connection, meeting_id, items)
        return ["evidence.json"], {"items": len(items)}

    run.stage(
        "consolidating",
        digest(CONSOLIDATION_VERSION, run.outputs["extracting"]),
        consolidating,
    )

    def summarizing() -> tuple[list[str], dict[str, Any]]:
        evidence = ConsolidatedEvidence.model_validate_json(
            run.read("consolidating", "evidence.json")
        )
        summary = summarize(evidence.items, client, settings, run.check)
        write_json(run.artifact("summary.json"), summary.model_dump(mode="json"))
        return ["summary.json"], {"points": len(summary.points), "omitted": summary.omitted_items}

    run.stage(
        "summarizing",
        digest(
            SUMMARY_VERSION,
            run.outputs["consolidating"],
            model_key,
            settings.ollama_context,
            settings.ollama_predict,
        ),
        summarizing,
    )
    # Always re-rendered: titles and confirmed speaker names can change without reprocessing.
    run.stage("exporting", None, lambda: (export_notes(settings, run.connection, meeting_id), {}))
    if embed_key is None:
        return

    def indexing() -> tuple[list[str], dict[str, Any]]:
        count = embed_segments(
            run.connection,
            meeting_id,
            transcript,
            client,
            settings.embedding_model,
            embed_key,
            run.check,
        )
        return [], {"vectors": count}

    # Optional semantic index after the notes exist, so an embedding failure never costs notes.
    run.stage("indexing", digest(EMBED_VERSION, transcript_sha, embed_key), indexing)


def run_job(
    settings: Settings,
    connection: sqlite3.Connection,
    job: Mapping[str, Any],
    check: Callable[[], None] = lambda: None,
) -> None:
    """Execute one claimed job. The caller holds the meeting's processing lock."""
    meeting = get_meeting(connection, job["meeting_id"])
    directory = meeting_dir(settings.data_dir, meeting["id"])
    if job["mode"] in {"align", "compare"} and not meeting["attachment_kind"]:
        raise NotesError("This mode requires an attached transcript")
    hashes = HashCache(settings.data_dir / ".hash-cache.json" if settings.hash_cache else None)
    try:
        verify_originals(directory, meeting, hashes)
        # Check local models before spending hours on audio stages.
        client = OllamaClient(settings)
        model_key = client.require_local_model()
        embed_key = (
            client.require_local_model(settings.embedding_model)
            if settings.embedding_model
            else None
        )
        run = Run(settings, connection, job["id"], meeting, directory, bool(job["force"]), check)
        run.hashes = hashes
        if job["kind"] == "full":
            run.paths = verify_models(settings.models_dir, cache=settings.hash_cache)
            run.models = model_fingerprints(settings.models_dir)
            _audio_stages(run, job["mode"])
        else:
            found = verified_checkpoint(
                connection, meeting["id"], directory, "attributing", hashes=hashes
            )
            if found is None:
                raise NotesError(
                    "Attributed transcript missing or failed integrity verification; "
                    "run full processing"
                )
            run.outputs["attributing"] = found
        _notes_stages(run, client, model_key, embed_key)
    finally:
        hashes.save()


def reindex(
    settings: Settings,
    connection: sqlite3.Connection,
    embedder: OllamaClient | None = None,
) -> dict[str, int]:
    """Rebuild search indexes from verified artifacts (and embeddings when an embedder is
    given). Meetings with a running job are skipped, never waited on."""
    counts = {"indexed": 0, "embedded": 0, "busy": 0, "unprocessed": 0}
    embed_key = (
        embedder.require_local_model(settings.embedding_model)
        if embedder and settings.embedding_model
        else None
    )
    meetings = [r[0] for r in connection.execute("SELECT id FROM meetings ORDER BY imported_at")]
    for meeting_id in meetings:
        directory = meeting_dir(settings.data_dir, meeting_id)
        with try_lock(directory / ".processing.lock") as acquired:
            if not acquired:
                counts["busy"] += 1
                continue
            found = verified_checkpoint(connection, meeting_id, directory, "attributing")
            if found is None:
                counts["unprocessed"] += 1
                continue
            transcript = Transcript.model_validate_json((directory / "transcript.json").read_text())
            store_segments(connection, meeting_id, transcript)
            if verified_checkpoint(connection, meeting_id, directory, "consolidating"):
                evidence = ConsolidatedEvidence.model_validate_json(
                    (directory / "evidence.json").read_text()
                )
                store_items(connection, meeting_id, evidence.items)
            counts["indexed"] += 1
            if embed_key and embedder:
                embed_segments(
                    connection,
                    meeting_id,
                    transcript,
                    embedder,
                    settings.embedding_model,
                    embed_key,
                )
                counts["embedded"] += 1
    return counts
