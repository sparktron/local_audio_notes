"""SQLite FTS5 search plus optional local-embedding similarity. No search or vector server.

User text is never passed to FTS5 as query syntax: it is split into words and each word is
quoted, so operators, column filters and parentheses are matched literally (or dropped).
"""

import math
import re
import sqlite3
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError
from local_audio_notes.schemas import EvidenceItem, Transcript
from local_audio_notes.speakers.mapping import SpeakerView, load_view
from local_audio_notes.storage.db import transaction

# Private-use markers around matches; stripped from indexed text so content cannot forge them.
MARK_START, MARK_END = "", ""
EMBED_VERSION = "embed-v1"
EMBED_BATCH = 8  # keeps each /api/embed response under the client's 2 MiB cap
EMBED_TEXT_LIMIT = 4000
_WORD = re.compile(r"[^\W_]+")


class Embedder(Protocol):
    def require_local_model(self, name: str | None = None) -> str: ...

    def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...


@dataclass
class Hit:
    kind: str  # meeting | item | transcript | semantic
    meeting_id: str
    title: str | None
    meeting_date: str | None
    ref: str | None = None  # segment or item ID
    start: float | None = None
    speaker: str | None = None
    category: str | None = None
    snippet: str = ""
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(text: str) -> str:
    return text.replace(MARK_START, "").replace(MARK_END, "")


def fts_query(text: str) -> str:
    words = _WORD.findall(text)[:16]
    if not words:
        raise NotesError("Search needs at least one word")
    return " ".join(f'"{w}"' for w in words)


def index_segments(connection: sqlite3.Connection, meeting_id: str, transcript: Transcript) -> None:
    """Caller runs this inside its transaction."""
    connection.execute("DELETE FROM segment_search WHERE meeting_id = ?", (meeting_id,))
    connection.executemany(
        "INSERT INTO segment_search (text, meeting_id, segment_id) VALUES (?, ?, ?)",
        [(_clean(s.text), meeting_id, s.id) for s in transcript.segments],
    )


def index_items(connection: sqlite3.Connection, meeting_id: str, items: list[EvidenceItem]) -> None:
    """Caller runs this inside its transaction."""
    connection.execute("DELETE FROM item_search WHERE meeting_id = ?", (meeting_id,))
    connection.executemany(
        "INSERT INTO item_search (text, meeting_id, item_id, category) VALUES (?, ?, ?, ?)",
        [
            (
                _clean(" ".join([i.statement or "", *(e.quote for e in i.evidence)])),
                meeting_id,
                i.id,
                i.category,
            )
            for i in items
        ],
    )


def _normalized(vector: list[float]) -> array:
    values = array("f", vector)
    norm = math.sqrt(sum(x * x for x in values)) or 1.0
    return array("f", (x / norm for x in values))


def embed_segments(
    connection: sqlite3.Connection,
    meeting_id: str,
    transcript: Transcript,
    transcript_sha: str,
    client: Embedder,
    model: str,
    model_key: str,
    check: Callable[[], None] = lambda: None,
) -> int:
    """Vectors are bound to the transcript digest; search ignores vectors whose transcript is
    no longer the meeting's verified attributed transcript (reprocessed or in progress)."""
    rows = []
    segments = transcript.segments
    for start in range(0, len(segments), EMBED_BATCH):
        check()
        batch = segments[start : start + EMBED_BATCH]
        vectors = client.embed(model, [s.text[:EMBED_TEXT_LIMIT] for s in batch])
        for segment, vector in zip(batch, vectors, strict=True):
            values = _normalized(vector)
            rows.append(
                (meeting_id, segment.id, model_key, transcript_sha, len(values), values.tobytes())
            )
    with transaction(connection):
        connection.execute("DELETE FROM segment_embeddings WHERE meeting_id = ?", (meeting_id,))
        connection.executemany("INSERT INTO segment_embeddings VALUES (?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


class _Views:
    """Effective speaker labels/names per meeting, honouring confirmed corrections."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.cache: dict[str, SpeakerView] = {}

    def get(self, meeting_id: str) -> SpeakerView:
        if meeting_id not in self.cache:
            row = self.connection.execute(
                "SELECT artifacts FROM checkpoints WHERE meeting_id = ? AND stage = 'attributing'",
                (meeting_id,),
            ).fetchone()
            sha = re.search(r'"transcript\.json":\s*"([0-9a-f]{64})"', row[0]) if row else None
            self.cache[meeting_id] = (
                load_view(self.connection, meeting_id, sha.group(1)) if sha else SpeakerView()
            )
        return self.cache[meeting_id]

    def speaker(self, meeting_id: str, segment_id: str, original: str) -> str:
        view = self.get(meeting_id)
        return view.display(view.speaker(segment_id, original))


def _date_clause(date_from: str, date_to: str) -> tuple[str, list[str]]:
    clause, params = "", []
    if date_from:
        clause += " AND m.meeting_date >= ?"
        params.append(date_from)
    if date_to:
        clause += " AND m.meeting_date <= ?"
        params.append(date_to)
    return clause, params


def search(
    connection: sqlite3.Connection,
    settings: Settings,
    text: str,
    *,
    participant: str = "",
    category: str = "",
    date_from: str = "",
    date_to: str = "",
    semantic: bool = False,
    embedder: Embedder | None = None,
    limit: int = 50,
) -> list[Hit]:
    """Meeting titles/dates, extracted items and transcript segments, best matches first.

    `participant` matches speaker labels or confirmed names (substring, case-insensitive).
    Dates filter on the user-set meeting date (YYYY-MM-DD). `category` restricts results to
    extracted items of that category.
    """
    limit = max(1, min(limit, 200))
    views = _Views(connection)
    wanted = participant.strip().casefold()
    dates, date_params = _date_clause(date_from, date_to)
    hits: list[Hit] = []

    if not category and not semantic:
        pattern = "%" + re.sub(r"([\\%_])", r"\\\1", text.strip()[:200]) + "%"
        for row in connection.execute(
            "SELECT m.id, m.title, m.meeting_date FROM meetings m "
            "WHERE (ifnull(m.title, '') LIKE ? ESCAPE '\\' OR ifnull(m.meeting_date, '') LIKE ? "
            "ESCAPE '\\')" + dates + " ORDER BY m.imported_at DESC LIMIT ?",
            (pattern, pattern, *date_params, limit),
        ):
            if not wanted:
                hits.append(Hit("meeting", row[0], row[1], row[2], snippet=row[1] or ""))

    if semantic and category:
        # Embeddings exist for transcript segments only; never return out-of-scope results.
        raise NotesError("Similar-meaning search covers transcripts; clear the category filter")
    if semantic:
        hits += _semantic(connection, settings, text, embedder, views, wanted, dates, date_params)
        return hits[:limit]

    query = fts_query(text)
    snippet = f"snippet({{table}}, 0, '{MARK_START}', '{MARK_END}', '…', 16)"
    item_rows = connection.execute(
        "SELECT i.meeting_id, i.item_id, i.category, "
        + snippet.format(table="item_search")
        + ", bm25(item_search), m.title, m.meeting_date FROM item_search i "
        "JOIN meetings m ON m.id = i.meeting_id "
        # Items only while their consolidation is current: a rerun that changed the
        # transcript drops that checkpoint, and old evidence IDs would point at new segments.
        "JOIN checkpoints c ON c.meeting_id = i.meeting_id AND c.stage = 'consolidating' "
        "WHERE item_search MATCH ?"
        + (" AND i.category = ?" if category else "")
        + dates
        + " ORDER BY bm25(item_search)",
        (query, *([category] if category else []), *date_params),
    )
    # Stream ranked candidates and filter by effective speaker before counting, so a
    # participant filter never loses matches that rank below an arbitrary row cap.
    found = 0
    for meeting_id, item_id, kind, text_snippet, score, title, meeting_date in item_rows:
        if found >= limit:
            break
        evidence = connection.execute(
            "SELECT e.segment_id, g.start_s, g.speaker FROM item_evidence e JOIN segments g "
            "ON g.meeting_id = e.meeting_id AND g.segment_id = e.segment_id "
            "WHERE e.meeting_id = ? AND e.item_id = ? ORDER BY e.position",
            (meeting_id, item_id),
        ).fetchall()
        speakers = [views.speaker(meeting_id, s, label) for s, _, label in evidence]
        if wanted and not any(wanted in s.casefold() for s in speakers):
            continue
        found += 1
        hits.append(
            Hit(
                "item",
                meeting_id,
                title,
                meeting_date,
                ref=item_id,
                start=evidence[0][1] if evidence else None,
                speaker=", ".join(dict.fromkeys(speakers)) or None,
                category=kind,
                snippet=text_snippet,
                score=-score,
            )
        )

    if not category:
        segment_rows = connection.execute(
            "SELECT s.meeting_id, s.segment_id, "
            + snippet.format(table="segment_search")
            + ", bm25(segment_search), g.start_s, g.speaker, m.title, m.meeting_date "
            "FROM segment_search s JOIN segments g ON g.meeting_id = s.meeting_id "
            "AND g.segment_id = s.segment_id JOIN meetings m ON m.id = s.meeting_id "
            "WHERE segment_search MATCH ?" + dates + " ORDER BY bm25(segment_search)",
            (query, *date_params),
        )
        found = 0
        for meeting_id, segment_id, text_snippet, score, start, label, title, mdate in segment_rows:
            if found >= limit:
                break
            speaker = views.speaker(meeting_id, segment_id, label)
            if wanted and wanted not in speaker.casefold():
                continue
            found += 1
            hits.append(
                Hit(
                    "transcript",
                    meeting_id,
                    title,
                    mdate,
                    ref=segment_id,
                    start=start,
                    speaker=speaker,
                    snippet=text_snippet,
                    score=-score,
                )
            )
    return hits[:limit]


def _semantic(
    connection: sqlite3.Connection,
    settings: Settings,
    text: str,
    embedder: Embedder | None,
    views: _Views,
    wanted: str,
    dates: str,
    date_params: list[str],
    top: int = 50,
) -> list[Hit]:
    if not settings.embedding_model or embedder is None:
        raise NotesError("Semantic search needs embedding_model set to an installed local model")
    model_key = embedder.require_local_model(settings.embedding_model)
    query = _normalized(embedder.embed(settings.embedding_model, [text[:EMBED_TEXT_LIMIT]])[0])
    rows = connection.execute(
        "SELECT e.meeting_id, e.segment_id, e.vector, g.start_s, g.speaker, m.title, "
        "m.meeting_date FROM segment_embeddings e JOIN segments g ON g.meeting_id = e.meeting_id "
        "AND g.segment_id = e.segment_id JOIN meetings m ON m.id = e.meeting_id "
        # Only vectors computed from the meeting's current verified transcript.
        "JOIN checkpoints c ON c.meeting_id = e.meeting_id AND c.stage = 'attributing' "
        "AND json_extract(c.artifacts, '$.\"transcript.json\"') = e.transcript_sha256 "
        "WHERE e.model = ? AND e.dims = ?" + dates,
        (model_key, len(query), *date_params),
    ).fetchall()
    if not rows:
        raise NotesError("No embeddings for the current embedding model; process or reindex first")
    scores = _similarities(query, [r[2] for r in rows])
    ranked = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    hits = []
    for index in ranked:
        meeting_id, segment_id, _, start, label, title, mdate = rows[index]
        speaker = views.speaker(meeting_id, segment_id, label)
        if wanted and wanted not in speaker.casefold():
            continue
        hits.append(
            Hit(
                "semantic",
                meeting_id,
                title,
                mdate,
                ref=segment_id,
                start=start,
                speaker=speaker,
                score=round(float(scores[index]), 4),
            )
        )
        if len(hits) >= top:
            break
    # One scan of the FTS content for all snippets (UNINDEXED columns have no index).
    keys = {(h.meeting_id, h.ref) for h in hits}
    marks = ",".join("?" * len(keys))
    texts = (
        {
            (m, s): t
            for m, s, t in connection.execute(
                "SELECT meeting_id, segment_id, text FROM segment_search "
                f"WHERE meeting_id || ' ' || segment_id IN ({marks})",
                [f"{m} {s}" for m, s in keys],
            )
        }
        if keys
        else {}
    )
    for hit in hits:
        hit.snippet = texts.get((hit.meeting_id, hit.ref), "")
    return hits


def _similarities(query: array, blobs: list[bytes]) -> list[float]:
    """Cosine similarity of unit vectors. Uses numpy when installed (inference extra)."""
    try:
        import numpy as np
    except ImportError:
        result = []
        for blob in blobs:
            vector = array("f")
            vector.frombytes(blob)
            result.append(sum(a * b for a, b in zip(query, vector, strict=True)))
        return result
    matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(blobs), len(query))
    return (matrix @ np.frombuffer(query.tobytes(), dtype=np.float32)).tolist()
