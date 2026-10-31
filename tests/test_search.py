import contextlib
import json
import re

import pytest
from fastapi.testclient import TestClient

from local_audio_notes.api import app as web
from local_audio_notes.cli import main as cli
from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.jobs.queue import process
from local_audio_notes.pipeline import reindex
from local_audio_notes.search.index import MARK_END, MARK_START, fts_query, search
from local_audio_notes.speakers import mapping
from local_audio_notes.storage.db import connect, transaction


@pytest.fixture
def indexed(audio, settings, mocked):
    identifier = ingest(audio, settings)
    process(settings, identifier)
    with contextlib.closing(connect(settings.data_dir)) as connection:
        with transaction(connection):
            connection.execute(
                "UPDATE meetings SET title = 'Storage review', meeting_date = '2026-10-30' "
                "WHERE id = ?",
                (identifier,),
            )
        yield identifier, connection


def kinds(hits):
    return [(h.kind, h.ref) for h in hits]


def test_transcript_items_and_titles_are_searchable(indexed, settings):
    identifier, connection = indexed
    hits = search(connection, settings, "migration")
    assert ("transcript", "s000002") in kinds(hits) and ("transcript", "s000008") in kinds(hits)
    item = next(h for h in hits if h.kind == "item")
    assert item.category == "action" and item.start == 6.0 and item.meeting_id == identifier
    transcript = next(h for h in hits if h.ref == "s000002")
    assert transcript.speaker == "Speaker 3" and transcript.title == "Storage review"
    assert f"{MARK_START}migration{MARK_END}" in transcript.snippet
    assert kinds(search(connection, settings, "storage review"))[0] == ("meeting", None)


def test_diacritics_and_case_insensitive(indexed, settings):
    _, connection = indexed
    assert search(connection, settings, "MIGRÁTION")


def test_filters_category_date_and_participant(indexed, settings):
    identifier, connection = indexed
    actions = search(connection, settings, "Alice", category="action")
    assert actions and all(h.kind == "item" and h.category == "action" for h in actions)
    assert not search(connection, settings, "migration", category="decision")
    assert search(connection, settings, "migration", date_from="2026-10-30", date_to="2026-10-30")
    assert not search(connection, settings, "migration", date_from="2026-11-01")
    by_speaker = search(connection, settings, "review", participant="speaker 1")
    assert {h.ref for h in by_speaker if h.kind == "transcript"} == {"s000003"}
    # Confirmed names and corrections are searchable participants.
    sha = json.loads(
        connection.execute(
            "SELECT artifacts FROM checkpoints WHERE meeting_id = ? AND stage = 'attributing'",
            (identifier,),
        ).fetchone()[0]
    )["transcript.json"]
    mapping.set_name(connection, identifier, sha, "Speaker 1", "Bob")
    mapping.correct_segment(connection, identifier, sha, "s000006", "Speaker 1")
    by_name = search(connection, settings, "blocker", participant="bob")
    assert [h.ref for h in by_name if h.kind == "transcript"] == ["s000006"]
    assert by_name[-1].speaker == "Bob [Speaker 1]"


@pytest.mark.parametrize(
    "query",
    [
        'alice "',
        "migration OR blocker",
        "text:migration",
        "NEAR(alice bob)",
        "migr*",
        "a AND (",
        "-x",
    ],
)
def test_fts_syntax_is_treated_literally(indexed, settings, query):
    _, connection = indexed
    search(connection, settings, query)  # never an FTS5 syntax error
    assert '"OR"' in fts_query("a OR b") and ":" not in fts_query("text:migration")


def test_empty_query_rejected(indexed, settings):
    _, connection = indexed
    with pytest.raises(NotesError, match="at least one word"):
        search(connection, settings, "  ¡¿ ")


def test_index_text_cannot_forge_highlight_markers(audio, settings, mocked):
    calls, ollama = mocked
    identifier = ingest(audio, settings)
    process(settings, identifier)
    with contextlib.closing(connect(settings.data_dir)) as connection:
        text = connection.execute(
            "SELECT text FROM segment_search WHERE meeting_id = ? LIMIT 1", (identifier,)
        ).fetchone()[0]
    assert MARK_START not in text


def test_reindex_rebuilds_and_skips_busy_meetings(indexed, settings):
    identifier, connection = indexed
    with transaction(connection):
        connection.execute("DELETE FROM segment_search")
        connection.execute("DELETE FROM item_search")
    assert not search(connection, settings, "migration")
    assert reindex(settings, connection)["indexed"] == 1
    assert search(connection, settings, "migration")
    from local_audio_notes.jobs.queue import lock_path
    from local_audio_notes.storage.locking import try_lock

    with try_lock(lock_path(settings, identifier)):
        assert reindex(settings, connection)["busy"] == 1


def test_semantic_search_with_local_embeddings(audio, settings, mocked):
    calls, ollama = mocked
    settings = settings.model_copy(update={"embedding_model": "embed:latest"})
    identifier = ingest(audio, settings)
    process(settings, identifier)
    with contextlib.closing(connect(settings.data_dir)) as connection:
        stored = connection.execute(
            "SELECT count(*), min(model) FROM segment_embeddings WHERE meeting_id = ?",
            (identifier,),
        ).fetchone()
        hits = search(
            connection,
            settings,
            "Alice will write the migration by Friday",
            semantic=True,
            embedder=ollama,
        )
        stages = [
            r[0]
            for r in connection.execute("SELECT stage FROM stage_runs WHERE state = 'succeeded'")
        ]
    assert tuple(stored) == (9, "embed:latest@sha256:fixture")
    assert hits[0].kind == "semantic" and hits[0].ref in {"s000002", "s000008"}
    assert hits[0].score > hits[-1].score and "migration" in hits[0].snippet
    assert "indexing" in stages
    # Rerun reuses the embedding checkpoint.
    embedded = ollama.embedded
    process(settings, identifier)
    assert ollama.embedded == embedded


def test_semantic_search_requires_configured_model(indexed, settings):
    _, connection = indexed
    with pytest.raises(NotesError, match="embedding_model"):
        search(connection, settings, "storage", semantic=True)


def test_semantic_similarity_without_numpy(monkeypatch):
    import builtins
    from array import array

    from local_audio_notes.search import index

    real_import = builtins.__import__

    def no_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    query = array("f", [1.0, 0.0])
    blobs = [array("f", [1.0, 0.0]).tobytes(), array("f", [0.0, 1.0]).tobytes()]
    assert index._similarities(query, blobs) == [1.0, 0.0]


def test_cli_search_prints_bracketed_matches(indexed, settings, tmp_path, capsys, monkeypatch):
    config = tmp_path / "search.toml"
    web.write_config(config, settings)
    monkeypatch.setattr(cli, "enable_runtime", lambda **kwargs: None)
    assert cli.main(["--config", str(config), "search", "migration", "--limit", "3"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 3 and any("[migration]" in r["snippet"] for r in rows)
    assert cli.main(["--config", str(config), "search", "x", "--from", "30/10/2026"]) == 1


def test_web_search_escapes_and_highlights(indexed, settings, tmp_path):
    identifier, connection = indexed
    with transaction(connection):
        connection.execute(
            "INSERT INTO segment_search (text, meeting_id, segment_id) VALUES (?, ?, ?)",
            ("<script>alert(1)</script> migration", identifier, "s000001"),
        )
    config = tmp_path / "web.toml"
    web.write_config(config, settings)

    class Launcher:
        def running(self):
            return True

        def ensure(self):
            pass

    app = web.create_app(settings, config, 8765, "t", Launcher())
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    client.get("/login?token=t")
    page = client.get("/search", params={"q": "migration", "participant": "speaker"}).text
    assert "<mark>migration</mark>" in page
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert re.search(rf'href="/meetings/{identifier}#s000002"', page)
    assert client.get("/search", params={"q": "x", "category": "rm"}).status_code == 400
    assert client.get("/search", params={"q": "x", "date_from": "soon"}).status_code == 400
    semantic = client.get("/search", params={"q": "storage", "semantic": "on"}).text
    assert "embedding_model" in semantic


def test_embeddings_from_an_older_transcript_are_never_matched(
    audio, settings, mocked, monkeypatch
):
    """Reprocessing that changes the transcript but stops before indexing must not let old
    vectors be joined to new segments by ID."""
    from conftest import TURNS

    import local_audio_notes.pipeline as pipeline
    from local_audio_notes.schemas import SpeakerTurn

    calls, ollama = mocked
    settings = settings.model_copy(update={"embedding_model": "embed:latest"})
    identifier = ingest(audio, settings)
    process(settings, identifier)
    with contextlib.closing(connect(settings.data_dir)) as connection:
        assert search(connection, settings, "migration", semantic=True, embedder=ollama)

    def transcript_sha():
        with contextlib.closing(connect(settings.data_dir)) as connection:
            return connection.execute(
                "SELECT artifacts FROM checkpoints WHERE meeting_id = ? AND stage = 'attributing'",
                (identifier,),
            ).fetchone()[0]

    before = transcript_sha()
    # One voice throughout: every segment gets a different label, so the transcript changes.
    solo = [SpeakerTurn(start=t.start, end=t.end, speaker="solo") for t in TURNS]
    monkeypatch.setattr(pipeline, "diarize", lambda *args: solo)
    ollama.fail_on_call = len(ollama.payloads) + 1
    with pytest.raises(NotesError, match="extracting"):
        process(settings, identifier, force=True)
    assert transcript_sha() != before
    with contextlib.closing(connect(settings.data_dir)) as connection:
        assert connection.execute("SELECT count(*) FROM segment_embeddings").fetchone()[0] == 9
        with pytest.raises(NotesError, match="No embeddings for the current"):
            search(connection, settings, "migration", semantic=True, embedder=ollama)
    ollama.fail_on_call = None
    process(settings, identifier)
    with contextlib.closing(connect(settings.data_dir)) as connection:
        assert search(connection, settings, "migration", semantic=True, embedder=ollama)


def test_participant_filter_reaches_matches_beyond_any_row_cap(settings):
    from local_audio_notes.ingestion.importer import register
    from local_audio_notes.pipeline import store_segments
    from local_audio_notes.schemas import Segment, Transcript

    count = 2_500
    transcript = Transcript(
        language="en",
        segments=[
            Segment(
                id=f"s{i:06d}",
                start=i,
                end=i + 1,
                text="budget review",
                speaker="Speaker 2" if i == count - 1 else "Speaker 1",
            )
            for i in range(count)
        ],
    )
    identifier = "00000000-0000-4000-8000-000000000002"
    with contextlib.closing(connect(settings.data_dir)) as connection:
        register(
            connection,
            {
                "id": identifier,
                "sha256": "b" * 64,
                "duration": count,
                "format": ["wav"],
                "imported_at": "now",
            },
        )
        store_segments(connection, identifier, transcript)
        hits = search(connection, settings, "budget", participant="speaker 2")
        assert [h.ref for h in hits] == [f"s{count - 1:06d}"]
        assert len(search(connection, settings, "budget", limit=30)) == 30


def test_semantic_search_with_category_is_rejected(indexed, settings):
    _, connection = indexed
    with pytest.raises(NotesError, match="clear the category"):
        search(connection, settings, "storage", semantic=True, category="action")
