import json

import pytest
from conftest import FakeOllama

from local_audio_notes.errors import NotesError
from local_audio_notes.extraction.consolidate import consolidate
from local_audio_notes.extraction.evidence import extract
from local_audio_notes.notes.render import export_json, render_markdown
from local_audio_notes.notes.summary import summarize
from local_audio_notes.schemas import Segment, SummaryRecord, Transcript
from local_audio_notes.speakers.mapping import SpeakerView


@pytest.fixture
def items(transcript, settings):
    outcome = extract(transcript, FakeOllama(), settings, "model@digest")
    return consolidate(outcome.items, transcript.segments)


def test_summary_keeps_only_grounded_cited_points(items, settings):
    client = FakeOllama()
    summary = summarize(items, client, settings)
    assert [p.text for p in summary.points] == ["Decided to use SQLite for meeting metadata."]
    assert summary.rejections == {"ungrounded_point": 1}
    assert summary.title == "Local storage format"
    sent = client.payloads[0]["untrusted_evidence_items"]
    assert {e["id"] for e in sent} == {i.id for i in items}  # evidence only, no transcript


@pytest.mark.parametrize(
    "point, reason",
    [
        ({"text": "Use SQLite.", "item_ids": ["i9999"]}, "unknown_item"),
        ({"text": "Upload to a cloud service was agreed.", "item_ids": None}, "decision_lang"),
    ],
)
def test_summary_rejects_unknown_ids_and_unsupported_decision_language(
    items, settings, point, reason
):
    proposal = next(i.id for i in items if i.category == "proposal")
    point = {**point, "item_ids": point["item_ids"] or [proposal]}

    def respond(entries):
        return {"title": "Invented Title 2027", "points": [point]}

    summary = summarize(items, FakeOllama(summary=respond), settings)
    assert summary.points == [] and summary.title is None
    expected = "unknown_item" if reason == "unknown_item" else "unsupported_decision_language"
    assert summary.rejections == {expected: 1, "ungrounded_title": 1}


def test_summary_skips_model_without_items(settings):
    client = FakeOllama()
    assert summarize([], client, settings) == SummaryRecord(title=None, points=[])
    assert client.payloads == []


def test_summary_invalid_schema(items, settings):
    client = FakeOllama(summary=lambda entries: {"title": "x", "points": [], "shell": "id"})
    with pytest.raises(NotesError, match="summary schema"):
        summarize(items, client, settings)


def test_markdown_sections_sources_and_markers(items, transcript, settings):
    summary = summarize(items, FakeOllama(), settings)
    view = SpeakerView(names={"Speaker 3": "Alice"}, overrides={"s000006": "Speaker 1"})
    meeting = {"id": "m-1", "title": None, "meeting_date": "2026-10-30"}
    output = render_markdown(meeting, transcript, items, summary, {"quote_not_found": 1}, view)
    assert output.startswith("# Local storage format\n")
    assert "Title: model-suggested" in output
    assert "Alice \\[Speaker 3\\] (confirmed)" in output
    assert "Speaker 1, Speaker 2 (anonymous labels)" in output
    for heading in (
        "Executive summary",
        "Key discussion points",
        "Decisions made",
        "Proposed decisions (not approved)",
        "Action items",
        "Risks and blockers",
        "Outstanding questions",
        "Follow-up agenda",
        "Source references",
    ):
        assert f"## {heading}" in output
    assert (
        "| Alice | Friday | Alice \\[Speaker 3\\] | s000002 00:00:06; s000008 00:00:23 |" in output
    )
    assert "| Bob | November 6 | Speaker 1 |" in output
    assert "s000005 [00:00:15–00:00:17] Uncertain (overlapping speech)" in output
    # Segment correction applies to rendering only.
    assert "s000006 [00:00:18–00:00:20] Speaker 1" in output
    assert "quote_not_found: 1, summary_ungrounded_point: 1" in output
    assert "Carol" not in output and "Kubernetes" not in output
    decisions = output.split("## Decisions made")[1].split("##")[0]
    assert "SQLite" in decisions and "cloud" not in decisions


def test_markdown_active_content_escaped(settings):
    transcript = Transcript(
        language="en",
        segments=[
            Segment(
                id="s000000",
                start=0,
                end=2,
                speaker="Speaker 1",
                text="<script>alert(1)</script> ![x](https://example.com) | cell",
            )
        ],
    )
    raw = {
        "fact": [
            {
                "category": "fact",
                "statement": "<script>alert(1)</script> ![x](https://example.com) | cell",
                "basis": "explicit",
                "evidence": [
                    {"segment_id": "s000000", "quote": "![x](https://example.com) | cell"}
                ],
            }
        ]
    }
    client = FakeOllama(canned={"s000000": raw["fact"]})
    items = consolidate(extract(transcript, client, settings, "m").items, transcript.segments)
    meeting = {"id": "m", "title": "<b>Title</b>", "meeting_date": None}
    output = render_markdown(meeting, transcript, items, None, {}, SpeakerView())
    assert "<script>" not in output and "![x]" not in output and "<b>" not in output
    assert "\\|" in output and "Not generated." in output


def test_json_export_is_self_describing(items, transcript, settings):
    summary = summarize(items, FakeOllama(), settings)
    view = SpeakerView(names={"Speaker 1": "Bob"})
    meeting = {"id": "m", "title": "Storage", "meeting_date": None}
    value = export_json(meeting, transcript, items, summary, {}, view, {"model": "m@d"})
    json.dumps(value, allow_nan=False)
    assert value["draft"] is True and value["meeting"]["title"] == "Storage"
    review = next(i for i in value["items"] if i["owner"] == "Bob")
    assert review["evidence"][0]["speaker_name"] == "Bob"
    assert review["evidence"][0]["start"] == 10
