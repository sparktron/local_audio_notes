"""Deterministic Markdown and JSON exports from validated evidence. No model calls."""

import html
import re
from typing import Any

from local_audio_notes.schemas import EvidenceItem, SummaryRecord, Transcript
from local_audio_notes.speakers.mapping import LABEL, SpeakerView

RENDER_VERSION = 2


def timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02}:{total % 3600 // 60:02}:{total % 60:02}"


def escape(value: str) -> str:
    # Markdown is a plain local artifact; prevent embedded HTML, images and active links.
    value = html.escape(value).replace("\n", " ").replace("\r", " ")
    value = "".join(c for c in value if c >= " " or c == "\t")
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", value)


def participants(transcript: Transcript, view: SpeakerView) -> list[dict[str, str | None]]:
    seen = dict.fromkeys(
        view.speaker(s.id, s.speaker)
        for s in transcript.segments
        if LABEL.fullmatch(view.speaker(s.id, s.speaker))
    )
    return [{"label": label, "name": view.names.get(label)} for label in seen]


def _speakers(item: EvidenceItem, view: SpeakerView) -> str:
    said = dict.fromkeys(view.display(view.speaker(e.segment_id, e.speaker)) for e in item.evidence)
    return ", ".join(said)


def _sources(item: EvidenceItem) -> str:
    return "; ".join(f"{e.segment_id} {timestamp(e.start)}" for e in item.evidence)


def _text(item: EvidenceItem) -> str:
    if item.statement:
        return escape(item.statement)
    return "“" + escape(" … ".join(e.quote for e in item.evidence)) + "”"


def _marks(item: EvidenceItem) -> str:
    marks = []
    if item.basis == "inferred":
        marks.append(" *(inferred)*")
    if item.disputed:
        marks.append(" **(classification disputed — verify)**")
    return "".join(marks)


def render_markdown(
    meeting: dict[str, Any],
    transcript: Transcript,
    items: list[EvidenceItem],
    summary: SummaryRecord | None,
    rejections: dict[str, int],
    view: SpeakerView,
) -> str:
    title = meeting.get("title") or (summary and summary.title)
    named = [p for p in participants(transcript, view) if p["name"]]
    anonymous = [p["label"] for p in participants(transcript, view) if not p["name"]]
    lines = [
        f"# {escape(title) if title else 'Meeting ' + meeting['id']}",
        "",
        f"- Date: {escape(meeting.get('meeting_date') or 'Not established')}",
        f"- Meeting ID: {meeting['id']}",
    ]
    if title and not meeting.get("title"):
        lines.append("- Title: model-suggested from extracted evidence")
    people = [f"{escape(p['name'] or '')} \\[{p['label']}\\] (confirmed)" for p in named]
    if anonymous:
        people.append(", ".join(anonymous) + " (anonymous labels)")
    lines += [
        f"- Participants: {'; '.join(people) or 'Not established'}",
        "",
        "> Draft notes generated locally. Every item cites transcript segments; statements are",
        "> model wording checked only for unsupported names, numbers and dates. Verify against",
        "> the recording before acting.",
        "",
        "## Executive summary",
        "",
    ]
    if summary and summary.points:
        lines += [f"- {escape(p.text)} ({', '.join(p.item_ids)})" for p in summary.points]
    else:
        lines.append("Not generated.")
    if summary and summary.omitted_items:
        lines += ["", f"{summary.omitted_items} items exceeded the summary context bound."]

    def section(title: str, categories: set[str]) -> None:
        lines.extend(["", f"## {title}", ""])
        selected = [i for i in items if i.category in categories]
        lines.extend(
            f"- {i.id}: {_text(i)}{_marks(i)} — {escape(_speakers(i, view))}, {_sources(i)}"
            for i in selected
        )
        if not selected:
            lines.append("None extracted.")

    section("Key discussion points", {"purpose", "topic", "fact", "requirement"})
    section("Decisions made", {"decision"})
    section("Proposed decisions (not approved)", {"proposal"})
    lines += [
        "",
        "## Action items",
        "",
        "| ID | Commitment | Owner | Deadline | Said by | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in (i for i in items if i.category == "action"):
        lines.append(
            f"| {item.id} | {_text(item)}{_marks(item)} | {escape(item.owner or 'Not stated')} | "
            f"{escape(item.deadline or 'Not stated')} | {escape(_speakers(item, view))} | "
            f"{_sources(item)} |"
        )
    section("Risks and blockers", {"risk", "blocker"})
    section("Outstanding questions", {"question"})
    section("Follow-up agenda", {"follow_up"})
    lines += ["", "## Source references", ""]
    texts = {s.id: s for s in transcript.segments}
    for item in items:
        for e in item.evidence:
            segment = texts[e.segment_id]
            speaker = escape(view.display(view.speaker(e.segment_id, e.speaker)))
            overlap = " (overlapping speech)" if segment.overlap else ""
            lines.append(
                f"- {item.id} ← {e.segment_id} [{timestamp(e.start)}–{timestamp(e.end)}] "
                f"{speaker}{overlap}: “{escape(e.quote)}”"
            )
    if not items:
        lines.append("No items cited.")
    review = dict(rejections)
    if summary:
        review.update({f"summary_{k}": v for k, v in summary.rejections.items()})
    if review:
        counts = ", ".join(f"{k}: {v}" for k, v in sorted(review.items()))
        lines += ["", "## Extraction review", "", f"Model output rejected or stripped — {counts}."]
    if view.stale_entries:
        lines += [
            "",
            f"{view.stale_entries} speaker names/corrections belong to an earlier transcript "
            "and were not applied.",
        ]
    lines += ["", "Full transcript: transcript.json in the meeting directory.", ""]
    return "\n".join(lines)


def export_json(
    meeting: dict[str, Any],
    transcript: Transcript,
    items: list[EvidenceItem],
    summary: SummaryRecord | None,
    rejections: dict[str, int],
    view: SpeakerView,
    generated_by: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": RENDER_VERSION,
        "draft": True,
        "meeting": {
            "id": meeting["id"],
            "title": meeting.get("title"),
            "suggested_title": summary.title if summary else None,
            "date": meeting.get("meeting_date"),
        },
        "participants": participants(transcript, view),
        "generated_by": generated_by,
        "summary": summary.model_dump(mode="json") if summary else None,
        "items": [
            {
                **item.model_dump(mode="json", exclude={"evidence"}),
                "evidence": [
                    {
                        **e.model_dump(mode="json"),
                        "speaker": view.speaker(e.segment_id, e.speaker),
                        "speaker_name": view.names.get(view.speaker(e.segment_id, e.speaker)),
                    }
                    for e in item.evidence
                ],
            }
            for item in items
        ],
        "rejections": rejections,
    }
