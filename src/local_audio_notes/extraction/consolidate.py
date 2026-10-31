"""Cross-chunk consolidation: merge repeats, keep distinct commitments, preserve sources."""

from itertools import combinations

from local_audio_notes.extraction.grounding import normalize, words
from local_audio_notes.schemas import CATEGORIES, Evidence, EvidenceItem, Segment

CONSOLIDATION_VERSION = "consolidate-v1"
# Overlapping chunks that classify the same excerpt both ways are surfaced, not resolved.
CONFLICTS = (frozenset({"decision", "proposal"}), frozenset({"action", "proposal"}))
# Shorter identical quotes ("Okay.", "Why?") in different segments are not one item.
RESTATEMENT_MIN_WORDS = 4


def _fold(text: str) -> str:
    return normalize(text).casefold()


def _refs_match(a: Evidence, b: Evidence) -> bool:
    qa, qb = _fold(a.quote), _fold(b.quote)
    if a.segment_id == b.segment_id:
        return qa in qb or qb in qa
    # Same commitment restated later in the meeting.
    return qa == qb and len(words(qa)) >= RESTATEMENT_MIN_WORDS


def _evidence_match(a: EvidenceItem, b: EvidenceItem) -> bool:
    return any(_refs_match(x, y) for x in a.evidence for y in b.evidence)


def _compatible(x: str | None, y: str | None) -> bool:
    return x is None or y is None or _fold(x) == _fold(y)


def same_item(a: EvidenceItem, b: EvidenceItem) -> bool:
    """Different owners or deadlines are distinct commitments even for one excerpt."""
    return (
        a.category == b.category
        and _compatible(a.owner, b.owner)
        and _compatible(a.deadline, b.deadline)
        and _evidence_match(a, b)
    )


def _merge(first: EvidenceItem, other: EvidenceItem, position: dict[str, int]) -> EvidenceItem:
    evidence = list(first.evidence)
    for reference in other.evidence:
        if reference not in evidence:
            evidence.append(reference)
    # Drop a quote fully contained in a longer quote from the same segment.
    evidence = [
        e
        for e in evidence
        if not any(
            o is not e
            and o.segment_id == e.segment_id
            and _fold(e.quote) in _fold(o.quote)
            and _fold(e.quote) != _fold(o.quote)
            for o in evidence
        )
    ]
    evidence.sort(key=lambda e: (position[e.segment_id], e.quote))
    return first.model_copy(
        update={
            "statement": first.statement or other.statement,
            "basis": "explicit" if "explicit" in (first.basis, other.basis) else "inferred",
            "owner": first.owner or other.owner,
            "deadline": first.deadline or other.deadline,
            "evidence": evidence,
            "disputed": first.disputed or other.disputed,
        }
    )


def consolidate(items: list[EvidenceItem], segments: list[Segment]) -> list[EvidenceItem]:
    position = {s.id: i for i, s in enumerate(segments)}
    clusters: list[EvidenceItem] = []
    for item in items:
        # Merging can bridge earlier clusters; re-check compatibility after each merge.
        for cluster in [c for c in clusters if same_item(c, item)]:
            if same_item(cluster, item):
                clusters.remove(cluster)
                item = _merge(cluster, item, position)
        clusters.append(item)
    disputed = set()
    for a, b in combinations(range(len(clusters)), 2):
        pair = frozenset({clusters[a].category, clusters[b].category})
        if pair in CONFLICTS and _evidence_match(clusters[a], clusters[b]):
            disputed |= {a, b}
    clusters = [
        c.model_copy(update={"disputed": True}) if i in disputed else c
        for i, c in enumerate(clusters)
    ]
    clusters.sort(
        key=lambda c: (
            min(position[e.segment_id] for e in c.evidence),
            CATEGORIES.index(c.category),
            c.statement or c.evidence[0].quote,
        )
    )
    return [c.model_copy(update={"id": f"i{n:04d}"}) for n, c in enumerate(clusters, start=1)]
