from collections import defaultdict

from local_audio_notes.schemas import SpeakerTurn, Transcript


def _assignment(start: float, end: float, turns: list[SpeakerTurn]) -> tuple[str, bool]:
    coverage: dict[str, float] = defaultdict(float)
    for turn in turns:
        amount = max(0.0, min(end, turn.end) - max(start, turn.start))
        if amount:
            coverage[turn.speaker] += amount
    if not coverage:
        return "Unknown", False
    if len(coverage) > 1:
        # Multiple plausible voices: do not pretend a single identity is certain.
        simultaneous = any(
            a.speaker != b.speaker and max(start, a.start, b.start) < min(end, a.end, b.end)
            for i, a in enumerate(turns)
            for b in turns[i + 1 :]
            if a.start < end and a.end > start and b.start < end and b.end > start
        )
        return "Uncertain", simultaneous
    label, amount = next(iter(coverage.items()))
    return (label if amount >= (end - start) * 0.5 else "Uncertain"), False


def attribute(transcript: Transcript, turns: list[SpeakerTurn]) -> Transcript:
    result = transcript.model_copy(deep=True)
    # Keep speaker names stable across all recording windows, ordered by first appearance.
    names = {}
    for turn in sorted(turns, key=lambda t: (t.start, t.speaker)):
        if turn.speaker not in names:
            names[turn.speaker] = f"Speaker {len(names) + 1}"
    normalized = sorted(
        [t.model_copy(update={"speaker": names[t.speaker]}) for t in turns],
        key=lambda t: t.start,
    )
    active: list[SpeakerTurn] = []
    position = 0
    for segment in result.segments:
        active = [t for t in active if t.end > segment.start]
        while position < len(normalized) and normalized[position].start < segment.end:
            active.append(normalized[position])
            position += 1
        relevant = [t for t in active if t.start < segment.end and t.end > segment.start]
        segment.speaker, segment.overlap = _assignment(segment.start, segment.end, relevant)
        for word in segment.words:
            if word.start is not None and word.end is not None and word.end > word.start:
                word.speaker, word.overlap = _assignment(word.start, word.end, relevant)
    return result
