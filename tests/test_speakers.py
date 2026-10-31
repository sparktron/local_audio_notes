from local_audio_notes.alignment.speakers import attribute
from local_audio_notes.schemas import Segment, SpeakerTurn, Transcript, Word


def test_three_speakers_overlap_unknown_and_words():
    transcript = Transcript(
        language="en",
        segments=[
            Segment(
                id="s000000",
                start=0,
                end=2,
                text="Hello",
                words=[Word(text="Hello", start=0.2, end=0.8)],
            ),
            Segment(id="s000001", start=2, end=4, text="Second"),
            Segment(id="s000002", start=4, end=6, text="Third"),
            Segment(id="s000003", start=6, end=8, text="Overlapping"),
            Segment(id="s000004", start=9, end=10, text="Unknown"),
        ],
    )
    turns = [
        SpeakerTurn(start=0, end=2, speaker="model_A"),
        SpeakerTurn(start=2, end=4, speaker="model_B"),
        SpeakerTurn(start=4, end=8, speaker="model_C"),
        SpeakerTurn(start=6, end=8, speaker="model_A"),
    ]
    result = attribute(transcript, turns)
    assert [s.speaker for s in result.segments] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 3",
        "Uncertain",
        "Unknown",
    ]
    assert result.segments[3].overlap
    assert result.segments[0].words[0].speaker == "Speaker 1"
    assert transcript.segments[0].speaker == "Unknown"  # source is preserved
    assert "confidence" not in result.segments[0].model_dump()


def test_partial_coverage_is_uncertain():
    transcript = Transcript(
        language="en", segments=[Segment(id="s000000", start=0, end=10, text="Too little coverage")]
    )
    result = attribute(transcript, [SpeakerTurn(start=0, end=1, speaker="A")])
    assert result.segments[0].speaker == "Uncertain"
