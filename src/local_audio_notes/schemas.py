from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Interval(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> "Interval":
        if self.end <= self.start:
            raise ValueError("Interval must have positive duration")
        return self


class Word(StrictModel):
    text: str
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    speaker: str = "Unknown"
    overlap: bool = False
    # Alignment confidence is not speaker confidence. No fabricated speaker score.
    alignment_score: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> "Word":
        if (self.start is None) != (self.end is None):
            raise ValueError("Both word timestamps or neither are required")
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("Invalid word interval")
        return self


class Segment(Interval):
    id: str = Field(pattern=r"^s\d{6,}$")
    text: str = Field(min_length=1, max_length=10000)
    words: list[Word] = Field(default_factory=list)
    speaker: str = "Unknown"
    overlap: bool = False


class Transcript(StrictModel):
    language: str
    segments: list[Segment]

    @model_validator(mode="after")
    def unique(self) -> "Transcript":
        ids = [s.id for s in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate segment IDs")
        if any(a.start > b.start for a, b in zip(self.segments, self.segments[1:], strict=False)):
            raise ValueError("Transcript must be ordered by start time")
        return self


class SpeakerTurn(Interval):
    speaker: str


class DraftItem(StrictModel):
    category: Literal[
        "purpose",
        "topic",
        "decision",
        "proposal",
        "action",
        "risk",
        "blocker",
        "question",
        "requirement",
        "fact",
        "follow_up",
    ]
    # Extractive Phase 1 output: quote itself is displayed, not an unchecked paraphrase.
    quote: str = Field(min_length=1, max_length=2000)
    segment_id: str
    owner: str | None = None
    deadline: str | None = None


class DraftNotes(StrictModel):
    items: list[DraftItem] = Field(max_length=100)
