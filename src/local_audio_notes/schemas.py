from typing import Literal, get_args

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


Category = Literal[
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
CATEGORIES: tuple[str, ...] = get_args(Category)
# Only explicit commitments may carry an owner or deadline.
COMMITMENTS = frozenset({"action", "follow_up"})
# Never accepted as inferences: these must be stated in the recording.
EXPLICIT_ONLY = frozenset({"decision", "action", "follow_up"})


# --- Model-facing schemas (Ollama `format`). No regex patterns: grammar support varies.


class QuoteRef(StrictModel):
    segment_id: str = Field(min_length=1, max_length=32)
    quote: str = Field(min_length=1, max_length=1000)


class ExtractedItem(StrictModel):
    category: Category
    statement: str = Field(min_length=1, max_length=400)
    basis: Literal["explicit", "inferred"]
    evidence: list[QuoteRef] = Field(min_length=1, max_length=4)
    owner: str | None = Field(default=None, max_length=120)
    deadline: str | None = Field(default=None, max_length=120)


class ExtractionResult(StrictModel):
    items: list[ExtractedItem] = Field(max_length=60)


class SummaryPoint(StrictModel):
    text: str = Field(min_length=1, max_length=400)
    item_ids: list[str] = Field(min_length=1, max_length=8)


class MeetingSummary(StrictModel):
    title: str = Field(min_length=1, max_length=120)
    points: list[SummaryPoint] = Field(max_length=8)


# --- Validated evidence. Times and speakers come from the transcript, never the model.


class Evidence(StrictModel):
    segment_id: str
    quote: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str


class EvidenceItem(StrictModel):
    id: str = ""
    category: Category
    # Model paraphrase that passed the grounding check, else None (render the quote).
    statement: str | None
    basis: Literal["explicit", "inferred"]
    owner: str | None = None
    deadline: str | None = None
    evidence: list[Evidence] = Field(min_length=1)
    # Overlapping chunks classified the same excerpt differently; needs human review.
    disputed: bool = False


class ExtractionOutcome(StrictModel):
    items: list[EvidenceItem]
    # Counts of rejected items or stripped fields by reason. Never the rejected text.
    rejections: dict[str, int]
    chunks: int = 0
    cached_chunks: int = 0


class SummaryRecord(StrictModel):
    title: str | None
    points: list[SummaryPoint]
    omitted_items: int = 0
    rejections: dict[str, int] = Field(default_factory=dict)


class ConsolidatedEvidence(StrictModel):
    items: list[EvidenceItem]
    rejections: dict[str, int]
    generated_by: dict[str, str]
