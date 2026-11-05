import json
import struct
import wave
from pathlib import Path

import pytest

from local_audio_notes.config import Settings
from local_audio_notes.schemas import SpeakerTurn, Transcript

FIXTURE = Path(__file__).parent / "fixtures/meeting.json"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data", models_dir=tmp_path / "models", ollama_model="qwen2.5:14b"
    )


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack("<h", 0) * 32000)
    return path


@pytest.fixture
def transcript() -> Transcript:
    return Transcript.model_validate_json(FIXTURE.read_text())


def item(category, statement, quotes, owner=None, deadline=None, basis="explicit"):
    """A raw model-shaped extraction item: quotes is [(segment_id, quote), ...]."""
    return {
        "category": category,
        "statement": statement,
        "basis": basis,
        "evidence": [{"segment_id": s, "quote": q} for s, q in quotes],
        "owner": owner,
        "deadline": deadline,
    }


# What a well-behaved model would return for each synthetic fixture segment, plus one
# hallucination per kind the validator must catch.
CANNED = {
    "s000000": [
        item(
            "purpose",
            "Choose a local storage format.",
            [("s000000", "choose a local storage format")],
        )
    ],
    "s000001": [
        item(
            "decision",
            "Use SQLite for meeting metadata.",
            [("s000001", "We have decided to use SQLite for meeting metadata.")],
        )
    ],
    "s000002": [
        item(
            "action",
            "Alice will write the migration by Friday.",
            [("s000002", "Alice will write the migration by Friday.")],
            "Alice",
            "Friday",
        )
    ],
    "s000003": [
        item(
            "action",
            "Bob will review it by November 6.",
            [("s000003", "Bob will review it by November 6.")],
            "Bob",
            "November 6",
        ),
        # Invented deadline on a grounded commitment: the field is stripped.
        item(
            "action",
            "Bob will review it.",
            [("s000003", "Bob will review it")],
            "Bob",
            "2026-11-06",
        ),
    ],
    "s000004": [
        item(
            "proposal",
            "Upload it to a cloud service.",
            [("s000004", "We could upload it to a cloud service")],
        ),
        # Fabricated commitment: quote absent from the transcript, so the item is rejected.
        item(
            "action",
            "Carol will upload it by Monday.",
            [("s000004", "Carol will upload it by Monday")],
            "Carol",
            "Monday",
        ),
        # Proposal presented as an inferred decision: rejected.
        item(
            "decision",
            "Cloud upload approved.",
            [("s000004", "upload it to a cloud service")],
            basis="inferred",
        ),
    ],
    "s000005": [
        item(
            "question", "Is there consent for that?", [("s000005", "do we have consent for that?")]
        )
    ],
    "s000006": [
        item(
            "blocker",
            "The driver upgrade is still a blocker.",
            [("s000006", "The driver upgrade is still a blocker.")],
        )
    ],
    "s000008": [
        item(
            "action",
            "Alice will write the migration by Friday.",
            [("s000008", "Alice will write the migration by Friday.")],
            "Alice",
            "Friday",
        )
    ],
}


def default_summary(items):
    decision = next(i["id"] for i in items if i["category"] == "decision")
    return {
        "title": "Local storage format",
        "points": [
            {"text": "Decided to use SQLite for meeting metadata.", "item_ids": [decision]},
            {"text": "Invented: Carol approved the Kubernetes rollout.", "item_ids": [decision]},
        ],
    }


class FakeOllama:
    """Mocked localhost Ollama: answers per chunk from CANNED, records payloads."""

    def __init__(self, canned=None, summary=default_summary):
        self.canned = CANNED if canned is None else canned
        self.summary = summary
        self.payloads: list[dict] = []
        self.fail_on_call: int | None = None

    def require_local_model(self, name=None) -> str:
        return f"{name or 'qwen2.5:14b'}@sha256:fixture"

    def embed(self, model, texts):
        """Deterministic bag-of-words vectors: shared words give similar vectors."""
        self.embedded = getattr(self, "embedded", 0) + len(texts)
        vectors = []
        for text in texts:
            vector = [0.0] * 32
            for word in text.lower().split():
                vector[sum(map(ord, word.strip(".,?!"))) % 32] += 1.0
            vectors.append(vector)
        return vectors

    def generate(self, system, content, schema):
        self.payloads.append(json.loads(content))
        if self.fail_on_call == len(self.payloads):
            return "not json"
        data = self.payloads[-1]
        if "untrusted_transcript" in data:
            ids = [s["id"] for s in data["untrusted_transcript"]]
            return json.dumps({"items": [x for i in ids for x in self.canned.get(i, [])]})
        return json.dumps(self.summary(data["untrusted_evidence_items"]))

    @property
    def extraction_calls(self) -> int:
        return sum("untrusted_transcript" in p for p in self.payloads)


# Diarization turns that reproduce the fixture's speakers, overlap and gaps.
TURNS = [
    SpeakerTurn(start=0, end=3, speaker="spk_a"),
    SpeakerTurn(start=3, end=6, speaker="spk_b"),
    SpeakerTurn(start=6, end=10, speaker="spk_c"),
    SpeakerTurn(start=10, end=13, speaker="spk_a"),
    SpeakerTurn(start=13, end=16.5, speaker="spk_b"),
    SpeakerTurn(start=16, end=17, speaker="spk_c"),
    SpeakerTurn(start=20, end=23, speaker="spk_a"),
    SpeakerTurn(start=23, end=26, speaker="spk_c"),
]


class Calls:
    def __init__(self):
        self.transcribe = self.align = self.diarize = 0


@pytest.fixture
def mocked(monkeypatch, transcript, tmp_path):
    """Real ffmpeg, SQLite and job queue; mocked model inference."""
    import local_audio_notes.pipeline as pipeline
    from local_audio_notes.models import KINDS

    calls = Calls()
    ollama = FakeOllama()
    plain = transcript.model_copy(deep=True)
    for segment in plain.segments:
        segment.speaker, segment.overlap = "Unknown", False

    class Backend:
        device = "cpu"

        def __init__(self, *args):
            pass

        def transcribe(self, path, check=lambda: None):
            calls.transcribe += 1
            check()
            return plain.model_copy(deep=True)

        def align(self, path, value):
            calls.align += 1
            return value

    def diarize(*args):
        calls.diarize += 1
        return TURNS

    monkeypatch.setattr(
        pipeline, "verify_models", lambda root, cache=False: {"diarization": tmp_path}
    )
    monkeypatch.setattr(pipeline, "model_fingerprints", lambda root: dict.fromkeys(KINDS, "fp"))
    monkeypatch.setattr(pipeline, "WhisperXBackend", Backend)
    monkeypatch.setattr(pipeline, "diarize", diarize)
    monkeypatch.setattr(pipeline, "OllamaClient", lambda settings: ollama)
    return calls, ollama
