import json
import time
from pathlib import Path
from typing import Any, Literal

from local_audio_notes.alignment.speakers import attribute
from local_audio_notes.audio.media import normalize
from local_audio_notes.config import Settings
from local_audio_notes.diarization.community import diarize
from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.transcripts import parse_transcript
from local_audio_notes.llm.ollama import OllamaClient
from local_audio_notes.models import verify_models
from local_audio_notes.notes.generate import generate_notes, render_markdown
from local_audio_notes.schemas import Transcript
from local_audio_notes.storage.files import atomic_write, meeting_dir, sha256, write_json
from local_audio_notes.storage.locking import exclusive_lock
from local_audio_notes.transcription.whisperx_backend import WhisperXBackend


def load_meeting(settings: Settings, identifier: str) -> tuple[Path, dict[str, Any]]:
    directory = meeting_dir(settings.data_dir, identifier)
    if not (directory / "meeting.json").is_file():
        raise NotesError("Meeting not found")
    metadata = json.loads((directory / "meeting.json").read_text())
    if sha256(directory / "original.audio") != metadata["sha256"]:
        raise NotesError("Original recording integrity verification failed")
    attachment = metadata.get("attachment")
    if attachment and sha256(directory / "attached.transcript") != attachment["sha256"]:
        raise NotesError("Attached transcript integrity verification failed")
    return directory, metadata


def process(
    settings: Settings,
    identifier: str,
    mode: Literal["transcribe", "align", "compare"] = "transcribe",
    notes_only: bool = False,
) -> None:
    directory = meeting_dir(settings.data_dir, identifier)
    with exclusive_lock(directory / ".processing.lock"):
        _process(settings, identifier, mode, notes_only)


def _process(
    settings: Settings,
    identifier: str,
    mode: Literal["transcribe", "align", "compare"],
    notes_only: bool,
) -> None:
    directory, metadata = load_meeting(settings, identifier)
    stage = "preflight"
    started = time.monotonic()
    timings: dict[str, float] = {}

    def status(value: str) -> None:
        nonlocal stage, started
        timings[stage] = time.monotonic() - started
        stage, started = value, time.monotonic()
        metadata.update(state=value, timings=timings)
        write_json(directory / "meeting.json", metadata)

    try:
        # Check model availability before decoding or transcribing anything.
        client = OllamaClient(settings)
        client.require_local_model()
        if notes_only:
            if sha256(directory / "transcript.json") != metadata.get("transcript_sha256"):
                raise NotesError("Attributed transcript integrity verification failed")
            transcript = Transcript.model_validate_json((directory / "transcript.json").read_text())
        else:
            paths = verify_models(settings.models_dir)
            if mode in {"align", "compare"} and not metadata.get("attachment"):
                raise NotesError("This mode requires an attached transcript")
            provided = None
            if mode == "align" or (mode == "compare" and metadata["attachment"]["kind"] != ".txt"):
                provided = parse_transcript(
                    directory / "attached.transcript",
                    metadata["attachment"]["kind"],
                    metadata["duration"],
                    settings.alignment_language,
                )
            status("normalizing")
            normalized = directory / "normalized.wav"
            # Manual full rerun recomputes stages; only notes-only reuses attributed transcript.
            normalized.unlink(missing_ok=True)
            normalize(directory / "original.audio", normalized, settings)
            backend = WhisperXBackend(settings, paths)
            status("transcribing")
            transcript = provided if mode == "align" else backend.transcribe(normalized)
            if not transcript or not transcript.segments:
                raise NotesError("No speech was transcribed; notes were not generated")
            write_json(directory / "verbatim.json", transcript.model_dump(mode="json"))
            if mode == "compare":
                # Exact comparison only; no quality metric or assumed word-error-rate reference.
                original = (directory / "attached.transcript").read_text(encoding="utf-8-sig")
                if provided:
                    original = " ".join(s.text for s in provided.segments)
                generated = " ".join(s.text for s in transcript.segments)
                write_json(
                    directory / "comparison.json",
                    {
                        "method": "exact-text",
                        "identical": original.strip() == generated.strip(),
                        "provided_text": original,
                        "local_text": generated,
                    },
                )
            status("aligning")
            transcript = backend.align(normalized, transcript)
            write_json(directory / "aligned.json", transcript.model_dump(mode="json"))
            status("diarizing")
            turns = diarize(normalized, paths["diarization"], settings, backend.device)
            write_json(directory / "diarization.json", [t.model_dump(mode="json") for t in turns])
            transcript = attribute(transcript, turns)
            write_json(directory / "transcript.json", transcript.model_dump(mode="json"))
            metadata["transcript_sha256"] = sha256(directory / "transcript.json")
        status("notes")
        draft = generate_notes(transcript, client, settings.chunk_chars)
        sources = {s.id: s for s in transcript.segments}
        write_json(
            directory / "notes.json",
            {
                "version": 1,
                "meeting_id": identifier,
                "draft": True,
                "items": [
                    {
                        **item.model_dump(mode="json"),
                        "start": sources[item.segment_id].start,
                        "end": sources[item.segment_id].end,
                    }
                    for item in draft.items
                ],
            },
        )
        atomic_write(
            directory / "notes.md",
            render_markdown(identifier, metadata["meeting_date"], transcript, draft.items),
        )
        status("complete")
        metadata.pop("error", None)
        metadata.pop("failed_stage", None)
        metadata["configuration"] = settings.model_dump(mode="json")
        write_json(directory / "meeting.json", metadata)
    except BaseException as error:
        timings[stage] = time.monotonic() - started
        metadata.update(
            state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            failed_stage=stage,
            error=type(error).__name__,
            timings=timings,
        )
        write_json(directory / "meeting.json", metadata)
        raise
