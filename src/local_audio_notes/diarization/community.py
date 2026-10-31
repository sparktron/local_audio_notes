from pathlib import Path

from local_audio_notes.config import Settings
from local_audio_notes.schemas import SpeakerTurn
from local_audio_notes.transcription.whisperx_backend import release_gpu


def diarize(path: Path, model_path: Path, settings: Settings, device: str) -> list[SpeakerTurn]:
    import torch
    from pyannote.audio import Pipeline

    pipeline = None
    try:
        pipeline = Pipeline.from_pretrained(str(model_path))
        pipeline.to(torch.device(device))
        result = pipeline(
            str(path), min_speakers=settings.min_speakers, max_speakers=settings.max_speakers
        )
        # Keep the regular output so overlapping speech is retained.
        annotation = result.speaker_diarization
        return [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=str(label))
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
    finally:
        del pipeline
        release_gpu()
