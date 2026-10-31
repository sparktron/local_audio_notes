from pathlib import Path

from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError
from local_audio_notes.schemas import SpeakerTurn
from local_audio_notes.transcription.whisperx_backend import is_oom, release_gpu

BATCH_ATTRIBUTES = ("segmentation_batch_size", "embedding_batch_size")


def _halve_batches(pipeline: object) -> bool:
    """Halve pyannote's inference batch sizes; False once they cannot shrink further."""
    changed = False
    for name in BATCH_ATTRIBUTES:
        size = getattr(pipeline, name, None)
        if isinstance(size, int) and size > 1:
            setattr(pipeline, name, size // 2)
            changed = True
    return changed


def diarize(path: Path, model_path: Path, settings: Settings, device: str) -> list[SpeakerTurn]:
    import torch
    from pyannote.audio import Pipeline

    pipeline = None
    try:
        pipeline = Pipeline.from_pretrained(str(model_path))
        # GPU OOM: retry with smaller batches (same model, same output semantics), then
        # optionally on CPU. Never silently skip diarization.
        devices = [device] + (["cpu"] if device == "cuda" and settings.oom_cpu_fallback else [])
        for candidate in devices:
            pipeline.to(torch.device(candidate))
            while True:
                try:
                    result = pipeline(
                        str(path),
                        min_speakers=settings.min_speakers,
                        max_speakers=settings.max_speakers,
                    )
                except RuntimeError as error:
                    if not is_oom(error):
                        raise
                    release_gpu()
                    if not _halve_batches(pipeline):
                        break
                    continue
                # Keep the regular output so overlapping speech is retained.
                annotation = result.speaker_diarization
                return [
                    SpeakerTurn(start=turn.start, end=turn.end, speaker=str(label))
                    for turn, _, label in annotation.itertracks(yield_label=True)
                ]
        raise NotesError(
            "GPU out of memory during diarization; free VRAM or set oom_cpu_fallback = true"
        )
    finally:
        del pipeline
        release_gpu()
