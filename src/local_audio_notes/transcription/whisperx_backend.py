import gc
import os
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError
from local_audio_notes.schemas import Segment, Transcript, Word


def audio_windows(path: Path, seconds: int) -> Iterator[tuple[float, Any]]:
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (16000, 1, 2):
            raise NotesError("Expected normalized 16 kHz mono PCM audio")
        offset = 0
        while raw := handle.readframes(seconds * 16000):
            yield offset / 16000, np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
            offset += len(raw) // 2


def local_vad(path: Path) -> Any:
    import torch
    from silero_vad import get_speech_timestamps
    from whisperx.diarize import Segment as VadSegment
    from whisperx.vads import Vad

    class LocalSilero(Vad):
        def __init__(self) -> None:
            super().__init__(0.5)
            self.model = torch.jit.load(str(path / "silero_vad.jit"), map_location="cpu")
            self.model.eval()

        def __call__(self, audio: dict[str, Any], **kwargs: Any) -> list[Any]:
            timestamps = get_speech_timestamps(
                audio["waveform"],
                self.model,
                sampling_rate=16000,
                threshold=0.5,
                max_speech_duration_s=30,
            )
            return [VadSegment(t["start"] / 16000, t["end"] / 16000, "UNKNOWN") for t in timestamps]

        @staticmethod
        def preprocess_audio(audio: Any) -> Any:
            return audio

        @staticmethod
        def merge_chunks(segments: list[Any], chunk_size: int, **kwargs: Any) -> list[Any]:
            if not segments:
                return []
            return Vad.merge_chunks(segments, chunk_size, **kwargs)

    return LocalSilero()


def release_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(error: Exception) -> bool:
    return "out of memory" in str(error).lower()


class WhisperXBackend:
    def __init__(self, settings: Settings, paths: dict[str, Path]):
        import torch

        self.settings = settings
        self.paths = paths
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise NotesError("CUDA requested but PyTorch cannot use the GPU")
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if settings.device == "auto"
            else settings.device
        )
        torch.set_num_threads(settings.cpu_threads)
        os.environ["NLTK_DATA"] = str(paths["tokenizer"])
        import nltk

        nltk.data.path[:] = [str(paths["tokenizer"])]

        # Prevent WhisperX's implicit NLTK fallback download, even if an asset is missing.
        def no_download(*args: Any, **kwargs: Any) -> None:
            raise NotesError("Missing tokenizer asset; run explicit assets-install")

        nltk.download = no_download

    def transcribe(self, path: Path) -> Transcript:
        import whisperx

        model = None
        segments = []
        language = None if self.settings.language == "auto" else self.settings.language
        batch = self.settings.batch_size
        try:
            model = whisperx.load_model(
                str(self.paths["whisper"]),
                self.device,
                compute_type=self.settings.compute_type if self.device == "cuda" else "int8",
                language=language,
                local_files_only=True,
                vad_model=local_vad(self.paths["vad"]),
                threads=self.settings.cpu_threads,
            )
            for offset, audio in audio_windows(path, self.settings.audio_window_seconds):
                while True:
                    try:
                        result = model.transcribe(
                            audio,
                            batch_size=batch,
                            language=language,
                            print_progress=False,
                            verbose=False,
                        )
                        break
                    except RuntimeError as error:
                        if not is_oom(error) or batch == 1:
                            raise
                        batch = max(1, batch // 2)
                        release_gpu()
                language = result["language"]
                for row in result["segments"]:
                    if row["text"].strip():
                        segments.append(
                            Segment(
                                id=f"s{len(segments):06d}",
                                text=row["text"],
                                start=offset + row["start"],
                                end=offset + row["end"],
                            )
                        )
            return Transcript(language=language or "en", segments=segments)
        finally:
            del model
            release_gpu()

    def align(self, path: Path, transcript: Transcript) -> Transcript:
        import numpy as np
        import whisperx

        if transcript.language != self.settings.alignment_language:
            raise NotesError(
                "Detected language does not match installed alignment model configuration"
            )
        model = None
        result_segments = []
        try:
            model, metadata = whisperx.load_align_model(
                transcript.language,
                self.device,
                model_name=str(self.paths["alignment"]),
                model_cache_only=True,
            )
            with wave.open(str(path), "rb") as audio:
                for segment in transcript.segments:
                    audio.setpos(int(segment.start * 16000))
                    raw = audio.readframes(int((segment.end - segment.start) * 16000))
                    waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                    rows = [{"start": 0.0, "end": len(waveform) / 16000, "text": segment.text}]
                    aligned = whisperx.align(
                        rows,
                        model,
                        metadata,
                        waveform,
                        self.device,
                        return_char_alignments=False,
                        print_progress=False,
                    )
                    words = []
                    for row in aligned["segments"]:
                        for item in row.get("words", []):
                            timed = "start" in item and "end" in item
                            words.append(
                                Word(
                                    text=item["word"],
                                    start=segment.start + item["start"] if timed else None,
                                    end=segment.start + item["end"] if timed else None,
                                    alignment_score=item.get("score"),
                                )
                            )
                    # Preserve original text/IDs and segment boundaries even when alignment fails.
                    result_segments.append(segment.model_copy(update={"words": words}))
            return Transcript(language=transcript.language, segments=result_segments)
        finally:
            del model
            release_gpu()
