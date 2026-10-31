"""Local asset inventory. Only install_model/install_assets can use the internet."""

import importlib.resources
import json
import os
import re
from pathlib import Path
from typing import Literal

from local_audio_notes.errors import NotesError
from local_audio_notes.storage.files import private_dir, sha256, write_json

ModelKind = Literal["whisper", "alignment", "diarization", "tokenizer", "vad"]
KINDS = ("whisper", "alignment", "diarization", "tokenizer", "vad")


def _inventory(path: Path) -> dict[str, str]:
    hashes = {}
    for item in sorted(path.rglob("*")):
        if ".cache" in item.relative_to(path).parts or "__pycache__" in item.parts:
            continue
        if item.is_symlink():
            raise NotesError("Model assets must be materialized files, not symlinks")
        if item.is_file():
            hashes[item.relative_to(path).as_posix()] = sha256(item)
    if not hashes:
        raise NotesError("Model directory contains no artifacts")
    return hashes


def register_model(root: Path, kind: str, path: Path, revision: str, source: str) -> None:
    private_dir(root)
    if kind not in KINDS or not path.is_dir() or path.is_symlink():
        raise NotesError("Invalid model kind or local directory")
    path = path.resolve()
    inventory = _inventory(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[kind] = {
        "path": str(path),
        "revision": revision,
        "source": source,
        "sha256": inventory,
    }
    write_json(manifest_path, manifest)


def verify_models(root: Path) -> dict[str, Path]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise NotesError("Model manifest missing; run explicit model setup/register commands")
    try:
        manifest = json.loads(manifest_path.read_text())
        paths = {}
        for kind in KINDS:
            entry = manifest[kind]
            path = Path(entry["path"])
            if not path.is_absolute() or not path.is_dir() or path.is_symlink():
                raise ValueError
            if not entry["revision"] or _inventory(path) != entry["sha256"]:
                raise ValueError
            paths[kind] = path
        required = {
            "whisper": ("model.bin", "config.json", "tokenizer.json"),
            "alignment": ("config.json", "vocab.json", "preprocessor_config.json"),
            "diarization": ("config.yaml",),
            "tokenizer": ("tokenizers/punkt_tab/english/ortho_context.tab",),
            "vad": ("silero_vad.jit",),
        }
        for kind, names in required.items():
            if not all((paths[kind] / name).is_file() for name in names):
                raise ValueError
        return paths
    except (OSError, KeyError, TypeError, ValueError):
        raise NotesError(
            "Required model assets missing or changed; check the model manifest"
        ) from None


def install_model(root: Path, kind: str, repo_id: str, revision: str, allow_download: bool) -> None:
    if not allow_download:
        raise NotesError("Model downloads require --allow-download and prior license acceptance")
    if kind not in {"whisper", "alignment", "diarization"}:
        raise NotesError("Use assets-install for tokenizer and VAD assets")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", repo_id):
        raise NotesError("Expected an explicit Hugging Face repository ID")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise NotesError("Pin models using a full 40-character commit SHA")
    from huggingface_hub import snapshot_download

    destination = private_dir(root) / f"{kind}-{revision}"
    if destination.exists():
        raise NotesError("Destination exists; refusing to overwrite model artifacts")
    snapshot_download(
        repo_id=repo_id, revision=revision, local_dir=destination, token=os.environ.get("HF_TOKEN")
    )
    register_model(root, kind, destination, revision, repo_id)


def install_assets(root: Path, allow_download: bool) -> None:
    if not allow_download:
        raise NotesError("Tokenizer asset installation requires --allow-download")
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import nltk

    destination = private_dir(private_dir(root) / "nltk")
    if not nltk.download(
        "punkt_tab", download_dir=str(destination), quiet=True, raise_on_error=True
    ):
        raise NotesError("NLTK asset installation failed")
    register_model(
        root, "tokenizer", destination, "unversioned-dataset-recorded-by-sha256", "nltk/punkt_tab"
    )
    # silero-vad's wheel contains this model; do not call torch.hub.load (network/code fetch).
    import silero_vad

    vad = Path(str(importlib.resources.files("silero_vad.data")))
    register_model(root, "vad", vad, "silero-vad-6.2.0", "installed silero-vad wheel")
    del silero_vad
