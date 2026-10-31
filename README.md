# local_audio_notes

Private meeting audio → local transcription and alignment → anonymous speakers →
local Ollama draft notes → Markdown/JSON. **Scaffold and Phase 1 only.**

This is a working CLI implementation with tested media handling and mocked inference
boundaries. Actual model inference, GPU integration, diarization quality and isolated
offline inference are **not yet verified**. No models were downloaded during development.
Do not treat this scaffold as a finished production service.

Implemented: WAV/M4A/MP3/FLAC imports; preserved originals and SHA-256; local WhisperX
adapters; local Community-1 diarization; word attribution with Unknown/Uncertain labels;
strict timed transcript import; extractive, cited Ollama draft notes; private artifacts;
stage/error metrics and notes-only retries. No web server, database, search or automatic
speaker identity assignment. See [scope](docs/PLAN.md) and [validation](docs/VALIDATION.md).

## Ubuntu installation

Do not replace system Python or change NVIDIA/Ollama services. Inspect first:

```bash
cd /home/mythos/repos/local_audio_notes
python3 --version
command -v ffmpeg ffprobe ollama nvidia-smi nvcc
nvidia-smi
nvcc --version
ffmpeg -version
```

Ubuntu 22.04's default Python 3.10 is too old. On the inspected workstation, use the
already installed Python 3.12, with no system changes:

```bash
PYTHON_BIN=/home/mythos/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 \
  bash scripts/setup.sh
cp -n config.example.toml config.toml
.venv/bin/local-audio-notes --config config.toml doctor
```

For another machine with uv already installed, explicitly installing a managed interpreter
is an optional dependency installation step: `uv python install 3.12`, then
`PYTHON_BIN="$(uv python find 3.12)" bash scripts/setup.sh`. If ffmpeg is absent, explicitly
install it through Ubuntu (`sudo apt-get update` followed by `sudo apt-get install ffmpeg`).
Setup checks for ffmpeg and never installs system packages or services itself.

The initial setup installs package/test dependencies only. To intentionally install the
inference stack, including the Silero VAD weights bundled with its wheel:

```bash
PYTHON_BIN=/home/mythos/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 \
  bash scripts/setup.sh --inference --allow-model-assets
.venv/bin/python -m pip check
```

WhisperX is pinned to 3.8.6, which declares torch/torchaudio 2.8, pyannote 4 and compatible
torchcodec dependencies. This dependency set is based on
[upstream packaging](https://github.com/m-bain/whisperX/blob/v3.8.6/pyproject.toml), and
the heavyweight stack has not been installed or tested here. Resolve it in this venv;
do not upgrade/downgrade system CUDA or drivers speculatively. CUDA toolkit detection
does not demonstrate PyTorch/CTranslate2 inference. CPU mode is available in config.

## Explicit model preparation (network-enabled setup only)

Normal commands never fetch models. Before downloads, review each model's license and
provenance. Community-1 requires accepting its Hugging Face user conditions and authorization
for installation; its published model card documents
[local offline loading](https://huggingface.co/pyannote/speaker-diarization-community-1).
Enter a least-privilege read token without storing it in history or repository files:

```bash
read -r -s -p 'Hugging Face read token: ' HF_TOKEN
export HF_TOKEN
```

The following commands deliberately contact Hugging Face, resolve immutable revisions,
download assets to this project's models directory and inventory their SHA-256 hashes.
Run them only after explicitly choosing to download these models and accepting conditions.
No model is pulled by the application as a side effect of processing audio.

```bash
whisper_revision=$(.venv/bin/python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Systran/faster-whisper-large-v3").sha)')
alignment_revision=$(.venv/bin/python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("facebook/wav2vec2-base-960h").sha)')
diarization_revision=$(.venv/bin/python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("pyannote/speaker-diarization-community-1").sha)')

.venv/bin/local-audio-notes --config config.toml models-install \
  --kind whisper --repo Systran/faster-whisper-large-v3 --revision "$whisper_revision" --allow-download
.venv/bin/local-audio-notes --config config.toml models-install \
  --kind alignment --repo facebook/wav2vec2-base-960h --revision "$alignment_revision" --allow-download
.venv/bin/local-audio-notes --config config.toml models-install \
  --kind diarization --repo pyannote/speaker-diarization-community-1 --revision "$diarization_revision" --allow-download
.venv/bin/local-audio-notes --config config.toml assets-install --allow-download
unset HF_TOKEN
.venv/bin/local-audio-notes --config config.toml models-check
```

`assets-install` explicitly fetches NLTK punkt_tab and records it by content hash (the
dataset is not revision-pinned); it inventories the installed Silero JIT model. Later
runs validate those hashes. Repeating setup will not overwrite an existing HF snapshot;
investigate an interrupted/changed destination manually. Never remove recordings to fix setup.

Already installed, trusted models can be used without downloading. Register each required
kind (`whisper`, `alignment`, `diarization`, `tokenizer`, `vad`) with materialized files:

```bash
.venv/bin/local-audio-notes --config config.toml models-register \
  --kind whisper --path /absolute/path/to/local-faster-whisper-large-v3 --revision YOUR_RECORDED_REVISION
```

Each directory is fully hashed. Hash checks detect changes, not malicious model provenance.
No `trust_remote_code` or remote torch.hub repository is used. Model loading can execute
trusted upstream Python/native code, so treat installed packages and models as trusted inputs.

## Configure local inference

The inspected host had no Ollama executable on PATH and no service at 127.0.0.1:11434.
Locate/start your intended installation; this project will not alter or install it.
For a manually managed Ollama process, keep it on loopback and disable cloud routing:

```bash
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 ollama serve
# In a second shell, from this repository:
.venv/bin/local-audio-notes --config config.toml ollama-models
```

If an existing server is running, use it; do not start a conflicting process. Configure
the existing server's local/cloud behavior deliberately. In config.toml set `ollama_model`
to an exact installed tag. Begin with a Qwen 14B-class model **if already installed**
(for example `qwen2.5:14b`); compare it to other installed models using consented audio
before selecting it. No Qwen model or alternative was benchmarked or downloaded here.
The CLI lists installed models/digests and rejects cloud/remote aliases. It never calls pull.

Defaults: local `large-v3` Whisper directory; English; batch 8; CUDA float16 when
PyTorch supports it, otherwise CPU int8; speaker bounds 1–8; 2 GiB/8-hour input limits;
10-minute transcription windows; at most 8192 Ollama context tokens and 2048 output
tokens by default. `whisper_model` is a descriptive label: the registered local model
directory actually selects the weights. Change both label and registration for an alternative.
Set `language="auto"` to detect language; alignment_language and registered CTC model
must match. No alignment model is downloaded automatically on language mismatch.

## Import, process and export

No web service is launched in Phase 1. The CLI is the supported interface:

```bash
meeting_id=$(.venv/bin/local-audio-notes --config config.toml ingest /absolute/path/meeting.m4a)
.venv/bin/local-audio-notes --config config.toml run "$meeting_id"
.venv/bin/local-audio-notes --config config.toml status "$meeting_id"
# Output paths:
printf '%s\n' "data/$meeting_id/notes.md" "data/$meeting_id/notes.json" "data/$meeting_id/transcript.json"
```

`notes.md` includes all required note sections, sourced excerpts and an action table.
Unknown meeting date/participants remain unestablished; ingestion time is not fabricated
as the meeting date. `notes.json` includes segment IDs and source start/end seconds.
`transcript.json` contains per-segment and per-word speaker labels/timestamps.
`verbatim.json` remains separate from notes. Files are already exports; copy them to
another trusted local location if desired. No external export destination is implemented.

Existing transcripts are optional:

```bash
meeting_id=$(.venv/bin/local-audio-notes --config config.toml ingest /absolute/path/meeting.m4a \
  --transcript /absolute/path/export.srt)
.venv/bin/local-audio-notes --config config.toml run "$meeting_id" --mode align
# Or re-transcribe locally and save an exact-text comparison:
.venv/bin/local-audio-notes --config config.toml run "$meeting_id" --mode compare
# Retry a failed LLM stage using the saved, hash-verified attributed transcript:
.venv/bin/local-audio-notes --config config.toml run "$meeting_id" --notes-only
```

Supported alignment input: UTF-8 SRT with `HH:MM:SS,mmm` timestamps, or this explicit JSON
schema using numeric **seconds**, monotonic segment starts, and segments no longer than
300 seconds within the recording:

```json
{"segments": [{"start": 0.0, "end": 3.2, "text": "Verbatim spoken words."}]}
```

UTF-8 TXT is supported for compare mode, not direct alignment: missing time anchors
are not guessed. WhisperBoard export schemas vary; arbitrary vendor JSON, timestamps
in milliseconds/strings and untimed content are rejected for alignment. Compare mode
saves an exact comparison, not a diarization/transcription quality score. CLI errors return
nonzero; Ctrl-C returns 130 and preserves originals/artifacts. Only scratch copies of a
failed import are cleaned up; caller-owned recordings are never deleted.

## Validation and offline use

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/local-audio-notes --config config.toml offline-check
bash scripts/verify_offline.sh --config config.toml
```

The strict script is expected to fail in the normal host namespace. See
[security instructions](docs/SECURITY.md) for an isolated namespace containing both
the CLI and Ollama. Application guards do not prove native-library or server isolation.
`requirements-dev.lock` records the exact lightweight development environment tested on
Python 3.12; it does not include the untested inference stack or package artifact hashes.

A real audio integration test is opt-in, with private data stored under your configured
data directory. Run it in a separate pytest process after model setup:

```bash
LAN_TEST_CONFIG=/absolute/path/config.toml LAN_TEST_AUDIO=/absolute/path/consented-meeting.wav \
  .venv/bin/python -m pytest -q -m integration
```

This checks real stage completion and emits actual wall time locally. Human comparison
against time-aligned reference audio is still needed to measure transcription/diarization
quality. The synthetic three-speaker fixture tests logic, not real quality or GPU performance.

## Security limits and next phases

No encryption at rest is provided; use encrypted underlying storage such as LUKS and
encrypted backups. Trusted packages/models, a trusted local Ollama daemon and private
same-user filesystem storage remain necessary. Native decoders and libraries are not
OS-sandboxed by the default CLI. Draft classifications can still be wrong despite quote
validation; review audio before treating extracted commitments as authoritative.

Phase 2 adds SQLite, durable recovery and richer evidence; Phase 3 adds the loopback
FastAPI UI and speaker corrections; Phase 4 adds search/hardening. These are intentionally
unimplemented. See [architecture](docs/ARCHITECTURE.md), [security](docs/SECURITY.md),
[troubleshooting](docs/TROUBLESHOOTING.md), and [status](docs/STATUS.md).
