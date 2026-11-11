# Local Audio Notes

Turn a local recording into a transcript and cited draft notes using models running on your
computer. Upload an audio file, review the transcript and anonymous speaker labels, and
export the notes as Markdown or JSON. **Audio alone is the default workflow**; you do not
need to supply a transcript.

The application offers a local web interface and a command-line interface. It can also use
a timestamped transcript as the notes source or compare a supplied transcript with a new
transcription. Speaker labels are anonymous until you confirm names yourself, and generated
notes and citations need review against the recording.

## Before you begin

You will need:

- Linux (the setup instructions below use Ubuntu), Python **3.11–3.13**, `ffmpeg` and
  `ffprobe`. A virtual environment keeps Python dependencies separate from system Python.
- Disk space and memory for the inference dependencies and local speech, alignment,
  diarization, and language models. Downloads can be large. A compatible GPU can speed up
  processing. CUDA GPU is the default; CPU mode is supported but may be slow.
- A local Ollama, LM Studio, or Bionic server with an installed language model. The
  application connects only to a loopback address and does not pull language models.
- A recording you are authorized to process. WAV, M4A, MP3, and FLAC are supported.

Setup requires network access to install dependencies and, if needed, explicitly download
models. Processing uses the models you have prepared locally. Local processing alone does
not prove OS-level network isolation; see [Privacy and offline use](#privacy-and-offline-use).

## Setup

Run these commands from a terminal in the repository. First check the prerequisites:

```bash
cd /path/to/local_audio_notes
python3 --version
ffmpeg -version
ffprobe -version
```

If Ubuntu does not have ffmpeg and ffprobe, install the `ffmpeg` package using your normal
system package manager. Python 3.10 is too old; choose a Python 3.11, 3.12, or 3.13
executable without replacing system Python. For example, if `python3.12` is available:

```bash
PYTHON_BIN=python3.12 bash scripts/setup.sh --inference --allow-model-assets
.venv/bin/python -m pip install -e '.[web]'
cp -n config.example.toml config.toml
.venv/bin/local-audio-notes --config config.toml doctor
```

The first command creates `.venv` if needed and installs the development and inference
dependencies, including the Silero VAD weights bundled with its wheel. The second installs
the optional web dependencies. These commands download Python packages; they do not install
system packages or start services. If `.venv` already exists, setup reuses it: check its
Python version with `.venv/bin/python --version`. `cp -n` preserves an existing configuration.
For a CLI-only installation, omit the web dependency command. Check dependency consistency
with `.venv/bin/python -m pip check`.

Next, **prepare the local models** using [Explicit model preparation](#explicit-model-preparation-network-enabled-setup-only),
then configure a local language model using [Configure local inference](#configure-local-inference).
Both steps are required before processing audio. `doctor` reports environment information;
`models-check` verifies the registered speech assets, and `ollama-models` lists language
models available from the configured server.

## Your first recording

After setup and model preparation, start the web interface:

```bash
.venv/bin/local-audio-notes --config config.toml serve --port 8765
```

1. Open the sign-in link printed in the terminal. Keep the server running while using the UI.
2. Open **Import**, select your audio file, and leave the default **transcribe** mode selected.
   A transcript attachment is optional. Start processing or import first and process later.
3. Open the meeting from the dashboard. Follow its status until processing completes.
4. Listen to the recording and review the transcript, citations, and speaker labels. Confirm
   names or correct segments only when you know who spoke.
5. Download the Markdown or JSON notes from the meeting page.

The application copies and preserves the original recording under its configured data
directory. The default is `data/`; treat that directory and its database, transcripts, and
notes as private. Processing time depends on recording length, hardware, models, and settings.

For a discussion without formal decisions or action items, select **Conversation** on the
web Settings page (or set `notes_style = "conversation"` in `config.toml`) before processing.
The default **Meeting** style includes decision and action sections. Choose **Training** (or
`notes_style = "training"`) for teaching sessions: extraction emphasizes concepts, explanations,
examples, procedures and substantive questions, with citations to the transcript. Questions and
answers are captured as separate items when supported; speaker roles are not inferred. Coverage is
bounded by the model and its context, so review the full transcript and recording for omissions.
All styles produce drafts that require review.

Style is a global setting for subsequent processing. After changing it, process an existing recording
again with **Notes only** (or `run ID --notes-only`) to reuse the transcript while regenerating
style-specific evidence and summary. Exporting alone only changes the layout; it cannot recover
teaching details that were not previously extracted.

### Import a folder of recordings

On the **Import** page, use **Import a folder** to select a folder. The browser includes
supported WAV, M4A, MP3 and FLAC files from its subfolders; other files are ignored. Each file
is checked by its audio content against existing meetings, regardless of filename or transcript
attachment. Existing recordings are skipped and are not queued again. The results list shows
imported, skipped and failed files. Folder import uses audio transcription and does not attach
transcripts; use the single-recording form if you need one. Files are handled one at a time,
and the original files are never modified.

### Use the command line instead

Replace the example below with the path to your audio file. `ingest` prints an ID; `run`
transcribes and processes that recording, and `status` reports progress or safe error details.

```bash
meeting_id=$(.venv/bin/local-audio-notes --config config.toml ingest /absolute/path/recording.wav)
.venv/bin/local-audio-notes --config config.toml run "$meeting_id"
.venv/bin/local-audio-notes --config config.toml status "$meeting_id"
```

By default, outputs are in `data/<meeting_id>/`, including `notes.md`, `notes.json`, and
`transcript.json`. See [Import, process and export (CLI)](#import-process-and-export-cli)
for retries, speaker corrections, and exports.

### Choose how an existing transcript is used

| Mode | Source of notes | Use of an attachment |
| --- | --- | --- |
| **transcribe** (default) | The app transcribes the audio. | Ignored for notes. |
| **align** | The supplied timestamped SRT or supported JSON. | Uses its existing time anchors with the recording. |
| **compare** | The app transcribes the audio. | Compares the new transcript to the attached TXT, SRT, or JSON. |

Untimed plain text cannot currently be used as an aligned notes source: the application
will not invent timestamps. Comparison is an exact-text comparison, not a quality score.
See the [CLI transcript examples](#import-process-and-export-cli) for supported formats.

## Inference dependencies

The inference setup installs WhisperX 3.8.6, pyannote.audio and their dependencies in the
virtual environment. PyTorch, CTranslate2 and native audio dependencies must be compatible
with your machine. Detecting a CUDA toolkit or GPU does not prove inference will work.
If GPU setup fails, see [troubleshooting](docs/TROUBLESHOOTING.md); CPU mode is configurable.

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

Install and start your chosen language-model server separately; this project does not
install or manage that service. For Ollama, first check whether your intended server is
already running.
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

### LM Studio or Bionic instead of Ollama

Set `llm_backend = "lmstudio"` and point `ollama_url` at the server's loopback port; the
network guard then allows only that port. Bionic serves its API on `http://127.0.0.1:1235`
(the `port` in `~/.lmstudio/apps/bionic/.internal/http-server-config.json`); the LM Studio
app's default is `http://127.0.0.1:1234`. Bionic also listens on another, changing port
for its own use, which does not serve this API. Notes use
`/v1/chat/completions` with a `json_schema` response format, embeddings use `/v1/embeddings`,
and `ollama-models` lists `/api/v0/models`. Set `ollama_model` (and `embedding_model`) to a
model ID from that list. Only models with a local weight format (GGUF or MLX) are accepted;
cloud models are refused. The other `ollama_*` keys apply to this backend too.

LM Studio cannot set a context length per request. Load the model with at least
`ollama_context` tokens, because a server set to truncate on overflow would drop part of
the transcript without an error. A model loaded on demand uses the context saved in the
app's settings for that model; the client refuses a loaded model reporting less than
`ollama_context`, but cannot check a model that is not loaded yet.
Each request sends `ttl` 60, so a model the server loaded on demand unloads a minute after
the last request, freeing GPU memory as Ollama's `keep_alive` 0 does. A model you loaded by
hand in the app stays loaded: unload it before transcribing if GPU memory is tight.
If the server logs prompts (`logSensitiveData`), meeting transcripts end up in its logs, and
with CORS on, web pages in a local browser can call it: turn both off first.
Set `LLM_PORT` to the same port when using `scripts/observe_network.sh`.

Defaults: local `large-v3` Whisper directory; English; batch 8; CUDA float16; speaker bounds 1–8; 2 GiB/8-hour input limits;
10-minute transcription windows; at most 8192 Ollama context tokens and 2048 output
tokens by default. `whisper_model` is a descriptive label: the registered local model
directory actually selects the weights. Change both label and registration for an alternative.
If CUDA is unavailable, the default fails with an error. Set `device = "auto"` to use CUDA
when available and otherwise CPU, or `device = "cpu"` to use CPU explicitly (int8). Existing
configuration files keep their chosen setting. This controls the speech pipeline; GPU use by
the language model is configured separately in the model server.
Set `language="auto"` to detect language; alignment_language and registered CTC model
must match. No alignment model is downloaded automatically on language mismatch.

## Web interface

```bash
.venv/bin/python -m pip install -e '.[web]'      # explicit, network-enabled dependency install
.venv/bin/local-audio-notes --config config.toml serve --port 8765
```

Open the `http://127.0.0.1:8765/login?token=…` link the command prints. The token changes
every start; other local users cannot sign in without it. The server only ever listens on
127.0.0.1; for another machine use SSH port forwarding (`ssh -L 8765:127.0.0.1:8765 host`),
never a public bind. Pages:

- **Dashboard**: recent meetings, status, title/date filter, job counts, stage timings.
- **Import**: upload audio and an optional WhisperBoard TXT/SRT/JSON transcript, choose a
  mode, optionally start processing. Uploads need JavaScript; otherwise use `ingest`.
- **Meeting**: audio player, process/cancel, title and date, notes with clickable source
  timestamps, action-item table, speaker names, per-segment speaker corrections, Markdown
  and JSON downloads.
- **Settings**: models, language, GPU, diarization and chunking options. Saving rewrites
  the configuration file (comments are not kept). Paths, the Ollama address, offline
  enforcement and size limits are only editable in the file.

Processing started from the browser runs in a separate `worker` process, which exits when
the queue is empty and keeps running if the server stops. See
[ADR-0003](docs/decisions/0003-phase3-loopback-web-interface.md) for the security model.

### Remove a test meeting

On the meeting page, open **Remove this meeting from the database**, review the scope, and
confirm removal. Its database records, job history, search entries and speaker corrections are
removed; the original recording and generated files stay in the meeting's data directory.
Cancel any active job first. Recovery will not automatically re-add a removed meeting; import
the original audio again to create a new meeting. Retained files still contain private data.

## Search

```bash
.venv/bin/local-audio-notes --config config.toml search "migration deadline"
.venv/bin/local-audio-notes --config config.toml search "review" --participant "Speaker 2" --from 2026-10-01
.venv/bin/local-audio-notes --config config.toml search "owner" --category action
.venv/bin/local-audio-notes --config config.toml reindex     # rebuild indexes from verified artifacts
```

Results cover meeting titles and dates, transcript segments and extracted items, each with
the meeting ID and the segment/item and start time; the web Search page links straight to the
timestamp in the meeting's player. Words match literally (no query operators). Participants
match speaker labels or confirmed names, including segment corrections. Meetings processed
before upgrading are indexed by `reindex`.

Semantic search is optional. Set `embedding_model` to an embedding model **already installed**
in your local Ollama (for example one listed by `ollama-models`); nothing is pulled. New jobs
then store embeddings after the notes, and `reindex --embeddings` adds them to existing
meetings. Use `search --semantic` or the "Similar meaning" box.

## Import, process and export (CLI)

The CLI covers everything the web interface does:

```bash
meeting_id=$(.venv/bin/local-audio-notes --config config.toml ingest /absolute/path/meeting.m4a)
.venv/bin/local-audio-notes --config config.toml meeting-set "$meeting_id" --title "Storage review" --date 2026-10-30
.venv/bin/local-audio-notes --config config.toml run "$meeting_id"
.venv/bin/local-audio-notes --config config.toml status "$meeting_id"
# Output paths:
printf '%s\n' "data/$meeting_id/notes.md" "data/$meeting_id/notes.json" "data/$meeting_id/transcript.json"
```

`run` queues a job and processes it in the foreground. Rerunning resumes: every stage whose
inputs, settings, models and outputs still verify is reused (`--force` recomputes all).
To queue work and process it later or in another terminal:

```bash
.venv/bin/local-audio-notes --config config.toml enqueue "$meeting_id"   # prints a job ID
.venv/bin/local-audio-notes --config config.toml worker                  # runs queued jobs, then exits
.venv/bin/local-audio-notes --config config.toml jobs                    # recent jobs and safe errors
.venv/bin/local-audio-notes --config config.toml cancel JOB_ID           # stops at the next safe point
.venv/bin/local-audio-notes --config config.toml recover                 # requeue jobs whose worker died
```

`worker` and `run` recover abandoned jobs automatically before starting. A crash, kill or
SIGTERM returns a running job to the queue (up to `job_max_attempts`); Ctrl-C cancels it.
Failed jobs are not retried automatically: fix the cause and `run` again to resume.
`status` and `jobs` show stage, attempts, durations and safe error text, never transcript
content, titles or speaker names.

`notes.md` contains the title and date (only if you set them; a model-suggested title is
labelled), participants (anonymous labels unless confirmed), a cited executive summary,
key points, decisions, proposals (kept separate from decisions), an action-item table
(owner and deadline only when spoken in the cited quote; "Said by" is the speaker label,
never assumed to be the owner), risks and blockers, open questions, follow-ups, source
references (segment ID, timestamps, speaker, verbatim quote) and counts of model output
that was rejected or stripped. Inferred items and disputed classifications are marked.
`notes.json` carries the same items with evidence start/end seconds. `verbatim.json`,
`transcript.json`, `extraction.json`, `evidence.json` and `summary.json` remain separate.

Confirm speaker names or fix a segment's speaker for this meeting only. Existing notes
are re-rendered without calling any model:

```bash
.venv/bin/local-audio-notes --config config.toml speakers "$meeting_id"
.venv/bin/local-audio-notes --config config.toml speakers "$meeting_id" \
  --name "Speaker 1=Alice" --assign "s000042=Speaker 2"
.venv/bin/local-audio-notes --config config.toml export "$meeting_id" --format json \
  --output /absolute/path/new-file.json   # creates a new 0600 file; never overwrites
```

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

`transcribe` creates notes from audio and ignores any attachment; `align` uses the attached
timed text as the notes source; `compare` transcribes the audio for notes and compares that
new transcript with the attachment. UTF-8 TXT is supported for compare mode, not direct alignment: missing time anchors
are not guessed. WhisperBoard export schemas vary; arbitrary vendor JSON, timestamps
in milliseconds/strings and untimed content are rejected for alignment. Compare mode
saves an exact comparison, not a diarization/transcription quality score. CLI errors return
nonzero; Ctrl-C returns 130, SIGTERM 143; both preserve originals and completed stages. Only scratch copies of a
failed import are cleaned up; caller-owned recordings are never deleted.

Set `notes_style = "conversation"` in the TOML configuration, or choose Conversation on
the web Settings page, to group items as discussion and perspectives without meeting decision
or action sections. The default `meeting` style retains those sections. The setting applies
when notes are generated or re-exported; model classifications still require human review.

## Validation and offline use

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/local-audio-notes --config config.toml offline-check
bash scripts/verify_offline.sh --config config.toml
bash scripts/observe_network.sh -- .venv/bin/local-audio-notes --config config.toml run "$meeting_id"
.venv/bin/python scripts/benchmark_local.py --dir data   # synthetic hashing/search timings
```

`observe_network.sh` runs a command under strace and lists every network endpoint it and its
children contacted; it exits 1 for anything but loopback Ollama or a port the command bound.
It needs the `strace` package and observes only the run you perform.
`models-check` reuses cached hashes for unchanged files; `models-check --full` rehashes all.

The strict script is expected to fail in the normal host namespace. See
[security instructions](docs/SECURITY.md) for an isolated namespace containing both
the CLI and Ollama. Application guards do not prove native-library or server isolation.
`requirements-dev.lock` records the exact lightweight development environment tested on
Python 3.12; it does not lock the inference stack or package artifact hashes.

A real audio integration test is opt-in, with private data stored under your configured
data directory. Run it in a separate pytest process after model setup:

```bash
LAN_TEST_CONFIG=/absolute/path/config.toml LAN_TEST_AUDIO=/absolute/path/consented-meeting.wav \
  .venv/bin/python -m pytest -q -m integration
```

This checks real stage completion and emits actual wall time locally. Human comparison
against time-aligned reference audio is still needed to measure transcription/diarization
quality. The synthetic three-speaker fixture tests logic, not real quality or GPU performance.

## Privacy and offline use

No encryption at rest is provided; use encrypted underlying storage such as LUKS and
encrypted backups. Trusted packages/models, a trusted local Ollama daemon and private
same-user filesystem storage remain necessary. Native decoders and libraries are not
OS-sandboxed by the default CLI. Draft classifications can still be wrong despite quote
validation; review audio before treating extracted commitments as authoritative.

Inference and note quality depend on the local models and recording; review outputs against
the audio. See [status](docs/STATUS.md) for current validation and known limitations. The SQLite database holds full transcript text (for
search), quotes, titles and speaker names: protect it like the recordings. See [architecture](docs/ARCHITECTURE.md), [security](docs/SECURITY.md),
[troubleshooting](docs/TROUBLESHOOTING.md), and [status](docs/STATUS.md).
