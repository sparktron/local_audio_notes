# Troubleshooting

- **Python 3.10**: Ubuntu 22.04 ships it. Use an isolated Python 3.11–3.13. The inspected
  machine already has `/home/mythos/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`.
  Do not replace `/usr/bin/python3`. See README setup.
- **Snap uv fails in a sandbox**: use the existing Python path directly for venv/setup.
  No system installation is necessary.
- **Ollama unavailable**: inspect `command -v ollama` and the existing deployment yourself.
  No daemon was reachable during development. Start the intended local Ollama process;
  do not change unrelated services. Use `OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1`
  for a separately launched process; do not start another on an occupied port.
- **No local models**: run `ollama-models`. The application never pulls Ollama models.
  Select an exact installed tag in config.toml. Obtain approval before downloading any
  new model. Prefer comparing an already-installed Qwen 14B-class model against other
  installed models on the same consented recording; record real wall time, GPU peak use
  and human source-grounding review. No benchmark has been obtained here.
- **Missing manifest/assets**: use explicit models-install/assets-install, or register a
  trusted materialized local directory. A Hugging Face cache containing symlinks must
  be copied/materialized first. `models-check` hashes every asset; hashes do not prove
  licensing, provenance or model-code safety.
- **Gated Community-1 download**: accept the model's conditions on Hugging Face and use a
  least-privilege read token via `HF_TOKEN`. No token is needed for offline runtime.
- **Language mismatch**: automatic detection is available, but the alignment model must
  support that language. Install/register an appropriate local CTC model and set
  alignment_language. Runtime will fail instead of downloading a language model.
- **Unsupported WhisperBoard export**: Phase 1 accepts UTF-8 TXT for comparison and
  strictly documented numeric-seconds JSON/SRT for alignment. Arbitrary vendor JSON or
  untimed TXT cannot be assumed compatible. Convert deliberately using the schema in README.
- **GPU out of memory**: transcription retries with smaller batches to 1, then fails
  with original/stage artifacts retained. Reduce batch_size, use int8_float16, a smaller
  installed Whisper model, or explicitly device="cpu". Alignment/diarization OOM needs
  a manual CPU rerun; there is no silent quality-reducing skip. The pipeline releases
  each audio model before loading the next. Unrelated Ollama models may already occupy VRAM;
  this CLI does not forcibly unload them. Ollama calls use keep_alive=0 for their own model.
- **CUDA detected but inference fails**: nvcc/nvidia-smi do not prove PyTorch/CTranslate2
  compatibility. Check the isolated environment's torch version and `torch.cuda.is_available()`.
  WhisperX 3.8.6 uses torch/torchaudio 2.8 and CUDA wheels/dependency libraries must match.
  No driver/toolkit changes are performed by setup.sh.
- **Torchcodec/ffmpeg loading**: pyannote 4 uses native audio dependencies. The existing
  ffmpeg 4.4 binary is verified for import/normalization only; torchcodec's shared library
  compatibility remains untested. Install matching trusted wheels in the venv, not system
  replacements, and test a real fixture.
- **Failed notes**: `status ID` reports the safe error class/stage. Fix model/config and
  `run ID --notes-only`; verbatim and attributed transcripts remain intact. If a prior
  successful notes file exists, it stays as the prior output until replacement succeeds;
  check meeting state before using it. Full run recomputes audio stages.
- **No word timestamps for a word**: alignment can fail on punctuation, names, numbers,
  unsupported language or bad audio. Missing timestamps/labels remain null/Unknown.
  Segment intervals do not imply word-level confidence. Review against original audio.
- **Enforced offline fails**: expected on the normal host. Use a private network namespace
  with both processes on its loopback. Ubuntu AppArmor/user namespace policy can reject
  unshare; do not disable host security controls automatically. An administrator may need
  to provision an appropriate sandbox. No successful OS-isolated inference is claimed.
- **Sensitive errors suppressed**: native errors can include transcript text, tokens or
  paths, so the CLI emits only a safe message/error class. Inspect private artifacts locally.
  Avoid publishing them in bug reports. `doctor` prints only tool/package/hardware information.
