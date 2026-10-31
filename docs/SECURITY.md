# Security and privacy boundaries

Runtime processing has no external inference API, analytics, automatic update check,
model fetch or external UI asset. The optional web interface listens on 127.0.0.1 only. The only supported HTTP target is
`http://127.0.0.1:11434`. HTTP redirects and environment proxies are disabled. Cloud
Ollama model tags/remote aliases are rejected using both inventory and model metadata.
The separately running Ollama server is a trust boundary: use a trusted installation,
local model weights and cloud-disabled settings, and isolate it along with this CLI for
stronger assurance. The client cannot control a server's outbound traffic or configuration.

The CLI sets Hugging Face offline flags, disables pyannote telemetry/OpenTelemetry,
and blocks Python socket connects/DNS except its fixed Ollama endpoint. It explicitly
loads local VAD assets and prevents NLTK's download fallback. ML libraries are lazy-imported
after these controls. Changing permissive environment values cannot enable downloads.

These are **application controls**, not proof of zero network activity. Native C/C++
libraries and subprocesses can bypass Python audit hooks. Supply-chain changes can alter
behavior. No packet capture or actual offline model inference has been completed here.
Never infer OS-level isolation from a successful unit test or disabled network cable.

## Optional fail-closed network isolation

`offline_enforced = true` (or `offline-check --enforced`) refuses operation unless Linux
exposes only a loopback interface. This is intended for a private network namespace.
It refuses the ordinary workstation namespace even when Wi-Fi is disconnected.
This checks the isolation prerequisite; it does not create a namespace or firewall.

After installing dependencies/models, an administrator can validate an unprivileged
user/network namespace, with Ollama **inside the same namespace**. Example Ubuntu shell
session (do not use sudo to run the application; commands change only this new namespace):

```bash
unshare --user --map-current-user --net bash
ip link set lo up
export OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1
ollama serve > /dev/null 2>&1 &
lan_ollama_pid=$!
trap 'kill "$lan_ollama_pid" 2>/dev/null || true' EXIT
bash scripts/verify_offline.sh --config config.toml
# Set offline_enforced = true in config.toml before processing in this shell.
.venv/bin/local-audio-notes --config config.toml ollama-models
# Then ingest/run as in README. Exit this shell to tear down the namespace.
```

This has **not** been validated on this machine: user namespace/AppArmor restrictions
can prevent unshare or GPU access. If it fails, enforced mode must remain failing; do not
pretend isolation succeeded. See troubleshooting. A host Ollama service cannot be reached
from isolated loopback; do not add a host-network proxy that defeats isolation. Host
administrators/capability holders and malicious local software remain outside the threat model.
Inspect `ip addr`/`ip route` in the namespace and perform an actual egress observation with
`strace -f -e trace=network` or namespace-aware capture during real inference before making
claims about native-library networking. Logs/captures are sensitive; store privately.

## Untrusted input and output

Imports copy through a byte limit, reject nonregular files/symlinks, inspect signatures
and actual containers, enforce duration bounds, and never place caller names in subprocess
commands or destination paths. ffmpeg only receives argument arrays and local file/pipe
protocols. Its decoders are native parsers: malformed media remains a dependency risk.
Maintain patched ffmpeg and trusted Python wheels. Process untrusted media in an OS sandbox
for stronger protection. Limits bound ordinary workloads, not every decoder exploit or
GPU kernel behavior; no cgroup CPU/RAM/disk quota is installed by this scaffold.

Prompt data is untrusted JSON in a user role, with fixed system instructions and no tools.
Stage 1 sends only segment IDs and text (no speaker labels, so owners cannot be inferred from
who spoke). Every returned item is validated in code: each quote must exist in its cited
segment (case/whitespace-insensitive; the stored quote is the transcript's own text);
decisions, actions and follow-ups must be explicit; owners and deadlines must appear in a
cited quote and are dropped if they are pronouns or attached to non-commitments; statements
introducing capitalized words or numbers absent from the quotes are discarded in favour of
the quote. Stage 2 sees only validated evidence; summary points must cite known items, may
not introduce such words, and may not use decision wording unless a cited item is a
decision. Rejections are recorded as counts by reason, never as content.

These checks stop fabricated quotes, names, numbers and dates. They do **not** prove that a
real quote was classified correctly, and lowercase paraphrase is unchecked: prompt injection
can still cause misclassification of genuine transcript text. Notes are drafts for human
review; source references always show the verbatim quote. Markdown escapes HTML, images,
links and table metacharacters; do not later render raw transcript HTML.

## Database

`data/notes.sqlite3` is created 0600 by the application (symlinks refused) and, like the
artifacts, contains transcript-derived text: extracted quotes, cached model responses,
titles and confirmed speaker names. `status`/`jobs` output and stored error messages are
limited to stage names, counts, durations and the application's own fixed error texts;
library exception messages are never stored. `secure_delete` is enabled, but on SSDs and
copy-on-write filesystems overwritten pages can persist: rely on full-volume encryption.
Speaker names are entered explicitly by the user per meeting; no voiceprints are computed.

## Files, credentials and confidentiality

Run as an ordinary user. Storage directories use 0700 and files 0600; CLI umask 077.
Model/recording directories and config are gitignored. A model setup token is read from
`HF_TOKEN` only during explicit installation. Never paste tokens in commands/history,
source files or logs. Accept gated-model licenses yourself. Setup downloads are opt-in
and intentionally network-enabled; no recording is sent during model installation.

No encryption at rest is implemented. Use LUKS or another encrypted underlying volume
for data/models and encrypted backups. Swap, core dumps, terminal capture, GPU memory,
process inspection and privileged users can expose content. This scaffold does not erase
VRAM, provide secure deletion, authenticate files, or defend against a malicious same-UID
process. Do not share its writable data directory across users. No retention automation
or automatic deletion is enabled; capacity and backup management are your responsibility.

## Web interface

`serve` binds 127.0.0.1 itself and cannot listen elsewhere. Access needs the per-launch
sign-in link printed in the terminal; it sets an HttpOnly, SameSite=Strict session cookie.
Requests with a Host other than 127.0.0.1/localhost are rejected (DNS rebinding). State
changes are refused when Origin or Sec-Fetch-Site indicate another site and require a
per-session CSRF token. Responses carry a strict Content-Security-Policy (no inline or
third-party script, style or media; no framing), nosniff, no-referrer and no-store headers.
FastAPI's interactive docs (CDN assets) and its built-in OpenTelemetry instrumentation are
disabled, and `FASTAPI_OTEL_AUTO_CONFIGURE` is forced off. Templates autoescape transcript
text, titles and names. Uploads stream into the private data directory under the configured
byte limits, are validated by `ingest`, and the staged copies are deleted. Errors show the
application's own messages or an exception class name, never library exception text.
Request access logs are disabled. See ADR-0003 for the threat model and its limits: a
malicious process running as the same user can read the token and the data directly.
