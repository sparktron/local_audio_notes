# Security and privacy boundaries

Runtime processing has no external inference API, analytics, automatic update check,
model fetch, external UI asset or web service. The only supported HTTP target is
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
Returned quotes must exist verbatim in cited segments. Owners/deadlines must appear in
those quotes; dates are never normalized or invented. These checks do **not** prove that
a quote semantically approves an action. Category errors and prompt injection remain possible.
Notes are drafts for human review. Proposals have their own section. Markdown escapes
HTML, images, links and table metacharacters; do not later render raw transcript HTML.

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
