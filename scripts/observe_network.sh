#!/usr/bin/env bash
# Observe the network system calls one command (and every child process) actually makes.
#
#   bash scripts/observe_network.sh -- .venv/bin/local-audio-notes --config config.toml run ID
#
# Records connect/sendto/sendmsg/bind with strace, then prints every distinct endpoint.
# Exit status: 0 when no IP traffic goes anywhere except loopback port 11434 (Ollama) or a
# loopback port the command itself bound (the web server); 1 otherwise; 2 on usage errors.
# This observes only the run you perform: other code paths, other machines and the Ollama
# server process are not covered. The trace (socket paths, ports) is deleted afterwards.
set -euo pipefail
if [[ "${1:-}" == "--" ]]; then shift; fi
if [[ $# -eq 0 ]]; then
  echo "usage: $0 -- COMMAND [ARGS...]" >&2
  exit 2
fi
if ! command -v strace >/dev/null; then
  echo "strace is required (sudo apt-get install strace)" >&2
  exit 2
fi
workdir=$(mktemp -d)
trap 'rm -rf -- "$workdir"' EXIT
status=0
strace -f -qq -s 256 -e trace=connect,sendto,sendmsg,bind -o "$workdir/trace" -- "$@" \
  >"$workdir/stdout" 2>"$workdir/stderr" || status=$?
python3 -I - "$workdir/trace" "$status" <<'PY'
import json, re, sys
trace, status = sys.argv[1], int(sys.argv[2])
inet = re.compile(r'sin6?_port=htons\((\d+)\).*?(?:sin_addr=inet_addr\("([^"]+)"\)|inet_pton\(AF_INET6, "([^"]+)")')
calls = re.compile(r"^\d+\s+(connect|sendto|sendmsg|bind)\((\d+), \{sa_family=(AF_\w+)(.*)$")
seen, bound = {}, set()
for line in open(trace, encoding="utf-8", errors="replace"):
    match = calls.match(line)
    if not match:
        continue
    call, _, family, rest = match.groups()
    if family in {"AF_INET", "AF_INET6"}:
        found = inet.search(rest)
        if not found:
            continue
        port, address = int(found.group(1)), found.group(2) or found.group(3)
        if call == "bind":
            bound.add(port)
            continue
        key = f"{family} {address}:{port}"
    elif family == "AF_UNIX":
        path = re.search(r'sun_path=@?"([^"]*)"', rest)
        key = f"AF_UNIX {path.group(1) if path else '?'}"
    else:
        key = family  # e.g. AF_NETLINK interface enumeration (kernel, not network)
    seen[key] = seen.get(key, 0) + 1
loopback = ("127.", "::1")
allowed = {11434} | bound
violations = sorted(
    k for k in seen
    if k.startswith("AF_INET")
    and not (k.split()[1].startswith(loopback) and int(k.rsplit(":", 1)[1]) in allowed)
)
print(json.dumps({
    "command_exit_status": status,
    "endpoints": dict(sorted(seen.items())),
    "loopback_ports_bound_by_command": sorted(bound),
    "unexpected_ip_endpoints": violations,
}, indent=2))
sys.exit(1 if violations else 0)
PY
