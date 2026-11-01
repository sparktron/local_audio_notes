#!/usr/bin/env bash
# Observe the network system calls one command (and every child process) actually makes.
#
#   bash scripts/observe_network.sh -- .venv/bin/local-audio-notes --config config.toml run ID
#
# Records connect/sendto/sendmsg/sendmmsg/bind with strace, then prints every distinct endpoint
# (parsed by network_trace.py, including unconnected UDP destinations such as DNS queries).
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
strace -f -qq -s 256 -e trace=connect,sendto,sendmsg,sendmmsg,bind -o "$workdir/trace" -- "$@" \
  >"$workdir/stdout" 2>"$workdir/stderr" || status=$?
python3 -I "$(dirname -- "$0")/network_trace.py" "$workdir/trace" "$status"
