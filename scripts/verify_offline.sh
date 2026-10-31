#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Does not assert zero packets. Strict mode fails unless already in an isolated namespace.
exec .venv/bin/local-audio-notes "$@" offline-check --enforced
