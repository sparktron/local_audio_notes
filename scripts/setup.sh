#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ${EUID} == 0 ]]; then
  echo 'Run as an unprivileged user.' >&2
  exit 1
fi
python_bin=${PYTHON_BIN:-python3.11}
"$python_bin" -c 'import sys; assert (3,11) <= sys.version_info < (3,14), "Python 3.11–3.13 required"'
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
if [[ ${1:-} == --inference ]]; then
  if [[ ${2:-} != --allow-model-assets ]]; then
    echo 'Inference dependencies include bundled Silero VAD weights. Pass --inference --allow-model-assets to opt in.' >&2
    exit 1
  fi
  install_spec='.[dev,inference]'
else
  install_spec='.[dev]'
fi
if [[ ! -d .venv ]]; then
  "$python_bin" -m venv .venv
fi
# Explicit setup operation: network access for dependencies only. No sudo or service changes.
.venv/bin/python -m pip install --disable-pip-version-check -e "$install_spec"
echo 'Setup complete. Configure config.toml and explicitly prepare models; see README.md.'
