#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

VENV_PYTHON="${PROJECT_ROOT}/.venv-gpu/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing Python runtime: $VENV_PYTHON" >&2
  echo "Create/install the GPU environment first." >&2
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DEBUG="${DEBUG:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

case "${DEBUG,,}" in
  1|0|true|false|yes|no|on|off|y|n|t|f|"")
    ;;
  *)
    DEBUG="0"
    ;;
esac

mkdir -p "${DATA_DIR:-$PROJECT_ROOT/data}"

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  echo "[run.sh] running preflight..."
  "$VENV_PYTHON" scripts/preflight.py
fi

exec env DEBUG="$DEBUG" "$VENV_PYTHON" -m chainlit run app.py --host "$HOST" --port "$PORT"
