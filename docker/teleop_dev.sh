#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT="$(cd -- "${ISAACLAB_ROOT}/.." >/dev/null 2>&1 && pwd)"

COMMAND="${1:-start}"
if [[ $# -gt 0 ]]; then
  shift
fi

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-isaaclab3teleop}"
export TELEOP_OUT_HOST="${TELEOP_OUT_HOST:-${WORKSPACE_ROOT}/out}"
mkdir -p "${TELEOP_OUT_HOST}"

STATE_FILE="${SCRIPT_DIR}/.container.cfg"
if ! grep -q "X11_FORWARDING_ENABLED" "${STATE_FILE}" 2>/dev/null; then
  cat >"${STATE_FILE}" <<'EOF'
[X11]
X11_FORWARDING_ENABLED = 0
EOF
fi

if [[ "$(id -u)" != "1000" ]]; then
  echo "[WARN] IsaacLab3 containers run as uid/gid 1000. Ensure TELEOP_OUT_HOST is writable by uid 1000:"
  echo "       ${TELEOP_OUT_HOST}"
fi

cd "${ISAACLAB_ROOT}"
exec ./docker/container.py "${COMMAND}" base \
  --suffix 300b2 \
  --files docker-compose.teleop-local.yaml \
  --env-files .env.teleop-local \
  "$@"
