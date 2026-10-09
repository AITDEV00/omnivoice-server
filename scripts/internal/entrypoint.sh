#!/usr/bin/env bash
# =============================================================================
# Internal-cluster entrypoint for the fine-tuned omnivoice-server image.
#
# Inherits the deployed image's environment (port 8080, UID 10000, triton/sage,
# baked whisper ASR) and adds:
#   * REQ-4 compliance: every runtime write is forced under /tmp
#   * clone voices: seeds the baked speaker profiles into a writable dir so
#     ProfileService can list them and users can still add profiles at runtime
#
# The root filesystem is expected to be read-only (REQ-3); /tmp is a tmpfs.
# =============================================================================
set -euo pipefail

# ── REQ-4: force all runtime writes into /tmp ───────────────────────────────
export HOME="${HOME:-/tmp/runner}"
export TMPDIR="${TMPDIR:-/tmp/runner/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/runner/.cache}"
export HF_HOME="${HF_HOME:-/tmp/runner/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/runner/hf/hub}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/runner/numba}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/runner/triton}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" \
         "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR"

# ── Clone voices: seed baked profiles into the writable profile dir ─────────
PROFILE_DIR="${OMNIVOICE_PROFILE_DIR:-/tmp/runner/profiles}"
BAKED_PROFILES="${OMNIVOICE_BAKED_PROFILES:-/home/runner/profiles-baked}"

mkdir -p "$PROFILE_DIR"
if [ -d "$BAKED_PROFILES" ]; then
  cp -rn "$BAKED_PROFILES"/. "$PROFILE_DIR"/ 2>/dev/null || true
  count=$(find "$PROFILE_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  echo "[entrypoint] seeded ${count} clone voices into ${PROFILE_DIR}"
fi

echo "[entrypoint] model=${OMNIVOICE_MODEL_ID:-unset} asr=${OMNIVOICE_ASR_MODEL_ID:-unset}"
echo "[entrypoint] triton=${OMNIVOICE_ENABLE_TRITON:-false} mode=${OMNIVOICE_TRITON_MODE:-} sage=${OMNIVOICE_TRITON_ENABLE_SAGE:-false}"
echo "[entrypoint] starting on ${OMNIVOICE_HOST:-0.0.0.0}:${OMNIVOICE_PORT:-8080}"

# ── REQ-5: single foreground process (PID 1 owns the container lifecycle) ───
# Chain through the NVIDIA entrypoint so its GPU driver check still runs,
# then it execs the server command (image CMD passed as "$@").
if [ "$#" -gt 0 ]; then
  exec /opt/nvidia/nvidia_entrypoint.sh "$@"
fi

exec /opt/nvidia/nvidia_entrypoint.sh \
  omnivoice-server \
  --host "${OMNIVOICE_HOST:-0.0.0.0}" \
  --port "${OMNIVOICE_PORT:-8080}" \
  --profile-dir "$PROFILE_DIR" \
  --device "${OMNIVOICE_DEVICE:-cuda}"
