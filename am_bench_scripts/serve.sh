#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

EXP_NAME=${1:?usage: serve.sh <exp_name> <step>}
STEP=${2:?usage: serve.sh <exp_name> <step>}
PORT=${PORT:-8000}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${OPENPI_ROOT}/checkpoints/pi05_am_bench_press_button/${EXP_NAME}/${STEP}"}

cd "${OPENPI_ROOT}"

uv run scripts/serve_policy.py policy:checkpoint \
    --port="${PORT}" \
    --policy.config=pi05_am_bench_press_button \
    --policy.dir="${CHECKPOINT_DIR}"
