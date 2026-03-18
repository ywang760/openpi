#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG_NAME="pi05_am_bench_peg_in_hole"
PORT="8000"

if [ "$#" -ne 2 ]; then
    echo "usage: serve.sh <exp_name> <step>" >&2
    exit 1
fi

EXP_NAME="$1"
STEP="$2"
CHECKPOINT_DIR="${OPENPI_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${STEP}"

cd "${OPENPI_ROOT}"

uv run scripts/serve_policy.py policy:checkpoint \
    --port="${PORT}" \
    --policy.config="${CONFIG_NAME}" \
    --policy.dir="${CHECKPOINT_DIR}"
