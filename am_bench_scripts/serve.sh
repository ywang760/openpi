#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG_NAME="pi05_am_bench_press_button"
PORT="8000"
CUDA_DEVICE=1

EXP_NAME="pressbutton_2"
STEP=20000
CHECKPOINT_DIR="${OPENPI_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${STEP}"

cd "${OPENPI_ROOT}"

env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" uv run scripts/serve_policy.py \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config="${CONFIG_NAME}" \
    --policy.dir="${CHECKPOINT_DIR}"
