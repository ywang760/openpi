#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Training config selection.
CONFIG_NAME="pi05_am_bench_peg_in_hole"

# Hardware / launch.
NUM_GPUS="4"

# Common training overrides.
BATCH_SIZE="32"
NUM_TRAIN_STEPS="20000"
NUM_WORKERS="2"
LOG_INTERVAL="100"
SAVE_INTERVAL="1000"
KEEP_PERIOD="5000"

# Boolean training switches.
RESUME="false"
OVERWRITE="false"

if [ "$#" -ne 1 ]; then
    echo "usage: train.sh <exp_name>" >&2
    exit 1
fi

EXP_NAME="$1"

cd "${OPENPI_ROOT}"

CMD=(
    uv run torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}"
    scripts/train_pytorch.py "${CONFIG_NAME}"
    --exp_name "${EXP_NAME}"
    --batch-size "${BATCH_SIZE}"
    --num-train-steps "${NUM_TRAIN_STEPS}"
    --num-workers "${NUM_WORKERS}"
    --log-interval "${LOG_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --keep-period "${KEEP_PERIOD}"
)

if [ "${RESUME}" = "true" ]; then
    CMD+=(--resume)
else
    CMD+=(--no-resume)
fi

if [ "${OVERWRITE}" = "true" ]; then
    CMD+=(--overwrite)
else
    CMD+=(--no-overwrite)
fi

"${CMD[@]}"
