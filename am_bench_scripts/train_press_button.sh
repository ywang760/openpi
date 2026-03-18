#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

EXP_NAME=${1:?usage: train_press_button.sh <exp_name>}

cd "${OPENPI_ROOT}"

uv run scripts/train_pytorch.py pi05_am_bench_press_button \
    --exp_name "${EXP_NAME}"
