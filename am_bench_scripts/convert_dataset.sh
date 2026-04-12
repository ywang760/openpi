#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

INPUT_DIR="/usr0/yutongw3/Desktop/am_isaac/datasets/PressButtonEERelPID/demo-20260128_021613"
REPO_ID="am_bench/press_button"
TASK_PROMPT="press the button"
SOURCE_FPS="120"
TARGET_FPS="30"

cd "${OPENPI_ROOT}"

uv run am_bench_scripts/hdf5_to_lerobot.py \
    --input_dir "${INPUT_DIR}" \
    --repo_id "${REPO_ID}" \
    --task_prompt "${TASK_PROMPT}" \
    --fps "${TARGET_FPS}" \
    --source_fps "${SOURCE_FPS}"
