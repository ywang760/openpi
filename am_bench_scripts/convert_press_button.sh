#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

INPUT_DIR=${INPUT_DIR:?Set INPUT_DIR to the raw AM Isaac HDF5 dataset directory.}
REPO_ID=${REPO_ID:-am_bench/press_button}
TASK_PROMPT=${TASK_PROMPT:-press the button}
FPS=${FPS:-30}

cd "${OPENPI_ROOT}"

uv run am_bench_scripts/convert_press_button_hdf5_to_lerobot.py \
    --input_dir "${INPUT_DIR}" \
    --repo_id "${REPO_ID}" \
    --task_prompt "${TASK_PROMPT}" \
    --fps "${FPS}"
