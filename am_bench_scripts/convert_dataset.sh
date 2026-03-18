#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

INPUT_DIR="/path/to/raw/peg_in_hole_hdf5"
REPO_ID="am_bench/peg_in_hole"
TASK_PROMPT="insert the peg into the hole"
FPS="120"

cd "${OPENPI_ROOT}"

uv run am_bench_scripts/hdf5_to_lerobot.py \
    --input_dir "${INPUT_DIR}" \
    --repo_id "${REPO_ID}" \
    --task_prompt "${TASK_PROMPT}" \
    --fps "${FPS}"
