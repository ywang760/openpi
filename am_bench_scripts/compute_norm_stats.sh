#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

cd "${OPENPI_ROOT}"

uv run scripts/compute_norm_stats.py --config-name pi05_am_bench_press_button
