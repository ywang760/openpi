#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OUTPUT_PATH="${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch"
CONFIG_NAME="pi05_am_bench_peg_in_hole"
CHECKPOINT_DIR="gs://openpi-assets/checkpoints/pi05_base"

echo "OPENPI_ROOT: ${OPENPI_ROOT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "CONFIG_NAME: ${CONFIG_NAME}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"

cd "${OPENPI_ROOT}"

TRANSFORMERS_DIR=$(uv run python - <<'PY'
import pathlib
import transformers

print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)

cp -r "${OPENPI_ROOT}/src/openpi/models_pytorch/transformers_replace/"* "${TRANSFORMERS_DIR}/"

uv run examples/convert_jax_model_to_pytorch.py \
    --config_name "${CONFIG_NAME}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_path "${OUTPUT_PATH}"
