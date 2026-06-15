# `am_bench` First Fine-Tuning Run Guide

This is the current step-by-step path for launching the first `pi-0.5` PyTorch fine-tuning run for `am_bench` on a server.

## Files That Matter

### Python
- `ext/openpi/src/openpi/policies/am_bench_policy.py`
  - runtime/training adapter for `am_bench` observations and actions
- `ext/openpi/src/openpi/training/config.py`
  - defines `LeRobotAmBenchDataConfig`
  - defines `pi05_am_bench_press_button`
- `scripts/data/export_lerobot_to_openpi.py` in the `am_isaac` repo
  - current LeRobot v3 export path for canonical recordings
  - exports one or more `session_root/lerobot` datasets into OpenPI's AM-Bench schema
- `ext/openpi/am_bench_scripts/direct_lerobot_to_legacy.py`
  - preferred path for current canonical LeRobot recordings when training with this OpenPI fork
  - reads canonical LeRobot parquet directly, applies AM-Bench delta-action resampling, and writes OpenPI's pinned legacy LeRobot layout
- `ext/openpi/am_bench_scripts/v3_lerobot_to_legacy.py`
  - rewrites the current LeRobot v3 export into the older LeRobot layout pinned by this OpenPI fork
- `ext/openpi/am_bench_scripts/make_legacy_task_subset.py`
  - filters an OpenPI legacy LeRobot dataset down to one task and rewrites episode/task/index metadata
- `ext/openpi/am_bench_scripts/hdf5_to_lerobot.py`
  - legacy compatibility path for older raw AM Isaac HDF5 episodes

### Shell
- `ext/openpi/am_bench_scripts/prepare_pi05_base_pytorch.sh`
  - patches `transformers` inside the local `uv` environment and converts the base JAX checkpoint to PyTorch
- `ext/openpi/am_bench_scripts/convert_dataset.sh`
  - legacy wrapper for converting older raw AM Isaac HDF5 episodes
- `ext/openpi/am_bench_scripts/compute_norm_stats.sh`
  - computes norm stats for the default `pi05_am_bench_press_button` config
- `ext/openpi/am_bench_scripts/train.sh`
  - launches PyTorch fine-tuning with single-node multi-GPU `torchrun`
  - exposes a few common `train_pytorch.py` CLI options near the top of the file for easy editing
- `ext/openpi/am_bench_scripts/serve.sh`
  - serves a trained checkpoint

## Step 1: Prepare The Base PyTorch Checkpoint

Run:

```bash
./am_bench_scripts/prepare_pi05_base_pytorch.sh
```

What this does:

- creates or reuses the local `uv` environment
- patches the installed `transformers` package used by OpenPI
- converts `gs://openpi-assets/checkpoints/pi05_base` into a local PyTorch checkpoint

Expected output path:

```bash
~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

## Step 2: Export Canonical LeRobot Data For OpenPI

For current AM-Bench recordings, use the direct canonical LeRobot to legacy OpenPI export. This avoids writing an intermediate LeRobot v3 OpenPI-schema dataset and then rewriting it again for this OpenPI fork.

Example:

```bash
cd /path/to/am_isaac/ext/openpi
uv run am_bench_scripts/direct_lerobot_to_legacy.py \
  --dataset_roots /path/to/am_isaac/datasets/PressButtonEEDeltaPID/demo-YYYYMMDD_HHMMSS/lerobot \
  --repo_id am_bench/press_button \
  --target_hz 30 \
  --task_prompt_map /path/to/am_isaac/scripts/data/am_bench_language_instructions.json \
  --require_task_prompt_map \
  --omit_base_image \
  --overwrite
```

Current OpenPI-original reproduction setting:

- record canonical `EEAbs` datasets
- export at `20 Hz`
- train with action horizon `50`
- evaluate with `n_action_steps=8`
- use `--action_representation ee_local_relative` so training chunks are local-relative SE(3) targets
  derived from absolute EE setpoints

Example for the validated PressButton path:

```bash
cd /path/to/am_isaac/ext/openpi
uv run am_bench_scripts/direct_lerobot_to_legacy.py \
  --dataset_roots /path/to/am_isaac/datasets/PressButtonEEAbsPID/demo-YYYYMMDD_HHMMSS/lerobot \
  --repo_id am_bench/press_button_openpi_original_20hz_ee_local_relative \
  --target_hz 20 \
  --action_representation ee_local_relative \
  --task_prompt_map /path/to/am_isaac/scripts/data/am_bench_language_instructions.json \
  --require_task_prompt_map \
  --omit_base_image \
  --overwrite
```

Example for the prepared LemonHarvesting dataset:

```bash
cd /path/to/am_isaac/ext/openpi
uv run am_bench_scripts/direct_lerobot_to_legacy.py \
  --dataset_roots /path/to/am_isaac/datasets/LemonHarvestingEEAbsPID/demo-20260524_070724/lerobot \
  --repo_id am_bench/lemon_harvesting_openpi_original_20hz_ee_local_relative \
  --target_hz 20 \
  --action_representation ee_local_relative \
  --task_prompt_map /path/to/am_isaac/scripts/data/am_bench_language_instructions.json \
  --require_task_prompt_map \
  --omit_base_image \
  --overwrite
```

The prepared LemonHarvesting export uses `stride=6` from the 120 Hz raw dataset, producing
46 episodes / 13294 frames at 20 Hz.

For multi-task fine-tuning, pass multiple canonical dataset roots to `--dataset_roots`.

Example multi-task input:

```bash
uv run am_bench_scripts/direct_lerobot_to_legacy.py \
  --dataset_roots \
    /path/to/am_isaac/datasets/OpenDoorEEDeltaPID/demo-YYYYMMDD_HHMMSS/lerobot \
    /path/to/am_isaac/datasets/PullLeverEEDeltaPID/demo-YYYYMMDD_HHMMSS/lerobot \
  --repo_id am_bench/multitask \
  --target_hz 30 \
  --task_prompt_map /path/to/am_isaac/scripts/data/am_bench_language_instructions.json \
  --require_task_prompt_map \
  --omit_base_image \
  --overwrite
```

## Legacy: Current LeRobot V3 Bridge

The two-step current-LeRobot bridge still exists for debugging, but it is slower because images are written once into the v3 export and then decoded/re-written into the legacy OpenPI layout.

Example:

```bash
cd /path/to/am_isaac
source ../IsaacLab/env_isaaclab/bin/activate
python scripts/data/export_lerobot_to_openpi.py \
  --dataset_roots datasets/PressButtonEEDeltaPID/demo-YYYYMMDD_HHMMSS/lerobot \
  --repo_id am_bench/press_button_v3 \
  --output_root /tmp/am_bench_press_button_v3 \
  --target_hz 30 \
  --overwrite
```

For multi-task fine-tuning, pass multiple canonical dataset roots to `--dataset_roots`; per-frame task prompts are preserved unless `--task_prompt` is set.

Example multi-task input:

```bash
python scripts/data/export_lerobot_to_openpi.py \
  --dataset_roots \
    datasets/OpenDoorEEDeltaPID/demo-YYYYMMDD_HHMMSS/lerobot \
    datasets/PullLeverEEDeltaPID/demo-YYYYMMDD_HHMMSS/lerobot \
  --repo_id am_bench/multitask_v3 \
  --output_root /tmp/am_bench_multitask_v3 \
  --target_hz 30 \
  --overwrite
```

Then bridge to OpenPI's pinned LeRobot layout:

This OpenPI fork is pinned to an older LeRobot API that expects JSONL metadata and one parquet file per episode. Convert the v3 export before running norm stats or training:

```bash
cd /path/to/am_isaac/ext/openpi
uv run am_bench_scripts/v3_lerobot_to_legacy.py \
  --input_root /tmp/am_bench_press_button_v3 \
  --repo_id am_bench/press_button \
  --overwrite
```

Default bridged dataset:

```bash
~/.cache/huggingface/lerobot/am_bench/press_button
```

## Legacy: Convert Raw HDF5 Data To LeRobot

Use this only for older data sessions that were not recorded as canonical LeRobot datasets.

Edit the variables at the top of `./am_bench_scripts/convert_dataset.sh`, then run:

```bash
./am_bench_scripts/convert_dataset.sh
```

What this does:

- reads the AM Isaac HDF5 episodes
- validates the action shape is 7D
- writes a LeRobot dataset

Default output dataset:

```bash
~/.cache/huggingface/lerobot/am_bench/press_button
```

Quick check:

```bash
find ~/.cache/huggingface/lerobot/am_bench/press_button -maxdepth 2 -type f | sort
```


## Step 3: Compute Norm Stats

Run:

```bash
./am_bench_scripts/compute_norm_stats.sh
```

What this does:

- loads the converted `am_bench/press_button` LeRobot dataset
- runs the default `pi05_am_bench_press_button` data path
- writes normalization statistics for `state` and `actions`

Expected output:

```bash
./assets/pi05_am_bench_press_button/am_bench/press_button/norm_stats.json
```

Quick check:

```bash
ls ./assets/pi05_am_bench_press_button/am_bench/press_button/norm_stats.json
```

For the promoted OpenPI-original local-relative config, compute stats directly with the config name:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative
```

## Step 4: Launch Training

Choose an experiment name and run:

```bash
./am_bench_scripts/train.sh press_button_v1
```

What this does:

- loads `pi05_am_bench_press_button`
- launches `torchrun` with `--nnodes=1 --nproc_per_node=4`
- loads the converted base PyTorch checkpoint from:

```bash
~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

- trains using the LeRobot dataset `am_bench/press_button`

Batching note:

- the config currently uses `batch_size=32`
- this trainer treats that as total batch size across all GPUs
- with 4 GPUs, the effective batch size is `8` per GPU
- writes checkpoints under:

```bash
./checkpoints/pi05_am_bench_press_button/press_button_v1
```

Quick check during training:

```bash
find ./checkpoints/pi05_am_bench_press_button/press_button_v1 -maxdepth 2 -type f | head
```

Things to watch:

- disk space in `./checkpoints`
- GPU memory
- whether the model loads `model.safetensors` from the base checkpoint path above

If this step fails early:

- confirm Step 1 created `~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors`
- confirm Step 3 created the norm stats file
- confirm the LeRobot dataset exists at `~/.cache/huggingface/lerobot/am_bench/press_button`

### Where To Change Training Parameters

For this pipeline, most training changes should go in:

- `ext/openpi/src/openpi/training/config.py`

The main block to edit is the `TrainConfig(...)` with:

- `name="pi05_am_bench_press_button"`

Useful fields in that block:

- `data.repo_id`
  - changes which LeRobot dataset is used for norm stats and training
  - keep this in sync with `convert_dataset.sh`
- `batch_size`
  - changes the total batch size across all GPUs
  - if you hit GPU memory issues, this is one of the first things to reduce
- `pytorch_weight_path`
  - changes which converted base PyTorch checkpoint is loaded before fine-tuning
- `model.action_horizon`
  - changes the predicted action chunk length
  - this should stay aligned with your dataset/task assumptions
- `pytorch_training_precision`
  - defaults higher up in the `TrainConfig` dataclass
  - can be set to `"bfloat16"` or `"float32"` if you want to trade memory for stability
- `lr_schedule`
  - controls warmup, peak learning rate, decay length, and final learning rate
- `optimizer`
  - controls optimizer settings such as weight decay and gradient clipping
- `save_interval`
  - defined in the `TrainConfig` dataclass defaults
  - controls how often checkpoints are written
- `resume`
  - defined in the `TrainConfig` dataclass defaults
  - controls whether training resumes from the latest checkpoint in the experiment directory

Some practical examples:

- If training runs out of memory:
  - reduce `batch_size`
  - optionally switch `pytorch_training_precision` to `bfloat16` if it is not already
- If learning looks unstable:
  - reduce `lr_schedule.peak_lr`
  - consider using `float32`

## Step 5: Serve A Trained Checkpoint

Once training has written a checkpoint directory for a specific step, serve it with (Default port is `8000`):

```bash
./am_bench_scripts/serve.sh press_button_v1 1000
```

This expects the checkpoint at,

```bash
./checkpoints/pi05_am_bench_press_button/press_button_v1/1000
```
