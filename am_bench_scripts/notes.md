# `am_bench` First Fine-Tuning Run Guide

This is the current step-by-step path for launching the first `pi-0.5` PyTorch fine-tuning run for `am_bench` on a server.

The current setup is intentionally narrow:

- single-task first run
- PressButton task
- relative action dataset only
- raw source data in AM Isaac HDF5 format
- OpenPI fine-tuning backend is PyTorch

## What This Path Expects

Your raw dataset should contain one HDF5 file per episode, with filenames matching either:

- `demo_*.hdf5`
- `demo_env_*_idx_*.hdf5`

Each episode file must contain:

- `obs/ee_pos`
- `obs/ee_quat`
- `obs/gripper_width`
- `actions`
- `images/ee_camera`

Optional:

- `images/base_camera`

Important constraints:

- `actions` must be relative 7D actions
- this path does not support absolute action datasets
- this path is currently hard-coded around `repo_id=am_bench/press_button`

## Files That Matter

- `ext/openpi/src/openpi/policies/am_bench_policy.py`
  - runtime/training adapter for `am_bench` observations and actions
- `ext/openpi/src/openpi/training/config.py`
  - defines `LeRobotAmBenchDataConfig`
  - defines `pi05_am_bench_press_button`
- `ext/openpi/am_bench_scripts/convert_press_button_hdf5_to_lerobot.py`
  - converts raw AM Isaac HDF5 episodes into LeRobot format
- `ext/openpi/am_bench_scripts/prepare_pi05_base_pytorch.sh`
  - patches `transformers` inside the local `uv` environment and converts the base JAX checkpoint to PyTorch
- `ext/openpi/am_bench_scripts/compute_norm_stats.sh`
  - computes norm stats for `pi05_am_bench_press_button`
- `ext/openpi/am_bench_scripts/train_press_button.sh`
  - launches PyTorch fine-tuning
- `ext/openpi/am_bench_scripts/serve.sh`
  - serves a trained checkpoint

## Before You Start On The Server

1. Start from the OpenPI repo root:

```bash
cd /path/to/am_isaac/ext/openpi
```

2. Use a persistent session. `tmux` or `screen` is recommended.

3. If you hit `lerobot` Git LFS smudge errors during the first `uv run`, export this once in the shell before running the steps below:

```bash
export GIT_LFS_SKIP_SMUDGE=1
```

I needed that on this machine during validation.

4. Make sure you are not sharing this Python environment with something else important.

Reason:
- `prepare_pi05_base_pytorch.sh` copies replacement files into the installed `transformers` package inside the local `uv` environment.
- If the environment is recreated later, rerun that prepare step.

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

Quick check:

```bash
ls ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

You should see at least:

- `model.safetensors`
- `assets/`
- `config.json`

If this step fails:

- check network access to `gs://openpi-assets`
- check whether `uv run` is failing during dependency bootstrap
- if the error mentions Git LFS / `lerobot`, retry with `GIT_LFS_SKIP_SMUDGE=1`

## Step 2: Convert Raw HDF5 Data To LeRobot

For the first run, keep the default repo ID. Do not change it unless you also update the training config.

Run:

```bash
INPUT_DIR=/path/to/raw/press_button_hdf5 \
./am_bench_scripts/convert_press_button.sh
```

Optional overrides:

```bash
INPUT_DIR=/path/to/raw/press_button_hdf5 \
TASK_PROMPT="press the button" \
FPS=30 \
./am_bench_scripts/convert_press_button.sh
```

What this does:

- reads the AM Isaac HDF5 episodes
- validates the action shape is 7D
- writes a LeRobot dataset with:
  - `ee_image`
  - `base_image`
  - `ee_pos`
  - `ee_quat`
  - `gripper_width`
  - `actions`
  - `task`

Default output dataset:

```bash
~/.cache/huggingface/lerobot/am_bench/press_button
```

Quick check:

```bash
find ~/.cache/huggingface/lerobot/am_bench/press_button -maxdepth 2 -type f | sort
```

You should see `meta/` files and a parquet file under `data/`.

If this step fails:

- confirm your files are named `demo_*.hdf5` or `demo_env_*_idx_*.hdf5`
- confirm `images/ee_camera` exists
- confirm `actions.shape[-1] == 7`

## Step 3: Compute Norm Stats

Run:

```bash
./am_bench_scripts/compute_norm_stats.sh
```

What this does:

- loads the converted `am_bench/press_button` LeRobot dataset
- runs the `pi05_am_bench_press_button` data path
- writes normalization statistics for `state` and `actions`

Expected output:

```bash
./assets/pi05_am_bench_press_button/am_bench/press_button/norm_stats.json
```

Quick check:

```bash
ls ./assets/pi05_am_bench_press_button/am_bench/press_button/norm_stats.json
```

If this step fails:

- make sure Step 2 finished successfully
- make sure the config still points at `repo_id="am_bench/press_button"`
- if your dataset is very small, note that the current config batch size is `32`

For tiny debug datasets, this script can fail if dataset size is smaller than batch size. For a real training run that should usually not be a problem.

## Step 4: Launch Training

Choose an experiment name and run:

```bash
./am_bench_scripts/train_press_button.sh press_button_v1
```

What this does:

- loads `pi05_am_bench_press_button`
- loads the converted base PyTorch checkpoint from:

```bash
~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

- trains using the LeRobot dataset `am_bench/press_button`
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

## Step 5: Serve A Trained Checkpoint

Once training has written a checkpoint directory for a specific step, serve it with:

```bash
./am_bench_scripts/serve.sh press_button_v1 1000
```

This expects the checkpoint at:

```bash
./checkpoints/pi05_am_bench_press_button/press_button_v1/1000
```

Default port:

- `8000`

To change the port:

```bash
PORT=9000 ./am_bench_scripts/serve.sh press_button_v1 1000
```

To serve a checkpoint from a non-default path:

```bash
CHECKPOINT_DIR=/custom/path/to/checkpoint \
./am_bench_scripts/serve.sh press_button_v1 1000
```

## Recommended First-Run Order

Run these in order, from `ext/openpi`:

```bash
export GIT_LFS_SKIP_SMUDGE=1
./am_bench_scripts/prepare_pi05_base_pytorch.sh
INPUT_DIR=/path/to/raw/press_button_hdf5 ./am_bench_scripts/convert_press_button.sh
./am_bench_scripts/compute_norm_stats.sh
./am_bench_scripts/train_press_button.sh press_button_v1
./am_bench_scripts/serve.sh press_button_v1 1000
```

## What I Validated Locally

I validated:

- the new `am_bench` config loads
- the converter works on synthetic HDF5 data
- norm stats can be computed for `pi05_am_bench_press_button`
- the repack and input transforms produce the expected `state`, `image`, `image_mask`, `prompt`, and `(10, 7)` action chunk

I did not run:

- a full real training run on your actual server dataset
- a real serving pass from a real fine-tuned checkpoint

## If You Need To Change The Integration

The main places to edit are:

- `ext/openpi/src/openpi/policies/am_bench_policy.py`
- `ext/openpi/src/openpi/training/config.py`
- `ext/openpi/am_bench_scripts/convert_press_button_hdf5_to_lerobot.py`

If you change the LeRobot dataset schema or the repo ID, keep the converter and `pi05_am_bench_press_button` config in sync.
