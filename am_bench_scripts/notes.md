# `am_bench` First Fine-Tuning Run Guide

This is the current step-by-step path for launching the first `pi-0.5` PyTorch fine-tuning run for `am_bench` on a server.

## Files That Matter

### Python
- `ext/openpi/src/openpi/policies/am_bench_policy.py`
  - runtime/training adapter for `am_bench` observations and actions
- `ext/openpi/src/openpi/training/config.py`
  - defines `LeRobotAmBenchDataConfig`
  - defines `pi05_am_bench_peg_in_hole`
- `ext/openpi/am_bench_scripts/hdf5_to_lerobot.py`
  - converts raw AM Isaac HDF5 episodes into LeRobot format

### Shell
- `ext/openpi/am_bench_scripts/prepare_pi05_base_pytorch.sh`
  - patches `transformers` inside the local `uv` environment and converts the base JAX checkpoint to PyTorch
- `ext/openpi/am_bench_scripts/convert_dataset.sh`
  - converts raw AM Isaac HDF5 episodes into a LeRobot dataset under `~/.cache/huggingface/lerobot`
- `ext/openpi/am_bench_scripts/compute_norm_stats.sh`
  - computes norm stats for the default `pi05_am_bench_peg_in_hole` config
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

## Step 2: Convert Raw HDF5 Data To LeRobot

For the first run, keep the default repo ID. Do not change it unless you also update the training config.

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
~/.cache/huggingface/lerobot/am_bench/peg_in_hole
```

Quick check:

```bash
find ~/.cache/huggingface/lerobot/am_bench/peg_in_hole -maxdepth 2 -type f | sort
```


## Step 3: Compute Norm Stats

Run:

```bash
./am_bench_scripts/compute_norm_stats.sh
```

What this does:

- loads the converted `am_bench/peg_in_hole` LeRobot dataset
- runs the default `pi05_am_bench_peg_in_hole` data path
- writes normalization statistics for `state` and `actions`

Expected output:

```bash
./assets/pi05_am_bench_peg_in_hole/am_bench/peg_in_hole/norm_stats.json
```

Quick check:

```bash
ls ./assets/pi05_am_bench_peg_in_hole/am_bench/peg_in_hole/norm_stats.json
```

## Step 4: Launch Training

Choose an experiment name and run:

```bash
./am_bench_scripts/train.sh peg_in_hole_v1
```

What this does:

- loads `pi05_am_bench_peg_in_hole`
- launches `torchrun` with `--nnodes=1 --nproc_per_node=4`
- loads the converted base PyTorch checkpoint from:

```bash
~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

- trains using the LeRobot dataset `am_bench/peg_in_hole`

Batching note:

- the config currently uses `batch_size=32`
- this trainer treats that as total batch size across all GPUs
- with 4 GPUs, the effective batch size is `8` per GPU
- writes checkpoints under:

```bash
./checkpoints/pi05_am_bench_peg_in_hole/peg_in_hole_v1
```

Quick check during training:

```bash
find ./checkpoints/pi05_am_bench_peg_in_hole/peg_in_hole_v1 -maxdepth 2 -type f | head
```

Things to watch:

- disk space in `./checkpoints`
- GPU memory
- whether the model loads `model.safetensors` from the base checkpoint path above

If this step fails early:

- confirm Step 1 created `~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors`
- confirm Step 3 created the norm stats file
- confirm the LeRobot dataset exists at `~/.cache/huggingface/lerobot/am_bench/peg_in_hole`

### Where To Change Training Parameters

For this pipeline, most training changes should go in:

- `ext/openpi/src/openpi/training/config.py`

The main block to edit is the `TrainConfig(...)` with:

- `name="pi05_am_bench_peg_in_hole"`

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
./am_bench_scripts/serve.sh peg_in_hole_v1 1000
```

This expects the checkpoint at,

```bash
./checkpoints/pi05_am_bench_peg_in_hole/peg_in_hole_v1/1000
```
