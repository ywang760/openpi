"""Legacy HDF5 compatibility converter for OpenPI AM-Bench data.

Prefer `scripts/data/export_lerobot_to_openpi.py` from the am_isaac repo for
new experiments. The default AM-Bench recording path already writes canonical
LeRobot datasets, so this script should only be used for older HDF5 sessions.

This script expects single-task AM Isaac episode files with:
- `obs/ee_pos`
- `obs/ee_quat`
- `obs/gripper_width`
- `actions`
- `images/ee_camera`
- optional `images/base_camera`

When `source_fps` is higher than `fps`, 7D EE delta actions are composed
over each raw downsampling window instead of simply subsampled.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro


def _load_am_isaac_resampling():
    repo_root = Path(__file__).resolve().parents[3]
    for source_path in (
        repo_root / "source" / "am_isaac_il",
        repo_root / "source" / "am_isaac",
    ):
        if source_path.is_dir() and str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))

    try:
        from am_isaac_il.processor.action_resampling import compose_delta_action_chunks
        from am_isaac_il.processor.action_resampling import compute_stride
    except ImportError as exc:
        raise RuntimeError(
            "Could not import am_isaac_il.processor.action_resampling. Run this "
            "converter from the am_isaac checkout with source/am_isaac and "
            "source/am_isaac_il importable, or install those packages into the "
            "OpenPI environment."
        ) from exc

    return compose_delta_action_chunks, compute_stride


def _compose_delta_action_chunks(actions: np.ndarray, stride: int) -> np.ndarray:
    import torch

    compose_delta_action_chunks, _ = _load_am_isaac_resampling()
    action_tensor = torch.as_tensor(actions, dtype=torch.float32)
    return (
        compose_delta_action_chunks(action_tensor, stride)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def _find_episode_files(input_dir: Path) -> list[Path]:
    episode_files = sorted(input_dir.glob("demo_*.hdf5"))
    if not episode_files:
        episode_files = sorted(input_dir.glob("demo_env_*_idx_*.hdf5"))
    if not episode_files:
        raise FileNotFoundError(f"No AM Isaac episode files found in {input_dir}")
    return episode_files


def _load_episode(file_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(file_path, "r") as episode:
        if "obs" not in episode:
            raise KeyError(f'Missing "obs" group in {file_path}')
        if "images" not in episode or "ee_camera" not in episode["images"]:
            raise KeyError(f'Missing required "images/ee_camera" dataset in {file_path}')

        obs = episode["obs"]
        data = {
            "ee_image": episode["images"]["ee_camera"][()].astype(np.uint8),
            "ee_pos": obs["ee_pos"][()].astype(np.float32),
            "ee_quat": obs["ee_quat"][()].astype(np.float32),
            "gripper_width": obs["gripper_width"][()].astype(np.float32),
            "actions": episode["actions"][()].astype(np.float32),
        }

        if data["actions"].shape[-1] != 7:
            raise ValueError(
                f"Expected relative 7D actions in {file_path}, got {data['actions'].shape}"
            )

        if "base_camera" in episode["images"]:
            data["base_image"] = episode["images"]["base_camera"][()].astype(np.uint8)
        else:
            data["base_image"] = np.zeros_like(data["ee_image"], dtype=np.uint8)

    num_steps = data["ee_pos"].shape[0]
    for key in ("ee_image", "base_image", "ee_quat", "gripper_width", "actions"):
        if data[key].shape[0] != num_steps:
            raise ValueError(
                f"Mismatched episode length for {key} in {file_path}: expected {num_steps}, "
                f"got {data[key].shape[0]}"
            )

    return data


def _get_downsample_stride(source_fps: int, target_fps: int) -> int:
    _, compute_stride = _load_am_isaac_resampling()
    return compute_stride(source_fps, target_fps)


def _get_downsample_indices(num_steps: int, source_fps: int, target_fps: int) -> np.ndarray:
    stride = _get_downsample_stride(source_fps, target_fps)
    num_full_windows = num_steps // stride
    if num_full_windows < 1:
        raise ValueError(
            f"Episode has {num_steps} steps, which is shorter than one "
            f"downsample window of {stride} raw steps."
        )
    return np.arange(0, num_full_windows * stride, stride, dtype=np.int64)


def main(
    input_dir: str,
    repo_id: str,
    task_prompt: str = "insert the peg into the hole",
    fps: int = 30,
    source_fps: int | None = None,
) -> None:
    input_path = Path(input_dir).expanduser().resolve()
    if source_fps is None:
        source_fps = fps
    stride = _get_downsample_stride(source_fps, fps)

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    episode_files = _find_episode_files(input_path)
    sample = _load_episode(episode_files[0])
    ee_image_shape = tuple(sample["ee_image"].shape[1:])
    base_image_shape = tuple(sample["base_image"].shape[1:])

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="am_bench",
        fps=fps,
        features={
            "ee_image": {
                "dtype": "image",
                "shape": ee_image_shape,
                "names": ["height", "width", "channel"],
            },
            "base_image": {
                "dtype": "image",
                "shape": base_image_shape,
                "names": ["height", "width", "channel"],
            },
            "ee_pos": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["ee_pos"],
            },
            "ee_quat": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["ee_quat"],
            },
            "gripper_width": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_width"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for episode_file in episode_files:
        episode = _load_episode(episode_file)
        num_steps = episode["ee_pos"].shape[0]
        downsample_indices = _get_downsample_indices(num_steps, source_fps, fps)
        usable_steps = len(downsample_indices) * stride
        actions = _compose_delta_action_chunks(episode["actions"][:usable_steps], stride)
        for logical_idx, step_idx in enumerate(downsample_indices):
            dataset.add_frame(
                {
                    "ee_image": episode["ee_image"][step_idx],
                    "base_image": episode["base_image"][step_idx],
                    "ee_pos": episode["ee_pos"][step_idx],
                    "ee_quat": episode["ee_quat"][step_idx],
                    "gripper_width": episode["gripper_width"][step_idx],
                    "actions": actions[logical_idx],
                    "task": task_prompt,
                }
            )
        dataset.save_episode()


if __name__ == "__main__":
    tyro.cli(main)
