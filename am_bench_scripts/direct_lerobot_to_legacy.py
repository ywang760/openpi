"""Directly export canonical AM-Bench LeRobot recordings to OpenPI's legacy layout.

This avoids the slower two-step path:

1. canonical LeRobot -> current LeRobot OpenPI-schema export
2. current LeRobot -> OpenPI's pinned legacy LeRobot layout

The script reads canonical AM-Bench parquet files directly, applies the same
delta-action downsampling semantics used by the AM-Bench resampler, and writes
the older JSONL-metadata / one-parquet-per-episode layout expected by this
OpenPI fork.
"""

from __future__ import annotations

import argparse
import functools
import importlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

DEFAULT_EE_IMAGE_KEY = "observation.images.ee_camera"
DEFAULT_BASE_IMAGE_KEY = "observation.images.base_camera"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"
TASK_PROMPT_MAP_KEYS = ("task_prompts", "task_prompt_map", "prompts")
EE_STATE_KEYS = ("ee_pos", "ee_quat", "gripper_width")
BASE_JOINT_STATE_KEYS = ("base_pos", "base_quat", "arm_joint_pos", "gripper_width")
STATE_KEY_DIMS = {
    "ee_pos": 3,
    "ee_quat": 4,
    "base_pos": 3,
    "base_quat": 4,
    "arm_joint_pos": 4,
    "gripper_width": 1,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@functools.cache
def _load_rotation_math():
    source_path = _repo_root() / "source" / "am_isaac_il"
    if source_path.is_dir() and str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

    try:
        rotation_math = importlib.import_module("am_isaac_il.utils.rotation_math")
    except ImportError as exc:
        raise RuntimeError(
            "Could not import AM-Bench rotation utilities from source/am_isaac_il. "
            "Run this script from the am_isaac checkout or keep source/am_isaac_il available."
        ) from exc

    return (
        rotation_math.normalize_quat,
        rotation_math.quat_mul,
        rotation_math.quat_to_rotvec,
        rotation_math.rotvec_to_quat,
    )


def compute_stride(raw_fps: int, target_fps: int) -> int:
    if target_fps < 1:
        raise ValueError(f"`target_hz` must be >= 1. Got {target_fps}.")
    if raw_fps < 1:
        raise ValueError(f"Source dataset fps must be >= 1. Got {raw_fps}.")
    if raw_fps % target_fps != 0:
        raise ValueError(f"Only integer downsampling is supported. Got source fps={raw_fps}, target_hz={target_fps}.")
    return raw_fps // target_fps


def compose_delta_action_chunks(actions: np.ndarray, stride: int) -> np.ndarray:
    normalize_quat, quat_mul, quat_to_rotvec, rotvec_to_quat = _load_rotation_math()

    action_tensor = torch.as_tensor(actions, dtype=torch.float32)
    if action_tensor.ndim != 2 or action_tensor.shape[1] != 7:
        raise ValueError(f"Expected delta actions with shape (N, 7). Got {tuple(action_tensor.shape)}.")
    if action_tensor.shape[0] % stride != 0:
        raise ValueError(
            "Expected raw action length to be divisible by stride. "
            f"Got length={action_tensor.shape[0]}, stride={stride}."
        )

    chunks = action_tensor.reshape(-1, stride, 7)
    delta_xyz = chunks[:, :, 0:3].sum(dim=1)

    rot_quats = rotvec_to_quat(chunks[:, :, 3:6].reshape(-1, 3)).reshape(-1, stride, 4)
    total_quat = rot_quats[:, 0, :]
    for step_index in range(1, stride):
        total_quat = normalize_quat(quat_mul(rot_quats[:, step_index, :], total_quat))
    delta_rot = quat_to_rotvec(total_quat).to(dtype=torch.float32)

    gripper = chunks[:, -1, 6:7]
    return torch.cat((delta_xyz, delta_rot, gripper), dim=1).cpu().numpy().astype(np.float32, copy=False)


def resolve_dataset_root(dataset_root: Path) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    if (dataset_root / "meta" / "info.json").is_file():
        return dataset_root

    canonical_root = dataset_root / "lerobot"
    if (canonical_root / "meta" / "info.json").is_file():
        return canonical_root

    raise FileNotFoundError(
        f"Could not find a LeRobot dataset under '{dataset_root}'. "
        "Expected either 'meta/info.json' directly or a 'lerobot/meta/info.json' child."
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_info(dataset_root: Path) -> dict[str, Any]:
    return load_json(dataset_root / "meta" / "info.json")


def resolve_state_slices(info: dict[str, Any], state_key: str) -> dict[str, slice]:
    state_shape = tuple(info["features"][state_key].get("shape", ()))
    if not state_shape:
        raise ValueError(f"Expected {state_key} to have a shape entry in source info.json.")

    state_dim = int(state_shape[0])
    am_isaac_info = info.get("am_isaac", {})
    state_keys = am_isaac_info.get("state_keys") if isinstance(am_isaac_info, dict) else None

    if isinstance(state_keys, list) and state_keys:
        slices: dict[str, slice] = {}
        offset = 0
        for key in state_keys:
            if not isinstance(key, str):
                raise ValueError(f"Expected am_isaac.state_keys to contain strings. Got {state_keys!r}.")
            if key not in STATE_KEY_DIMS:
                raise ValueError(f"Unsupported AM-Bench state key {key!r} in source metadata.")
            width = STATE_KEY_DIMS[key]
            slices[key] = slice(offset, offset + width)
            offset += width
        if offset > state_dim:
            raise ValueError(
                f"am_isaac.state_keys describe {offset} state dims, but {state_key} only has shape {state_shape}."
            )
        return slices

    if state_dim == 8:
        return {
            "ee_pos": slice(0, 3),
            "ee_quat": slice(3, 7),
            "gripper_width": slice(7, 8),
        }
    if state_dim == 12:
        return {
            "base_pos": slice(0, 3),
            "base_quat": slice(3, 7),
            "arm_joint_pos": slice(7, 11),
            "gripper_width": slice(11, 12),
        }

    raise ValueError(
        f"Could not infer state layout for {state_key} shape {state_shape}. "
        "Expected am_isaac.state_keys metadata, 8D EE state, or 12D Base+joints state."
    )


def require_state_keys(state_slices: dict[str, slice], required_keys: tuple[str, ...], *, context: str) -> None:
    missing = [key for key in required_keys if key not in state_slices]
    if missing:
        raise ValueError(f"{context} requires state keys {required_keys}, but metadata is missing {missing}.")


def load_task_prompt_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    raw = load_json(path.expanduser().resolve())
    for key in TASK_PROMPT_MAP_KEYS:
        nested = raw.get(key)
        if isinstance(nested, dict):
            raw = nested
            break

    prompts: dict[str, str] = {}
    for source_task_raw, prompt_raw in raw.items():
        if not isinstance(source_task_raw, str) or not isinstance(prompt_raw, str):
            raise ValueError(f"Task prompt map entries must be string pairs. Got {source_task_raw!r}: {prompt_raw!r}.")
        source_task = source_task_raw.strip()
        prompt = prompt_raw.strip()
        if not source_task or not prompt:
            raise ValueError(f"Task prompt map contains an empty key or prompt: {path}")
        prompts[source_task] = prompt
    return prompts


def load_tasks(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing canonical LeRobot task table: {tasks_path}")

    tasks = pq.read_table(tasks_path).to_pydict()
    task_indices = tasks.get("task_index")
    task_names = tasks.get("task") or tasks.get("__index_level_0__")
    if task_indices is None or task_names is None:
        raise ValueError(f"Could not parse task table columns from {tasks_path}")

    return {int(index): str(task) for index, task in zip(task_indices, task_names, strict=True)}


def load_episodes(dataset_root: Path) -> list[dict[str, Any]]:
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No canonical LeRobot episode tables under {dataset_root / 'meta' / 'episodes'}")

    episodes: list[dict[str, Any]] = []
    for episode_file in episode_files:
        episodes.extend(pq.read_table(episode_file).to_pylist())
    return sorted(episodes, key=lambda episode: int(episode["episode_index"]))


def source_data_file(dataset_root: Path, info: dict[str, Any], episode: dict[str, Any]) -> Path:
    data_path = str(info.get("data_path", ""))
    if not data_path:
        raise ValueError(f"Source dataset info.json is missing data_path: {dataset_root}")

    chunk_index = int(episode.get("data/chunk_index", episode.get("meta/episodes/chunk_index", 0)))
    file_index = int(episode.get("data/file_index", episode.get("meta/episodes/file_index", episode["episode_index"])))
    episode_index = int(episode["episode_index"])
    try:
        relative_path = data_path.format(
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )
    except KeyError as exc:
        raise KeyError(f"Unsupported data_path template '{data_path}' in {dataset_root}") from exc

    file_path = dataset_root / relative_path
    if not file_path.is_file():
        raise FileNotFoundError(f"Episode parquet file does not exist: {file_path}")
    return file_path


def validate_source_info(
    dataset_root: Path,
    info: dict[str, Any],
    *,
    ee_image_key: str,
    base_image_key: str,
    state_key: str,
    action_key: str,
    action_representation: str,
) -> None:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Source info.json has no feature dictionary: {dataset_root}")

    for key in (ee_image_key, state_key, action_key):
        if key not in features:
            raise KeyError(f"Source dataset {dataset_root} is missing required feature '{key}'.")

    state_shape = tuple(features[state_key].get("shape", ()))
    action_shape = tuple(features[action_key].get("shape", ()))
    if not state_shape:
        raise ValueError(f"Expected {state_key} to have a vector shape. Got shape {state_shape}.")

    action_semantics = info.get("am_isaac", {}).get("action_semantics")
    if action_representation == "delta":
        require_state_keys(
            resolve_state_slices(info, state_key),
            EE_STATE_KEYS,
            context="Direct OpenPI delta export",
        )
        if action_shape != (7,):
            raise ValueError(f"Expected 7D EE delta actions in {action_key}. Got shape {action_shape}.")
        if action_semantics not in (None, "ee_delta"):
            raise ValueError(
                f"Direct OpenPI delta export expects ee_delta source actions. "
                f"Dataset {dataset_root} reports action_semantics={action_semantics!r}."
            )
    elif action_representation == "ee_local_relative":
        require_state_keys(
            resolve_state_slices(info, state_key),
            EE_STATE_KEYS,
            context=f"Direct OpenPI {action_representation} export",
        )
        if action_shape != (8,):
            raise ValueError(f"Expected 8D EE absolute actions in {action_key}. Got shape {action_shape}.")
        if action_semantics != "ee_absolute":
            raise ValueError(
                f"Direct OpenPI {action_representation} export expects ee_absolute source actions. "
                f"Dataset {dataset_root} reports action_semantics={action_semantics!r}."
            )
    elif action_representation == "base_joint_relative":
        require_state_keys(
            resolve_state_slices(info, state_key),
            BASE_JOINT_STATE_KEYS,
            context="Direct OpenPI base_joint_relative export",
        )
        if action_shape != (12,):
            raise ValueError(f"Expected 12D Base+joints absolute actions in {action_key}. Got shape {action_shape}.")
        if action_semantics != "base_joint_absolute":
            raise ValueError(
                f"Direct OpenPI base_joint_relative export expects base_joint_absolute source actions. "
                f"Dataset {dataset_root} reports action_semantics={action_semantics!r}."
            )
    else:
        raise ValueError(f"Unsupported action representation: {action_representation}")

    if base_image_key not in features:
        print(f"Source dataset has no {base_image_key}; base_image will be omitted or zero-filled.")


def legacy_features(
    sample_shape: tuple[int, int, int],
    *,
    include_base_image: bool,
    action_dim: int,
    action_representation: str,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "ee_image": {
            "dtype": "image",
            "shape": sample_shape,
            "names": ["height", "width", "channel"],
        },
        "gripper_width": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_width"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    }
    if action_representation == "base_joint_relative":
        features.update(
            {
                "base_pos": {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": ["base_pos"],
                },
                "base_quat": {
                    "dtype": "float32",
                    "shape": (4,),
                    "names": ["base_quat"],
                },
                "arm_joint_pos": {
                    "dtype": "float32",
                    "shape": (4,),
                    "names": ["arm_joint_pos"],
                },
            }
        )
    else:
        features.update(
            {
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
            }
        )
    if include_base_image:
        features["base_image"] = {
            "dtype": "image",
            "shape": sample_shape,
            "names": ["height", "width", "channel"],
        }
    return features


def decode_image(value: Any, dataset_root: Path) -> Image.Image:
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is not None:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        image_path = value.get("path")
        if image_path is not None:
            return Image.open(dataset_root / image_path).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)}")


def resolve_prompt(
    source_task: str,
    *,
    task_prompt: str,
    prompt_map: dict[str, str],
    require_prompt_map: bool,
) -> str:
    if task_prompt:
        return task_prompt
    if source_task in prompt_map:
        return prompt_map[source_task]
    if require_prompt_map:
        raise KeyError(
            f"Missing language instruction for source task '{source_task}'. "
            "Add it to --task_prompt_map or remove --require_task_prompt_map."
        )
    return source_task


def selected_episode_rows(
    dataset_root: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    *,
    columns: list[str],
) -> list[dict[str, Any]]:
    file_path = source_data_file(dataset_root, info, episode)
    expected_length = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
    episode_index = int(episode["episode_index"])

    read_columns = list(dict.fromkeys([*columns, "episode_index"]))
    rows = pq.read_table(file_path, columns=read_columns).to_pylist()
    if len(rows) != expected_length:
        rows = [row for row in rows if int(row["episode_index"]) == episode_index]

    if len(rows) != expected_length:
        raise ValueError(
            f"Unexpected row count in {file_path} for episode {episode_index}: "
            f"expected {expected_length}, got {len(rows)} after filtering by episode_index."
        )

    if "episode_index" not in columns:
        for row in rows:
            row.pop("episode_index", None)
    return rows


def create_dataset(
    *,
    repo_id: str,
    output_root: Path,
    target_hz: int,
    image_shape: tuple[int, int, int],
    include_base_image: bool,
    image_writer_threads: int,
    image_writer_processes: int,
    action_dim: int,
    action_representation: str,
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        robot_type="am_bench",
        fps=target_hz,
        features=legacy_features(
            image_shape,
            include_base_image=include_base_image,
            action_dim=action_dim,
            action_representation=action_representation,
        ),
        use_videos=False,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def export_episode(
    dataset: LeRobotDataset,
    *,
    dataset_root: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    tasks: dict[int, str],
    target_hz: int,
    stride: int,
    ee_image_key: str,
    base_image_key: str,
    state_key: str,
    action_key: str,
    include_base_image: bool,
    task_prompt: str,
    prompt_map: dict[str, str],
    require_prompt_map: bool,
    zero_base_image: np.ndarray | None,
    action_representation: str,
) -> int:
    columns = [state_key, action_key, ee_image_key, "task_index"]
    has_base_image = base_image_key in info["features"]
    if include_base_image and has_base_image:
        columns.append(base_image_key)

    rows = selected_episode_rows(dataset_root, info, episode, columns=columns)
    usable_raw_steps = (len(rows) // stride) * stride
    if usable_raw_steps < stride:
        return 0

    state_slices = resolve_state_slices(info, state_key)
    actions = np.asarray([row[action_key] for row in rows[:usable_raw_steps]], dtype=np.float32)
    if action_representation == "delta":
        exported_actions = compose_delta_action_chunks(actions, stride)
    elif action_representation in ("ee_local_relative", "base_joint_relative"):
        exported_actions = actions[::stride][: usable_raw_steps // stride].astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported action representation: {action_representation}")
    prompt_cache: dict[int, str] = {}

    for logical_index, raw_index in enumerate(range(0, usable_raw_steps, stride)):
        row = rows[raw_index]
        state = np.asarray(row[state_key], dtype=np.float32).reshape(-1)

        task_index = int(row["task_index"])
        if task_index not in prompt_cache:
            source_task = tasks[task_index]
            prompt_cache[task_index] = resolve_prompt(
                source_task,
                task_prompt=task_prompt,
                prompt_map=prompt_map,
                require_prompt_map=require_prompt_map,
            )

        frame = {
            "ee_image": decode_image(row[ee_image_key], dataset_root),
            "gripper_width": state[state_slices["gripper_width"]],
            "actions": exported_actions[logical_index],
            "task": prompt_cache[task_index],
        }
        if action_representation == "base_joint_relative":
            frame.update(
                {
                    "base_pos": state[state_slices["base_pos"]],
                    "base_quat": state[state_slices["base_quat"]],
                    "arm_joint_pos": state[state_slices["arm_joint_pos"]],
                }
            )
        else:
            frame.update(
                {
                    "ee_pos": state[state_slices["ee_pos"]],
                    "ee_quat": state[state_slices["ee_quat"]],
                }
            )
        if include_base_image:
            if has_base_image:
                frame["base_image"] = decode_image(row[base_image_key], dataset_root)
            elif zero_base_image is not None:
                frame["base_image"] = zero_base_image
        dataset.add_frame(frame)

    dataset.save_episode()
    return usable_raw_steps // stride


def export_datasets(
    *,
    dataset_roots: list[Path],
    repo_id: str,
    output_root: Path | None,
    target_hz: int,
    ee_image_key: str,
    base_image_key: str,
    state_key: str,
    action_key: str,
    include_base_image: bool,
    task_prompt: str,
    task_prompt_map: dict[str, str],
    require_task_prompt_map: bool,
    overwrite: bool,
    max_episodes_per_source: int | None,
    image_writer_threads: int,
    image_writer_processes: int,
    action_representation: str,
) -> Path:
    output_root = (output_root or (HF_LEROBOT_HOME / repo_id)).expanduser().resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    resolved_roots = [resolve_dataset_root(root) for root in dataset_roots]
    source_infos = [load_info(root) for root in resolved_roots]
    for root, info in zip(resolved_roots, source_infos, strict=True):
        validate_source_info(
            root,
            info,
            ee_image_key=ee_image_key,
            base_image_key=base_image_key,
            state_key=state_key,
            action_key=action_key,
            action_representation=action_representation,
        )

    image_shape = tuple(source_infos[0]["features"][ee_image_key]["shape"])
    if len(image_shape) != 3:
        raise ValueError(f"Expected 3D image shape for {ee_image_key}, got {image_shape}.")
    action_dim = int(source_infos[0]["features"][action_key]["shape"][0])
    for root, info in zip(resolved_roots[1:], source_infos[1:], strict=True):
        root_shape = tuple(info["features"][ee_image_key]["shape"])
        if root_shape != image_shape:
            raise ValueError(f"Image shape mismatch: {root} has {root_shape}, expected {image_shape}.")
        root_action_dim = int(info["features"][action_key]["shape"][0])
        if root_action_dim != action_dim:
            raise ValueError(f"Action dim mismatch: {root} has {root_action_dim}, expected {action_dim}.")

    dataset = create_dataset(
        repo_id=repo_id,
        output_root=output_root,
        target_hz=target_hz,
        image_shape=image_shape,
        include_base_image=include_base_image,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
        action_dim=action_dim,
        action_representation=action_representation,
    )

    export_report: dict[str, Any] = {
        "repo_id": repo_id,
        "output_root": str(output_root),
        "target_hz": target_hz,
        "include_base_image": include_base_image,
        "action_representation": action_representation,
        "action_dim": action_dim,
        "sources": [],
    }

    zero_base_image = np.zeros(image_shape, dtype=np.uint8) if include_base_image else None
    total_frames = 0
    try:
        for source_index, (root, info) in enumerate(zip(resolved_roots, source_infos, strict=True)):
            source_fps = int(info["fps"])
            stride = compute_stride(source_fps, target_hz)
            tasks = load_tasks(root)
            episodes = load_episodes(root)
            if max_episodes_per_source is not None:
                episodes = episodes[:max_episodes_per_source]

            source_frames = 0
            for episode in episodes:
                written_frames = export_episode(
                    dataset,
                    dataset_root=root,
                    info=info,
                    episode=episode,
                    tasks=tasks,
                    target_hz=target_hz,
                    stride=stride,
                    ee_image_key=ee_image_key,
                    base_image_key=base_image_key,
                    state_key=state_key,
                    action_key=action_key,
                    include_base_image=include_base_image,
                    task_prompt=task_prompt,
                    prompt_map=task_prompt_map,
                    require_prompt_map=require_task_prompt_map,
                    zero_base_image=zero_base_image,
                    action_representation=action_representation,
                )
                source_frames += written_frames
                total_frames += written_frames
                if dataset.meta.total_episodes % 10 == 0:
                    print(
                        f"progress: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames written"
                    )

            source_tasks = sorted(set(tasks.values()))
            export_report["sources"].append(
                {
                    "dataset_root": str(root),
                    "source_index": source_index,
                    "source_fps": source_fps,
                    "stride": stride,
                    "episodes": len(episodes),
                    "frames": source_frames,
                    "task_prompts": {
                        task: resolve_prompt(
                            task,
                            task_prompt=task_prompt,
                            prompt_map=task_prompt_map,
                            require_prompt_map=require_task_prompt_map,
                        )
                        for task in source_tasks
                    },
                }
            )
    finally:
        dataset.stop_image_writer()

    if total_frames == 0:
        raise ValueError("No frames were exported.")

    report_path = output_root / "am_isaac_direct_legacy_export_report.json"
    report_path.write_text(json.dumps(export_report, indent=2), encoding="utf-8")
    print(f"Exported {dataset.meta.total_episodes} episodes / {dataset.meta.total_frames} frames.")
    print(f"Wrote direct legacy export report to {report_path}")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly export canonical AM-Bench LeRobot datasets to OpenPI's pinned LeRobot layout."
    )
    parser.add_argument("--dataset_roots", type=Path, nargs="+", required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--target_hz", type=int, required=True)
    parser.add_argument("--ee_image_key", type=str, default=DEFAULT_EE_IMAGE_KEY)
    parser.add_argument("--base_image_key", type=str, default=DEFAULT_BASE_IMAGE_KEY)
    parser.add_argument("--state_key", type=str, default=DEFAULT_STATE_KEY)
    parser.add_argument("--action_key", type=str, default=DEFAULT_ACTION_KEY)
    parser.add_argument(
        "--omit_base_image",
        action="store_true",
        help="Do not write base_image. Use this for current EE-camera-only AM-Bench datasets.",
    )
    parser.add_argument(
        "--action_representation",
        type=str,
        default="delta",
        choices=("delta", "ee_local_relative", "base_joint_relative"),
        help=(
            "Policy action representation to prepare. 'delta' expects 7D ee_delta source actions and "
            "composes them over --target_hz strides. 'ee_local_relative' expects 8D ee_absolute source "
            "actions and keeps absolute setpoints for quaternion-safe local-relative transforms during "
            "OpenPI training. 'base_joint_relative' expects 12D base_joint_absolute source actions "
            "and keeps absolute setpoints for Base+joints relative transforms during OpenPI training."
        ),
    )
    parser.add_argument("--task_prompt", type=str, default="")
    parser.add_argument("--task_prompt_map", type=Path, default=None)
    parser.add_argument("--require_task_prompt_map", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max_episodes_per_source",
        type=int,
        default=None,
        help="Optional smoke-test limit. Exports only the first N episodes from each source.",
    )
    parser.add_argument("--image_writer_threads", type=int, default=16)
    parser.add_argument("--image_writer_processes", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_prompt_map = load_task_prompt_map(args.task_prompt_map)
    if args.require_task_prompt_map and not task_prompt_map and not args.task_prompt:
        raise ValueError("--require_task_prompt_map requires --task_prompt_map unless --task_prompt is set.")
    if args.max_episodes_per_source is not None and args.max_episodes_per_source < 1:
        raise ValueError("--max_episodes_per_source must be >= 1 when set.")

    output_root = export_datasets(
        dataset_roots=args.dataset_roots,
        repo_id=args.repo_id,
        output_root=args.output_root,
        target_hz=args.target_hz,
        ee_image_key=args.ee_image_key,
        base_image_key=args.base_image_key,
        state_key=args.state_key,
        action_key=args.action_key,
        include_base_image=not args.omit_base_image,
        task_prompt=args.task_prompt,
        task_prompt_map=task_prompt_map,
        require_task_prompt_map=args.require_task_prompt_map,
        overwrite=args.overwrite,
        max_episodes_per_source=args.max_episodes_per_source,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        action_representation=args.action_representation,
    )
    print(f"Wrote OpenPI legacy LeRobot dataset to {output_root}")


if __name__ == "__main__":
    main()
