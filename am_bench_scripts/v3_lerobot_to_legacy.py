"""Bridge AM-Bench LeRobot v3 exports into OpenPI's pinned LeRobot layout.

`scripts/data/export_lerobot_to_openpi.py` in am_isaac writes the current
canonical LeRobot layout, which stores tasks and episode metadata as parquet.
This OpenPI fork is pinned to an older LeRobot revision that expects JSONL
metadata and one parquet file per episode. This script rewrites an already
resampled OpenPI-schema AM-Bench dataset into that older layout.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import Any, Iterator

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

REQUIRED_FEATURES = ("ee_image", "ee_pos", "ee_quat", "gripper_width", "actions")
OPTIONAL_FEATURES = ("base_image",)


def _load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _legacy_features(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Input LeRobot info.json is missing a features dictionary.")

    legacy_features: dict[str, dict[str, Any]] = {}
    for key in (*REQUIRED_FEATURES, *OPTIONAL_FEATURES):
        if key in OPTIONAL_FEATURES and key not in features:
            continue
        if key not in features:
            raise KeyError(f"Input dataset is missing required OpenPI AM-Bench feature '{key}'.")
        feature = dict(features[key])
        feature["shape"] = tuple(feature["shape"])
        if feature["dtype"] == "image" and len(feature["shape"]) == 3:
            feature["names"] = ["channel", "height", "width"]
        legacy_features[key] = feature
    return legacy_features


def _task_mapping(root: Path) -> dict[int, str]:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot v3 task table: {tasks_path}")

    tasks = pq.read_table(tasks_path).to_pydict()
    task_indices = tasks.get("task_index")
    task_names = tasks.get("task") or tasks.get("__index_level_0__")
    if task_indices is None or task_names is None:
        raise ValueError(f"Could not parse task table columns from {tasks_path}")

    return {int(index): str(task) for index, task in zip(task_indices, task_names, strict=True)}


def _iter_rows(root: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def _decode_image(value: Any, root: Path) -> Image.Image:
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is not None:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        image_path = value.get("path")
        if image_path is not None:
            return Image.open(root / image_path).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)}")


def _float32(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(shape)


def _legacy_frame(row: dict[str, Any], root: Path, tasks: dict[int, str]) -> dict[str, Any]:
    task_index = int(row["task_index"])
    if task_index not in tasks:
        raise KeyError(f"Frame references task_index={task_index}, which is missing from meta/tasks.parquet.")

    frame = {
        "ee_image": _decode_image(row["ee_image"], root),
        "ee_pos": _float32(row["ee_pos"], (3,)),
        "ee_quat": _float32(row["ee_quat"], (4,)),
        "gripper_width": _float32(row["gripper_width"], (1,)),
        "actions": _float32(row["actions"], (7,)),
        "task": tasks[task_index],
    }
    if "base_image" in row:
        frame["base_image"] = _decode_image(row["base_image"], root)
    return frame


def convert_dataset(
    *,
    input_root: Path,
    repo_id: str,
    output_root: Path | None,
    overwrite: bool,
    batch_size: int,
) -> Path:
    input_root = input_root.expanduser().resolve()
    output_root = (output_root or (HF_LEROBOT_HOME / repo_id)).expanduser().resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}")
        shutil.rmtree(output_root)

    info = _load_info(input_root)
    tasks = _task_mapping(input_root)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        robot_type=str(info.get("robot_type", "am_bench")),
        fps=int(info["fps"]),
        features=_legacy_features(info),
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    current_episode: int | None = None
    num_frames = 0
    for row in _iter_rows(input_root, batch_size):
        episode_index = int(row["episode_index"])
        if current_episode is None:
            current_episode = episode_index
        elif episode_index != current_episode:
            dataset.save_episode()
            current_episode = episode_index

        dataset.add_frame(_legacy_frame(row, input_root, tasks))
        num_frames += 1

    if current_episode is None:
        raise ValueError(f"No frames found under {input_root}")

    dataset.save_episode()
    print(f"Converted {num_frames} frames to OpenPI legacy LeRobot layout.")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3 AM-Bench OpenPI-schema exports to OpenPI's pinned LeRobot layout."
    )
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = convert_dataset(
        input_root=args.input_root,
        repo_id=args.repo_id,
        output_root=args.output_root,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )
    print(f"Wrote OpenPI legacy LeRobot dataset to {output_root}")


if __name__ == "__main__":
    main()
