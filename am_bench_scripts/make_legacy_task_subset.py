#!/usr/bin/env python3
"""Create a single-task subset from an OpenPI legacy LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _repo_root(repo_id: str) -> Path:
    hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if not hf_lerobot_home:
        hf_lerobot_home = str(Path.home() / ".cache" / "huggingface" / "lerobot")
    return Path(hf_lerobot_home).expanduser().resolve() / repo_id


def _episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    relative_path = str(info["data_path"]).format(
        episode_chunk=episode_index // chunks_size,
        chunk_index=episode_index // chunks_size,
        episode_index=episode_index,
        file_index=episode_index,
    )
    return root / relative_path


def _index_stats(start: int, length: int) -> dict[str, list[float] | list[int]]:
    end = start + length - 1
    std = math.sqrt((length * length - 1) / 12) if length > 1 else 0.0
    return {
        "min": [start],
        "max": [end],
        "mean": [(start + end) / 2],
        "std": [std],
        "count": [length],
    }


def _constant_stats(value: int, length: int) -> dict[str, list[float] | list[int]]:
    return {
        "min": [value],
        "max": [value],
        "mean": [float(value)],
        "std": [0.0],
        "count": [length],
    }


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing expected column: {name}")
    return table.set_column(index, name, values)


def _rewrite_episode_table(table: pa.Table, *, episode_index: int, task_index: int, start_index: int) -> pa.Table:
    length = table.num_rows
    table = _replace_column(table, "episode_index", pa.array([episode_index] * length, type=pa.int64()))
    table = _replace_column(table, "task_index", pa.array([task_index] * length, type=pa.int64()))
    table = _replace_column(table, "frame_index", pa.array(range(length), type=pa.int64()))
    table = _replace_column(table, "index", pa.array(range(start_index, start_index + length), type=pa.int64()))
    return table


def create_subset(
    *,
    source_root: Path,
    target_root: Path,
    target_repo_id: str,
    task: str,
    overwrite: bool,
) -> None:
    source_meta = source_root / "meta"
    info = json.loads((source_meta / "info.json").read_text(encoding="utf-8"))
    episodes = _jsonl(source_meta / "episodes.jsonl")
    episode_stats_by_index = {
        int(row["episode_index"]): row for row in _jsonl(source_meta / "episodes_stats.jsonl")
    }

    selected = [episode for episode in episodes if task in episode.get("tasks", [])]
    if not selected:
        raise ValueError(f"No episodes found for task {task!r} in {source_root}")

    if target_root.exists():
        if not overwrite:
            raise FileExistsError(f"Target dataset already exists: {target_root}")
        shutil.rmtree(target_root)

    (target_root / "meta").mkdir(parents=True)
    (target_root / "data" / "chunk-000").mkdir(parents=True)

    new_episodes: list[dict[str, Any]] = []
    new_episode_stats: list[dict[str, Any]] = []
    global_index = 0

    for new_episode_index, old_episode in enumerate(selected):
        old_episode_index = int(old_episode["episode_index"])
        source_episode_path = _episode_path(source_root, info, old_episode_index)
        target_episode_path = _episode_path(target_root, info, new_episode_index)
        target_episode_path.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(source_episode_path)
        table = _rewrite_episode_table(
            table,
            episode_index=new_episode_index,
            task_index=0,
            start_index=global_index,
        )
        pq.write_table(table, target_episode_path)

        length = table.num_rows
        new_episodes.append({"episode_index": new_episode_index, "tasks": [task], "length": length})

        stats_row = json.loads(json.dumps(episode_stats_by_index[old_episode_index]))
        stats_row["episode_index"] = new_episode_index
        stats = stats_row["stats"]
        stats["episode_index"] = _constant_stats(new_episode_index, length)
        stats["task_index"] = _constant_stats(0, length)
        stats["frame_index"] = _index_stats(0, length)
        stats["index"] = _index_stats(global_index, length)
        new_episode_stats.append(stats_row)

        global_index += length

    new_info = json.loads(json.dumps(info))
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_frames"] = global_index
    new_info["total_tasks"] = 1
    new_info["total_chunks"] = 1
    new_info["splits"] = {"train": f"0:{len(new_episodes)}"}

    _write_json(target_root / "meta" / "info.json", new_info)
    _write_jsonl(target_root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(target_root / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(target_root / "meta" / "episodes_stats.jsonl", new_episode_stats)
    _write_json(
        target_root / "am_isaac_subset_report.json",
        {
            "source_root": str(source_root),
            "target_repo_id": target_repo_id,
            "target_root": str(target_root),
            "task": task,
            "episodes": len(new_episodes),
            "frames": global_index,
        },
    )

    print(f"Wrote {len(new_episodes)} episodes / {global_index} frames to {target_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_repo_id", required=True)
    parser.add_argument("--target_repo_id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source_root", type=Path, default=None)
    parser.add_argument("--target_root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve() if args.source_root else _repo_root(args.source_repo_id)
    target_root = args.target_root.expanduser().resolve() if args.target_root else _repo_root(args.target_repo_id)
    create_subset(
        source_root=source_root,
        target_root=target_root,
        target_repo_id=args.target_repo_id,
        task=args.task,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
