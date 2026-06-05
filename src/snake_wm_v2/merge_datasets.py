from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .generate_dataset import assert_no_terminal_tint


ARRAYS = ("actions", "rewards", "dones", "lengths", "prev_rewards", "policy_ids")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge indexed no-overlay Snake transition datasets")
    p.add_argument("--out", required=True)
    p.add_argument("datasets", nargs="+")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_all = []
    context_all = []
    next_all = []
    episode_all = []
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ARRAYS}
    inputs = []
    frame_offset = 0
    episode_offset = 0

    for raw_path in args.datasets:
        path = Path(raw_path)
        meta = read_json(path / "dataset_meta.json")
        frames = np.load(path / "frames.npy")
        context_indices = np.load(path / "context_indices.npy") + frame_offset
        next_indices = np.load(path / "next_frame_indices.npy") + frame_offset
        episode_ids = np.load(path / "episode_ids.npy") + episode_offset
        frames_all.append(frames)
        context_all.append(context_indices)
        next_all.append(next_indices)
        episode_all.append(episode_ids)
        for key in ARRAYS:
            arrays[key].append(np.load(path / f"{key}.npy"))
        inputs.append({"path": path.as_posix(), "transitions": int(len(next_indices)), "frames": int(len(frames)), "meta": meta})
        frame_offset += int(len(frames))
        episode_offset = int(np.max(episode_ids) + 1 if len(episode_ids) else episode_offset)

    frames_merged = np.concatenate(frames_all, axis=0)
    assert_no_terminal_tint([frames_merged[i] for i in range(len(frames_merged))])
    np.save(out_dir / "frames.npy", frames_merged.astype(np.uint8))
    np.save(out_dir / "context_indices.npy", np.concatenate(context_all, axis=0).astype(np.int64))
    np.save(out_dir / "next_frame_indices.npy", np.concatenate(next_all, axis=0).astype(np.int64))
    np.save(out_dir / "episode_ids.npy", np.concatenate(episode_all, axis=0).astype(np.int64))
    for key, chunks in arrays.items():
        np.save(out_dir / f"{key}.npy", np.concatenate(chunks, axis=0))

    status_counts: dict[str, int] = {}
    for item in inputs:
        for status, count in item["meta"].get("status_counts", {}).items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
    write_json(out_dir / "dataset_meta.json", {
        "format": "indexed_npy_v2_merged_no_terminal_overlay",
        "path": out_dir.as_posix(),
        "inputs": inputs,
        "transitions": int(sum(item["transitions"] for item in inputs)),
        "frames": int(len(frames_merged)),
        "max_context": 5,
        "status_counts": status_counts,
        "corner_colors": sorted([list(c) for c in set(tuple(map(int, f[0, 0])) for f in frames_merged)]),
    })
    print(out_dir.as_posix())
    print(f"transitions={sum(item['transitions'] for item in inputs)} frames={len(frames_merged)}")


if __name__ == "__main__":
    main()
