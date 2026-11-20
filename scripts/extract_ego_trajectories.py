"""Extract local ego trajectories from nuScenes metadata tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_table(metadata_dir: Path, name: str) -> list[dict]:
    with (metadata_dir / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def yaw_from_quaternion(rotation: list[float]) -> float:
    w, x, y, z = rotation
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def local_xy(origin: dict, pose: dict) -> dict[str, float]:
    dx = pose["translation"][0] - origin["translation"][0]
    dy = pose["translation"][1] - origin["translation"][1]
    yaw = yaw_from_quaternion(origin["rotation"])
    return {
        "x": math.cos(yaw) * dx + math.sin(yaw) * dy,
        "y": -math.sin(yaw) * dx + math.cos(yaw) * dy,
    }


def interpolate(points: list[tuple[float, dict[str, float]]], target_time: float) -> dict[str, float]:
    if target_time <= points[0][0]:
        return dict(points[0][1])
    if target_time >= points[-1][0]:
        return dict(points[-1][1])
    for (left_time, left), (right_time, right) in zip(points, points[1:]):
        if left_time <= target_time <= right_time:
            alpha = (target_time - left_time) / max(1e-9, right_time - left_time)
            return {
                "x": left["x"] + alpha * (right["x"] - left["x"]),
                "y": left["y"] + alpha * (right["y"] - left["y"]),
            }
    return dict(points[-1][1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/trajectories_with_ground_truth.jsonl"))
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.2)
    args = parser.parse_args()

    samples = {row["token"]: row for row in load_table(args.metadata_dir, "sample")}
    sample_data_rows = load_table(args.metadata_dir, "sample_data")
    sample_data = {row["token"]: row for row in sample_data_rows}
    front_sample_data = {}
    for row in sample_data_rows:
        is_front = row.get("channel") == "CAM_FRONT" or "/CAM_FRONT/" in row.get("filename", "")
        if is_front:
            front_sample_data[row["sample_token"]] = row
    ego_poses = {row["token"]: row for row in load_table(args.metadata_dir, "ego_pose")}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    with args.predictions.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            row = json.loads(line)
            frame_token = row["frame_token"]
            if frame_token in samples:
                sample = samples[frame_token]
            elif frame_token in sample_data:
                sample = samples[sample_data[frame_token]["sample_token"]]
            else:
                raise KeyError(f"Frame token is not a sample or sample_data token: {frame_token}")
            origin_sample_data = front_sample_data[sample["token"]]
            origin = ego_poses[origin_sample_data["ego_pose_token"]]
            origin_time = sample["timestamp"] / 1_000_000

            points = [(0.0, {"x": 0.0, "y": 0.0})]
            current = sample
            while current.get("next") and current["timestamp"] / 1_000_000 - origin_time < args.horizon:
                current = samples[current["next"]]
                current_sd = front_sample_data[current["token"]]
                pose = ego_poses[current_sd["ego_pose_token"]]
                relative_time = current["timestamp"] / 1_000_000 - origin_time
                points.append((relative_time, local_xy(origin, pose)))

            if len(points) == 1:
                continue
            ground_truth = [
                interpolate(points, step * args.dt)
                for step in range(round(args.horizon / args.dt) + 1)
            ]
            destination.write(
                json.dumps({**row, "ground_truth_trajectory": ground_truth})
                + "\n"
            )
            converted += 1

    print(f"Extracted ground-truth trajectories: {converted}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
