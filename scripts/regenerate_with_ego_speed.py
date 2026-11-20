"""Regenerate planner trajectories using nuScenes-derived initial ego speed."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from drivelm.action_labels import normalize_behavior_action
from drivelm.pipeline import generate_action_trajectory


def load_table(root: Path, name: str) -> list[dict]:
    with (root / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def ego_speed(sample: dict, samples: dict, front_data: dict, poses: dict) -> float:
    previous_token = sample.get("prev")
    if not previous_token:
        return 10.0
    previous = samples[previous_token]
    current_pose = poses[front_data[sample["token"]]["ego_pose_token"]]
    previous_pose = poses[front_data[previous["token"]]["ego_pose_token"]]
    dt = (sample["timestamp"] - previous["timestamp"]) / 1_000_000
    if dt <= 0:
        return 10.0
    dx = current_pose["translation"][0] - previous_pose["translation"][0]
    dy = current_pose["translation"][1] - previous_pose["translation"][1]
    return math.hypot(dx, dy) / dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/predicted_trajectories_pose_speed.jsonl"))
    args = parser.parse_args()

    samples = {row["token"]: row for row in load_table(args.metadata_dir, "sample")}
    sample_data_rows = load_table(args.metadata_dir, "sample_data")
    sample_data = {row["token"]: row for row in sample_data_rows}
    front_data = {
        row["sample_token"]: row
        for row in sample_data_rows
        if row.get("channel") == "CAM_FRONT" or "/CAM_FRONT/" in row.get("filename", "")
    }
    poses = {row["token"]: row for row in load_table(args.metadata_dir, "ego_pose")}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            row = json.loads(line)
            frame_token = row["frame_token"]
            sample = (
                samples[frame_token]
                if frame_token in samples
                else samples[sample_data[frame_token]["sample_token"]]
            )
            action = normalize_behavior_action(row["prediction"])
            if action is None:
                continue
            speed = ego_speed(sample, samples, front_data, poses)
            artifact = generate_action_trajectory(action, current_speed=speed)
            destination.write(
                json.dumps(
                    {
                        **row,
                        **artifact,
                        "initial_speed_from_ego_pose_mps": speed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
