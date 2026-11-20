"""Convert model behavior predictions into planner trajectory artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from drivelm.action_labels import normalize_behavior_action
from drivelm.pipeline import generate_action_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/predicted_trajectories.jsonl"))
    parser.add_argument("--current-speed", type=float, default=10.0)
    args = parser.parse_args()

    total = 0
    converted = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            row = json.loads(line)
            total += 1
            action = normalize_behavior_action(row["prediction"])
            if action is None:
                continue
            artifact = generate_action_trajectory(
                action,
                current_speed=args.current_speed,
            )
            destination.write(
                json.dumps(
                    {
                        "index": row.get("index"),
                        "scene_token": row.get("scene_token"),
                        "frame_token": row.get("frame_token"),
                        "reference_behavior": row.get("reference"),
                        "predicted_behavior": row.get("prediction"),
                        **artifact,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            converted += 1

    print(f"Read predictions: {total}")
    print(f"Converted trajectories: {converted}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
