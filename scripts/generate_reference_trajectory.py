"""Generate an action-conditioned trajectory along a nuScenes lane path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drivelm.planner import plan_reference_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_path", type=Path)
    parser.add_argument("action")
    parser.add_argument("--current-speed", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("reports/reference_trajectory.json"))
    args = parser.parse_args()

    map_artifact = json.loads(args.reference_path.read_text(encoding="utf-8"))
    trajectory = plan_reference_trajectory(
        args.action,
        map_artifact["reference_path"],
        current_speed=args.current_speed,
    )
    artifact = {
        **map_artifact,
        "action": args.action.upper(),
        "trajectory": [point.__dict__ for point in trajectory],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Action: {artifact['action']}")
    print(f"Reference points: {len(map_artifact['reference_path'])}")
    print(f"Trajectory points: {len(trajectory)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
