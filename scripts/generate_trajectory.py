"""Generate a planner trajectory for one canonical DriveLM action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drivelm.pipeline import generate_action_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="Canonical action, e.g. STOP or ACCELERATE")
    parser.add_argument("--current-speed", type=float, default=10.0)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("reports/trajectory.json"))
    args = parser.parse_args()

    artifact = generate_action_trajectory(
        args.action,
        current_speed=args.current_speed,
        horizon=args.horizon,
        dt=args.dt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    final_point = artifact["trajectory"][-1]
    print(f"Action: {artifact['action']}")
    print(f"Samples: {len(artifact['trajectory'])}")
    print(f"Final x: {final_point['x']:.3f} m")
    print(f"Final y: {final_point['y']:.3f} m")
    print(f"Final speed: {final_point['speed']:.3f} m/s")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
