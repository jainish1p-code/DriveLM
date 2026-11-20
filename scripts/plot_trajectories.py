"""Plot predicted planner trajectories for the report and presentation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/predicted_trajectories.png"))
    args = parser.parse_args()

    grouped = defaultdict(list)
    with args.trajectories.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[row["action"]].append(row["trajectory"])

    actions = sorted(grouped)
    colors = {action: plt.get_cmap("tab10")(index) for index, action in enumerate(actions)}
    figure, (trajectory_axis, speed_axis) = plt.subplots(1, 2, figsize=(15, 6))
    legend_handles = []
    for action in actions:
        color = colors[action]
        first = True
        for trajectory in grouped[action]:
            trajectory_axis.plot(
                [point["x"] for point in trajectory],
                [point["y"] for point in trajectory],
                color=color,
                alpha=0.35,
                label=action if first else None,
            )
            speed_axis.plot(
                [point["time"] for point in trajectory],
                [point["speed"] for point in trajectory],
                color=color,
                alpha=0.35,
            )
            first = False
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=2, label=action)
        )

    trajectory_axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    trajectory_axis.set_title("Action-conditioned XY paths")
    trajectory_axis.set_xlabel("Longitudinal distance x (m)")
    trajectory_axis.set_ylabel("Lateral offset y (m)")
    trajectory_axis.grid(True, alpha=0.3)

    speed_axis.set_title("Speed profiles")
    speed_axis.set_xlabel("Time (s)")
    speed_axis.set_ylabel("Speed (m/s)")
    speed_axis.grid(True, alpha=0.3)
    speed_axis.legend(handles=legend_handles, fontsize="small")

    figure.suptitle("DriveLM Action-Conditioned Planner Trajectories")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
