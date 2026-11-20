"""Compute ADE/FDE from predicted and ground-truth trajectory JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drivelm.trajectory_metrics import average_displacement_error, final_displacement_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", type=Path)
    args = parser.parse_args()

    rows = []
    with args.trajectories.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(row)

    if not rows or "ground_truth_trajectory" not in rows[0]:
        raise ValueError("Input must contain ground_truth_trajectory for every row")
    ade_values = [
        average_displacement_error(row["trajectory"], row["ground_truth_trajectory"])
        for row in rows
    ]
    fde_values = [
        final_displacement_error(row["trajectory"], row["ground_truth_trajectory"])
        for row in rows
    ]
    print(f"Samples: {len(rows)}")
    print(f"ADE: {sum(ade_values) / len(ade_values):.3f} m")
    print(f"FDE: {sum(fde_values) / len(fde_values):.3f} m")


if __name__ == "__main__":
    main()
