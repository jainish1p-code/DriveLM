"""Trajectory error metrics for planner evaluation."""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def _xy(point: Mapping[str, float]) -> tuple[float, float]:
    return float(point["x"]), float(point["y"])


def average_displacement_error(
    predicted: Iterable[Mapping[str, float]],
    ground_truth: Iterable[Mapping[str, float]],
) -> float:
    """Mean Euclidean distance between corresponding trajectory points."""

    errors = [
        math.dist(_xy(predicted_point), _xy(target_point))
        for predicted_point, target_point in zip(predicted, ground_truth)
    ]
    if not errors:
        raise ValueError("Trajectories must contain at least one corresponding point")
    return sum(errors) / len(errors)


def final_displacement_error(
    predicted: Iterable[Mapping[str, float]],
    ground_truth: Iterable[Mapping[str, float]],
) -> float:
    """Euclidean distance between the final corresponding points."""

    predicted_points = list(predicted)
    ground_truth_points = list(ground_truth)
    if not predicted_points or not ground_truth_points:
        raise ValueError("Trajectories must contain at least one point")
    return math.dist(_xy(predicted_points[-1]), _xy(ground_truth_points[-1]))
