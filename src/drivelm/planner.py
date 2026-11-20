"""Small deterministic baseline planner for the tactical-action interface."""

from dataclasses import dataclass
import math

from .actions import Action, action_profile


@dataclass(frozen=True)
class TrajectoryPoint:
    """A Cartesian trajectory sample relative to the ego vehicle."""

    time: float
    x: float
    y: float
    speed: float


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def plan_trajectory(
    action: Action | str,
    *,
    current_speed: float = 10.0,
    horizon: float = 3.0,
    dt: float = 0.2,
) -> list[TrajectoryPoint]:
    """Generate a smooth straight-road baseline trajectory.

    This is intentionally map-independent. It provides a tested contract for
    the learned tactical policy before the full nuScenes/Frenet implementation
    is connected.
    """

    if horizon <= 0 or dt <= 0:
        raise ValueError("horizon and dt must be positive")
    if current_speed < 0:
        raise ValueError("current_speed cannot be negative")

    profile = action_profile(action)
    count = max(1, math.ceil(horizon / dt))
    points: list[TrajectoryPoint] = []
    for index in range(count + 1):
        time = min(index * dt, horizon)
        phase = time / horizon
        blend = _smoothstep(phase)
        speed = current_speed + (profile.target_speed - current_speed) * blend
        distance = current_speed * time + 0.5 * (profile.target_speed - current_speed) * time * blend
        lateral = profile.lateral_offset * blend
        points.append(TrajectoryPoint(time=time, x=distance, y=lateral, speed=max(0.0, speed)))
    return points


def plan_reference_trajectory(
    action: Action | str,
    reference_path: list[dict[str, float]],
    *,
    current_speed: float = 10.0,
    horizon: float = 3.0,
    dt: float = 0.2,
) -> list[TrajectoryPoint]:
    """Follow a local lane reference path while applying a tactical action."""

    if len(reference_path) < 2:
        raise ValueError("reference_path must contain at least two points")
    if current_speed < 0 or horizon <= 0 or dt <= 0:
        raise ValueError("current_speed must be non-negative; horizon and dt must be positive")

    profile = action_profile(action)
    cumulative = [0.0]
    for left, right in zip(reference_path, reference_path[1:]):
        cumulative.append(
            cumulative[-1] + math.hypot(right["x"] - left["x"], right["y"] - left["y"])
        )

    def reference_at(distance: float) -> tuple[float, float, float]:
        distance = min(distance, cumulative[-1])
        for index in range(1, len(cumulative)):
            if distance <= cumulative[index]:
                span = max(1e-9, cumulative[index] - cumulative[index - 1])
                alpha = (distance - cumulative[index - 1]) / span
                left, right = reference_path[index - 1], reference_path[index]
                return (
                    left["x"] + alpha * (right["x"] - left["x"]),
                    left["y"] + alpha * (right["y"] - left["y"]),
                    left.get("yaw", 0.0) + alpha * (right.get("yaw", 0.0) - left.get("yaw", 0.0)),
                )
        last = reference_path[-1]
        return last["x"], last["y"], last.get("yaw", 0.0)

    count = max(1, math.ceil(horizon / dt))
    points: list[TrajectoryPoint] = []
    for index in range(count + 1):
        time = min(index * dt, horizon)
        phase = time / horizon
        blend = _smoothstep(phase)
        speed = current_speed + (profile.target_speed - current_speed) * blend
        distance = current_speed * time + 0.5 * (profile.target_speed - current_speed) * time * blend
        x, y, yaw = reference_at(distance)
        lateral = profile.lateral_offset * blend
        x -= math.sin(yaw) * lateral
        y += math.cos(yaw) * lateral
        points.append(TrajectoryPoint(time=time, x=x, y=y, speed=max(0.0, speed)))
    return points

