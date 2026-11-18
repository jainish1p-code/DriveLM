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

