"""End-to-end action-to-trajectory interface."""

from __future__ import annotations

from typing import Any

from .actions import Action, action_profile
from .planner import plan_trajectory


def generate_action_trajectory(
    action: Action | str,
    *,
    current_speed: float = 10.0,
    horizon: float = 3.0,
    dt: float = 0.2,
) -> dict[str, Any]:
    """Convert one canonical action into a serializable trajectory artifact."""

    normalized_action = action if isinstance(action, Action) else Action(action.upper())
    profile = action_profile(normalized_action)
    trajectory = plan_trajectory(
        normalized_action,
        current_speed=current_speed,
        horizon=horizon,
        dt=dt,
    )
    return {
        "action": normalized_action.value,
        "current_speed_mps": current_speed,
        "target_speed_mps": profile.target_speed,
        "target_lateral_offset_m": profile.lateral_offset,
        "horizon_seconds": horizon,
        "dt_seconds": dt,
        "trajectory": [point.__dict__ for point in trajectory],
    }
