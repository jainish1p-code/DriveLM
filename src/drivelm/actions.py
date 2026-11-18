"""Canonical tactical actions shared by the VLM and motion planner."""

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    STOP = "STOP"
    YIELD = "YIELD"
    MAINTAIN_SPEED = "MAINTAIN_SPEED"
    ACCELERATE = "ACCELERATE"
    DECELERATE = "DECELERATE"
    LANE_CHANGE_LEFT = "LANE_CHANGE_LEFT"
    LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"
    SLIGHT_LEFT_NUDGE = "SLIGHT_LEFT_NUDGE"
    SLIGHT_RIGHT_NUDGE = "SLIGHT_RIGHT_NUDGE"
    TURNING_MANEUVER = "TURNING_MANEUVER"


@dataclass(frozen=True)
class ActionProfile:
    """Planner targets associated with one discrete tactical action.

    ``target_speed`` is in metres/second and ``lateral_offset`` is in metres.
    The offset is relative to the current reference lane centre.
    """

    target_speed: float
    lateral_offset: float = 0.0


ACTION_PROFILES: dict[Action, ActionProfile] = {
    Action.STOP: ActionProfile(0.0),
    Action.YIELD: ActionProfile(2.0),
    Action.MAINTAIN_SPEED: ActionProfile(10.0),
    Action.ACCELERATE: ActionProfile(14.0),
    Action.DECELERATE: ActionProfile(5.0),
    Action.LANE_CHANGE_LEFT: ActionProfile(10.0, 3.5),
    Action.LANE_CHANGE_RIGHT: ActionProfile(10.0, -3.5),
    Action.SLIGHT_LEFT_NUDGE: ActionProfile(10.0, 0.75),
    Action.SLIGHT_RIGHT_NUDGE: ActionProfile(10.0, -0.75),
    Action.TURNING_MANEUVER: ActionProfile(6.0, 1.5),
}


def action_profile(action: Action | str) -> ActionProfile:
    """Return the planner profile for an action or raise a useful error."""

    try:
        normalized = action if isinstance(action, Action) else Action(action.upper())
    except (AttributeError, ValueError) as exc:
        valid = ", ".join(item.value for item in Action)
        raise ValueError(f"Unknown action {action!r}; expected one of: {valid}") from exc
    return ACTION_PROFILES[normalized]

