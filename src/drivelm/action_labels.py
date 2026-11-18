"""Heuristic normalization of DriveLM behavior answers to canonical actions."""

from __future__ import annotations

import re

from .actions import Action


def normalize_behavior_action(answer: str) -> Action | None:
    """Map a natural-language behavior answer to one canonical action.

    This conservative first pass returns ``None`` when the answer does not
    contain enough evidence for a single action. Unmapped examples should be
    manually reviewed before being used as ground-truth classification labels.
    """

    text = re.sub(r"\s+", " ", answer.lower().replace("_", " ")).strip()

    # Specific lateral maneuvers must be checked before generic turn phrases.
    if re.search(r"lane change|changes? lanes?", text):
        if re.search(r"left", text):
            return Action.LANE_CHANGE_LEFT
        if re.search(r"right", text):
            return Action.LANE_CHANGE_RIGHT
        return None
    if re.search(r"slight(ly)?\s+(nudge|move|veer|偏)?\s*left|nudge.*left|veer.*left", text):
        return Action.SLIGHT_LEFT_NUDGE
    if re.search(r"slight(ly)?\s+(nudge|move|veer|偏)?\s*right|nudge.*right|veer.*right", text):
        return Action.SLIGHT_RIGHT_NUDGE
    if re.search(r"steer(ing)?|turn(ing)?", text):
        if re.search(r"left", text) and re.search(r"slight", text):
            return Action.SLIGHT_LEFT_NUDGE
        if re.search(r"right", text) and re.search(r"slight", text):
            return Action.SLIGHT_RIGHT_NUDGE
        return Action.TURNING_MANEUVER
    if re.search(r"turn|turning|make a .* maneuver", text):
        return Action.TURNING_MANEUVER
    if re.search(r"stop|halt|come to a stop|full stop|not moving|stationary", text):
        return Action.STOP
    if re.search(r"yield|give way|right of way", text):
        return Action.YIELD
    if re.search(r"accelerat|speed up|increase speed", text):
        return Action.ACCELERATE
    if re.search(r"decelerat|slow down|driving slowly|brak|reduce speed", text):
        return Action.DECELERATE
    if re.search(r"driving (?:very )?fast|speeding|accelerat|speed up|increase speed", text):
        return Action.ACCELERATE
    if re.search(
        r"maintain speed|normal speed|keep (?:a )?steady speed|"
        r"continue straight|proceed straight|going straight",
        text,
    ):
        return Action.MAINTAIN_SPEED
    return None

