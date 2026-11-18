from drivelm.action_labels import normalize_behavior_action
from drivelm.actions import Action


def test_action_label_mapping():
    assert normalize_behavior_action("The ego vehicle comes to a full stop.") == Action.STOP
    assert normalize_behavior_action("The vehicle changes lanes to the right.") == Action.LANE_CHANGE_RIGHT
    assert normalize_behavior_action("The car should yield the right of way.") == Action.YIELD
    assert normalize_behavior_action("The vehicle continues straight at a steady speed.") == Action.MAINTAIN_SPEED


def test_ambiguous_action_is_not_forced():
    assert normalize_behavior_action("The ego vehicle moves through the scene.") is None

