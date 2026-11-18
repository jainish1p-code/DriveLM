from drivelm.action_labels import normalize_behavior_action


def test_behavior_action_examples_are_mappable():
    assert normalize_behavior_action("The ego vehicle is not moving.").value == "STOP"
    assert normalize_behavior_action("The ego vehicle is driving fast.").value == "ACCELERATE"

