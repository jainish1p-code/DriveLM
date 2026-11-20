from drivelm.pipeline import generate_action_trajectory
from drivelm.planner import plan_reference_trajectory


def test_pipeline_generates_stop_trajectory():
    artifact = generate_action_trajectory("STOP", current_speed=8.0)

    assert artifact["action"] == "STOP"
    assert artifact["trajectory"][-1]["speed"] == 0.0
    assert len(artifact["trajectory"]) == 16


def test_reference_planner_follows_reference_path():
    path = [
        {"x": 0.0, "y": 0.0, "yaw": 0.0},
        {"x": 5.0, "y": 1.0, "yaw": 0.2},
        {"x": 10.0, "y": 3.0, "yaw": 0.3},
    ]
    trajectory = plan_reference_trajectory("MAINTAIN_SPEED", path)

    assert trajectory[-1].y > 0
