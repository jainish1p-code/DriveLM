from drivelm.actions import Action, action_profile
from drivelm.planner import plan_trajectory


def test_all_proposal_actions_have_profiles():
    for action in Action:
        assert action_profile(action).target_speed >= 0


def test_stop_reaches_zero_speed_and_preserves_horizon():
    trajectory = plan_trajectory(Action.STOP, current_speed=8.0, horizon=3.0, dt=0.5)

    assert trajectory[0].time == 0.0
    assert trajectory[-1].time == 3.0
    assert trajectory[-1].speed == 0.0
    assert all(point.speed >= 0 for point in trajectory)


def test_lane_change_moves_to_expected_side():
    left = plan_trajectory(Action.LANE_CHANGE_LEFT, current_speed=10.0)
    right = plan_trajectory(Action.LANE_CHANGE_RIGHT, current_speed=10.0)

    assert left[-1].y > 0
    assert right[-1].y < 0

