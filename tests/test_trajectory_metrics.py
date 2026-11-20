from drivelm.trajectory_metrics import average_displacement_error, final_displacement_error


def test_zero_error_for_identical_trajectories():
    trajectory = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}]

    assert average_displacement_error(trajectory, trajectory) == 0.0
    assert final_displacement_error(trajectory, trajectory) == 0.0
