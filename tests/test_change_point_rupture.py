from src.change_point.change_point_rupture import *
import pytest
import ruptures as rpt
import matplotlib.pyplot as plt
@pytest.mark.parametrize("bandit, observations, width, model, min_size, jump, pen, expected_result", [
    (GaussianThompson(2), rpt.pw_constant(100, 3, 4, noise_std=.1)[0], 2, 'l1', 2, 5, 2, 5),
    (GaussianThompson(2), rpt.pw_constant(100, 3, 1, noise_std=.1)[0], 2, 'l1', 2, 5, 2, 2),
])
def test_window(bandit, observations, width, model, min_size, jump, pen, expected_result):
    """
    Test window function finds the correct number of changepoints as the input observations
    """
    changepoints = window(bandit=bandit, observations=observations)
    if len(changepoints[0]) > 1:
        assert changepoints[1] is not None
        assert changepoints[2] is not None

    assert len(changepoints[0]) == expected_result

@pytest.mark.parametrize("bandit, observations, width, model, pen, expected_result", [
    (GaussianThompson(2), rpt.pw_constant(100, 3, 4, noise_std=0.1)[0], 2, 'l1', 5, 5),
    (GaussianThompson(2), rpt.pw_constant(100, 3, 1, noise_std=0.1)[0], 2, 'l1', 5, 2),
])
def test_pelt(bandit, observations, width, model, pen, expected_result):
    """
    Test Pelt function finds the correct number of changepoints as the input observations
    """
    changepoints = pelt(bandit=bandit, observations=observations)
    if len(changepoints[0]) > 1:
        assert changepoints[1] is not None
        assert changepoints[2] is not None

    assert len(changepoints[0]) == expected_result
