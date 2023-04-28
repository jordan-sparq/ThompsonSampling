from src.Bandit import *
import pytest
import numpy as np
@pytest.mark.parametrize("GaussianThompsonObject, expected_result", [
    (GaussianThompson([1, 2, 3]), [0, 0]),  # standard use case
    (GaussianThompson([1]), [0, 0]),  # a single q input
    (GaussianThompson(1), [0, 0]),  # a scalar input
    (GaussianThompson([1, 2, 3], variance=1), [0, 0])  # include variance as an input
])
def test_GaussianThompson_initialize(GaussianThompsonObject, expected_result):
    GaussianThompsonObject.initialize()
    assert [GaussianThompsonObject.Q, GaussianThompsonObject.n]

@pytest.mark.parametrize("GaussianThompsonObject, mu_vary, expected_result", [
    (GaussianThompson([1, 2, 3]), None, [True, True, True]),  # standard use case
    (GaussianThompson([1, 2, 3]), 1, [True, True, True]),  # with a mu_vary
])
def test_GaussianThompson_simulate_observation(GaussianThompsonObject, mu_vary, expected_result):
    """
    This is a rough test to see if the random value drawn from a normal is within a sensible range given the mean
    """
    GaussianThompsonObject.initialize()
    result=[-1*GaussianThompsonObject.q[0]-5 <= GaussianThompsonObject.simulate_observation(mu_vary=mu_vary)[0] <= GaussianThompsonObject.q[0] + 5,
           -1*GaussianThompsonObject.q[1]-5 <= GaussianThompsonObject.simulate_observation(mu_vary=mu_vary)[1] <= GaussianThompsonObject.q[1] + 5,
           -1*GaussianThompsonObject.q[2]-5 <= GaussianThompsonObject.simulate_observation(mu_vary=mu_vary)[2] <= GaussianThompsonObject.q[2] + 5
           ]
    assert result

@pytest.mark.parametrize("GaussianThompsonObject, mu_vary, expected_result", [
    (GaussianThompson([1, 2, 3]), None, 3),  # standard use case
    (GaussianThompson([1, 2, 3]), 1, 3),  # with a mu_vary
    (GaussianThompson([1]), 1, 1),  # one value
])
def test_GaussianThompson_simulate_observation_shape(GaussianThompsonObject, mu_vary, expected_result):
    """
    This is a rough test to see if the shape of the output of simulate_observation is correct
    This test would fail if the input is a scalar but user can do that
    """
    result = len(GaussianThompsonObject.simulate_observation(mu_vary=mu_vary))
    assert result

@pytest.mark.parametrize("GaussianThompsonObject, reward, expected_result", [
    (GaussianThompson(2), 100, [100, 100, 2]),  # standard use case
    (GaussianThompson(10), 10, [10, 10, 2]),  # with a mu_vary
])
def test_GaussianThompson_update(GaussianThompsonObject, reward, expected_result):
    """
    The update function should give you the reward when rounded as n=1 and that is all it has to go off
    """
    GaussianThompsonObject.initialize()
    GaussianThompsonObject.update(reward)
    assert [GaussianThompsonObject.Q, np.ceil(GaussianThompsonObject.mu_0), np.ceil(GaussianThompsonObject.tau_0)] == expected_result

@pytest.mark.parametrize("GaussianThompsonObject, arm_idle_count, scale_factor, expected_result", [
    (GaussianThompson(2), 1, 1, True),  # standard use case
    (GaussianThompson(10), 100, 1, True),  # with a mu_vary
])
def test_GaussianThompson_sample(GaussianThompsonObject, arm_idle_count, scale_factor, expected_result):
    GaussianThompsonObject.initialize()
    assert isinstance(GaussianThompsonObject.sample(arm_idle_count, scale_factor), float)


def test_plot_arms():
    assert True
