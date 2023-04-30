from src.Bandit import *
import pytest
import numpy as np
@pytest.mark.parametrize("GaussianThompsonObject, expected_result", [
    ([GaussianThompson((q * 1),) for q in [1, 2, 3]], [[0, 0, 0], [0, 0, 0]]),  # standard use case
    ([GaussianThompson((q * 1),) for q in [1]], [[0, 0, 0], [0, 0, 0]]),  # a single q input
    ([GaussianThompson((q * 1), variance = 1) for q in [1, 2, 3]], [[0, 0, 0], [0, 0, 0]]), # include variance as an input
])
def test_GaussianThompson_initialize(GaussianThompsonObject, expected_result):
    # initialise
    [x.initialise() for x in GaussianThompsonObject]
    Qs = [x.Q for x in GaussianThompsonObject]
    ns = [x.n for x in GaussianThompsonObject]
    assert [Qs, ns]

@pytest.mark.parametrize("GaussianThompsonObject, mu_vary, expected_result", [
    ([GaussianThompson((q * 1),) for q in [1, 2, 3]], None, [True, True, True]),  # standard use case
    ([GaussianThompson((q * 1),) for q in [1, 2, 3]], 1, [True, True, True]),  # with a mu_vary
])
def test_GaussianThompson_simulate_observation(GaussianThompsonObject, mu_vary, expected_result):
    """
    This is a rough test to see if the random value drawn from a normal is within a sensible range given the mean
    """
    # initialise
    [x.initialise() for x in GaussianThompsonObject]

    result=[-1*GaussianThompsonObject[0].q - 5 <= GaussianThompsonObject[0].simulate_observation(mu_vary=mu_vary) <= GaussianThompsonObject[0].q + 5,
           -1*GaussianThompsonObject[1].q - 5 <= GaussianThompsonObject[1].simulate_observation(mu_vary=mu_vary) <= GaussianThompsonObject[1].q + 5,
           -1*GaussianThompsonObject[2].q - 5 <= GaussianThompsonObject[2].simulate_observation(mu_vary=mu_vary) <= GaussianThompsonObject[2].q + 5
           ]

    assert result

@pytest.mark.parametrize("GaussianThompsonObject, reward, expected_result", [
    (GaussianThompson(2), 100, [100, 100, 2]),  # standard use case
    (GaussianThompson(10), 10, [10, 10, 2]),  # with a mu_vary
])
def test_GaussianThompson_update(GaussianThompsonObject, reward, expected_result):
    """
    The update function should give you the reward when rounded as n=1 and that is all it has to go off
    """
    GaussianThompsonObject.initialise()
    GaussianThompsonObject.update(reward)
    assert [GaussianThompsonObject.Q, np.ceil(GaussianThompsonObject.mu_0), np.ceil(GaussianThompsonObject.tau_0)] == expected_result

@pytest.mark.parametrize("GaussianThompsonObject, arm_idle_count, scale_factor, expected_result", [
    (GaussianThompson(2), 1, 1, True),  # standard use case
    (GaussianThompson(10), 100, 1, True),  # with a mu_vary
])
def test_GaussianThompson_sample(GaussianThompsonObject, arm_idle_count, scale_factor, expected_result):
    GaussianThompsonObject.initialise()
    assert isinstance(GaussianThompsonObject.sample(arm_idle_count, scale_factor), float)


def test_plot_arms():
    assert True
