from Bandit import *
import ruptures as rpt
import scipy.stats

def window(bandit, observations, width=10, model='l1', min_size=2, jump=5):
    """ wrapper for the ruptures window function """
    assert len(observations) >= width, f"observations must be > width ({width})"
    # change point detection
    algo = rpt.Window(width=width, model=model, min_size=min_size, jump=jump).fit(np.array(observations))
    # result of change point detection. Returns a list of change points
    result = algo.predict(pen=10)
    # if result has a length > 1 then a change point has been found
    if len(result) > 1:
        print('Change point has been found')
        # how far is this observation from the expected mean?
        # 0th element of result is the change point location
        distance = observations[result[0]] - bandit.mu_0
        # what is the probability of this not being a change point
        prob = scipy.stats.norm.pdf(observations[result[0]], mu=bandit.mu_0, scale=1/bandit.tau_0)
        return result, distance, prob
    else:
        return None, None, None
