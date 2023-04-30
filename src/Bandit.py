"""
Created on Sunday Apr 2 2023
@author: Jordan Palmer
"""

import numpy as np
import random
from abc import ABC, abstractmethod
import scipy.stats as stats
import matplotlib.pyplot as plt


class Bandit(ABC):
    """ the base bandit class """

    def __init__(self, q):
        self.q = q  # the true reward value
        self.initialize()  # reset the arm

    def initialize(self):
        """ initialise bandit arm """
        self.Q = 0  # the estimate of this arm's reward value
        self.n = 0  # the number of times this arm has been tried

    @abstractmethod
    def simulate_observation(self):
        """ return a random amount of charge """
        pass

    @abstractmethod
    def update(self, R):
        """ update this arm after it has returned reward value 'R' """
        pass

    @abstractmethod
    def sample(self):
        """ return an estimate of the arm's reward value """
        pass

    @abstractmethod
    def plot_arms(self, x: np.ndarray, arms, true_values: list):
        """ plot probability distributions for each arm """
        pass


class GaussianThompson(Bandit):
    """
    Gaussian Thompson Bandit

    This bandit expects a reward distribution to be Gaussian.
    This assumes we know the variance of our reward distribution
    """

    def __init__(self, q, variance=1):
        self.tau_0 = 0.00001  # the posterior precision
        self.mu_0 = q  # the posterior mean
        self.tau = 1 / variance
        # pass the true reward value to the base Bandit class
        super().__init__(q)

    def sample(self, arm_idle_count=0, scale_factor=.1) -> float:
        """
        Return a value from the posterior normal distribution

        This function samples from our expected reward distribution. In this function, we
        can actively modify the variance to force the heuristic to more likely choose an arm.
        We do this in the cases where arms are not sampled enough and to account for an uncertainty
        that comes with not pulling an arm for a long period of time.

        :param arm_idle_count: Number of time steps since the arm was last pulled
        :param scale_factor: additional scale factor to adjust the variance
        :return: sample from the normal
        """
        # adjust variance by scale factor * idle count for every time step an arm is idle
        # if you don't want to adjust, set arm_idle_count to 0 or scalefactor
        print(self.tau_0)
        adjusted_variance = (1 / self.tau_0) + (1 / self.tau_0) * arm_idle_count * scale_factor
        adjusted_scale = np.sqrt(adjusted_variance)
        self.tau_0 = (1 / adjusted_scale ** 2)
        # sample from a normal as this is the expected distribution of our reward
        return np.random.normal(loc=self.mu_0, scale=adjusted_scale)

    def update(self, R) -> None:
        """
        Update the mean and precision of the normal

        :param R: observed reward
        :return: None
        """
        """ update this arm after it has returned reward value 'R' """
        # do a standard update of the estimated mean
        self.n += 1
        # the new estimate of the mean is calculated from the old estimate
        self.Q = (1 - 1.0 / self.n) * self.Q + (1.0 / self.n) * R
        # update the mean and precision of the posterior
        # can take these update functions from wiki
        self.mu_0 = ((self.tau_0 * self.mu_0) + (self.n * self.Q * self.tau)) / (self.tau_0 + self.n * self.tau)
        self.tau_0 += self.tau

    def simulate_observation(self, mu_vary=None) -> float:
        """
        Simulate an observation when pulling this arm

        :param mu_vary: vary the true mean of the reward distribution. Use for simulating change point
        :return: float
        """
        if mu_vary is None:
            mu_vary = self.q

        # the reward is a Gaussian distribution with unit variance
        # around the true value 'q'
        value = np.random.normal(loc=mu_vary, scale=1 / np.sqrt(self.tau))
        return value

    def plot_arms(self, x: np.ndarray, arms, true_values: list) -> None:
        """
        plot arms probability distributions

        :param x: domain to plot distributions over
        :param arm: list of class objects
        :param true_values: list of true means of each arm
        :return: plot, return 0
        """
        trials = sum([arm.n for arm in arms])
        norm = stats.norm

        for count, arm in enumerate(arms):
            y = norm.pdf(x, arm.mu_0, np.sqrt(1. / arm.tau_0))
            p = plt.plot(x, y, lw=2, label=f'{arm.n}/{trials}')
            c = p[0].get_markeredgecolor()
            plt.fill_between(x, y, 0, color=c, alpha=0.2)
            plt.axvline(true_values[count], linestyle='--', color=c)
            plt.autoscale(tight="True")
            plt.title(f"{trials} Trials")
            plt.legend()
            plt.autoscale(tight=True)


class BernoulliThompson(Bandit):
    """
    Bernoulli Thompson Bandit

    This bandit is useful when your expected reward is fixed between 0 - 1
    For example, conversion rate.

    To Do: implement ability to simulate change point detection. Not sure how to do this for Bernoulli trials.
           You can not scale the variance for idle arms!
    """

    def __init__(self, q):
        self.alpha = 1  # the number of times this socket returned a charge        
        self.beta = 1  # the number of times no charge was returned

        # pass the true reward value to the base PowerSocket             
        super().__init__(q)

    def simulate_observation(self) -> float:
        """ return some charge with the socket's predefined probability """
        return np.random.random() < self.q

    def update(self, R) -> None:
        """
        Increase the number of times this arm has been used and
            update the counts of the number of times the arm has and
            has not returned a unit reward (alpha and beta)
        :param R: observed reward
        :return: None
        """
        self.n += 1
        # update number of successes
        self.alpha += R
        # update number of failures (if R=0 --> failure --> increment beta)
        self.beta += (1 - R)

    def sample(self) -> float:
        """ return a value sampled from the beta distribution """
        return np.random.beta(self.alpha, self.beta)

    def plot_arms(self, x: np.ndarray, arms, true_values: list) -> None:
        """
        plot arms probability distributions

        :param x: domain to plot distributions over
        :param arms: list of class objects
        :param true_values: list of true means of each arm
        :return: plot, return 0
        """

        trials = sum([arm.n for arm in arms])
        beta = stats.beta

        for count, arm in enumerate(arms):
            y = beta(arms[count].alpha, arms[count].beta)
            p = plt.plot(x, y.pdf(x), lw=2, label=f'{arms[count].alpha - 1}/{arms[count].n}')
            c = p[0].get_markeredgecolor()
            plt.fill_between(x, y.pdf(x), 0, color=c, alpha=0.2)
            plt.axvline(true_values[count], linestyle='--', color=c)
            plt.autoscale(tight="True")
            plt.title(f"{trials} Trials")
            plt.legend()
            plt.autoscale(tight=True)
