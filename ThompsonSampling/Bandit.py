"""
Created on Sunday Apr 2 2023
@author: Jordan Palmer
"""

import numpy as np
import random
from abc import ABC, abstractmethod
import scipy.stats as stats
import matplotlib.pyplot as plt

class Bandit:
    """ the base bandit class """

    def __init__(self, q):
        self.q = q  # the true reward value
        self.initialize()  # reset the arm

    def initialize(self):
        """ initialise bandit arm """
        self.Q = 0  # the estimate of this arm's reward value
        self.n = 0  # the number of times this arm has been tried

    def simulate_observation(self):
        """ return a random amount of charge """
        pass

    def update(self, R):
        """ update this arm after it has returned reward value 'R' """
        pass

    def sample(self):
        """ return an estimate of the arm's reward value """
        pass

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
        self.tau_0 = 0.0001  # the posterior precision
        self.mu_0 = 1  # the posterior mean
        self.tau = 1/variance
        # pass the true reward value to the base Bandit class
        super().__init__(q)

    def sample(self):
        """ return a value from the posterior normal distribution """
        return np.random.normal(loc=self.mu_0, scale=1/np.sqrt(self.tau_0))

    def update(self, R):
        """ update this arm after it has returned reward value 'R' """
        # do a standard update of the estimated mean
        self.n += 1
        # the new estimate of the mean is calculated from the old estimate
        self.Q = (1 - 1.0/self.n) * self.Q + (1.0/self.n) * R
        # update the mean and precision of the posterior
        self.mu_0 = ((self.tau_0 * self.mu_0) + (self.n * self.Q * self.tau)) / (self.tau_0 + self.n * self.tau)
        self.tau_0 += self.tau

    def simulate_observation(self):
        """ return a random observation """
        # the reward is a Gaussian distribution with unit variance
        # around the true value 'q'
        value = np.random.randn() + self.q
        return value

    def plot_arms(self, x: np.ndarray, arms, true_values: list):
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
        return 0