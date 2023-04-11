import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.core.pylabtools import figsize
import utils
from Bandit import *


class ArmTester:
    """ create and test a set of arms over a single test run """

    def __init__(self, bandit=GaussianThompson, arm_order: list = [], multiplier=2, **kwargs):

        # create supplied arm type with a mean value defined by arm order 
        self.arms = [bandit((q * multiplier) + 2, **kwargs) for q in arm_order]

        # set the number of arms equal to the number created
        self.number_of_arms = len(self.arms)

        # the index of the best arm is the last in the arm_order list
        # - this is a one-based value so convert to zero-based
        self.optimal_arm_index = (arm_order[-1] - 1)

        # by default a arm tester records 2 bits of information over a run
        self.number_of_stats = kwargs.pop('number_of_stats', 2)

    def initialize_run(self, number_of_steps):
        """ reset counters at the start of a run """

        # save the number of steps over which the run will take place
        self.number_of_steps = number_of_steps
        # reset the actual number of steps that the test ran for
        self.total_steps = 0
        # monitor the total reward obtained over the run
        self.total_reward = 0
        # the current total reward at each timestep of the run
        self.total_reward_per_timestep = []
        # the actual reward obtained at each timestep
        self.reward_per_timestep = []
        # stats for each time-step
        # - by default records: estimate, number of trials
        self.arm_stats = np.zeros(shape=(number_of_steps + 1,
                                         self.number_of_arms,
                                         self.number_of_stats))

        # ensure that all arms are re-initialized
        for arm in self.arms:
            arm.initialize()

    def simulate_observation(self, arm_index):
        return self.arms[arm_index].simulate_observation()

    def update(self, arm_index, reward):
        """ charge from & update the specified arm and associated parameters """

        # charge from the chosen arm and update its mean reward value
        self.arms[arm_index].update(reward)
        # update the total reward
        self.total_reward += reward
        # store the current total reward at this timestep
        self.total_reward_per_timestep.append(self.total_reward)
        # store the reward obtained at this timestep
        self.reward_per_timestep.append(reward)

    def get_arm_stats(self, t):
        """ get the current information from each arm """
        arm_stats = [[arm.Q, arm.n] for arm in self.arms]
        return arm_stats

    def get_mean_reward(self):
        """ the total reward averaged over the number of time steps """
        return (self.total_reward / self.total_steps)

    def get_total_reward_per_timestep(self):
        """ the cumulative total reward at each timestep of the run """
        return self.total_reward_per_timestep

    def get_reward_per_timestep(self):
        """ the actual reward obtained at each timestep of the run """
        return self.reward_per_timestep

    def get_estimates(self):
        """ get the estimate of each arm's reward at each timestep of the run """
        return self.arm_stats[:, :, 0]

    def get_number_of_trials(self):
        """ get the number of trials of each arm at each timestep of the run """
        return self.arm_stats[:, :, 1]

    def get_arm_percentages(self):
        """ get the percentage of times each arm was tried over the run """
        return (self.arm_stats[:, :, 1][self.total_steps] / self.total_steps)

    def get_optimal_arm_percentage(self):
        """ get the percentage of times the optimal arm was tried """
        final_trials = self.arm_stats[:, :, 1][self.total_steps]
        return (final_trials[self.optimal_arm_index] / self.total_steps)

    def get_time_steps(self):
        """ get the number of time steps that the test ran for """
        return self.total_steps

    def select_arm(self, t):
        """ Greedy arm Selection"""

        # choose the arm with the current highest mean reward or arbitrarily
        # select a arm in the case of a tie            
        arm_index = utils.random_argmax([arm.sample() for arm in self.arms])
        return arm_index

    def run(self, number_of_steps, maximum_total_reward=float('inf')):
        """ perform a single run, over the set of arms, 
            for the defined number of steps """

        # reset the run counters
        self.initialize_run(number_of_steps)

        # loop for the specified number of time-steps
        for t in range(number_of_steps):

            # get information about all arms at the start of the time step
            self.arm_stats[t] = self.get_arm_stats(t)

            # select a arm
            arm_index = self.select_arm(t)

            # simulate_observation from the chosen arm and update its mean reward value
            val = self.simulate_observation(arm_index)
            self.update(arm_index, val)

            # test if the accumulated total reward is greater than the maximum
            if self.total_reward > maximum_total_reward:
                break

        # save the actual number of steps that have been run
        self.total_steps = t

        # get the stats for each arm at the end of the run        
        self.arm_stats[t + 1] = self.get_arm_stats(t + 1)
        print(self.arm_stats[t + 1])
        # x = np.linspace(0, 20, 100)
        # self.arms[0].plot_arms(x, self.arms, [((q * self.multiplier) + 2) for q in self.arm_order])
        return self.total_steps, self.total_reward


if __name__ == '__main__':
    ArmTester(bandit=GaussianThompson, arm_order=[2, 1, 3, 5, 4], multiplier=2).run(100)
    ArmTester(bandit=BernoulliThompson, arm_order=[0.5, 0.1, 0.3, 0.2, 0.8], multiplier=1).run(100)