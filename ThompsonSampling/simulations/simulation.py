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
import change_point.change_point_rupture as changepoint


class ArmTester:
    """ create and test a set of arms over a single test run

    credit:  Steve Roberts
    modified from: https://github.com/WhatIThinkAbout/BabyRobot/blob/master/Multi_Armed_Bandits/PowerarmSystem.py
    """

    def __init__(self, bandit=GaussianThompson, arm_values: list = [], multiplier=2, **kwargs):

        # some quantities to track
        self.arm_stats = None
        self.total_reward_per_timestep = None
        self.reward_per_timestep = None
        self.total_reward = None
        self.total_steps = None
        self.number_of_steps = None

        # create supplied arm type with a mean value defined by arm order
        self.arms = [bandit((q * multiplier), **kwargs) for q in arm_values]

        # set the number of arms equal to the number created
        self.number_of_arms = len(self.arms)

        # the index of the best arm is the last in the arm_values list
        # - this is a one-based value so convert to zero-based
        self.optimal_arm_index = (arm_values[-1] - 1)

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

    def simulate_observation(self, arm_index, mu_vary=None):
        if mu_vary is None:
            return self.arms[arm_index].simulate_observation()
        else:
            # not all bandits will have this argument
            return self.arms[arm_index].simulate_observation(mu_vary=mu_vary)

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
        return self.arm_stats[:, :, 1][self.total_steps] / self.total_steps

    def get_optimal_arm_percentage(self):
        """ get the percentage of times the optimal arm was tried """
        final_trials = self.arm_stats[:, :, 1][self.total_steps]
        return final_trials[self.optimal_arm_index] / self.total_steps

    def get_time_steps(self):
        """ get the number of time steps that the test ran for """
        return self.total_steps

    def select_arm(self, t):
        """ Greedy arm Selection"""

        # choose the arm with the current highest mean reward or arbitrarily
        # select a arm in the case of a tie            
        arm_index = utils.random_argmax([arm.sample() for arm in self.arms])
        return arm_index

    def run(self,
            number_of_steps,
            maximum_total_reward=float('inf'),
            change_point_method: str = None,
            change_point_burn: int = 10,
            adjust_for_change_point: bool = False,
            change_point_window: int = None,
            simulate_change_point_dict: dict = {}
            ):
        """ perform a single run, over the set of arms, 
            for the defined number of steps """

        # check change point is a valid method
        change_point_methods = {'window': changepoint.window, 'Pelt': changepoint.pelt}
        assert (change_point_method in change_point_methods.keys()) or (change_point_method is None)

        # store rewards for change point detection
        arm_observations = [[] for _ in range(len(self.arms))]

        # reset the run counters
        self.initialize_run(number_of_steps)

        # loop for the specified number of time-steps
        for t in range(number_of_steps):

            # get information about all arms at the start of the time step
            self.arm_stats[t] = self.get_arm_stats(t)

            # select an arm
            arm_index = self.select_arm(t)

            # simulate change point
            if len(simulate_change_point_dict) != 0 \
                    and len(arm_observations[simulate_change_point_dict['simulate_changepoint_arm_index']]) >= simulate_change_point_dict['simulate_changepoint_step'] \
                    and arm_index == simulate_change_point_dict['simulate_changepoint_arm_index']:
                # check user wants a change point, and we have a change point at the desired step, and desired arm
                reward = self.simulate_observation(arm_index, mu_vary=simulate_change_point_dict['simulate_changepoint_vary'])

            else:
                # simulate_observation from the chosen arm and update its mean reward value
                reward = self.simulate_observation(arm_index)

            self.update(arm_index, reward)

            # CHANGE POINT DETECTION
            if change_point_method is not None:  # check for change point
                if (change_point_window is None) or (len(arm_observations[arm_index]) < change_point_window):
                    # append to unconstrained list if no change point window
                    # store reward for change point detection
                    arm_observations[arm_index].append(reward)
                else:
                    # if there is a change point window, only store that many rewards
                    arm_observations[arm_index].pop(0)
                    arm_observations[arm_index].append(reward)

                # change point detection - only when we have seen at least change_point_burn amount
                if len(arm_observations[arm_index]) >= change_point_burn:
                    result, distance, prob = change_point_methods[change_point_method](self.arms[arm_index],
                                                                                       arm_observations[arm_index])
                    if (result is not None) and adjust_for_change_point:
                        self.arms[arm_index].mu_0 += distance

            # test if the accumulated total reward is greater than the maximum
            if self.total_reward > maximum_total_reward:
                break

        # save the actual number of steps that have been run
        self.total_steps = t

        # get the stats for each arm at the end of the run        
        self.arm_stats[t + 1] = self.get_arm_stats(t + 1)
        # x = np.linspace(0, 20, 100)
        # self.arms[0].plot_arms(x, self.arms, [((q * self.multiplier) + 2) for q in self.arm_values])
        return self.total_steps, self.total_reward


class ArmExperiment:
    """ setup and run repeated arm tests to get the average results """

    def __init__(self,
                 arm_tester=ArmTester,
                 number_of_tests=1000,
                 number_of_steps=30,
                 maximum_total_reward=float('inf'),
                 **kwargs):

        self.reward_per_timestep = None
        self.cumulative_reward_per_timestep = None
        self.number_of_trials = None
        self.estimates = None
        self.arm_percentages = None
        self.mean_time_steps = None
        self.optimal_selected = None
        self.mean_total_reward = None
        self.arm_tester = arm_tester
        self.number_of_tests = number_of_tests
        self.number_of_steps = number_of_steps
        self.maximum_total_reward = maximum_total_reward
        self.number_of_arms = self.arm_tester.number_of_arms

    def initialize_run(self):

        # keep track of the average values over the run
        self.mean_total_reward = 0.
        self.optimal_selected = 0.
        self.mean_time_steps = 0.
        self.arm_percentages = np.zeros(self.number_of_arms)
        self.estimates = np.zeros(shape=(self.number_of_steps + 1, self.number_of_arms))
        self.number_of_trials = np.zeros(shape=(self.number_of_steps + 1, self.number_of_arms))
        # the cumulative total reward per timestep
        self.cumulative_reward_per_timestep = np.zeros(shape=self.number_of_steps)
        # the actual reward obtained at each timestep
        self.reward_per_timestep = np.zeros(shape=self.number_of_steps)

    def get_mean_total_reward(self):
        """ the final total reward averaged over the number of timesteps """
        return self.mean_total_reward

    def get_cumulative_reward_per_timestep(self):
        """ the cumulative total reward per timestep """
        return self.cumulative_reward_per_timestep

    def get_reward_per_timestep(self):
        """ the mean actual reward obtained at each timestep """
        return self.reward_per_timestep

    def get_optimal_selected(self):
        """ the mean times the optimal arm was selected """
        return self.optimal_selected

    def get_arm_percentages(self):
        """ the mean of the percentage times each arm was selected """
        return self.arm_percentages

    def get_estimates(self):
        """ per arm reward estimates """
        return self.estimates

    def get_number_of_trials(self):
        """ per arm number of trials """
        return self.number_of_trials

    def get_mean_time_steps(self):
        """ the average number of trials of each test """
        return self.mean_time_steps

    def update_mean_array(self, current_mean, new_value, n):
        """ calculate the new mean from the previous mean and the new value for an array """

        new_value = np.array(new_value)

        # pad the new array with its last value to make sure its the same length as the original           
        pad_length = (current_mean.shape[0] - new_value.shape[0])

        if pad_length > 0:
            new_array = np.pad(new_value, (0, pad_length), mode='constant', constant_values=new_value[-1])
        else:
            new_array = new_value

        return (1 - 1.0 / n) * current_mean + (1.0 / n) * new_array

    def update_mean(self, current_mean, new_value, n):
        """ calculate the new mean from the previous mean and the new value """
        return (1 - 1.0 / n) * current_mean + (1.0 / n) * new_value

    def record_test_stats(self, n):
        """ update the mean value for each statistic being tracked over a run """

        # calculate the new means from the old means and the new value
        tester = self.arm_tester
        self.mean_total_reward = self.update_mean(self.mean_total_reward, tester.get_mean_reward(), n)
        self.optimal_selected = self.update_mean(self.optimal_selected, tester.get_optimal_arm_percentage(), n)
        self.arm_percentages = self.update_mean(self.arm_percentages, tester.get_arm_percentages(), n)
        self.mean_time_steps = self.update_mean(self.mean_time_steps, tester.get_time_steps(), n)

        self.cumulative_reward_per_timestep = self.update_mean_array(self.cumulative_reward_per_timestep,
                                                                     tester.get_total_reward_per_timestep(), n)

        # check if the tests are only running until a maximum reward value is reached
        if self.maximum_total_reward == float('inf'):
            self.estimates = self.update_mean_array(self.estimates, tester.get_estimates(), n)
            self.cumulative_reward_per_timestep = self.update_mean_array(self.cumulative_reward_per_timestep,
                                                                         tester.get_total_reward_per_timestep(), n)
            self.reward_per_timestep = self.update_mean_array(self.reward_per_timestep,
                                                              tester.get_reward_per_timestep(), n)
            self.number_of_trials = self.update_mean_array(self.number_of_trials, tester.get_number_of_trials(), n)

    def run(self,
            maximum_total_reward=float('inf'),
            change_point_method: str = None,
            change_point_burn: int = 10,
            adjust_for_change_point: bool = False,
            change_point_window: int = None,
            simulate_change_point_dict = {}
            ):
        """ repeat the test over a set of arms for the specified number of trials """
        # do the specified number of runs for a single test
        self.initialize_run()
        for n in range(1, self.number_of_tests + 1):
            # do one run of the test
            self.arm_tester.run(self.number_of_steps,
                                maximum_total_reward=self.maximum_total_reward,
                                change_point_method=change_point_method,
                                change_point_burn=change_point_burn,
                                adjust_for_change_point=adjust_for_change_point,
                                change_point_window=change_point_window,
                                simulate_change_point_dict=simulate_change_point_dict,
                                )
            self.record_test_stats(n)


if __name__ == '__main__':
    """
    Example use cases
    """
    # basic use
    # ArmTester(bandit=GaussianThompson, arm_values=[2, 1, 3, 5, 4], multiplier=2).run(100)
    # ArmTester(bandit=BernoulliThompson, arm_values=[0.5, 0.1, 0.3, 0.2, 0.8], multiplier=1).run(100)