# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import scipy.stats as stats
from IPython.core.pylabtools import figsize
norm = stats.norm
gamma = stats.gamma
import utils
from change_point.change_point_rupture import *

def simulation_changepoint(adjust_for_changepoint = True, verbose =0):
    """
    simulate a bandit problem where there is a change point for one arm

    When we are at the 15th step, we adjust the mean of the true reward distribution from 5 to 15
    If adjust_for_changepoint is True, we shift the expected mean of the reward function based on the observation
    at the changepoint. i.e if we see 15 but epect 5, we shift by 10.

    :param adjust_for_changepoint: true if you want the expected mean to adjust to the observed changepoint
    :param verbose: print details about the changepoint of verbose = 1
    :return: plots
    """

    figsize(11.0, 10)

    x = np.linspace(0.0, 30.0, 200)
    # set the random seed to produce a recreatable graph
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # create 5 arms in a fixed order
    arm_order = [2, 1, 3, 5, 4]

    arm_true_values = [((q * 2) + 2) for q in arm_order]

    print(f"True Values = {arm_true_values}")

    arms = [GaussianThompson(q) for q in arm_true_values]
    # print('arms', arms)
    draw_samples = [1, 1, 3, 11, 10, 25, 150, 800]
    # store n observations for each arm for change point detection
    arm_observations = [[] for _ in range(len(arms))]

    for j, i in enumerate(draw_samples):
        plt.subplot(4, 2, j + 1)

        for k in range(i):
            # choose the arm with the current highest sampled value or arbitrary select a arm in the case of a tie
            arm_samples = [arm.sample() for arm in arms]
            arm_index = utils.random_argmax(arm_samples)
            # charge from the chosen arm and update its mean reward value
            # this will become a real observation but a simulation for now
            if len(arm_observations[arm_index]) >= 15:  # simulate breakpoint
                reward = arms[arm_index].simulate_observation(mu_vary=30)
            else:
                reward = arms[arm_index].simulate_observation()
            arms[arm_index].update(reward)
            # store observations
            arm_observations[arm_index].append(reward)
            # only want to store a rolling 30 observations
            # if len(arm_observations[arm_index]) >= 30:
            #     arm_observations[arm_index].pop(0)
            # only want to look at change points once we have more than 10 observations
            if len(arm_observations[arm_index]) >= 10:
                result, distance, prob = window(arms[arm_index], arm_observations[arm_index])
                if result is not None:
                    if verbose:
                        print(verbose)
                        print(f"Change point detected! At {result[0]}")
                    # adjust expected mean
                    if adjust_for_changepoint:
                        arms[arm_index].mu_0 += distance
                    # TO DO: adjust expected mean if we see a change point for x days

        arms[0].plot_arms(x, arms, arm_true_values)

        plt.autoscale(tight=True)

    plt.tight_layout()
    plt.show()

    plt.plot(arm_observations[3])
    plt.show()



if __name__ == '__main__':
    simulation_changepoint(adjust_for_changepoint=True, verbose=0)
    simulation_changepoint(adjust_for_changepoint=False, verbose=0)