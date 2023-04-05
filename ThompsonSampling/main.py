# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from Bandit import *
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
norm = stats.norm
gamma = stats.gamma
import scipy
import utils

if __name__ == '__main__':

    figsize(11.0, 10)

    x = np.linspace(0.0, 16.0, 200)
    # set the random seed to produce a recreatable graph
    
    seed = 15
    random.seed(seed)
    np.random.seed(seed)

    # create 5 arms in a fixed order
    arm_order = [2, 1, 3, 5, 4]

    arm_true_values = [((q * 2) + 2) for q in arm_order]

    print(f"True Values = {arm_true_values}")

    # create the arms
    # - the mean value of each arm is derived from the arm order index, which is doubled to give
    #   distinct values and offset by 2 to keep the distribution above zero

    arms = [GaussianThompson(q) for q in arm_true_values]
    # print('arms', arms)
    draw_samples = [1, 1, 3, 10, 10, 25, 150, 800]
    for j, i in enumerate(draw_samples):
        plt.subplot(4, 2, j + 1)

        for k in range(i):
            # choose the arm with the current highest sampled value or arbitrary select a arm in the case of a tie
            arm_samples = [arm.sample() for arm in arms]
            arm_index = utils.random_argmax(arm_samples)

            # charge from the chosen arm and update its mean reward value
            reward = arms[arm_index].simulate_observation()
            arms[arm_index].update(reward)

        arms[0].plot_arms(x, arms, arm_true_values)

        plt.autoscale(tight=True)

    plt.tight_layout()
    plt.show()