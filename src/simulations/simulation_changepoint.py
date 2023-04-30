from src.Bandit import *
import simulation

if __name__ == '__main__':
    """
    Example use cases
    """

    # if we want a change point to be simulated for one arm
    simulate_change_point_dict = {
        'simulate_changepoint_arm_index': 3,
        'simulate_changepoint_vary': 30,
        'simulate_changepoint_step': 20,
    }

    # basic use
    # ArmTester(bandit=GaussianThompson, arm_values=[2, 1, 3, 5, 4], multiplier=2).run(100)
    # ArmTester(bandit=BernoulliThompson, arm_values=[0.5, 0.1, 0.3, 0.2, 0.8], multiplier=1).run(100)

    # experiments with change point
    gaussian_thompson = simulation.ArmTester(bandit=GaussianThompson, arm_values=[2, 1, 3, 5, 4], multiplier=2)
    number_of_steps = 100
    experiment_gaussian = simulation.ArmExperiment(arm_tester=gaussian_thompson, number_of_steps=number_of_steps)
    experiment_gaussian.run(change_point_method='window', change_point_window=30,
                            simulate_change_point_dict=simulate_change_point_dict)

    # REGRET no change point
    cumulative_optimal_reward_nocp = [r * 10 for r in range(1, number_of_steps + 1)]
    regret_nochangepoint = cumulative_optimal_reward_nocp - experiment_gaussian.get_cumulative_reward_per_timestep()

    # REGRET with change point detection ( we know where the change point is)
    cumulative_optimal_reward = []
    for r in range(1, number_of_steps + 1):
        if r < simulate_change_point_dict['simulate_changepoint_step']:
            cumulative_optimal_reward.append(10)
        else:
            cumulative_optimal_reward.append(simulate_change_point_dict['simulate_changepoint_vary'])

    cumulative_optimal_reward = np.cumsum(cumulative_optimal_reward)
    regret = cumulative_optimal_reward - experiment_gaussian.get_cumulative_reward_per_timestep()

    fig = plt.figure(figsize=(20, 6))
    plt.suptitle(f'Epsilon Greedy Regret', fontsize=20, fontweight='bold')

    plt.subplot(1, 2, 1)
    plt.plot(experiment_gaussian.get_cumulative_reward_per_timestep(), label="Actual")
    plt.plot(cumulative_optimal_reward, label="Optimal")
    plt.plot(cumulative_optimal_reward_nocp, label='No change point')
    plt.plot(regret, label="Regret")
    plt.legend()
    plt.title('Cumulative Reward vs Time', fontsize=15)
    plt.xlabel('Time Steps')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.title('Regret vs Time', fontsize=15)
    plt.plot(regret, label='Accounting for change points')
    plt.plot(regret_nochangepoint, label='No change point detection')
    plt.xlabel('Time Steps')
    plt.ylabel('Regret')
    plt.legend()

    plt.show()
