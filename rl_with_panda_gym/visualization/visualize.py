import matplotlib.pyplot as plt
import numpy as np
import csv
from rl_with_panda_gym.utils import compute_moving_average
import seaborn as sns


def visualize(results_path,color="blue", title="", window_size=10000,save_path="", save=False):

    color = sns.color_palette("colorblind")[3]

    with open(results_path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header
        results = list(reader)
        

    #CALCOLO REWARD MOVING AVERAGE
    y = compute_moving_average(np.array([float(row[1]) for row in results]), window_size)
    x = np.arange(len(y))

    #CALCOLO SUCCESS RATE MOVING AVERAGE
    success_rate = np.array([1 if row[2] == "True" else 0 for row in results])
    success_rate_moving_average = compute_moving_average(success_rate, window_size)


    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    fig.suptitle(title)

    ax[0].plot(x, y, color=color)
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    ax[0].set_title("Moving Average Reward")
    ax[0].set_ylim(-50, 0)
    ax[0].grid(True)
    plt.tight_layout()


    ax[1].plot(x, success_rate_moving_average, color=color)
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('success_rate')
    ax[1].set_title('Moving Average Success Rate')
    ax[1].set_ylim(0, 1)
    ax[1].grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
if __name__ == "__main__":
    title="DQN Training Results"
    results_path = "results/train/panda_push_results/ddpg_her_push_sparse_training_results.csv"
    save_path = "plots/panda_reach_plots/reward_successrate_DQN_target_network.png"
    window_size = 1000
    color = sns.color_palette("colorblind")[3]
    
    visualize(results_path=results_path,save_path=save_path, window_size=window_size, title=title, color=color, save=False)