import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import csv
import seaborn as sns


from rl_with_panda_gym.utils import compute_moving_average


#test functions
from rl_with_panda_gym.test_models import test



def compare_training_results_reach():
    """
    Funzione per confrontare i risultati di addestramento dei modelli DQN e REINFORCE.
    Carica i risultati da file CSV, calcola le medie mobili e visualizza i grafici.
    """
    
    colors = sns.color_palette("colorblind")[:4]


    #PATH DEI RISULTATI
    results_DQN_single_net = "results/train/panda_reach_results/dqn_single_net_training_results.csv"
    results_DQN_target_net = "results/train/panda_reach_results/dqn_target_net_training_results.csv"
    results_reinforce = "results/train/panda_reach_results/reinforce_training_results.csv"
    results_reinforce_discrete = "results/train/panda_reach_results/reinforce_training_discrete_results.csv"

    moving_average_window = 1000

    with open(results_DQN_single_net, mode='r') as file:
        reader = csv.reader(file)
        _ = next(reader) 
        DQN_single_net = list(reader)

    with open(results_DQN_target_net, mode='r') as file:
        reader = csv.reader(file)
        
        _ = next(reader) 
        DQN_target_net = list(reader)
        
    with open(results_reinforce, mode='r') as file:
        reader = csv.reader(file)
        _ = next(reader)
        reinforce = list(reader)
        
    with open(results_reinforce_discrete, mode='r') as file:
        reader = csv.reader(file)
        _ = next(reader)
        reinforce_discrete = list(reader)
        
        
    """DQN SINGLE NET"""
    #CALCOLO REWARD MOVING AVERAGE
    y_DQN_single_net = compute_moving_average(np.array([float(row[1]) for row in DQN_single_net]), window_size=moving_average_window)
    x = np.arange(len(y_DQN_single_net))

    #CALCOLO SUCCESS RATE MOVING AVERAGE
    success_rate_DQN_single_net = np.array([int(row[2]) for row in DQN_single_net])
    success_rate_moving_average_DQN_single_net = compute_moving_average(success_rate_DQN_single_net, window_size=moving_average_window)

    """DQN TARGET NET"""

    #CALCOLO REWARD MOVING AVERAGE
    y_DQN_target_net = compute_moving_average(np.array([float(row[1]) for row in DQN_target_net]), window_size=moving_average_window)
    success_rate_DQN_target_net = np.array([int(row[2]) for row in DQN_target_net])
    success_rate_moving_average_DQN_target_net = compute_moving_average(success_rate_DQN_target_net, window_size=moving_average_window)

    moving_average_window = 10000

    """REINFORCE"""
    #CALCOLO REWARD MOVING AVERAGE
    y_reinforce = compute_moving_average(np.array([float(row[1]) for row in reinforce]), window_size=moving_average_window)
    x_reinforce = np.arange(len(y_reinforce))
    #CALCOLO SUCCESS RATE MOVING AVERAGE
    success_rate_reinforce = np.array([int(row[2]) for row in reinforce])
    success_rate_moving_average_reinforce = compute_moving_average(success_rate_reinforce, window_size=moving_average_window)
    """REINFORCE DISCRETE"""
    #CALCOLO REWARD MOVING AVERAGE
    y_reinforce_discrete = compute_moving_average(np.array([float(row[1]) for row in reinforce_discrete]), window_size=moving_average_window)
    x_reinforce_discrete = np.arange(len(y_reinforce_discrete))
    #CALCOLO SUCCESS RATE MOVING AVERAGE
    success_rate_reinforce_discrete = np.array([int(row[2]) for row in reinforce_discrete])
    success_rate_moving_average_reinforce_discrete = compute_moving_average(success_rate_reinforce_discrete, window_size=moving_average_window)

    fig, ax = plt.subplots(4, 2, figsize=(6, 12))
    fig.suptitle('Training Results Comparison')

    """DQN SINGLE NET"""
    ax[0,0].set_title('DQN Single Network')
    ax[0,0].plot(x, y_DQN_single_net, color=colors[0])
    ax[0,0].set_xlabel('Episodes')
    ax[0,0].set_ylabel('Reward')
    ax[0,0].set_ylim(-40, 0)
    ax[0,0].grid(True)

    ax[0,1].plot(x, success_rate_moving_average_DQN_single_net, color=colors[0])
    ax[0,1].set_xlabel('Episodes')
    ax[0,1].set_ylabel('Success Rate')
    ax[0,1].grid(True)
    ax[0,1].set_ylim(0, 1)
    
    
    """DQN TARGET NET"""
    
    ax[1,0].set_title('DQN Target Network')
    ax[1,0].plot(x, y_DQN_target_net, color=colors[1])
    ax[1,0].set_xlabel('Episodes')
    ax[1,0].set_ylabel('Reward')
    ax[1,0].set_ylim(-40, 0)
    ax[1,0].grid(True)

    ax[1,1].plot(x, success_rate_moving_average_DQN_target_net, color=colors[1])
    ax[1,1].set_xlabel('Episodes')
    ax[1,1].set_ylabel('Success Rate')
    ax[1,1].grid(True)
    ax[1,1].set_ylim(0, 1)
    
    
    """REINFORCE"""
    ax[2,0].set_title('REINFORCE Continuous')
    ax[2,0].plot(x_reinforce, y_reinforce, color=colors[2])
    ax[2,0].set_xlabel('Episodes')
    ax[2,0].set_ylabel('Reward')
    ax[2,0].tick_params(axis='x', labelrotation=30)
    ax[2,0].set_ylim(-40, 0)
    ax[2,0].grid(True)

    ax[2,1].plot(x_reinforce, success_rate_moving_average_reinforce, color=colors[2])
    ax[2,1].set_xlabel('Episodes')
    ax[2,1].set_ylabel('Success Rate')
    ax[2,1].tick_params(axis='x', labelrotation=30)
    ax[2,1].grid(True)
    ax[2,1].set_ylim(0, 1)
    
    
    """REINFORCE DISCRETE"""
    
    ax[3,0].set_title('REINFORCE Discrete')
    ax[3,0].plot(x_reinforce_discrete, y_reinforce_discrete, color=colors[3])
    ax[3,0].set_xlabel('Episodes')
    ax[3,0].set_ylabel('Reward')
    ax[3,0].set_ylim(-40, 0)
    ax[3,0].grid(True)

    ax[3,1].plot(x_reinforce_discrete, success_rate_moving_average_reinforce_discrete, color=colors[3])
    ax[3,1].set_xlabel('Episodes')
    ax[3,1].set_ylabel('Success Rate')
    ax[3,1].grid(True)
    ax[3,1].set_ylim(0, 1)
    
    moving_average_window = 1000

    
    ddpg_path = "results/train/panda_reach_results/ddpg_reach_sparse_training_results.csv"
    ddpg_her_path = "results/train/panda_reach_results/DDPG_HER_training_results.csv"
    
    with open(ddpg_path, mode='r') as file:
        reader = csv.reader(file)
        _ = next(reader)
        ddpg = list(reader)
        
    with open(ddpg_her_path, mode='r') as file:
        reader = csv.reader(file)
        _ = next(reader)
        ddpg_her = list(reader)
    
    
    """DDPG"""
    #CALCOLO REWARD MOVING AVERAGE
    y_ddpg = compute_moving_average(np.array([float(row[1]) for row in ddpg]), window_size=moving_average_window)
    x_ddpg = np.arange(len(y_ddpg))
    #CALCOLO SUCCESS RATE MOVING AVERAGE
    success_rate_ddpg = np.array([int(row[2]) for row in ddpg])
    success_rate_moving_average_ddpg = compute_moving_average(success_rate_ddpg, window_size=moving_average_window)
    """DDPG + HER   """
    #CALCOLO REWARD MOVING AVERAGE
    y_ddpg_her = compute_moving_average(np.array([float(row[1]) for row in ddpg_her]), window_size=moving_average_window)
    x_ddpg_her = np.arange(len(y_ddpg_her))
    #CALCOLO SUCCESS RATE MOVING AVERAGE
    success_rate_ddpg_her = np.array([1 if row[2] == "True" else 0  for row in ddpg_her])
    success_rate_moving_average_ddpg_her = compute_moving_average(success_rate_ddpg_her, window_size=moving_average_window)
    

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
    fig2.suptitle('DDPG and DDPG + HER Training Results Comparison')    
    
    ax2[0].plot(x_ddpg, y_ddpg, color=colors[0], label='DDPG')
    ax2[0].plot(x_ddpg_her, y_ddpg_her, color=colors[1], label='DDPG + HER')
    ax2[0].set_xlabel('Episodes')
    ax2[0].set_ylabel('Reward')
    ax2[0].tick_params(axis='x', labelrotation=30)
    ax2[0].set_ylim(-50, 0); ax2[0].set_xlim(-0.1, 10000)
    ax2[0].grid(True)
    ax2[0].legend()
    
    ax2[1].plot(x_ddpg, success_rate_moving_average_ddpg, color=colors[0], label='DDPG')
    ax2[1].plot(x_ddpg_her, success_rate_moving_average_ddpg_her, color=colors[1], label='DDPG + HER')
    ax2[1].set_xlabel('Episodes')
    ax2[1].set_ylabel('Success Rate')
    ax2[1].tick_params(axis='x', labelrotation=30)
    ax2[1].grid(True)
    ax2[1].set_ylim(0, 1.1); ax2[1].set_xlim(-0.1, 10000)
    ax2[1].legend()


    plt.tight_layout() 
    plt.show()
    
def compare_test_results_reach(new_test=False):
    models = ["DQN Single Net", "DQN Double Net",  "Reinforce Continuous","Reinforce Disc", "DDPG", "DDPG + HER"]
    colors = sns.color_palette("colorblind")[:6]
    
    
    data = read_csv('results/test/panda_reach_results/reinforce_discrete_test_results.csv')
    reward_rd = data['Reward'].tolist()
    success_rd = data['Success'].tolist()
    
    data = read_csv('results/test/panda_reach_results/dqn_double_network_test_result.csv')
    reward_dqn_tn = data['Reward'].tolist()
    success_dqn_tn = data['Success'].tolist()
    
    data = read_csv('results/test/panda_reach_results/dqn_single_network_test_result.csv')
    reward_dqn_sn = data['Reward'].tolist()
    success_dqn_sn = data['Success'].tolist()
    
    data = read_csv('results/test/panda_reach_results/reinforce_test_results.csv')
    reward_reinforce = data['Reward'].tolist()
    success_reinforce = data['Success'].tolist()

    data = read_csv('results/test/panda_reach_results/ddpg_test_results.csv')
    reward_ddpg = data['Reward'].tolist()
    success_ddpg = data['Success'].tolist()

    data = read_csv('results/test/panda_reach_results/ddpg_her_test_results.csv')
    reward_ddpg_her = data['Reward'].tolist()
    success_ddpg_her = data['Success'].tolist()
    
    #Rewards medi
    rewards = [reward_dqn_sn, reward_dqn_tn, reward_reinforce, reward_rd, reward_ddpg, reward_ddpg_her]
    #rewards_formatted = [f"{reward_dqn_sn:.3f}", f"{reward_dqn_tn:.3f}", f"{reward_reinforce:.3f}", f"{reward_rd:.3f}"]
    
    #Success rate medi
    success_rates = [np.mean(success_dqn_sn), np.mean(success_dqn_tn), np.mean(success_reinforce), np.mean(success_rd), np.mean(success_ddpg), np.mean(success_ddpg_her)]
    success_rates_formatted = [f"{np.mean(success_dqn_sn)*100:.2f}%", f"{np.mean(success_dqn_tn)*100:.2f}%", f"{np.mean(success_reinforce)*100:.2f}%", f"{np.mean(success_rd)*100:.2f}%", f"{np.mean(success_ddpg)*100:.2f}%", f"{np.mean(success_ddpg_her)*100:.2f}%"]

    fig, ax = plt.subplots(3, 2, figsize=(12, 6))
    
    
    bplot1 = ax[0,0].boxplot(rewards[0], patch_artist=True, )
    ax[0,0].set_title(models[0])
    ax[0,0].set_ylabel('Reward')
    ax[0,0].set_ylim(-50 , 0 )


    bplot2 = ax[0,1].boxplot(rewards[1], patch_artist=True, )
    ax[0,1].set_title(models[1])
    ax[0,1].set_ylabel('Reward')
    ax[0,1].set_ylim(-50 , 0 )

    bplot3 = ax[1,0].boxplot(rewards[2], patch_artist=True, )
    ax[1,0].set_title(models[2])
    ax[1,0].set_ylabel('Reward')
    ax[1,0].set_ylim(-50 , 0 )

    bplot4 = ax[1,1].boxplot(rewards[3], patch_artist=True, )
    ax[1,1].set_title(models[3])
    ax[1,1].set_ylabel('Reward')
    ax[1,1].set_ylim(-50 , 0 )
    
    bplot5 = ax[2,0].boxplot(rewards[4], patch_artist=True, )
    ax[2,0].set_title(models[4])
    ax[2,0].set_ylabel('Reward')
    ax[2,0].set_ylim(-50 , 0 )
    
    bplot6 = ax[2,1].boxplot(rewards[5], patch_artist=True, )
    ax[2,1].set_title(models[5])
    ax[2,1].set_ylabel('Reward')
    ax[2,1].set_ylim(-50 , 0 )
    
    
    bplots = [bplot1, bplot2, bplot3, bplot4, bplot5, bplot6]
    
    #coloro i boxplot e cambio colore delle mediane
    for i in range(len(models)):
        for patch in bplots[i]['boxes']:
            patch.set_facecolor(colors[i])
        for median in bplots[i]['medians']:
            median.set(color ='black', linewidth = 2)


    
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bars = ax2.bar(models, success_rates, color=colors, )
    ax2.set_ylabel('Success Rate')
    ax2.set_xlabel('Models')
    ax2.tick_params(axis='x', labelrotation=30)
    ax2.set_ylim(0 , 1 )
    ax2.bar_label(bars, labels=success_rates_formatted, label_type='center', color='black', fontsize=12)

    fig.suptitle("Test Results Comparison reward (100 Episodes)", fontsize=16)
    fig2.suptitle("Test Results Comparison success rate (100 Episodes)", fontsize=16)

    plt.tight_layout() 
    plt.show()
    
def compare_train_results_push():
    """
    Funzione per confrontare i risultati di addestramento dei modelli DQN e REINFORCE.
    Carica i risultati da file CSV, calcola le medie mobili e visualizza i grafici.
    """
    
    colors = sns.color_palette("colorblind")[:2]


    #PATH DEI RISULTATI
    results_dqn = "results/train/panda_push_results/dqn_target_net_push_training_results.csv"
    #results_ddpg = "results/train/panda_reach_results/dqn_target_net_training_results.csv"
    result_ddpg_her = "results/train/panda_push_results/ddpg_her_push_sparse_training_results.csv"
    
        
    with open(results_dqn, mode='r') as file:
        reader = csv.reader(file)
        _ = next(reader)
        dqn = list(reader)
        
    with open(result_ddpg_her, mode='r') as file:
        reader = csv.reader(file)
        _ = next(reader)
        ddpg_her = list(reader)
        
    moving_average_window = 1000

    """DQN"""
    #CALCOLO REWARD MOVING AVERAGE
    y_dqn = compute_moving_average(np.array([float(row[1]) for row in dqn]), window_size=moving_average_window)
    x_dqn = np.arange(len(y_dqn))
    #CALCOLO SUCCESS RATE MOVING AVERAGE
    success_rate_dqn = np.array([int(row[2]) for row in dqn])
    success_rate_moving_average_dqn = compute_moving_average(success_rate_dqn, window_size=moving_average_window)
    
    moving_average_window = 10000

    """DDPG HER"""
    #CALCOLO REWARD MOVING AVERAGE
    y_ddpg_her = compute_moving_average(np.array([float(row[1]) for row in ddpg_her]), window_size=moving_average_window)
    x_ddpg_her = np.arange(len(y_ddpg_her))
    #CALCOLO SUCCESS RATE MOVING AVERAGE
    success_rate_ddpg_her = np.array([1 if row[2] == "True" else 0 for row in ddpg_her])
    success_rate_moving_average_ddpg_her = compute_moving_average(success_rate_ddpg_her, window_size=moving_average_window)

    fig, ax = plt.subplots(1,2 , figsize=(12, 6))
    fig.suptitle('Training Results Comparison PandaPush task')
    
    
    """dqn"""

    ax[0].plot(x_dqn, success_rate_moving_average_dqn, color=colors[0])
    ax[0].set_title('DQN')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Success Rate')
    ax[0].tick_params(axis='x', labelrotation=30)
    ax[0].grid(True)
    ax[0].set_ylim(0, 1)
    
    
    """ddpg_her"""


    ax[1].plot(x_ddpg_her, success_rate_moving_average_ddpg_her, color=colors[1])
    ax[1].set_title('DDPG + HER')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Success Rate')
    ax[1].grid(True)
    ax[1].set_ylim(0, 1)


    plt.tight_layout() 
    plt.show()
    
def compare_test_results_push():
    models = ["DQN", "DDPG + HER"]
    colors = sns.color_palette("colorblind")[:2]
    
    
    data = read_csv('results/test/panda_push_results/dqn_results.csv')
    reward_dqn = data['reward'].tolist()
    success_dqn = data['success'].tolist()
    
    data = read_csv('results/test/panda_push_results/ddpg_her_results.csv')
    reward_ddpg_her = data['reward'].tolist()
    success_ddpg_her = data['success'].tolist()
    
    #Rewards medi
    rewards = [reward_dqn, reward_ddpg_her]
    
    #Success rate medi
    success_rates = [np.mean(success_dqn), np.mean(success_ddpg_her)]
    success_rates_formatted = [f"{np.mean(success_dqn)*100:.2f}%", f"{np.mean(success_ddpg_her)*100:.2f}%"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    fig.suptitle('Test Results Comparison PandaPush task - reward (100 Episodes)')
    
    bplot1 = ax[0].boxplot(rewards[0], patch_artist=True, )
    ax[0].set_title(models[0])
    ax[0].set_ylabel('Reward')
    ax[0].set_ylim(-51 , 0 )


    bplot2 = ax[1].boxplot(rewards[1], patch_artist=True, )
    ax[1].set_title(models[1])
    ax[1].set_ylabel('Reward')
    ax[1].set_ylim(-51 , 0 )

    
    bplots = [bplot1, bplot2]
    
    #coloro i boxplot e cambio colore delle mediane
    for i in range(len(models)):
        for patch in bplots[i]['boxes']:
            patch.set_facecolor(colors[i])
        for median in bplots[i]['medians']:
            median.set(color ='black', linewidth = 2)


    
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig2.suptitle('Test Results Comparison PandaPush task - success rate (100 Episodes)')
    bars = ax2.bar(models, success_rates, color=colors, )
    ax2.set_ylabel('Success Rate')
    ax2.set_xlabel('Models')
    ax2.tick_params(axis='x', labelrotation=30)
    ax2.set_ylim(0 , 1 )
    ax2.bar_label(bars, labels=success_rates_formatted, label_type='center', color='black', fontsize=12)
    
    plt.show()
    
    
if __name__ == "__main__":
    #compare_training_results_reach()
    #compare_test_results_reach(new_test=False)
    #compare_train_results_push()
    compare_test_results_push()