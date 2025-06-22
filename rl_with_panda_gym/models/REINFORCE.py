import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque
import panda_gym
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


from rl_with_panda_gym.utils import compute_moving_average, preprocess_obs



class DiscretePolicy(tf.keras.Model):
    def __init__(self, num_actions, name="discrete_policy"):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu") 
        self.dense3 = tf.keras.layers.Dense(32, activation="relu")
        self.output_layer = tf.keras.layers.Dense(num_actions, activation="softmax")
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

class ContinuousPolicy(tf.keras.Model):
    def __init__(self, action_dim, name="continuous_policy"):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(32, activation="relu")
        self.mu = tf.keras.layers.Dense(action_dim, activation="tanh")
        self.var = tf.keras.layers.Dense(action_dim, activation=tf.nn.softplus)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.mu(x), self.var(x)


class Reinforce:
    """
    Class per l'agente Reinforce per PandaGym

    Attributes:
        env (gym.Env): Ambiente PandaGym
        gamma (float): Fattore di sconto per i ritorni, di default 0.99
        learning_rate (float): Tasso di apprendimento per l'ottimizzatore, di default 0.001
        num_episodes (int): Numero di episodi da eseguire, di default 10000
        print_every (int): Frequenza con cui stampare i risultati, di default 100
        division (int): Numero di divisioni dello spazio delle azioni, di default 4

    """

    def __init__(
        self,
        env,
        result_path="./",
        model_path="./",
        plot_path="./",
        gamma=0.99,
        learning_rate=0.001,
        num_episodes=10000,
        print_every=100,
        division=4,
        discrete=False,
        store_data=True
    ):
        """
        Inizializza l'agente Reinforce per PandaGym
        """

        self.env = env
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.print_every = print_every
        self.store_data = store_data

        self.discrete = discrete
        print(f"Discretizzazione dello spazio delle azioni: {self.discrete}")

        # Imposto la dimensione dello spazio delle azioni
        self.action_dim = self.env.action_space.shape[0]

        # DISCRETIZATION
        self.division = division
        self.num_action = self.division**self.action_dim
        # Divido lo spazio delle azioni in division parti
        self.action_bins = np.linspace(
            self.env.action_space.low[0], self.env.action_space.high[0], self.division
        )
        # Ora bisogna salvarsi tutte le combinazioni possibili di azioni
        self.all_action_combination = np.array(
            np.meshgrid(self.action_bins, self.action_bins, self.action_bins)
        ).T.reshape(-1, 3)
        # CREO LA POLICY
        if self.discrete:
            self.policy = DiscretePolicy(
                num_actions=self.num_action)
        else:
            self.policy = ContinuousPolicy(action_dim=self.action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.success_track = []
        self.reward_track = []
        
        #PERCORSI PER I RISULTATI
        self.result_path = result_path
        self.model_path = model_path
        self.plot_path = plot_path

    def generate_episode(self, policy):
        """
        Genera un episodio eseguendo la policy nell'ambiente

        Args:
            policy (ContinousPolicy): Policy da eseguire nell'ambiente
        Returns:
            states (list): Lista degli stati osservati durante l'episodio
            actions (list): Lista delle azioni eseguite durante l'episodio
            rewards (list): Lista dei reward ottenuti durante l'episodio

        """

        obs, _ = self.env.reset()
        obs = preprocess_obs(obs)
        done = False
        states, action_indices, actions, rewards = [], [], [], []

        while not done:

            obs = obs[np.newaxis, :]  
            states.append(obs)
            action, action_idx = self.sample_action(policy, obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = preprocess_obs(obs)
            done = terminated or truncated

            action_indices.append(action_idx)
            actions.append(action)
            rewards.append(reward)

        self.success_track.append(info["is_success"])

        if self.discrete:
            return states, action_indices, rewards
        return states, actions, rewards

    def sample_action(self, policy, obs):
        """
        Campiona un'azione dalla policy data l'osservazione

        dopo aver predetto la probabilità di ogni azione discreta creiamo una distribuzione categorica e campioniamo un'azione
        per poi ritornare l'azione e il suo indice
        Args:
            policy (ContinuousPolicy): Policy da cui campionare l'azione
            obs (tf.Tensor): Osservazione dell'ambiente
        Returns:
            action (np.ndarray): Azione campionata dalla policy
            action_idx (int): Indice dell'azione campionata ( indice della combinazione di azioni discrete )


        """
        
        if self.discrete:
            x = policy(obs)
          
            dist = tfp.distributions.Categorical(probs=x)
            action_idx = dist.sample()[0]
            
            if action_idx >= self.num_action:
                print(f"Indice non valido {action_idx}, lo clippo al massimo possibile")
                action_idx = self.num_action - 1
                
            return self.all_action_combination[action_idx], action_idx

        else:
            mu, var = policy(obs)
            dist = tfp.distributions.Normal(mu, var)
            action = dist.sample()
            action = tf.clip_by_value(
                action, self.env.action_space.low, self.env.action_space.high
            )
            return action.numpy()[0], None


    def train_plicy(self, states, returns, action_indices=None, actions=None):
        if self.discrete:

            with tf.GradientTape() as tape:
                policy_loss = 0

                # Calcolo la log_probabilità delle azioni e la loss della policy
                for state, action_idx, G in zip(states, action_indices, returns):

                    # predico nuovamente la media e sigma
                    x = self.policy(state)

                    # creo la distribuzione normale
                    dist = tfp.distributions.Categorical(probs=x)
                    
                    # ESATTAMENTE LA RIGA 7 PSEUDOCODICE
                    # calcolo la log_probabilità dell'azione nella distribuzione
                    action_idx = tf.convert_to_tensor([action_idx], dtype=tf.float32)
                    log_prob = tf.reduce_sum(
                        dist.log_prob(action_idx)
                    )  # sommo le log_probabilità per ogni dimensione dell'azione per avere un singolo valore
                    policy_loss += -log_prob * G
                    if tf.math.is_nan(policy_loss):
                        print(policy_loss)
                        

                policy_loss /= len(states)

            # LINEA 8 PSEUDOCODICE
            gradients = tape.gradient(policy_loss, self.policy.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.policy.trainable_variables)
            )

        else:
            with tf.GradientTape() as tape:
                policy_loss = 0

                # Calcolo la log_probabilità delle azioni e la loss della policy
                for state, action, G in zip(states, actions, returns):
                    
                    # predico nuovamente la media e var
                    mu, var = self.policy(state)

                    # creo la distribuzione normale
                    dist = tfp.distributions.Normal(mu, var)

                    # ESATTAMENTE LA RIGA 7 PSEUDOCODICE
                    # calcolo la log_probabilità dell'azione nella distribuzione
                    action = tf.convert_to_tensor(action, dtype=tf.float32)
                    log_prob = tf.reduce_sum(
                        dist.log_prob(action)
                    )  # sommo le log_probabilità per ogni dimensione dell'azione per avere un singolo valore
                    policy_loss += -log_prob * G

                    if tf.math.is_nan(policy_loss):
                        print("NaN detected in policy_loss!")
                        break

                policy_loss /= len(states)
            gradients = tape.gradient(policy_loss, self.policy.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.policy.trainable_variables)
            )

    def train(self):
        """
        Addestramento dell'agente Reinforce per num_episodes episodi
        Durante l'addestramento:
            1. Per ogni episodio
            2. Genera un episodio seguendo la policy attuale
            3. Calcola i ritorni (rewards scontati) per ogni step dell'episodio generato
            4. Calcola la loss della policy basata sui ritorni
            5. Ottimizza la policy usando l'ottimizzatore Adam
        Args:
            None
        Returns:
            None


        """

        # LINEA 3 PSEUDOCODICE
        for i in range(self.num_episodes):

            # LINEA 4 PSEUDOCODICE, genera un episodio
            if self.discrete:
                states, action_indices, rewards = self.generate_episode(self.policy)
            else:
                states, actions, rewards = self.generate_episode(self.policy)

            self.reward_track.append(sum(rewards))

            returns = []

            # LINEA 6 PSEUDOCODICE, calcola i ritorni
            for t in range(len(rewards)):
                G = 0
                for k, r in enumerate(rewards[t:]):
                    G += r * (self.gamma**k)
                returns.append(G)

            # LINEE 7-8 PSEUDOCODICE, calcola la loss e ottimizza la policy
            # Calcola la loss della policy

            if self.discrete:
                self.train_plicy(states, returns, action_indices=action_indices)
            else:
                self.train_plicy(states, returns, actions=actions)

            if i % self.print_every == 0 and i > 0:
                avg_reward = np.mean(self.reward_track[-self.print_every :])
                success_rate = np.mean(self.success_track[-self.print_every :])
                print(
                    f"Episode {i + 1}/{self.num_episodes}\t"
                    f"Average Reward: {avg_reward:.2f}\t"
                    f"Success Rate: {success_rate*100:.1f}%\t"
                )

                if self.store_data:
                    self.save_metrics()
                    self.save_model()
                    self.plot_results()
                

    def save_metrics(self):
        results_df = pd.DataFrame(
            {
                "episode": range(1, len(self.reward_track) + 1),
                "reward": self.reward_track,
                "success_rate": self.success_track,
            }
        )

        # Salva in CSV
        results_df.to_csv(
            self.result_path + ".csv", index=False
        )
                
    def save_model(self):
        """
        Salva il modello della policy in un file HDF5.

        Args:
            path (str): Percorso del file in cui salvare il modello.

        Returns:
            None
        """
        self.policy.save_weights(self.model_path + ".weights.h5")

    def plot_results(self):
        """
        Crea un grafico dei reward e success rate durante l'addestramento.

        1. Calcola la media mobile dei reward e success rate.
        2. Crea un grafico con 2 subplot:
            - Moving average dei reward.
            - Success rate (media mobile).
        3. Salva il grafico in una cartella "plots"

        Args:
            None

        Returns:
            None

        """

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Moving average dei reward
        sma_reward = compute_moving_average(
            self.reward_track, window_size=min(50, len(self.reward_track) // 2)
        )

        ax[0].set_title("Reward Progress")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("Reward")
        ax[0].plot(self.reward_track, alpha=0.3, label="Raw Reward")
        ax[0].plot(sma_reward, label="Moving Average", linewidth=2)
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # Success rate
        sma_success_rate = compute_moving_average(
            self.success_track, window_size=min(50, len(self.success_track) // 2)
        )

        ax[1].set_title("Success Rate Progress")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Success Rate")
        ax[1].plot(self.success_track, alpha=0.3, label="Raw Success Rate")
        ax[1].plot(sma_success_rate, label="Moving Average", linewidth=2)
        ax[1].set_ylim(0, 1.1)
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        fig.savefig(
            self.plot_path + ".png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


# ESEMPIO DI UTILIZZO, decommentare per eseguire
# if __name__ == "__main__":
    
#     result_path_reinforce_disc = "results/reinforce_disc_training_results"
#     plot_path_reinforce_disc = "plots/reinforce_disc_results"
#     model_path_reinforce_disc = "models/reinforce_disc_model"

#     # Crea l'ambiente
#     env = gym.make("PandaReach-v3", reward_type="dense")

#     # Inizializza l'agente DDPG
#     agent = Reinforce(
#         env=env,
#         num_episodes=50000,
#         print_every=1000,
#         store_data=True,
#         discrete=True,
#         result_path=result_path_reinforce_disc,
#         model_path=model_path_reinforce_disc,
#         plot_path=plot_path_reinforce_disc,
#     )

#     agent.train()

   
