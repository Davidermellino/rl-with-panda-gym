import os
import tensorflow as tf
import gymnasium as gym
import panda_gym
from panda_gym.utils import distance
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

from rl_with_panda_gym.utils import compute_moving_average, preprocess_obs



tf.keras.utils.disable_interactive_logging()


class DDPG:

    def __init__(
        self,
        env,
        result_path="./",
        model_path="./",
        plot_path="./",
        noise_std=0.2,
        num_episodes=10000,
        actor_lr=0.0001,
        critic_lr=0.001,
        gamma=0.98,
        tau=0.005,
        batch_size=256,
        print_every=100,
        train_with_her=False,
        store_data=True,
    ):

        # ======ENVIRONMENT VARIABLE========
        self.env = env
        # servono per calcolare le dimensioni degli stati e delle azioni
        dummy_obs, _ = env.reset()
        dummy_action = env.action_space.sample()
        self.state_dim = len(preprocess_obs(dummy_obs))
        self.goal_dim = self.env.observation_space["desired_goal"].shape[0]
        self.achieved_goal_dim = self.env.observation_space["achieved_goal"].shape[0]
        self.observation_dim = self.env.observation_space["observation"].shape[0]
        self.action_dim = len(dummy_action)

        # ======TRAINING VARIABLE========
        self.print_every = print_every
        self.train_with_her = train_with_her
        self.store_data = store_data
        print(f"Utilizzao HER: {self.train_with_her}")
        print(f"Memorizzazione dati: {self.store_data}")

        # ======HYPERPARAMETERS========
        self.num_episodes = num_episodes
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.k = 4

        # ======MEMORIA========
        self.replay_buffer = deque(maxlen=1000000)
        self.batch_size = batch_size

        # ======RETI NEURALI========
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        # reti target
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        # copiamo i pesi delle reti in quelle target
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        # optimizer
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

        # ======EXPLORATION NOISE========
        self.noise_std = noise_std
        self.noise_decay = 0.9999
        self.noise_min = 0.01

        # ======METRICS========
        self.reward_track = []
        self.success_track = []
        self.noise_history = []  # Per tracciare la storia del rumore

        # ======PERCORSI PER RISULTATI========
        self.result_path = result_path
        self.model_path = model_path
        self.plot_path = plot_path

    def build_actor(self):
        """
        Costruisce la rete Actor per la policy deterministica.

        Returns:
            tf.keras.Model: Modello Actor.
        """
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))

        # Normalizzazione iniziale
        x = tf.keras.layers.LayerNormalization()(inputs)

        # Hidden layers
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        # Output layer con tanh per limitare le azioni
        outputs = tf.keras.layers.Dense(self.action_dim, activation="tanh")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def build_critic(self):
        """
        Costruisce la rete Critic per stimare i Q-values.

        Returns:
            tf.keras.Model: Modello Critic.
        """
        # Input per stato e azione
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        action_input = tf.keras.layers.Input(shape=(self.action_dim,))

        # Processamento dello stato
        state_h = tf.keras.layers.LayerNormalization()(state_input)
        state_h = tf.keras.layers.Dense(256, activation="relu")(state_h)

        # Processamento dell'azione
        action_h = tf.keras.layers.Dense(256, activation="relu")(action_input)

        # Concatenazione e hidden layers
        concat = tf.keras.layers.Concatenate()([state_h, action_h])
        x = tf.keras.layers.Dense(256, activation="relu")(concat)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        # Output layer (Q-value)
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=[state_input, action_input], outputs=outputs)
        return model

    def choose_action(self, state):
        """Sceglie azione con rumore"""
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        action_tensor = self.actor(state_tensor)[0]
        
        action = action_tensor.numpy()
        noise = np.random.normal(0, self.noise_std, size=self.action_dim)
        
        return np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)

    def update_target_networks(self):

        """ Aggiorna le reti target utilizzando il soft update."""
        
        #====AGGIORNO ACTOR====
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = (
                self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
            )
        self.target_actor.set_weights(target_actor_weights)

        #====AGGIORNO CRITIC====
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = (
                self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]
            )
        self.target_critic.set_weights(target_critic_weights)

    def generate_her_transitions(self, trainsitions):

        """
            Genera le transizioni HER (Hindsight Experience Replay) a partire dalle transizioni dell'episodio.
            
            tecnica: future
            
            
            Args:
                trainsitions (list): Lista delle transizioni dell'episodio
            Returns:
                her_transitions (list): Lista delle transizioni HER generate
        """

        her_transitions = []

        for i, (state, action, reward, next_state, done) in enumerate(trainsitions):
            for _ in range(self.k):

                # Genero goal casuale futuro
                if i + 1 < len(trainsitions):
                    future_transition = trainsitions[
                        random.randint(i + 1, len(trainsitions) - 1)
                    ]
                    future_goal = future_transition[0][0 : self.achieved_goal_dim]

                    # GENERO NUOVI STATI

                    new_state = np.concatenate(
                        [
                            state[: self.achieved_goal_dim],
                            future_goal,
                            state[-self.observation_dim :],
                        ]
                    )
                    new_next_state = np.concatenate(
                        [
                            next_state[: self.achieved_goal_dim],
                            future_goal,
                            next_state[-self.observation_dim :],
                        ]
                    )

                    # GENERO NUOVO REWARD
                    distance_threshold = 0.05
                    achieved_goal = state[: self.achieved_goal_dim]
                    d = distance(achieved_goal, future_goal)
                    new_reward = -np.array(d > distance_threshold, dtype=np.float32)

                    # GENERO NUOVA TERMINAZIONE
                    new_done = new_reward == 0  # Successo se reward = 0

                    her_transitions.append(
                        (new_state, action, new_reward, new_next_state, new_done)
                    )

        return her_transitions

    def generate_episode(self):
        """
        Genera un episodio eseguendo la policy nell'ambiente

        Returns:
            transitions (list): Lista delle transizioni dell'episodio, dove ogni transizione è una tupla

        """

        state, _ = self.env.reset()
        state = preprocess_obs(state)
        done = False
        states, episode_transitions = [], []
        reward_episode = 0

        while not done:

            states.append(state)
            action = self.choose_action(state)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = preprocess_obs(next_state)
            done = terminated or truncated

            reward_episode += reward
            episode_transitions.append((state, action, reward, next_state, done))

            state = next_state

        self.reward_track.append(reward_episode)
        self.success_track.append(info["is_success"])

        # Decay the exploration noise
        self.noise_history.append(self.noise_std)  # Aggiungo il rumore alla storia
        self.noise_std = max(self.noise_std * self.noise_decay, self.noise_min)

        return episode_transitions

    

    def train(self):
        
        """
        Addestra l'agente DDPG per un numero specificato di episodi.        
        """

        for i in range(self.num_episodes):
            episode_transitions = self.generate_episode()

            # MEMORIZZO LE TRANZIONI DELL'EPISODIO NEL REPLAY BUFFER
            for trainsition in episode_transitions:
                self.replay_buffer.append(trainsition)

            # GEMNERO LE TRANZIONI HER
            if self.train_with_her:
                her_transitions = self.generate_her_transitions(episode_transitions)
                for transition in her_transitions:
                    self.replay_buffer.append(transition)

            # Addestramento per ogni step dell'episodio
            for _ in range(len(episode_transitions)):
                self.train_batch()

            # MOSTRO RISULTATI
            if (i + 1) % self.print_every == 0 and i > 0:
                avg_reward = np.mean(self.reward_track[-self.print_every :])
                success_rate = np.mean(self.success_track[-self.print_every :])
                print(
                    f"Episode {i + 1}/{self.num_episodes}\t"
                    f"Average Reward: {avg_reward:.2f}\t"
                    f"Success Rate: {success_rate*100:.1f}%\t"
                    f"Noise: {self.noise_std:.3f}"
                )
                if self.store_data:
                    self.plot_results()
                    self.save_model()
                    self.save_metrics()

    @tf.function
    def train_actor(self, current_state):
        with tf.GradientTape() as tape:
            actions = self.actor(current_state, training=True)
            current_q_value = self.critic([current_state, actions], training=True)
            actor_loss = -tf.reduce_mean(current_q_value)

        grads_actor = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grads_actor, self.actor.trainable_variables)
        )

    @tf.function
    def train_critic(self, current_state, actions, target_q):
        with tf.GradientTape() as tape:
            current_q_value = self.critic([current_state, actions])
            critic_loss = tf.reduce_mean(
                tf.math.pow(target_q - tf.cast(current_q_value, tf.float32), 2)
            )

        grads_critic = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(grads_critic, self.critic.trainable_variables)
        )

    def train_batch(self):
        
        """
        Addestra l'agente DDPG su un batch di transizioni dal replay buffer.
        """
        
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)

        # PRENDO I DATI DAL BATCH
        current_state = tf.convert_to_tensor([x[0] for x in batch])
        actions = tf.convert_to_tensor([x[1] for x in batch])
        next_state = tf.convert_to_tensor([x[3] for x in batch])
        reward = tf.convert_to_tensor(
            np.array([x[2] for x in batch]).reshape(-1, 1), dtype=tf.float32
        )
        done = tf.convert_to_tensor(
            np.array([x[4] for x in batch], dtype=np.float32).reshape(-1, 1),
            dtype=tf.float32,
        )

        # CALCOLO I TARGET Q-VALUE
        q_actions = self.target_actor(next_state)
        target_q = tf.cast(reward, tf.float32) + (
            tf.cast(tf.constant(1.0), tf.float32) - tf.cast(done, tf.float32)
        ) * self.gamma * tf.cast(
            self.target_critic([next_state, q_actions]), tf.float32
        )

        # OTTIMIZZO IL CRITIC

        self.train_critic(current_state, actions, target_q)
        self.train_actor(current_state)

        self.update_target_networks()  # Aggiorno le reti target

    def plot_results(self):
        """
        Crea grafici dei risultati durante l'addestramento.
        """
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))

        # Moving average dei reward
        sma_reward = compute_moving_average(
            self.reward_track,
            window_size=min(
                self.print_every, len(self.reward_track) - len(self.reward_track) // 4
            ),
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
            self.success_track,
            window_size=min(self.print_every, len(self.success_track) // 2),
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
        fig.savefig(self.plot_path + ".png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_model(self):
        """
        Salva i pesi del modello in un file.

        """
        self.actor.save(self.model_path + ".keras")

    def save_metrics(self):
        """
        Salva le metriche in un file.

        """
        # creo dataframe con le metriche
        results_df = pd.DataFrame(
            {
                "episode": range(1, len(self.reward_track) + 1),
                "reward": self.reward_track,
                "success_rate": self.success_track,
                "noise": self.noise_history,
            }
        )

        results_df.to_csv(self.result_path + ".csv", index=False)


# ESEMPIO DI UTILIZZO, decommentare per eseguire
# if __name__ == "__main__":

#     result_path = (
#         "results/train/panda_reach_results/ddpg_her_reach_sparse_training_results.csv"
#     )
#     plot_path = "plots/panda_reach_plots/reward_successrate_DDPG_her_reach_sparse.png"
#     model_path = "models/panda_reach_models/actor_ddpg_her_reach_sparse.keras"
#     # Crea l'ambiente
#     env = gym.make("PandaReach-v3", reward_type="sparse")

#     # Inizializza l'agente DDPG
#     agent = DDPG(
#         env=env,
#         num_episodes=20000,
#         print_every=10,
#         store_data=False,
#         train_with_her=True,
#     )

#     agent.train()
