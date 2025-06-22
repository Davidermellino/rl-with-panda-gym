import os
import tensorflow as tf
import gymnasium as gym
import panda_gym
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

from rl_with_panda_gym.utils import compute_moving_average, preprocess_obs


tf.keras.utils.disable_interactive_logging()


class DQN:
    """
    Agente DQN con rete target per l'ambiente PandaGym.

    Attributes:
        env (gym.Env): L'ambiente PandaGym da utilizzare.
        num_episodes (int): Numero di episodi per l'addestramento.
        learning_rate (float): Tasso di apprendimento per le 2 reti neurali.
        gamma (float): Fattore di sconto per il calcolo del Q-value.
        epsilon (float): Valore iniziale di epsilon per l'esplorazione.
        epsilon_decay (float): Fattore di decrescita di epsilon ad ogni episodio (dopo 50 episodi di pura esplorazione).
        batch_size (int): Dimensione del batch per l'addestramento.
        division (int): Numero di divisioni dello spazio delle azioni (3 dimensioni, 3 azioni per dimensione).
        update_target_network_freq (int): Frequenza di aggiornamento della rete target.


    """

    def __init__(
        self,
        env,
        result_path="./",
        model_path="./",
        plot_path="./",
        num_episodes=5000,
        learning_rate=0.0005,
        gamma=0.95,
        epsilon=1,
        epsilon_decay=0.995,
        batch_size=64,
        division=4,
        update_target_network_freq=5000,
        print_every=100,
        store_data=True,
        use_target_network=False,
    ):
        """
        Inizializza l'agente DQN con rete target per l'ambiente PandaGym.

        Args:
            env (gym.Env): L'ambiente PandaGym da utilizzare.
            num_episodes (int): Numero di episodi per l'addestramento.
            learning_rate (float): Tasso di apprendimento per le 2 reti neurali.
            gamma (float): Fattore di sconto per il calcolo del Q-value.
            epsilon (float): Valore iniziale di epsilon per l'esplorazione.
            epsilon_decay (float): Fattore di decrescita di epsilon ad ogni episodio (dopo 50 episodi di pura esplorazione).
            batch_size (int): Dimensione del batch per l'addestramento.
            division (int): Numero di divisioni dello spazio delle azioni (3 dimensioni, 3 azioni per dimensione).
            update_target_network_freq (int): Frequenza di aggiornamento della rete target.

        """
        self.store_data = store_data
        self.use_target_network = use_target_network
        print(f"Memorizzazione dati: {self.store_data}")
        print(f"Rete target: {self.use_target_network}")
        self.print_every = print_every

        # INFORMAZIONI SULL'AMBIENTE
        self.env = env
        self.division = (
            division  # indica in quante parti dividiamo lo spazio delle azioni
        )
        self.num_actions = (
            self.division**3
        )  # numero di azioni possibili (3 dimensioni, 3 azioni per dimensione)
        self.state_dim = 12  # dimensione dello stato (3 achieved_goal + 3 desired_goal + 6 observation)

        # IPERPARAMETRI
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.update_target_network_freq = update_target_network_freq
        self.update_target_network_counter = 0
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01  # valore minimo di epsilon per l'esplorazione
        self.learning_rate = learning_rate

        # Divido lo spazio delle azioni in division parti
        self.action_bins = np.linspace(
            self.env.action_space.low[0], self.env.action_space.high[0], self.division
        )

        # Ora bisogna salvarsi tutte le combinazioni possibili di azioni
        self.all_action_combination = np.array(
            np.meshgrid(self.action_bins, self.action_bins, self.action_bins)
        ).T.reshape(-1, 3)

        # NETWORKS
        self.online_network = (
            self.build_model()
        )  # online network: usata per scegliere le azioni
        
        if self.use_target_network:
            self.target_network = (
                self.build_model()
            )  # target network: usata per calcolare il target del Q-learning
            self.target_network.set_weights(
                self.online_network.get_weights()
            )  # inizialmente i pesi sono uguali

        # REPLAY MEMORY
        self.memory = deque(maxlen=100000)
        self.batch_size = batch_size

        # METRICHE
        self.reward_track = []
        self.epsilon_history = [1]
        self.success_track = []
        
        # ======PERCORSI PER RISULTATI========
        self.result_path = result_path
        self.model_path = model_path
        self.plot_path = plot_path


    def choose_action(self, state):
        """
        Politica di scelta dell'azione dell'agente. Basata su epsilon-greedy.

        1. Con probabilità epsilon, l'agente esplora e sceglie un'azione casuale.
        2. Con probabilità 1-epsilon, l'agente sfrutta e sceglie l'azione con il massimo Q-value.
        3. Le azioni sono mappate dallo spazio continuo a uno spazio discreto utilizzando gli action_bins.

        Args:
            state (np.ndarray): Lo stato corrente dell'ambiente.
        Returns:
            np.ndarray: L'azione scelta dall'agente.

        """

        if random.random() < self.epsilon:

            # ESPLORATION: scelgo un'azione casuale
            action = self.env.action_space.sample()  # campioniamo un'azione cassuale
            action_idx = np.digitize(
                action, self.action_bins
            )  # Per ogni valore dell'azione lo mappiamo al valore dello spazio discreto
            return self.action_bins[
                action_idx - 1
            ]  # ritorno l'azione corrispondente alla combinazione di azioni
        else:
            # EXPLOITATION: scelgo l'azione con il massimo Q value
            q_values = self.online_network.predict(np.array([state]))
            action_idx = np.argmax(q_values)
            return self.all_action_combination[action_idx]

    def get_action_idx(self, actions):
        """
        Restituisce, data un'azione già discretizzata , l'indice dell'azione corrispondente al vettore di tutte le azioni possibili.

        Esempio:
            Se le combinazioni di azioni sono [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], ...]
            e l'azione presa è [0.1, 0.2, 0.3], allora restituisce 1 (l'indice della combinazione di azioni presa).

        Args:
            actions: azione presa
        Returns:
            scalar: Indice dell'azione corrispondente nell'array di tutte le combinazioni di azioni.


        """

        # 1. Per prima cosa creo un array booleano che indica a True nell'indice
        #   corrispondente alla combinazione di azioni che ho preso
        action_correspondences = np.all(self.all_action_combination == actions, axis=1)
        # ES. se ho preso l'azione [0.1, 0.2, 0.3] e le combinazioni sono
        #    [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], ...]
        #    allora action_correspondences sarà [False, True, ...]

        # 2 infine con np.where prendo l'indice dell'azione
        action_index = np.where(action_correspondences)[0][0]
        # ES. se ho preso l'azione [0.1, 0.2, 0.3] e le combinazioni sono
        #    [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], ...]
        #    allora action_indices sarà [1]
        #    perchè l'azione [0.1, 0.2, 0.3] è la seconda combinazione di azioni
        return action_index

    def build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.state_dim,)),
                # Normalizzazione iniziale (stabilizza gli input)
                tf.keras.layers.LayerNormalization(),
                # Primo blocco di dense + dropout
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                # Secondo blocco
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                # Più profondità
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                # Output layer lineare per Q-values
                tf.keras.layers.Dense(self.num_actions, activation="linear"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        return model

    def train(self):
        """
        Addestra l'agente DQN per un numero num_episodes di episodi.

        1. Per ogni episodio, resetta l'ambiente e ottieni lo stato iniziale.
        2. Scegli un'azione utilizzando la politica epsilon-greedy.
        3. Esegui l'azione nell'ambiente e ottieni il nuovo stato e il reward.
        4. Memorizza la transizione nella replay memory.
        5. Esegui un passo di training utilizzando un batch di transizioni dalla replay memory.
        6. Aggiorna lo stato corrente.
        7. Dopo ogni episodio, aggiorna le metriche e stampa i risultati.


        Args:
            None

        Returns:
            None

        """

        for i in range(self.num_episodes):

            # stato iniziale
            state, _ = self.env.reset()
            # Dato che PandaGym restituisce un dizionario, prendo lo stato concatenando i valori
            state = preprocess_obs(state)
            done = False
            episode_reward = 0

            # EPSILON DECAY
            if i > 50:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            while not done:

                # 1. SCELGO AZIONE
                action = self.choose_action(state)

                # 2. ESEGUO AZIONE E CALCOLO NUOVO STATO E REWARD
                next_state, reward, terminated, truncated, info = self.env.step(action)

                next_state = preprocess_obs(next_state)  # concateno il nuovo stato

                episode_reward += reward

                done = terminated or truncated  # controllo se l'episodio è finito

                # 3. MEMORIZZO LA TRANSIZIONE
                self.memory.append((state, action, reward, next_state, done))

                # 4. ESEGUO UNO STEP DI TRAINING
                self.train_batch()

                # 5. AGGIORNO LO STATO
                state = next_state

            self.reward_track.append(episode_reward)  # salvo il reward dell'episodio
            self.epsilon_history.append(self.epsilon)  # salvo l'epsilon dell'episodio
            self.success_track.append(
                info["is_success"]
            )  # salva 1 se successo, 0 altrimenti (True/False convertito in int)

            if i % self.print_every == 0 and i > 0:
                avg_reward = np.mean(self.reward_track[-self.print_every :])
                success_rate = np.mean(self.success_track[-self.print_every :])
                print(
                    f"Episode {i + 1}/{self.num_episodes}\t"
                    f"Average Reward: {avg_reward:.2f}\t"
                    f"Success Rate: {success_rate*100:.1f}%\t"
                    f"Epsilon: {self.epsilon:.1f}%\t"
                )
                if self.store_data:
                    self.plot_results()
                    self.save_metrics()
                    self.save_model()

    def train_batch(self):
        """
        Esegue un passo di training dell'agente DQN utilizzando un batch di transizioni dalla replay memory.

        1. Campiona un batch di transizioni dalla replay memory.
        2. Predice i Q-values per gli stati del batch utilizzando la rete online.
        3. Calcola i target Q-values utilizzando la rete target.
        4. Calcola il target Q-value per ogni transizione nel batch.
        5. Aggiorna i pesi della rete online utilizzando la loss tra i Q-values predetti e i target Q-values.
        6. Aggiorna la rete target se necessario.

        Args:
            None

        Returns:
            None

        """

        if len(self.memory) < self.batch_size:
            return

        # 1. CAMPIONO UN BATCH DI TRANSIZIONI
        batch = random.sample(self.memory, self.batch_size)

        # 2. PREDICO I Q VALUES CON L'ONLINE NETWORK
        observations = np.array(
            [transition[0] for transition in batch]
        )  # mi prendo gli stati
        actions = np.array(
            [transition[1] for transition in batch]
        )  # mi prendo le azioni

        q_values = self.online_network.predict(
            observations
        )  # predico i Q values per gli stati

        # 3. CALCOLO I TARGET Q VALUES CON IL TARGET NETWORK
        next_observations = np.array([transition[3] for transition in batch])
        if self.use_target_network:
            next_q_values = self.target_network.predict(next_observations)
        else:
            next_q_values = self.online_network.predict(next_observations)
        target_q_values = next_q_values.max(
            axis=1
        )  # prendo il massimo Q value per ogni stato successivo

        # 4. CALCOLO IL TARGET DELLA RETE

        # per non dover fare una loss custom, faccio una copia dei q_values e poi aggiorno solo gli indici delle azioni prese
        # ES. se ho 3 azioni e le azioni sono [0.1, 0.2, 0.3] e i q_values sono [1.0, 2.0, 3.0],
        #    allora target_q_values sarà [1.0, 2.0, 3.0] e poi aggiornerò solo l'indice corrispondente all'azione presa
        #    se l'azione presa è 0.2, allora target_q_values diventerà [1.0, reward + gamma*2.0, 3.0]
        #    dove reward è il reward

        target = (
            q_values.copy()
        )  # faccio una copia dei q_values per non modificarli direttamente

        dones = np.array([transition[4] for transition in batch])  # mi prendo i done
        rewards = np.array([transition[2] for transition in batch])
        for i in range(self.batch_size):

            # Calcolo il target Q value per ogni transizione
            y = rewards[i] + self.gamma * target_q_values[i] * (
                1 - dones[i]
            )  # calcolo il target Q value

            # Prendo l'indice dell'azione presa
            action_idx = self.get_action_idx(
                np.array(actions[i])
            )  # prendo l'indice dell'azione presa
            target[i, action_idx] = y

        # 5. AGGIORNO I PESI DELL'ONLINE NETWORK
        self.online_network.fit(
            observations, target, verbose=0, epochs=2
        )  # addestramento della rete online

        # 6. AGGIORNO IL TARGET NETWORK SE NECESSARIO
        if self.use_target_network:
            self.update_target_network_counter += 1
            if self.update_target_network_counter >= self.update_target_network_freq:
                self.target_network.set_weights(self.online_network.get_weights())
                self.update_target_network_counter = 0

    def save_metrics(self):
        results_df = pd.DataFrame(
            {
                "episode": range(1, len(self.reward_track) + 1),
                "reward": self.reward_track,
                "success_rate": self.success_track,
            }
        )

        # Salva in CSV
        results_df.to_csv(self.result_path + ".csv", index=False)


    def save_model(self):
        """
        Salva i pesi del modello in un file.

        """
        self.online_network.save(self.model_path + ".keras")

    def plot_results(self):
        """
        Crea un grafico dei reward, epsilon e success rate durante l'addestramento.

        1. Calcola la media mobile dei reward.
        2. Crea un grafico con tre subplot:
            - Moving average dei reward.
            - Epsilon decay nel tempo.
            - Success rate (media mobile).
        3. Salva il grafico in una cartella "plots" con nome "reward_epsilon_successrate_DQN_target_nw.png".

        Args:
            None

        Returns:
            None

        """

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))

        # Moving average dei reward
        sma_reward = compute_moving_average(
            self.reward_track, window_size=min(10, len(self.reward_track) // 2)
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
            self.success_track, window_size=min(10, len(self.success_track) // 2)
        )

        ax[1].set_title("Success Rate Progress")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Success Rate")
        ax[1].plot(self.success_track, alpha=0.3, label="Raw Success Rate")
        ax[1].plot(sma_success_rate, label="Moving Average", linewidth=2)
        ax[1].set_ylim(0, 1.1)
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        # Epsilon
        x_eps = np.arange(len(self.epsilon_history))
        ax[2].plot(x_eps, self.epsilon_history, "r-", label="Epsilon per episodio")
        ax[2].set_xlabel("Episodi")
        ax[2].set_ylabel("Epsilon")
        ax[2].set_ylim(0, 1)
        ax[2].set_title("Epsilon decay over num_episodes")
        ax[2].legend()
        ax[2].grid(True)
        ax[2].set_xlim(0, self.num_episodes - 1)

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

#     env = gym.make("PandaReach-v3", reward_type="dense")

# # Inizializza l'agente DQN
#     agent = DQN(
#         env=env,
#         num_episodes=10000,
#         print_every=100,
#         store_data=False,
#         use_target_network=False,
  
#     )

#     agent.train()
    