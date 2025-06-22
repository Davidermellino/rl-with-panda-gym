import numpy as np
import tensorflow as tf

def compute_moving_average(data, window_size=10):
        """
        Calcola la media mobile dei reward ottenuti durante l'addestramento.
        
        Args:
            data: lista dei reward per episodio
            window_size: dimensione della finestra per il calcolo della media mobile (default: 10).
        
        Returns:
            Array con le medie mobili. Il primo valore corrisponde alla media
            dei primi window_size episodi.
            
        """
        if len(data) < window_size:
            return np.array([])
        
        weights = np.ones(window_size) / window_size
        sma = np.convolve(data, weights, mode='valid')
        return sma
    
def preprocess_obs(observation):
        """
           Concatena "achieved_goal", "desired_goal" e "observation" dell'osservazione in un unico array.

        Args:
            observation (Dict): Osservazione dell'ambiente PandaGym contenente "achieved_goal", "desired_goal" e "observation".
        Returns:
            np.ndarray: un numpy array concatenato contenente achieved_goal, desired_goal e observation.

        """

        achieved_goal = observation["achieved_goal"]
        desired_goal = observation["desired_goal"]
        obs = observation["observation"]

        return np.concatenate([achieved_goal, desired_goal, obs])
    


def load_model(model_path, model=None, model_type="full"):
    """
    Carica i pesi di un modello, assicurandosi che sia costruito correttamente.
    
    Args:
        model_path (str): Percorso ai pesi salvati
        model: Istanza del modello (già creata)
        input_shape (tuple): Forma dell'input (batch_size, *dims)
        sample_input (tf.Tensor): Input di esempio per costruire il modello
    
    Returns:
        model: Modello con i pesi caricati
    """
  
    if model_type == "full":
        return tf.keras.models.load_model(model_path)
    elif model_type == "weights":
       
        model.build(input_shape=(None, 12))  # Modifica questa dimensione

        model.load_weights(model_path)

    return model
   


