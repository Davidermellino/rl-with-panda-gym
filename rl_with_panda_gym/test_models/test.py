import tensorflow as tf
import numpy as np
import panda_gym
import pandas as pd
import gymnasium as gym
from time import sleep

from rl_with_panda_gym.utils import preprocess_obs, load_model
from rl_with_panda_gym.models.DQN import DQN
from rl_with_panda_gym.models.REINFORCE import Reinforce, ContinuousPolicy, DiscretePolicy
from rl_with_panda_gym.models.DDPG import DDPG

def choose_action(model, agent, name, obs, env):
    
    if name == "DQN":
        q_values = model.predict(np.array([obs]))
        action_idx = np.argmax(q_values)
        return agent.all_action_combination[action_idx]

    elif name == "REINFORCE":
        state_tensor = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float32), 0)
        action,_ = agent.sample_action(model, state_tensor)
        
    elif name == "DDPG":
        state_tensor = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float32), 0)
        action_tensor = model(state_tensor)[0]
        action = action_tensor.numpy()
        
        return np.clip(action, env.action_space.low, env.action_space.high)
    else:
        raise ValueError("Modello non esistente, scegli tra: 'DQN', 'REINFORCE', o 'DDPG'.")
    
    return action

def test_model(model, name, agent, env,  render=True, num_episodes=100, save=False, save_path="./results"):
    

    reward_track = np.zeros(shape=(num_episodes,))
    success_track = np.zeros(shape=(num_episodes,))
    
    for i in range(num_episodes):
        
        # Reset dell'ambiente
        obs, _ = env.reset()
        obs = preprocess_obs(obs)
        num_step = 0
        done = False
        additional_step = 0
        while additional_step < 2:
            
            
            action = choose_action(model, agent, name, obs, env)
            

            obs, reward, terminated, truncated, info = env.step(action)
            obs = preprocess_obs(obs)
            done = terminated or truncated
            if done:
                additional_step += 1
                success_track[i] = info["is_success"]

            num_step += 1
            reward_track[i] += reward
            if render: sleep(0.1)
        
        if render: print(f"Episode {i+1}/{num_episodes} - Reward: {reward_track[i]:.2f} - Success: {success_track[i]} - Steps: {num_step}")
        if render: sleep(1)
            
    env.close()
    
    if save:
         #SALVO RICOMPENSE E SUCCESSO
        df = pd.DataFrame({"reward": reward_track, "success": success_track})
        df.to_csv(save_path + ".csv", index=False)
    
    return reward_track, success_track

if __name__ == "__main__":
    
    
    
    #CARICO MODELLO
    model_path = "models/panda_push_models/actor_ddpg_her_push_sparse.keras"
    save_path = "results/test/panda_push_results/dqn_results"
   
    
    
    model = load_model(model_path, model_type="full")

    #CREO AMBIENTE
    env = gym.make("PandaPush-v3", render_mode="human",reward_type="sparse")

    #CREO AGENTE
    agent = DDPG(env=env, noise_std=0)

    #TESTO IL MODELLO
    reward_track, success_track = test_model(model=model, name="DDPG", agent=agent, env=env, save=False, save_path=save_path, render=True)
    
    print(f"Reward medio: {np.mean(reward_track)}")
    print(f"Success rate: {np.mean(success_track)}")

