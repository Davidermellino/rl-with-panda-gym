# 🤖 Reinforcement Learning with PandaGym

This project explores the application of Deep Reinforcement Learning (DRL) algorithms to robotic control tasks using the [PandaGym](https://github.com/Farama-Foundation/Panda-Gym) simulation environment, which models the Franka Emika Panda robotic arm.

The goal is to train agents to solve robotic manipulation tasks using various RL algorithms. The focus is on two specific environments:

- **PandaReach**: the end-effector must reach a randomly generated point in 3D space.
- **PandaPush**: the robot must push a cube to a randomly chosen goal position.

We experimented with several RL techniques:

- **Value-based**: DQN (single and target network variants)
- **Policy-based**: REINFORCE (continuous and discrete)
- **Actor-Critic**: DDPG, DDPG + HER (Hindsight Experience Replay)

The best performance was achieved using **DDPG + HER**, which demonstrated strong results on both tasks

---

## 🎯 Results

Below are the results of trained agents on both environments.

### 🐼 PandaReach

The robot arm learns to reach the target in just a few hundred episodes using DQN target network and DDPG.

![PandaReach](video_test/gifs/DDPG_HER_REACH.gif)

### 📦 PandaPush

The agent successfully learns to push the cube to the goal using DDPG combined with HER.

![PandaPush](https://github.com/Davidermellino/rl-with-panda-gym/blob/main/video_test/gifs/DDPG_HER%20_PUSH.gif)

---

## ⚙️ Installation

Install the module using pip:

```bash
pip install rl-with-panda-gym
```

---

## 🚀 Example Usage

To train DDPG + HER on PandaReach ( takes few minutes to converge )

```python
from rl_with_panda_gym.models.DDPG import DDPG


# Create Environment
env = gym.make("PandaReach-v3", reward_type="sparse")

# Initialize DDPG Agent
agent = DDPG(
    env=env,
    num_episodes=2000,
    print_every=20,
    store_data=True,
    train_with_her=True,
    result_path=result_path_DDPG,
    model_path=model_path_DDPG,
    plot_path=plot_path_DDPG,
)

agent.train()

```
---

## ✅ Solved Tasks

- [x] PandaReach
- [x] PandaPush
- [ ] PandaSlide
- [ ] PandaPickAndPlace
- [ ] PandaStack
- [ ] PandaFlip

---

## 🔭 Future Work

- Apply DDPG + HER to more complex environments like PandaPickAndPlace
- Implement and benchmark advanced actor-critic methods such as PPO, A2C, and SAC
- Investigate sim-to-real transfer of trained policies to physical robots

---

## 👨‍💻 Author

Developed by Davide Ermellino - Università degli Studi di Cagliari 
For any questions or collaborations: ermellinodavide@gmail.com
