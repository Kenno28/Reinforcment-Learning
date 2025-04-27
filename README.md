# 🧠 Reinforcement Learning Playground
Welcome to my Reinforcement Learning (RL) Playground!  
This repository documents my journey into RL by implementing, analyzing, and improving various algorithms from scratch.  
The goal is not just to use RL — but to truly understand how and why it works.

---

## 📁 Projects

| Project                         | Description                                                                                       | Status       |
|----------------------------------|--------------------------------------------------------------------------------------------------|--------------|
| 🧊 **FrozenLake Q-Learning**     | Classic table-based Q-learning agent for the FrozenLake environment                              | ✅ Complete  |
| 🤖 **CartPole DQN**              | Deep Q-Network agent that learns to balance a pole using neural networks and replay memory       | ✅ Complete  |
| 🛷 **Simple1DEnv with Danger**   | Custom-built 1D environment with goal and danger zones, including Q-Learning and Slippery Mode   | ✅ Complete  |
| 🧠 **CartPole Double DQN**       | Implementation of Double DQN to reduce overestimation bias and stabilize learning on CartPole    | ✅ Complete |

---

## 🧊 FrozenLake Q-Learning

### 🔍 Overview

A simple but powerful implementation of Q-Learning in the FrozenLake-v1 environment (`is_slippery=True`).  
The agent has no prior knowledge of the environment and must learn the optimal path through **trial and error** using the Bellman equation and epsilon-greedy exploration.

### 💡 Concepts Learned

- How Q-Learning works under the hood
- The Bellman equation explained and implemented
- Exploration vs. Exploitation through epsilon decay
- Off-policy (Q-Learning) vs. On-policy (SARSA)

---

## 🤖 CartPole DQN

### 🔍 Overview

This project implements a Deep Q-Network (DQN) to solve the CartPole-v1 environment.  
Instead of using a Q-table, the agent uses a neural network to approximate Q-values in a continuous state space.  
This allows the agent to generalize better and scale to more complex environments.

### 🧠 Key Concepts Learned

- Building a Q-network using TensorFlow and Keras
- Using Replay Buffer for stabilized learning
- Epsilon-greedy strategy for exploration vs. exploitation
- Updating Q-values with the Bellman equation
- Training the model step-by-step during episodes

---

## 🛷 Simple1DEnv with Danger (Slippery Environment)

### 🔍 Overview

Custom-designed 1D grid environment where the agent must reach the goal while avoiding a "danger" tile.  
The environment also introduces stochasticity ("slippery" behavior), forcing the agent to plan under uncertainty.

### 🧠 Key Concepts Learned

- Designing custom environments for Reinforcement Learning
- Handling negative rewards (danger zones)
- Adding stochastic behavior (slippery transitions)
- Training Q-Learning agents in uncertain environments

---

## 🧠 CartPole Double DQN

### 🔍 Overview

Double DQN implementation to fix overestimation problems in standard DQN.  
This approach separates action selection and action evaluation to create more stable and reliable learning.

### 🚀 Concepts Learned

- Why DQN tends to overestimate Q-values
- How Double DQN corrects these issues
- Updating the target model periodically
- Better stability and final performance

---

# 🚀 Future Work

- Dueling DQN
- Prioritized Experience Replay
- Policy Gradient Methods (REINFORCE, PPO)
- Full Minecraft RL Agent
- RL applied to real-world finance datasets

---

**Follow the journey!** 😎🚀
