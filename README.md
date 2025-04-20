Welcome to my Reinforcement Learning (RL) Playground!  
This repository documents my journey into RL by implementing, analyzing, and improving various algorithms from scratch. The goal is not just to use RL — but to truly understand how and why it works.

---

## 📁 Projects

| Project                       | Description                                                                       | Status     |
|-------------------------------|-----------------------------------------------------------------------------------|------------|
| 🧊 **FrozenLake Q-Learning**  | Classic table-based Q-learning agent for the FrozenLake environment              | ✅ Complete |


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