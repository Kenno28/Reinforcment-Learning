import gymnasium as gym;
import numpy as np;
import random;
# Create Enviroment
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="ansi")

# Initialize the Q-table
# np.zeros to create a 2D-Matrix
# np.zeros((n_states, n_actions)) -> n_states = 16, because there are 16 States possible due to the field being 4x4 and n_actions = 4, because we have 4 possible actions
# Each state has 4 possible Actions, depending which of them has the highest Value the Agent will take that route
# the values within the table will later be replaced with the Bellman-Formula 
q_table = np.zeros((16,4))

# Hyperparameters:
# Name            | Example      | Effect
# ----------------|--------------|----------------------------------------------------
# alpha           | 0.1          | Learning rate – higher = faster learning, but less stable
# gamma           | 0.99         | Discount factor – how much future rewards are considered
# epsilon         | 1.0          | Exploration rate – how often to explore randomly
# epsilon_decay   | 0.995        | How quickly exploration decreases over time
# epsilon_min     | 0.01         | Minimum exploration – ensures agent never stops exploring
# n_episodes      | 100000       | Number of training episodes – more episodes = more learning
# max_steps       | 300          | Max steps per episode – too low = goal might not be reached
alpha = 0.1        
gamma = 0.99        
epsilon = 0.9
epsilon_decay = 0.995
epsilon_min = 0.01
n_episodes = 100000
max_steps = 300

# To track the success rate over episodes
rewards = []

# Training loop
for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Exploration vs. Exploitation:
        # With probability epsilon, take a random action (exploration)
        # Otherwise, take the best known action from the Q-table (exploitation)
        if random.random() < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(q_table[state])

        # Perform the action and observe the result
        new_state, reward, done, truncated, info = env.step(action)

        # update q-table with Bellman-Formula
        # (reward + gamma * np.max(q_table[new_state]) - q_table[state][action]) - The value of the new state is subtracted from the current state to find out how good the new step was 
        q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state][action])

        # Move to the next state
        state = new_state
        total_reward += reward
        if done:
            break
    # Decay epsilon – slowly reduce exploration over time, but never below epsilon_min
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

print(f"Success rate: {sum(rewards) / len(rewards)}")
