import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from collections import deque
from Env.Simple1DEnv import Simple1DEnv
import random

#Memory
q_table = np.zeros((5,2))
env = Simple1DEnv()

# Parameters
epsilon = 1
epislon_decay = 0.99
epsilon_min = 0.01
gamma = 0.8
alpha = 0.1

episodes = 500
successes = 0
for episode in range(episodes):
    state = env.reset()

    #steps
    steps = 300
    for step in range(steps):
        if  random.random() < epsilon:
            action = random.randint(0,1)
        else:
            action = np.argmax(q_table[state])
        # move
        new_state, reward, done = env.step(action)

        # Bellman Formel
        q_table[state] = q_table[state] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state])

        # If reached finish line break
        if done:
           successes += 1
           break
    
    epsilon = max(epsilon_min, epsilon * epislon_decay)
    print(f"Episode {episode+1} - Epsilon: {epsilon:.3f}")

print("\nFinal Q-Table:")
print(q_table)
print(f"\n✅ Success Rate: {successes / episodes * 100:.2f}%")
