import gymnasium as gym
import numpy as np
import tensorflow as tf
from collections import deque
import random

# Set up environment
env = gym.make("CartPole-v1")

# Parameters
state_size = env.observation_space.shape[0]   # Cart position, velocity, pole angle, etc.
action_size = env.action_space.n              # 2 actions: left or right

epsilon = 1.0             # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

gamma = 0.99              # Discount factor
alpha = 0.001             # Learning rate

batch_size = 64
# Memory structure of the agent
replay_buffer = deque(maxlen=2000) 

# Build the Q-network
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(state_size,)),
        tf.keras.layers.Dense(24, activation='relu'),
        # Last Layer to decide which action to choose either 0 or 1 (left or right)
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    # Compile Model with Adma optimaztion
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
    return model

model = build_model()

def get_action(state):
    # if the random number is less than the epsilon, do a random action
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)
    # if not use the models prediction
    # np.array([state]) -> changes the 1D-Vector to a 2D-Matrixs, becuase the model awaits a 2D-Matrix
    q_values = model.predict(np.array([state]), verbose=0)
    # Retrieve with 0 the prediciton from our model since it gives us the prediciton in batches
    return np.argmax(q_values[0])

def train_model():
    # Not enough samples in the replay buffer to create a full batch – skip training for now
    if len(replay_buffer) < batch_size:
        return

    # takes from the replay_buffer a random amount of 64 (batch_size) samples so our agent has a mixed experience form old and new Expriences
    minibatch = random.sample(replay_buffer, batch_size)
    states, targets = [], []

    for state, action, reward, next_state, done in minibatch:
        target = reward

        if not done:
            # Add discounted future reward to the current target.
            # We use the maximum predicted Q-value from the next state
            # → representing the best possible future action according to the current model.
            target += gamma * np.amax(model.predict(np.array([next_state]), verbose=0)[0])

        # Predicition from the model
        target_f = model.predict(np.array([state]), verbose=0)[0]

        # Replace the action value with the traget
        target_f[action] = target

        states.append(state)
        targets.append(target_f)

    # Train the Model
    model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

n_episodes = 10
for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0

    # The model has not been trained yet, so predictions will be based on random weights.
    # Due to epsilon being high at the start, the agent explores mostly at random anyway.
    for _ in range(500):

        # Select the next action (exploration vs. exploitation)
        action = get_action(state)

        # Perform the action in the environment
        next_state, reward, done, truncated, _ = env.step(action)

        # Store the experience (state transition) in the replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        # Move to the next state
        state = next_state
        total_reward += reward

        # Train the model if enough samples are available in the buffer
        train_model()

        # Stop the episode if the environment signals it's done
        if done:
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

env.close()