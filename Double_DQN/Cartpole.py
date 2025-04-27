
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
        tf.keras.layers.Dense(24, activation="relu", input_shape=(state_size,)),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(action_size, activation="linear")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss="mse")
    return model

# Main model (Student)
model = build_model()

# Target model (for evaluation) (Teacher)
target_model = build_model()
target_model.set_weights(model.get_weights())  # initialize with same weights

def train_model():
    # Not enough samples in the replay buffer to create a full batch – skip training for now
    if len(replay_buffer) < batch_size:
        return

    # takes from the replay_buffer a random amount of 64 (batch_size) samples so our agent has a mixed experience form old and new Expriences
    minibatch = random.sample(replay_buffer, batch_size)
    states,targets  = [], []

    for state, action, reward, next_state, done in minibatch:
        target = reward

        if not done:
            # Double DQN core idea:
            # 1. Choose best next action using the main model
            best_next_action = np.argmax(model.predict(np.array([next_state]), verbose=0)[0])
            # 2. Evaluate that action using the target model
            target += gamma * target_model.predict(np.array([next_state]), verbose=0)[0][best_next_action]
        
        # Predicition from the model
        target_f = model.predict(np.array([state]), verbose=0)[0]

        # Replace the action value with the traget
        target_f[action] = target

        states.append(state)
        targets.append(target_f)

    # Train the Model
    model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

def get_action(state):
    # if the random number is less than the epsilon, do a random action
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)
    # if not use the models prediction
    # np.array([state]) -> changes the 1D-Vector to a 2D-Matrixs, becuase the model awaits a 2D-Matrix
    q_values = model.predict(np.array([state]), verbose=0)
    # Retrieve with 0 the prediciton from our model since it gives us the prediciton in batches
    return np.argmax(q_values[0])

# Training loop
n_episodes = 500
target_update_freq = 10  # update target model every 10 episodes

for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(500):
        action = get_action(state)
        next_state, reward, done, truncated, _ = env.step(action)

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train_model()

        if done:
            break

    # Update target model every few episodes
    if episode % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    # Update epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

env.close()