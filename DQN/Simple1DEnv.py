import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from collections import deque
from Env.Simple1DEnv import Simple1DEnv
import random


env = Simple1DEnv()

# Memory Structure
replay_buffer = deque(maxlen=2000)
batch_size = 64


# Parameters
gamma = 0.5
alpha = 0.9
epsilon = 1
epsilon_min = 0.1
epsilon_decay = 0.9

# Model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24,"relu",input_shape=(1,)),
        tf.keras.layers.Dense(24,"relu"),
        tf.keras.layers.Dense(2, "linear")])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

model = build_model()

# train model
def train_model():

    if len(replay_buffer) < batch_size:
        return
    
    minibatch = random.sample(replay_buffer,batch_size)
    states, targets = [], []

    for state, reward, action, new_state, done in minibatch:
        target = reward

        if not done:
            target *= alpha
            target += gamma * np.max(model.predict(np.array([new_state]))[0])
        
        target_f  = model.predict(np.array([state]))[0]
        target_f[action] = target

        states.append(state)
        targets.append(target_f)
    
    model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

def action(state):
    if random.random() < epsilon:
        action = random.randint(0,1)
    else:
        action = np.argmax(model.predict(np.array([state])))

    return action

# training loop
for episode in range(300):
    state = env.reset()

    #step
    for step in range(300):
        #take action
        action_to_take = action(state)
        
        new_state, reward, done = env.step(action_to_take)

        replay_buffer.append([state,reward,action_to_take,new_state,done])

        train_model()
        
        if done:
            break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
