import random

class Simple1DEnv:
    def __init__(self, size=5, goal=4, danger =2, is_slippery = True, slip_prob= 0.2):
        self.size = size
        self.goal = goal
        self.danger = danger
        self.is_slippery = is_slippery
        self.slip_prob = slip_prob
        self.reset()

    def reset(self):
        self.agent_pos = 0
        return self.agent_pos
    
    def step(self, action):
        done = False
        reward = -0.01

        if random.random() < self.slip_prob and self.is_slippery:
            if action == 0:
                action = 1
            else:
                action = 0

        # Action: 0 = left, 1 = right
        if action == 0:
            self.agent_pos = max(0, self.agent_pos - 1)
        else:
            self.agent_pos = min(self.size - 1, self.agent_pos + 1)

        # End episode if the agent falls
        if self.agent_pos == self.danger:
            reward = -1
            done = True

        if self.agent_pos == self.goal:
            reward = 1.0
            done = True

        return self.agent_pos, reward, done