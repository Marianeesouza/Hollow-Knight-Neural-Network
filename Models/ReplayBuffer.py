import random
from collections import deque
import numpy as np

class ReplayBuffer:

    CAPACITY = 1000000

    def __init__(self):
        self.buffer = deque(maxlen=ReplayBuffer.CAPACITY)

    def push(self, old_state, action, reward, state, done: bool):
        self.buffer.append((old_state, action, reward, state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)