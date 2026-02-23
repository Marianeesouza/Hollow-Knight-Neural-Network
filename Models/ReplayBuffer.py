import random
import numpy as np
import pickle

from collections import deque

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

    def save_buffer(self, file_name: str = 'HK_Buffer.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('💾 ReplayBuffer saved')

    def load_buffer(self, file_name: str = 'HK_Buffer.pkl'):
        try:
            with open(file_name, 'rb') as f:
                self.buffer = pickle.load(f)
            print("💾 ReplayBuffer loaded")
        except FileNotFoundError:
            print("⚠️ ReplayBuffer file not found. Starting with empty buffer.")
            self.buffer = deque(maxlen=ReplayBuffer.CAPACITY)

    def __len__(self):
        return len(self.buffer)