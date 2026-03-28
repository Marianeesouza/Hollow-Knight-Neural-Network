import pickle
import os
from collections import deque

class FileUtilities:

    @staticmethod
    def save_file(file_name: str = 'stats.pkl', *data):

        file_path = os.path.join('Checkpoints', 'Statistic', file_name)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"💾 Stats saved to {file_path}")

    @staticmethod
    def load_file(file_name: str = 'stats.pkl'):

        file_path = os.path.join('Checkpoints', 'Statistic', file_name)

        try:
            with open(file_path, "rb") as file:
                print("💾 Stats loaded")
                data = pickle.load(file)

                if len(data) == 1 and isinstance(data[0], tuple):
                    data = data[0]
                    return data
                elif len(data) == 4:
                    episodes_counter, reward_stack, episode_stats, mean_stats = data
                    best_mean_reward = -float("inf")
                elif len(data) == 5:
                    episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward = data
                else:
                    raise ValueError("Invalid file format")

                return episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward

        except FileNotFoundError:
            print(f"⚠️ File not found at {file_path}. Creating new empty file")
            return 0, deque(maxlen=100), [], [], -float("inf")
