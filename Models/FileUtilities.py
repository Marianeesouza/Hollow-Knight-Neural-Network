import pickle
from collections import deque

class FileUtilities:

    @staticmethod
    def save_file(file_name, *data):
        with open(file_name, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            print("💾 Stats saved")

    @staticmethod
    def load_file(file_name):
        try:
            with open(file_name, "rb") as file:
                print("💾 Stats loaded")
                data = pickle.load(file)

                if len(data) == 4:
                    episodes_counter, reward_stack, episode_stats, mean_stats = data
                    best_mean_reward = -float("inf")
                elif len(data) == 5:
                    episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward = data
                else:
                    raise ValueError("Invalid file format")

                return episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward

        except FileNotFoundError:
            print("⚠️ File not found. Creating new file")
            return 0, deque(maxlen=100), [], [], -float("inf")
