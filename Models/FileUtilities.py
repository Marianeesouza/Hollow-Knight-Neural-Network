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
                return pickle.load(file)
        except FileNotFoundError:
            print("⚠️ File not found. Creating new file")
            return 0, deque(maxlen=100), [], []
