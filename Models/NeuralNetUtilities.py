import numpy as np

class NeuralNetUtilities:

    # Activation Functions
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        sig = NeuralNetUtilities.sigmoid(z)
        return sig * (1 - sig)

    @staticmethod
    def save_model(weights: list, biases: list, epsilon: float, episodes: int, file_name: str = 'HK_Model.npz'):

        w_arr = np.empty(len(weights), dtype=object)
        w_arr[:] = weights

        b_arr = np.empty(len(biases), dtype=object)
        b_arr[:] = biases

        np.savez(file_name, weights=w_arr, biases=b_arr, epsilon=epsilon, episodes=episodes)
        print("💾 Model saved")

    @staticmethod
    def load_model(file_name: str = 'HK_Model.npz'):
        try:
            data = np.load(file_name, allow_pickle=True)
            print("💾 Model loaded")
            return data['weights'].tolist(), data['biases'].tolist(), data['epsilon'].item(), data['episodes'].item()

        except FileNotFoundError:
            print("⚠️ File not found. Neural Network restarting learning")
            return None, None, 1


    # Reward
    @staticmethod
    def calculate_reward(old_state: dict, state: dict):
        reward = 0

        # attack boss
        boss_damage = old_state['bossHp'] - state['bossHp']

        # if the current boss hp is less than the previous boss hp
        if boss_damage > 0:
            reward += boss_damage # around 30 p

        # player hp
        player_damage = old_state['hp'] - state['hp']

        # if the player takes damage
        if player_damage > 0:
            reward -= 40 # 40 p

        # player healing
        player_heal = state['hp'] - old_state['hp']

        # if the player gained hp,
        if player_heal > 0:
            reward += 5 # around 5 points

        # Player death. If the previous state, the player had least one hp and now he is dead
        if old_state['hp'] > 0 and state['hp'] <= 0:
            return float(-1), True

        # Boss death.
        if old_state['bossHp'] > 0 and state['bossHp'] <= 0:
            return float(1), True

        # encouraging the player to be aggressive

        current_delta_x = state['bx'] - state['px']
        current_delta_y = state['by'] - state['py']

        if abs(current_delta_x) < 10 and abs(current_delta_y) < 5 and player_damage == 0:
            reward += 0.001 #

        # if the player don't act. Bleeding effect
        reward -= 0.003

        scaled_reward = reward / 100

        scaled_reward = np.clip(scaled_reward, -1, 1)

        return float(scaled_reward), False