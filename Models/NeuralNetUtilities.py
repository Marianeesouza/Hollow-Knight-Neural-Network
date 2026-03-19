import numpy as np
import torch
import os

class NeuralNetUtilities:

    @staticmethod
    def save_model(actor_critic, optimizer, episodes: int, file_name: str = 'HK_Model.pth'):
        checkpoint = {
            'model_state_dict': actor_critic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episodes': episodes
        }
        torch.save(checkpoint, file_name)
        print("💾 Model saved")

    @staticmethod
    def load_model(actor_critic, optimizer, file_name: str = 'HK_Model.pth'):
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name)
            actor_critic.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            episodes = checkpoint.get('episodes', 0)
            print("💾 Model loaded")
            return episodes
        else:
            print("⚠️ File not found. Neural Network restarting learning")
            return 0

    @staticmethod
    def calculate_ratio(old_state, current_state, epsilon, advantage):
        return np.minimum(((current_state / old_state) * advantage), np.clip(current_state / old_state, 1 - epsilon, 1 + epsilon) * advantage)


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

        return float(reward), False