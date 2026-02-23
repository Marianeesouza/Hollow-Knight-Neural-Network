import random
import numpy as np

from Models import ClientPipe
from Models.DataHandler import DataHandler
from Models.NeuralNet import NeuralNet
from Models.NeuralNetTraining import NeuralNetTraining
from Models.NeuralNetUtilities import NeuralNetUtilities
from Models.ReplayBuffer import ReplayBuffer
from Models.VirtualGamePad import VirtualGamePad

SAVE_INTERVAL = 3600
TARGET_UPDATE = 1000
BATCH_SIZE = 64
DISCOUNT_FACTOR = 0.99
BOSS_SCENE_HASH = 383855111

def main():
    pipe = ClientPipe.Pipe()
    pipe.connect()

    frames_count = 0

    neural_net = NeuralNet()
    target_net = NeuralNet()

    is_ai_running = False
    is_ai_training = True

    replay_buffer = ReplayBuffer()

    virtual_gamepad = VirtualGamePad()

    total_reward = 0
    episodes_counter = 0

    old_state = None
    old_data = None

    boss_scene = None

    try:
        while True:

            # retrieving the data from the pipe (Hollow Knight raw data)
            state = pipe.read_state()

            if state is None:
                print("⚠️ Connection lost.")
                if is_ai_running:
                    NeuralNetUtilities.save_model(neural_net.weights, neural_net.biases, NeuralNetTraining.epsilon, episodes_counter)
                break

            boss_scene = state['bossScene']

            # treating the data from the pipe
            data = DataHandler.treat_data(state)

            # only start the neural_net (is_ai_running) if the player is in the boss scene
            if (boss_scene is not None and boss_scene == BOSS_SCENE_HASH) and not is_ai_running:
                neural_net.initialize(DataHandler.stored_size(data))

                loaded_weights, loaded_biases, NeuralNetTraining.epsilon, episodes_counter = NeuralNetUtilities.load_model()

                replay_buffer.load_buffer()

                if loaded_weights is not None and loaded_biases is not None:
                    neural_net.weights = loaded_weights
                    neural_net.biases = loaded_biases

                target_net = neural_net.copy()
                is_ai_running = True

            if is_ai_running:

                # q_values from the neural net
                values = neural_net.forward(data)[0]

                if random.random() < NeuralNetTraining.epsilon:
                    next_actions = [i for i in range(len(values)) if random.random() > 0.5]
                else:
                    next_actions = [i for i, prob in enumerate(values) if prob > 0.5]

                virtual_gamepad.update_gamepad(next_actions)

                if is_ai_training:
                    executed_actions = np.zeros(len(values))
                    for action_idx in next_actions:
                        executed_actions[action_idx] = 1

                    if old_state is not None:
                        reward, done = NeuralNetUtilities.calculate_reward(old_state, state)
                        replay_buffer.push(old_data, executed_actions, reward, data, done)
                        #NeuralNetTraining.update_epsilon()

                        total_reward += reward

                        if len(replay_buffer) >= BATCH_SIZE:
                            b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(BATCH_SIZE)

                            current_q_values = neural_net.forward(b_states)
                            next_q_values = target_net.forward(b_next_states)

                            target_q_values = current_q_values.copy()

                            for i in range(BATCH_SIZE):
                                active_actions = [idx for idx, prob in enumerate(b_actions[i]) if prob > 0.5]

                                for action_idx in active_actions:
                                    if b_dones[i]:
                                        target_q_values[i][action_idx] = b_rewards[i]
                                    else:
                                        target_q_values[i][action_idx] = NeuralNetTraining.bellman(b_rewards[i], DISCOUNT_FACTOR, next_q_values[i])

                            neural_net.weights, neural_net.biases =  NeuralNetTraining.optimize(
                                neural_net.weights,
                                neural_net.biases,
                                b_states,
                                target_q_values)

                        if done:
                            NeuralNetTraining.update_epsilon()
                            old_state = None
                            old_data = None
                            episodes_counter += 1

                            NeuralNetUtilities.save_model(neural_net.weights, neural_net.biases,
                                                          NeuralNetTraining.epsilon, episodes_counter)

                            if episodes_counter % 100 == 0:
                                replay_buffer.save_buffer()

                            print('Epsilon: ', NeuralNetTraining.epsilon)
                            print('Reward: ', total_reward)
                            print('Episode: ', episodes_counter)
                            print('Buffer size: ', len(replay_buffer.buffer))

                            total_reward = 0
                            continue

                    old_state = state.copy()
                    old_data = data.copy()

                    frames_count += 1

                    if frames_count % TARGET_UPDATE == 0:
                        target_net = neural_net.copy()

                    if frames_count % SAVE_INTERVAL == 0:
                        NeuralNetUtilities.save_model(neural_net.weights, neural_net.biases, NeuralNetTraining.epsilon, episodes_counter)

    except KeyboardInterrupt:
        if is_ai_running:
            NeuralNetUtilities.save_model(neural_net.weights, neural_net.biases, NeuralNetTraining.epsilon, episodes_counter)
            replay_buffer.save_buffer()
    finally:
        pipe.disconnect()

if __name__ == "__main__":
    main()


