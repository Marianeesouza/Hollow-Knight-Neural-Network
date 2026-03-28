import torch
import numpy as np
import statistics

from Python.Src.Data import ClientPipe
from Python.Src.Data.DataHandler import DataHandler
from Python.Src.Models.ActorCritic import ActorCritic
from Python.Src.Utils.FileUtilities import FileUtilities
from Python.Src.Models.NeuralNetTraining import NeuralNetTraining
from Python.Src.Utils.NeuralNetUtilities import NeuralNetUtilities
from Python.Src.Data.RolloutBuffer import RolloutBuffer
from Python.Src.Utils.VirtualGamePad import VirtualGamePad
from Python.Src.Utils.StatsUtilities import StatsUtilities
from collections import deque

SAVE_INTERVAL = 90000
UPDATE_TIMESTEP = 4096
BOSS_SCENE_HASH = 423158243 # Hornet
NUM_FRAMES = 4

FILE_NAME = 'stats.pkl'

def save_data(is_ai_running, actor_critic, ppo_trainer, is_training, *stats):
    if is_ai_running and actor_critic is not None and is_training:
        NeuralNetUtilities.save_model(actor_critic, ppo_trainer.optimizer)
        FileUtilities.save_file(FILE_NAME, *stats)

def main():
    pipe = ClientPipe.Pipe()
    pipe.connect()

    frames_count = 0
    total_reward = 0
    episodes_counter = 0

    actor_critic = None
    ppo_trainer = None

    rollout_buffer = RolloutBuffer()
    virtual_gamepad = VirtualGamePad()

    frame_stack = deque(maxlen=NUM_FRAMES)
    reward_stack = deque(maxlen=100)

    episode_stats = []
    mean_stats = []
    best_mean_reward = -float('inf')

    is_ai_running = False
    is_training = False

    old_state = None
    old_data = None
    last_action = None
    last_log_prob = None
    last_value = None

    next_value = None

    try:
        while True:
            #retrieving the data from the pipe (Hollow Knight raw data)
            state = pipe.read_state()

            if state is None:
                print("⚠️ Connection lost.")
                save_data(is_ai_running, actor_critic, ppo_trainer, is_training, episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward)
                StatsUtilities.plot_graph(episode_stats, mean_stats)
                break
                # if is_ai_running and actor_critic is not None and is_training:
                #     NeuralNetUtilities.save_model(actor_critic, ppo_trainer.optimizer)
                #     FileUtilities.save_file(FILE_NAME, episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward)
                # break

            boss_scene = state['bossScene']
            data = DataHandler.treat_data(state)    # treating the data from the pipe

            # only start the neural_net (is_ai_running) if the player is in the boss scene
            if (boss_scene is not None and boss_scene == BOSS_SCENE_HASH) and not is_ai_running:

                input_dim = DataHandler.stored_size(data) * NUM_FRAMES
                output_dim = 9

                actor_critic = ActorCritic(input_dim, output_dim)
                ppo_trainer = NeuralNetTraining(actor_critic)

                NeuralNetUtilities.load_model(actor_critic, ppo_trainer.optimizer) # loading the current model

                episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward = FileUtilities.load_file()

                print('episodes_counter: ', episodes_counter)
                print('reward_stack amount: ', len(reward_stack))
                print('episode_stats amount: ', len(episode_stats))
                print('mean_stats amount: ', len(mean_stats))
                print('best performance: ', best_mean_reward)

                is_ai_running = True

            if not is_ai_running:
                continue

            if len(frame_stack) == 0:
                for _ in range(NUM_FRAMES):
                    frame_stack.append(data)
            else:
                frame_stack.append(data)

            stacked_data = np.concatenate(frame_stack)
            state_tensor = torch.FloatTensor(stacked_data).unsqueeze(0)

            with torch.no_grad():
                action, action_log_prob, state_value = actor_critic.act(state_tensor)

            action_array = action.squeeze(0).cpu().numpy()

            active_actions = [i for i, val in enumerate(action_array) if val == 1]  # check which actions probs are greater than 50%

            virtual_gamepad.update_gamepad(active_actions)  # execute those actions

            if is_training:
                # if we are not on the first frame
                if old_state is not None:
                    reward, done = NeuralNetUtilities.calculate_reward(old_state, state)  # calculate the reward of the current state based on the prev. one
                    total_reward += reward  # sum this reward value

                    # save those values into a buffer
                    rollout_buffer.push(
                        state=torch.FloatTensor(old_data),
                        action=last_action,
                        reward=reward,
                        is_terminal=done,
                        log_prob=last_log_prob,
                        value=last_value
                    )

                    # frame count for frame memory
                    frames_count += 1

                    if frames_count % UPDATE_TIMESTEP == 0:

                        if done:
                            next_value = torch.tensor(0.0, dtype=torch.float)
                        else:
                            with torch.no_grad():
                                next_value = actor_critic.get_value(state_tensor)

                        next_value = next_value.detach()

                        ppo_trainer.update(actor_critic, rollout_buffer, next_value)

                    if done:
                        episodes_counter += 1
                        #print(f'Episode: {episodes_counter} | Reward: {total_reward}')

                        reward_stack.append(total_reward)

                        current_mean = statistics.mean(reward_stack)

                        if len(reward_stack) == reward_stack.maxlen:
                            if current_mean > best_mean_reward:
                                best_mean_reward = current_mean
                                NeuralNetUtilities.save_model(actor_critic, ppo_trainer.optimizer, file_name='Checkpoints/Weights/HK_Model_best.pth')
                                print(f'New best performance: {current_mean}')

                        if episodes_counter % 10 == 0 and len(reward_stack) > 0:
                            episode_stats.append(episodes_counter)
                            mean_stats.append(current_mean)
                            print(f'Episode: {episodes_counter} | Mean Reward: {current_mean} | STD: {statistics.stdev(reward_stack)}')

                        total_reward = 0
                        old_state = None
                        old_data = None
                        frame_stack.clear()

                        continue

                old_state = state.copy()
                old_data = stacked_data.copy()

                last_action = action.squeeze(0).detach()
                last_log_prob = action_log_prob.squeeze(0).detach()
                last_value = state_value.squeeze(0).detach()

                if frames_count > 0 and frames_count % SAVE_INTERVAL == 0:
                    NeuralNetUtilities.save_model(actor_critic, ppo_trainer.optimizer)
                    FileUtilities.save_file(FILE_NAME, episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward)
                    StatsUtilities.plot_graph(episode_stats, mean_stats)

    except KeyboardInterrupt:
        save_data(is_ai_running, actor_critic, ppo_trainer, is_training, episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward)
        # if is_ai_running and actor_critic is not None and is_training:
        #     NeuralNetUtilities.save_model(actor_critic, ppo_trainer.optimizer)
        #     FileUtilities.save_file(FILE_NAME, episodes_counter, reward_stack, episode_stats, mean_stats, best_mean_reward)
    finally:
        pipe.disconnect()

if __name__ == "__main__":
    main()


