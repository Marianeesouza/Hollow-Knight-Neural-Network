import torch
import numpy as np

from Models import ClientPipe
from Models.DataHandler import DataHandler
from Models.ActorCritic import ActorCritic
from Models.NeuralNetTraining import NeuralNetTraining
from Models.NeuralNetUtilities import NeuralNetUtilities
from Models.RolloutBuffer import RolloutBuffer
from Models.VirtualGamePad import VirtualGamePad
from collections import deque

SAVE_INTERVAL = 90000
UPDATE_TIMESTEP = 2048
BOSS_SCENE_HASH = 423158243 # Hornet
NUM_FRAMES = 4

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

    is_ai_running = False

    old_state = None
    old_data = None
    last_action = None
    last_log_prob = None
    last_value = None

    try:
        while True:

            #retrieving the data from the pipe (Hollow Knight raw data)
            state = pipe.read_state()

            if state is None:
                print("⚠️ Connection lost.")
                if is_ai_running and actor_critic is not None:
                    NeuralNetUtilities.save_model(actor_critic, ppo_trainer.optimizer, episodes_counter)
                break

            boss_scene = state['bossScene']

            data = DataHandler.treat_data(state)    # treating the data from the pipe

            #only start the neural_net (is_ai_running) if the player is in the boss scene
            if (boss_scene is not None and boss_scene == BOSS_SCENE_HASH) and not is_ai_running:

                input_dim = DataHandler.stored_size(data) * NUM_FRAMES
                output_dim = 9

                actor_critic = ActorCritic(input_dim, output_dim)
                ppo_trainer = NeuralNetTraining(actor_critic)

                episodes_counter = NeuralNetUtilities.load_model(actor_critic, ppo_trainer.optimizer)

                #return
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
                    ppo_trainer.update(actor_critic, rollout_buffer)

                if done:
                    episodes_counter += 1
                    print(f'Episode: {episodes_counter} | Reward: {total_reward}')

                    total_reward = 0
                    old_state = None
                    old_data = None
                    frame_stack.clear()

                    #NeuralNetUtilities.save_model(actor_critic, ppo_trainer.optimizer, episodes_counter)

                    continue

            old_state = state.copy()
            old_data = stacked_data.copy()

            last_action = action.squeeze(0).detach()
            last_log_prob = action_log_prob.squeeze(0).detach()
            last_value = state_value.squeeze(0).detach()

            if frames_count > 0 and frames_count % SAVE_INTERVAL == 0:
                NeuralNetUtilities.save_model(actor_critic, ppo_trainer.optimizer, episodes_counter)

    except KeyboardInterrupt:
        if is_ai_running and actor_critic is not None:
            NeuralNetUtilities.save_model(actor_critic, ppo_trainer.optimizer, episodes_counter)
    finally:
        pipe.disconnect()

if __name__ == "__main__":
    main()


