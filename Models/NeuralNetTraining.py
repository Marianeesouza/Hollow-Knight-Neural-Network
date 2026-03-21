import torch
import torch.nn as nn

from Models.RolloutBuffer import RolloutBuffer


class NeuralNetTraining:
    def __init__(self, actor_critic, lr=0.0003, gamma=0.99, k_epochs=8, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def gae(self, rewards, values, dones, next_value, lam = 0.95):
        advantages = []
        gae = 0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]

            delta = rewards[step] + self.gamma * next_val * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + values.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, returns

    def update(self, actor_critic, buffer: RolloutBuffer, next_value):
        rewards = torch.tensor(buffer.rewards, dtype=torch.float32)
        dones = torch.tensor(buffer.is_terminals, dtype=torch.float32)
        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach()
        old_log_probs = torch.squeeze(torch.stack(buffer.log_probs, dim=0)).detach()
        old_values = torch.squeeze(torch.stack(buffer.values, dim=0)).detach()

        advantages, returns = self.gae(
            rewards = rewards,
            values = old_values,
            dones = dones,
            next_value=next_value
        )

        batch_size = old_states.size(0)
        mini_batch_size = 64

        for _ in range(self.k_epochs):

            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                batch_idx = indices[start:end]

                batch_states = old_states[batch_idx]
                batch_actions = old_actions[batch_idx]
                batch_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                #batch_rewards = rewards[batch_idx]
                batch_rewards = returns[batch_idx]

                log_probs, state_values, dist_entropy = actor_critic.evaluate(batch_states, batch_actions)

                state_values = state_values.view(-1)
                dist_entropy = dist_entropy.mean()

                ratios = torch.exp(log_probs - batch_log_probs)

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values.view(-1), batch_rewards) - 0.01 * dist_entropy
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        buffer.clear()