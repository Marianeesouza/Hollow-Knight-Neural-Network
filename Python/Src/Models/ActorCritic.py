import torch
import torch.nn as nn
from torch.distributions import Bernoulli

# Actions:
    # Idle
    # Move Up
    # Move Down
    # Move Left
    # Move Right
    # Jump
    # Dash
    # Attack
    # Cast
    # Heal

HIDDEN_DIM = 256

class ActorCritic(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(HIDDEN_DIM, output_dim),
            nn.Sigmoid(),
        )

        self.critic = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1),
        )

    def act(self, state):
        x = self.shared(state)
        action_probs = self.actor(x)

        dist = Bernoulli(action_probs)

        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        state_value = self.critic(x)

        return action, action_log_prob, state_value

    def evaluate(self, state, action):
        x = self.shared(state)
        action_probs = self.actor(x)
        dist = Bernoulli(action_probs)

        action_log_prob = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(x)

        return action_log_prob, state_values, dist_entropy

    def get_value(self, state):
        x = self.shared(state)
        return self.critic(x)
