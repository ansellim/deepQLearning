import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

random.seed(43)
np.random.seed(43)
torch.manual_seed(43)


class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, layer_widths=[128, 128, 32]):

        super(DeepQNetwork, self).__init__()

        widths = copy.deepcopy(layer_widths)
        widths.insert(0, state_size)
        widths.append(action_size)

        layer_list = []
        for i in range(len(widths) - 1):
            layer_list.append(nn.Linear(widths[i], widths[i + 1]))
            if i != len(widths) - 2:
                layer_list.append(nn.ReLU())

        self.layers = nn.Sequential(*layer_list)

    def forward(self, state_vector):
        return self.layers(state_vector)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque([], maxlen=buffer_size)  # capacity of the buffer is fixed, finite
        self.batch_size = batch_size

    def add_experience(self, state, action, reward, next_state, done):
        """
        Save a transition
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample_experiences(self):
        experiences = random.sample(self.memory, self.batch_size)
        states = np.vstack([experience[0] for experience in experiences if experience is not None])
        states = torch.FloatTensor(states)
        actions = np.vstack([exp[1] for exp in experiences if exp is not None])
        actions = torch.LongTensor(actions)
        rewards = np.vstack([exp[2] for exp in experiences if exp is not None])
        rewards = torch.FloatTensor(rewards)
        next_states = np.vstack([exp[3] for exp in experiences if exp is not None])
        next_states = torch.FloatTensor(next_states)
        dones = np.vstack([exp[4] for exp in experiences if exp is not None]).astype(np.uint8)
        dones = torch.LongTensor(dones)
        return states, actions, rewards, next_states, dones

    def get_num_elements(self):
        return len(self.memory)


def interpolate_networks(primary_network, secondary_network, tau):
    for source_parameters, target_parameters in zip(primary_network.parameters(), secondary_network.parameters()):
        target_parameters.data.copy_(tau * source_parameters.data + (1.0 - tau) * target_parameters.data)
