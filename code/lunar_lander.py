import time
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import ReplayBuffer, DeepQNetwork, interpolate_networks

np.random.seed(43)
torch.manual_seed(43)


class Agent:
    def __init__(self, state_size, action_size, buffer_size, batch_size, epsilon_min, epsilon, epsilon_decay_factor,
                 discount_factor, update_interval, tau, learning_rate, layer_widths):
        self.time_step = 0
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.discount_factor = discount_factor
        self.update_interval = update_interval
        self.tau = tau
        self.learning_rate = learning_rate

        self.memory_buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)
        self.primary_network = DeepQNetwork(state_size, action_size, layer_widths=layer_widths)
        self.secondary_network = DeepQNetwork(state_size, action_size, layer_widths=layer_widths)
        self.criterion = nn.MSELoss()
        self.primary_network_optimizer = optim.Adam(self.primary_network.parameters(), lr=self.learning_rate)
        self.last_100_game_scores = deque(maxlen=100)
        self.scores = []

    def update(self, state, action, reward, next_state, done):
        """
        Update the buffer, which consists of updating the buffer as well as periodically updating the Q networks.
        1) Add experience to the buffer.
        2) for every `update_interval` number of steps, perform learning procedure to update the Q networks, by calling the `learn` procedure.
        """
        self.memory_buffer.add_experience(state, action, reward, next_state, done)
        self.time_step += 1
        if self.time_step % self.update_interval == 1:
            if self.memory_buffer.get_num_elements() > self.batch_size:
                sampled_experiences = self.memory_buffer.sample_experiences()
                self.learn(sampled_experiences)

    def learn(self, experiences):
        """
        Sample from the buffer, then perform learning procedure, updating both Q networks.
        """
        states, actions, rewards, next_states, is_terminal = experiences
        q_values = self.secondary_network(next_states).detach()
        max_q_values = torch.max(q_values, dim=1)[0].unsqueeze(1)
        target_q_values = rewards + (self.discount_factor * max_q_values * (1 - is_terminal))
        predicted_q_values = self.primary_network(states).gather(1, actions)
        loss = self.criterion(predicted_q_values, target_q_values)
        self.primary_network_optimizer.zero_grad()
        loss.backward()
        self.primary_network_optimizer.step()
        interpolate_networks(self.primary_network, self.secondary_network, self.tau)

    def epsilon_greedy_action(self, observation):
        """
        Returns epsilon-greedy action given a state. If the random number generated is larger than epsilon, then select a greedy action based on the DQN.
        """
        if np.random.random() > self.epsilon:
            # greedy action
            state = torch.tensor(np.array([observation])).float().unsqueeze(0)
            self.primary_network.eval()
            with torch.no_grad():
                action_values = self.primary_network.forward(state)
            self.primary_network.train()
            action = torch.argmax(action_values).item()
        else:
            # take random action
            action = np.random.randint(self.action_size)
        return action

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_factor)

    def get_last_100_game_scores(self):
        return np.mean(self.last_100_game_scores)

    def add_game_score(self, score):
        self.last_100_game_scores.append(score)
        self.scores.append(score)

    def get_all_scores(self):
        return self.scores

    def save_model(self):
        torch.save(self.primary_network.state_dict(), "./models/model.pth")


def train_model(env, state_size, action_size, buffer_size, batch_size, epsilon_min, epsilon, epsilon_decay_factor,
                discount_factor, update_interval, tau, learning_rate, layer_widths, early_stopping):
    agent = Agent(state_size=state_size, action_size=action_size, buffer_size=buffer_size, batch_size=batch_size,
                  epsilon_min=epsilon_min, epsilon=epsilon, epsilon_decay_factor=epsilon_decay_factor,
                  discount_factor=discount_factor, update_interval=update_interval, tau=tau,
                  learning_rate=learning_rate, layer_widths=layer_widths)

    episode = 0

    not_solved = True

    while episode < MAX_EPISODES + 1 and not_solved:
        state = env.reset()
        score = 0
        time_step = 0
        while time_step < MAX_STEPS:
            action = agent.epsilon_greedy_action(state)
            next_state, reward, done, _ = env.step(action)  # take step, observe next state, reward, etc.
            score += reward  # track the cumulative reward within an episode
            agent.update(state, action, reward, next_state,
                         done)  # add the information to agent, periodically update Q networks
            state = next_state  # go to the next state
            if done:
                break
            agent.decay_epsilon()
            time_step += 1

        if episode >= PRINTING_INTERVAL and episode % PRINTING_INTERVAL == 0:
            mean_score = agent.get_last_100_game_scores()
            print("\rEpisode {}/{}, mean score in last 100 episodes: {}".format(episode, MAX_EPISODES, mean_score),
                  end="")
            if mean_score > MIN_SCORE:
                print('\nEnvironment solved in {} episodes'.format(episode), end="")
                if early_stopping:
                    not_solved = False
                agent.save_model()  # save model only if solved
                # break

        agent.add_game_score(score)
        episode += 1

    return agent


if __name__ == '__main__':
    # Instantiate lunar lander environment
    env = gym.make('LunarLander-v2')
    env.reset()

    # Configuration, hyperparameters
    STATE_SIZE = env.observation_space.shape[0]
    ACTION_SIZE = env.action_space.n
    BATCH_SIZE = 64
    LAYER_WIDTHS = [128, 128, 32]  # hidden layer widths for Q networks
    BUFFER_SIZE = 10000
    DISCOUNT_FACTOR = 0.99
    TAU = 1e-3  # interpolation parameter for updating neural networks
    LEARNING_RATE = 0.001  # learning rate in the neural network optimizer
    UPDATE_INTERVAL = 4  # update the Q network every UPDATE_INTERVAL number of steps
    MAX_EPISODES = 4000  # maximum number of games
    MAX_STEPS = 1000  # maximum number of steps in each episode
    MIN_SCORE = 200  # game is solved if score exceeds MIN_SCORE
    EPSILON = 1.0  # starting value for epsilon
    EPSILON_DECAY_FACTOR = 0.9  # decay factor for epsilon
    EPSILON_MIN = 0.01  # minimum epsilon for the epsilon-greedy selection of next state
    PRINTING_INTERVAL = 100  # interval for printing
    EARLY_STOPPING = True

    # Test out different hyperparameters, overwrite the base configuration above.
    # LEARNING_RATE = 0.01
    # EARLY_STOPPING = False
    # MAX_EPISODES = 2000
    # EPSILON_DECAY_FACTOR = 0.1

    start = time.time()
    agent = train_model(env=env, state_size=STATE_SIZE, action_size=ACTION_SIZE, buffer_size=BUFFER_SIZE,
                        batch_size=BATCH_SIZE,
                        epsilon_min=EPSILON_MIN, epsilon=EPSILON, epsilon_decay_factor=EPSILON_DECAY_FACTOR,
                        discount_factor=DISCOUNT_FACTOR, update_interval=UPDATE_INTERVAL, tau=TAU,
                        learning_rate=LEARNING_RATE, layer_widths=LAYER_WIDTHS, early_stopping=EARLY_STOPPING)
    end = time.time()
    print('Time taken for training: {} seconds'.format(end - start))

    # Training curve
    scores = agent.get_all_scores()
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(scores)), scores)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode')
    fig.savefig("./models/figure.jpg")
    plt.show()
