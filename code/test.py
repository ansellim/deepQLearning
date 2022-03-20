import torch
import gym
from utils import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt

# change model name
MODEL_NAME = "model.pth"

env = gym.make('LunarLander-v2')
state_dict = torch.load("./models/" + MODEL_NAME)

network = DeepQNetwork(state_size=8,action_size=4,layer_widths=[128,128,32])
network.load_state_dict(state_dict)

total_num_episodes = 100
num_episode = 0
max_num_time_steps = 1000

scores = []

while num_episode < total_num_episodes:
    state = env.reset()
    state = torch.Tensor(state)
    score = 0
    num_time_step = 0

    while num_time_step < max_num_time_steps:
        q_values = network.forward(state)
        action = torch.argmax(q_values)
        action = action.item()
        next_state,reward,done,_ = env.step(action)
        score += reward
        if done:
            break
        state = torch.Tensor(next_state)
        num_time_step += 1

    num_episode+=1

    scores.append(score)

fig,ax = plt.subplots()
ax.plot(np.arange(len(scores)),scores)
ax.set_ylabel('Score')
ax.set_xlabel('Episode')
ax.set_title("Score for 100 episodes using trained model")
fig.savefig("./models/figure.jpg")
plt.show()

