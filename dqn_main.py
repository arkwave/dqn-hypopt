import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

# taken from https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch, with modifications.

class Net(nn.Module):
    def __init__(self, num_state, num_action):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = self.fc2(x)
        return action_prob

# TODO: sanity check, make sure logic is right. 
class DQN(nn.Module):
    capacity = 8000
    learning_rate = 1e-3
    memory_count = 0
    batch_size = 256
    gamma = 0.995
    update_count = 0

    def __init__(self, num_state, num_action, params):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net(num_state, num_action), Net(num_state, num_action)
        self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')
        self.params = params
        self.num_actions = num_action 
        self.num_states = num_state


    def select_action(self,state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.act_net(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= 0.9: # epslion greedy
            action = np.random.choice(range(self.num_action), 1).item()
        return action

    def store_transition(self,transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self):
        if self.memory_count >= self.capacity:
            state = torch.Tensor([t.state for t in self.memory]).float()
            action = torch.LongTensor([t.action for t in self.memory]).view(-1,1).long()
            reward = torch.Tensor([t.reward for t in self.memory]).float()
            next_state = torch.Tensor([t.next_state for t in self.memory]).float()

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad():
                target_v = reward + self.gamma * self.target_net(next_state).max(1)[0]

            # Update
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                v = (self.act_net(state).gather(1, action))[index]
                loss = self.loss_func(target_v[index].unsqueeze(1), (self.act_net(state).gather(1, action))[index])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                self.update_count +=1
                if self.update_count % 100 ==0:
                    self.target_net.load_state_dict(self.act_net.state_dict())
        else:
            print("Too few instances in memory buffer.")


class Transition(object):
    def __init__(self, old_state, action, new_state, reward: float, terminate_: bool):
        self.old_state = old_state
        self.action = action
        self.new_state = new_state
        self.reward = reward
        self.terminate = terminate_

    @property
    def terminate(self):
        return self.__terminate

    @terminate.setter
    def terminate(self, terminate_):
        if isinstance(terminate_, bool):
            self.__terminate = terminate_
        else:
            raise TypeError(f'{terminate_} should be bool type')

    def get_transition_tuple(self) -> tuple:
        return self.old_state, self.action, self.new_state, self.reward, self.terminate


# TODO: make this work for both discrete and continuous games. 
def initialize_game(params):
    env_name = params.env_name
    # discrete action spaces.
    if env_name in ['MountainCar-v0', 'CartPole-v0', 'MountainCar-v1', 'LunarLander']:
        pass

    elif env_name in ['Pendulum-v0', 'LunarLander-Continuous', 'BipedalWalker']:
        pass 
    env = gym.make(env_name).unwrapped
    num_state = 0 
    num_action = 0
    dqn = DQN(num_state, num_action, params) 
    return env, num_state, num_action 


# TODO: include funcs to save logfiles etc 
def main(params):
    env, num_state, num_action, DQN = initialize_game(params)
    agent = DQN()
    for i_ep in range(params.num_episodes):
        state = env.reset()
        if params.render: env.render()
        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if params.render: env.render()
            transition = Transition(state, action, next_state, reward, done)
            agent.store_transition(transition)
            state = next_state
            if done or t >=9999:
                agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
                agent.update()
                if i_ep % 10 == 0:
                    print("episodes {}, step is {} ".format(i_ep, t))


# TODO: figure out which of these are unnecessary.
if __name__ == "__main__":

    # parse arguments. 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'

    # Supported Envs:
    # 1. Discrete: ['CartPole-v0', 'MountainCar-v0', 'LunarLander', 'Acrobot']
    # 2. Continuous: ['MountainCarContinuous-v0', 'LunarLanderContinuous', 'BipedalWalker', 'Pendulum-v0']
    parser.add_argument("--env_name", default="CartPole-v0")

    # model parameters. 
    parser.add_argument('--update_count', default=150, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
    parser.add_argument('--capacity', default=50000, type=int) # replay buffer size
    parser.add_argument('--num_iteration', default=100000, type=int) #  num of  games
    parser.add_argument('--batch_size', default=100, type=int) # mini batch size
    parser.add_argument('--seed', default=True, type=bool)
    parser.add_argument('--random_seed', default=9527, type=int)
    parser.add_argument('--sample_type', default='uniform', type=str)

    # optional parameters
    parser.add_argument('--num_hidden_layers', default=2, type=int)
    parser.add_argument('--sample_frequency', default=256, type=int)
    parser.add_argument('--render', default=False, type=bool) # show UI or not
    parser.add_argument('--log_interval', default=50, type=int) #
    parser.add_argument('--load', default=False, type=bool) # load model
    parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_delay', default=2, type=int)
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    args = parser.parse_args()

    print(args)

    # # pass args into the main function 
    # main(args)
    

