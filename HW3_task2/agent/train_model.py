# from .models import *
# from .dqn_agent import *
# from .replay_buffer import *
# from .dqn_env import *
from models import *
from dqn_agent import *
from replay_buffer import *
from dqn_env import *
from prototype import *

import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os
import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')


def compute_loss(model, target, states, actions, rewards, next_states, dones):
    '''
    FILL ME : This function should compute the DQN loss function for a batch of experiences.

    Input:
        * `model`       : model network to optimize
        * `target`      : target network
        * `states`      (`torch.tensor` [batch_size, channel, height, width])
        * `actions`     (`torch.tensor` [batch_size, 1])
        * `rewards`     (`torch.tensor` [batch_size, 1])
        * `next_states` (`torch.tensor` [batch_size, channel, height, width])
        * `dones`       (`torch.tensor` [batch_size, 1])

    Output: scalar representing the loss.

    References:
        * MSE Loss  : https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
        * Huber Loss: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
    '''

    indices = np.arange(len(rewards))
    q_state_values = model(states).gather(1, actions)

    q_next_state_values = target(next_states).max(1)[0].detach()

    # Create mask based on terminal state
    mask = dones.clone()
    mask[dones == 0.0] = 1.0
    mask[dones > 0.0] = 0.0

    # If its terminal(done), then set as reward only
    q_target_value = ((q_next_state_values * gamma) * mask[indices, [0]]) + rewards[indices, [0]]

    theloss = F.smooth_l1_loss(q_state_values, q_target_value.unsqueeze(1))

    return theloss

def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size, device)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

def test(model, env, max_episodes=600):
    '''
    Test the `model` on the environment `env` (GridDrivingEnv) for `max_episodes` (`int`) times.

    Output: `avg_rewards` (`float`): the average rewards
    '''
    rewards = []
    dqnagent = DQNAgent()
    for episode in range(max_episodes):
        state = env.reset()
        episode_rewards = 0.0
        for t in range(t_max):
            action = dqnagent.act(state)
            state, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
    avg_rewards = np.mean(rewards)
    print("{} episodes avg rewards : {:.1f}".format(max_episodes, avg_rewards))
    return avg_rewards

def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`.
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model


def save_model(model):
    '''
    Save `model` to disk. Location is specified in `model_path`.
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)

def train(model_class, env):
    '''
    Train a model of instance `model_class` on environment `env` (`GridDrivingEnv`).

    It runs the model for `max_episodes` times to collect experiences (`Transition`)
    and store it in the `ReplayBuffer`. It collects an experience by selecting an action
    using the `model.act` function and apply it to the environment, through `env.step`.
    After every episode, it will train the model for `train_steps` times using the
    `optimize` function.

    Output: `model`: the trained model.
    '''

    # Initialize model and target network
    model = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()
    dqnagent = DQNAgent()

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("train : step 1 , t_max : ", t_max)

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0
        #mtcs = MonteCarloTreeSearch(model, device, dqnagent, epsilon,env, 100, 1., 15)

        for t in range(t_max):
            # Model takes action
            action = dqnagent.act(model, device, state, epsilon)
            #root_node_state = GridWorldState(state, False)

            #action = mtcs.buildTreeAndReturnBestAction(initialState=root_node_state)
            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))

            state = next_state
            if t == t_max:
                print("train : t_max :")

            episode_rewards += reward
            if done:
                break
        #print("train : episode_rewards :", episode_rewards)
        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            if np.mean(rewards[print_interval:]) < 0:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval * 10:]), len(memory),
                    epsilon * 100))
    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    args = parser.parse_args()

    env = construct_task2_env();
    #sys.stdout = open("log.txt", "w")
    if args.train:
        model = train(AtariDQN, env)
        save_model(model)
    else:
        model = get_model()
    test(model, env, max_episodes=600)
    #sys.stdout.close()

