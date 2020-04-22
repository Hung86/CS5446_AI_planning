# from .models import *
# from .dqn_agent import *
# from .replay_buffer import *
# from .dqn_env import *
from models import *
from dqn_agent import *
from replay_buffer import *
from env import *
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

def compute_loss(paras, model, target, states, actions, rewards, next_states, dones):
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
    q_target_value = ((q_next_state_values * paras.gamma) * mask[indices, [0]]) + rewards[indices, [0]]

    theloss = F.smooth_l1_loss(q_state_values, q_target_value.unsqueeze(1))

    return theloss

def optimize(model, target, memory, optimizer, paras):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(paras.batch_size, device)
    loss = compute_loss(paras, model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_epsilon(episode, paras):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = paras.min_epsilon + (paras.max_epsilon - paras.min_epsilon) * math.exp(-1. * episode / paras.epsilon_decay)
    return epsilon

def train(model_class, running_env, paras):

    # Initialize model and target network
    model = model_class(running_env.observation_space.shape, running_env.action_space.n).to(device)
    target = model_class(running_env.observation_space.shape, running_env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer(paras.buffer_limit)
    dqnagent = DQNAgent()

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=paras.learning_rate)
    print("train : step 1 , t_max : ", t_max)

    for episode in range(paras.max_episodes):
        epsilon = compute_epsilon(episode, paras)
        state = running_env.reset()
        episode_rewards = 0.0

        for t in range(paras.t_max):
            # Model takes action
            action = dqnagent.act(model, device, state, epsilon)

            # Apply the action to the environment
            next_state, reward, done, info = running_env.step(action)

            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))

            state = next_state
            episode_rewards += reward
            if done:
                break
        #print("train : episode_rewards :", episode_rewards)
        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(memory) > paras.min_buffer:
            if np.mean(rewards[paras.print_interval:]) < 0:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(paras.train_steps):
                loss = optimize(model, target, memory, optimizer, paras)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % paras.target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % paras.print_interval == 0 and episode > 0:
            print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode, np.mean(rewards[paras.print_interval:]), np.mean(losses[paras.print_interval * 10:]), len(memory),
                    epsilon * 100))
    return model

def train_model(old_model, running_env, paras):

    # Initialize model and target network
    model = old_model
    target = old_model
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer(paras.buffer_limit)
    dqnagent = DQNAgent()

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=paras.learning_rate)
    print("train : step 1 , t_max : ", t_max)

    for episode in range(paras.max_episodes):
        epsilon = compute_epsilon(episode, paras)
        state = running_env.reset()
        episode_rewards = 0.0

        for t in range(paras.t_max):
            # Model takes action
            action = dqnagent.act(model, device, state, epsilon)

            # Apply the action to the environment
            next_state, reward, done, info = running_env.step(action)

            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))

            state = next_state
            episode_rewards += reward
            if done:
                break
        #print("train : episode_rewards :", episode_rewards)
        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(memory) > paras.min_buffer:
            if np.mean(rewards[paras.print_interval:]) < 0:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(paras.train_steps):
                loss = optimize(model, target, memory, optimizer, paras)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % paras.target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % paras.print_interval == 0 and episode > 0:
            print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode, np.mean(rewards[paras.print_interval:]), np.mean(losses[paras.print_interval * 10:]), len(memory),
                    epsilon * 100))
    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    parser.add_argument('--model', dest='model', action='store_true', help='train the agent')
    args = parser.parse_args()

    running_env = construct_task2_env();
    if args.train & args.model:
        print("train existed model")
        existed_model = get_model()
        model = train_model(existed_model, running_env, HyperParameter(hyper_paras[1]))
        save_model(model)
    if args.train:
        print("train new model")
        model = train(AtariDQN, running_env, HyperParameter(hyper_paras[0]))
        save_model(model)
    else:
        model = get_model()

