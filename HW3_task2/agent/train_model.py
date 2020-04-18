# from .models import *
# from .dqn_agent import *
# from .replay_buffer import *
# from .dqn_env import *
from models import *
from prototype import *
from actor_critic import  *
from dqn_env import *
from replay_buffer import *
from mtcs import *

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

#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# script_path = os.path.dirname(os.path.realpath(__file__))
# model_path = os.path.join(script_path, 'model.pt')

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

def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

def train(actor_critic_agent, env):
    '''
    Train a model of instance `model_class` on environment `env` (`GridDrivingEnv`).

    It runs the model for `max_episodes` times to collect experiences (`Transition`)
    and store it in the `ReplayBuffer`. It collects an experience by selecting an action
    using the `model.act` function and apply it to the environment, through `env.step`.
    After every episode, it will train the model for `train_steps` times using the
    `optimize` function.

    Output: `model`: the trained model.
    '''

    # Initialize rewards, losses, and optimizer
    rewards = []
    actor_losses = []
    critic_losses = []
    action_critic_losses = []
    print("train : step 1 , t_max : ", t_max)
    memory = ReplayBuffer()
    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0
        #mtcs = MonteCarloTreeSearch(model, device, dqnagent, epsilon,env, 100, 1., 15)
        for t in range(t_max):
            action = actor_critic_agent.choose_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            print("---info:",info)
            #memory.push(Transition(state, [action], [reward], next_state, [done]))

            actor_loss, critic_loss = actor_critic_agent.learn(state, reward, next_state, done)
            state = next_state
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            episode_rewards += reward

            if done:
                break
        rewards.append(episode_rewards)
        # Update target network every once in a while
        # # Train the model if memory is sufficient
        # if len(memory) > min_buffer:
        #     if np.mean(rewards[print_interval:]) < 0:
        #         print('Bad initialization. Please restart the training.')
        #         exit()
        #
        #     #actor_loss, critic_loss = actor_critic_agent.learn(state, reward, next_state, done)
        #     actor_loss, critic_loss, action_critic_loss = actor_critic_agent.learn(memory)
        #     actor_losses.append(actor_loss.item())
        #     critic_losses.append(critic_loss.item())
        #     action_critic_losses.append(action_critic_loss.item())


        if episode % print_interval == 0 and episode > 0:
            print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg actor loss: : {:.6f},\tavg critic loss: : {:.6f},\tavg action critic loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode, np.mean(rewards[print_interval:]), np.mean(actor_losses[print_interval * 10:]), np.mean(critic_losses[print_interval * 10:]), np.mean(action_critic_losses[print_interval * 10:]),
                    len(memory),epsilon*100))

        # for t in range(t_max):
        #     # Model takes action
        #     action = dqnagent.act(state, epsilon)
        #     #root_node_state = GridWorldState(state, False)
        #
        #     #action = mtcs.buildTreeAndReturnBestAction(initialState=root_node_state)
        #     # Apply the action to the environment
        #     next_state, reward, done, info = env.step(action)
        #
        #     # Save transition to replay buffer
        #     memory.push(Transition(state, [action], [reward], next_state, [done]))
        #
        #     state = next_state
        #     episode_rewards += reward
        #     if done:
        #         break
        # #print("train : episode_rewards :", episode_rewards)
        # rewards.append(episode_rewards)
        #
        # # # Train the model if memory is sufficient
        # # if len(memory) > min_buffer:
        # #     if np.mean(rewards[print_interval:]) < 0:
        # #         print('Bad initialization. Please restart the training.')
        # #         exit()
        # #     for i in range(train_steps):
        # #         loss = optimize(model, target, memory, optimizer)
        # #         losses.append(loss.item())
        #
        #
        # if episode % print_interval == 0 and episode > 0:
        #     print(
        #         "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
        #             episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval * 10:]), len(memory),
        #             epsilon * 100))
    return actor_critic_agent

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    args = parser.parse_args()

    env = construct_task2_env();

    #sys.stdout = open("log.txt", "w")

    print("Cuda version : ",  torch.cuda.is_available())
    if args.train:
        model = ActorCritic(env, AtariDQN)
        train(model, env)
        model.save_models()

        # save_model(model)
    else:
        model = get_model()
    # test(model, device, env, max_episodes=600)
    #sys.stdout.close()

