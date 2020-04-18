import torch
import torch.autograd as autograd
import torch.distributions as distributions
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os
import sys

from models import *
from prototype import *


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# script_path = os.path.dirname(os.path.realpath(__file__))
# model_path = os.path.join(script_path, 'model.pt')
#
# # Hyperparameters --- don't change, RL is very sensitive
# learning_rate = 0.001
# gamma         = 0.98
# buffer_limit  = 5000
# batch_size    = 32
# max_episodes  = 2000
# t_max         = 600
# min_buffer    = 1000
# target_update = 20 # episode(s)
# train_steps   = 10
# max_epsilon   = 1.0
# min_epsilon   = 0.01
# epsilon_decay = 500
# print_interval= 20
#
# Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
ENT_COEF = 1e-2


class ActorCritic():
    def __init__(self, env, network_model):
        self.log_probs = None
        self.actor = network_model(env.observation_space.shape, env.action_space.n).to(device)
        self.critic = network_model(env.observation_space.shape, 1).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)


    def choose_action_mtcs(self, state):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        output_actions = self.actor(state)
        output_action = torch.argmax(output_actions).item()
        return output_action

    def choose_action(self, state):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        probabilities = F.softmax(self.actor.forward(state), dim=1)
        action_probs = distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        return action.item()
        # if not isinstance(state, torch.FloatTensor):
        #     state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # '''
        # FILL ME : This function should return epsilon-greedy action.
        #
        # Input:
        #     * `state` (`torch.tensor` [batch_size, channel, height, width])
        #     * `epsilon` (`float`): the probability for epsilon-greedy
        #
        # Output: action (`Action` or `int`): representing the action to be taken.
        #         if action is of type `int`, it should be less than `self.num_actions`
        # '''
        # #Get a random number and determine if agent should exploit or explore
        # rand_num = np.random.random()
        # if(rand_num < epsilon):#explore by choosing random action
        #     output_action = np.random.randint(self.model.num_actions)
        # else:                   #exploit by choosing best action
        #     output_actions = self.model.forward(state)
        #     output_action = torch.argmax(output_actions).item()
        # return output_action

    def learn(self, state, reward, new_state, done):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        if not isinstance(new_state, torch.FloatTensor):
            new_state = torch.from_numpy(new_state).float().unsqueeze(0).to(device)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        critic_value = self.critic.forward(state)
        critic_value_next = self.critic.forward(new_state)
        critic_value_next_ = (reward + gamma * critic_value_next * (1 - int(done)))
        td_error = (critic_value_next_ - critic_value)

        actor_loss = -self.log_probs * td_error
        #critic_loss =  td_error**2
        critic_loss = F.smooth_l1_loss(critic_value, critic_value_next_)

        #print(actor_loss, critic_loss)
        (actor_loss + critic_loss).backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return actor_loss, critic_loss


    def save_models(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        actor_model_path = os.path.join(script_path, 'actor_model.pt')
        actor_data = (self.actor.__class__.__name__, self.actor.state_dict(), self.actor.input_shape, self.actor.num_actions)
        torch.save(actor_data, actor_model_path)

        critic_model_path = os.path.join(script_path, 'critic_model.pt')
        critic_data = (self.critic.__class__.__name__, self.critic.state_dict(), self.critic.input_shape, self.critic.num_actions)
        torch.save(critic_data, critic_model_path)