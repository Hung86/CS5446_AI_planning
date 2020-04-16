import torch
import torch.autograd as autograd
import torch.distributions as distributions

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


class ActorCritic():
    def __init__(self, env, network_model):
        self.log_probs = None
        self.actor = network_model(env.observation_space.shape, env.action_space.n).to(device)
        self.critic = network_model(env.observation_space.shape, env.action_space.n).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)


    def choose_action(self, state):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        probabilities = F.softmax(self.actor.forward(state))
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
        td_error = ((reward + gamma * critic_value_next * (1- int(done))) - critic_value)

        actor_loss = -self.log_probs * td_error
        critic_loss =  td_error**2

        print(actor_loss, critic_loss)
        (actor_loss + critic_loss).backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return actor_loss, critic_loss