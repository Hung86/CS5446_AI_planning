import numpy as np
import torch
from copy import deepcopy
import gym
from gym.utils import seeding
import gym_grid_driving
from gym_grid_driving.envs.grid_driving import LaneSpec, MaskSpec, Point
import math

random = None



class DQNAgent():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        '''
        FILL ME : This function should return epsilon-greedy action.

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''
        #Get a random number and determine if agent should exploit or explore
        rand_num = np.random.random()
        if(rand_num < epsilon):#explore by choosing random action
            output_action = np.random.randint(self.model.num_actions)
        else:                   #exploit by choosing best action
            output_actions = self.model.forward(state)
            output_action = torch.argmax(output_actions).item()
        return output_action
