'''
Name : Tran Khanh Hung (A0212253W)
Name : Lim Jia Xian Clarence (A0212209U)
'''
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
    def act(self, model,device, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #Get a random number and determine if agent should exploit or explore
        rand_num = np.random.random()
        if(rand_num < epsilon):#explore by choosing random action
            output_action = np.random.randint(model.num_actions)
        else:                   #exploit by choosing best action
            output_actions = model.forward(state)
            output_action = torch.argmax(output_actions).item()
        return output_action
