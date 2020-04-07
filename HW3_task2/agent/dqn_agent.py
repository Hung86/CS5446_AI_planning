import numpy as np
import torch


class DQNAgent():
    def act(self, model, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
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
            output_action = np.random.randint(model.num_actions)
        else:                   #exploit by choosing best action
            output_actions = model.forward(state)
            output_action = torch.argmax(output_actions).item()
        return output_action