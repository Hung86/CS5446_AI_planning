'''
Name : Tran Khanh Hung (A0212253W)
Name : Lim Jia Xian Clarence (A0212209U)
'''
import torch
import collections
import random

from .prototype import *

class ReplayBuffer():
    def __init__(self, buffer_limit=10000):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        # dequeu to limit buffer and auto remove element when full
        self.buffer = collections.deque(maxlen=buffer_limit)
        pass


    def push(self, transition):
        '''
        FILL ME : This function should store the transition of type `Transition` to the buffer `self.buffer`.

        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.

        Output:
            * None
        '''
        # Dequeue will append elements from the right and remove elements from the left when full
        self.buffer.append(transition)
        pass


    def sample(self, batch_size, device):
        '''
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''

        # random a batch of samples
        rand_transitions = random.sample(self.buffer, batch_size)

        # zip and unzip the elements together
        sample_batch = Transition(*zip(*rand_transitions))

        # Get the relevant elements of the batch
        state_tensor = torch.tensor(sample_batch.state, dtype=torch.float32).to(device)
        actions_tensor = torch.LongTensor(sample_batch.action).to(device)
        rewards_tensor = torch.tensor(sample_batch.reward, dtype=torch.float32).to(device)
        next_state_tensor = torch.tensor(sample_batch.next_state, dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(sample_batch.done, dtype=torch.float32).to(device)

        output_tuple = (state_tensor, actions_tensor, rewards_tensor, next_state_tensor, dones_tensor)

        return output_tuple

        pass


    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)