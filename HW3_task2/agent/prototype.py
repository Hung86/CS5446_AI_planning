import collections
import torch
import os
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.99
buffer_limit  = 5000
batch_size    = 32
max_episodes  = 800
t_max         = 600
min_buffer    = 1000
target_update = 20 # episode(s)
train_steps   = 10
max_epsilon   = 1.0
min_epsilon   = 0.01
epsilon_decay = 500
print_interval= 20

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
