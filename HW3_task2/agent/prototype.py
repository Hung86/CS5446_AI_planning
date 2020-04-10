import collections


# # Hyperparameters --- don't change, RL is very sensitive
# learning_rate = 1e-4
# gamma         = 0.99
# buffer_limit  = 100000
# batch_size    = 64
# max_episodes  = 100000
# t_max         = 600
# min_buffer    = 1000
# target_update = 20 # episode(s)
# train_steps   = 20
# max_epsilon   = 1.0
# min_epsilon   = 0.01
# epsilon_decay = 20000
# print_interval= 20

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 100000
batch_size    = 64
max_episodes  = 20000
t_max         = 600
min_buffer    = 1000
target_update = 20 # episode(s)
train_steps   = 20
max_epsilon   = 1.0
min_epsilon   = 0.01
epsilon_decay = 500
print_interval= 20

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
