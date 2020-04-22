import collections
import torch
import os


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

hyper_paras =[
    {"learning_rate" : 0.001 , "gamma" : 0.98 , "buffer_limit"  : 100000,
    "batch_size" : 64, "max_episodes" : 50000, "t_max" : 600, "min_buffer"    : 1000,
    "target_update" : 20, "train_steps"   : 10, "max_epsilon"   : 1.0, "min_epsilon"   : 0.01,
    "epsilon_decay" : 500, "print_interval" : 20},     {"learning_rate" : 0.0001 , "gamma" : 0.98 , "buffer_limit"  : 100000,
    "batch_size" : 64, "max_episodes" : 2000, "t_max" : 600, "min_buffer"    : 1000,
    "target_update" : 20, "train_steps"   : 10, "max_epsilon"   : 1.0, "min_epsilon"   : 0.01,
    "epsilon_decay" : 500, "print_interval" : 20}
     ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

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

class HyperParameter():
    def __init__(self, parameters):
        self.learning_rate = parameters["learning_rate"]
        self.gamma = parameters["gamma"]
        self.buffer_limit = parameters["buffer_limit"]
        self.batch_size = parameters["batch_size"]
        self.max_episodes = parameters["max_episodes"]
        self.t_max = parameters["t_max"]
        self.min_buffer = parameters["min_buffer"]
        self.target_update = parameters["target_update"]
        self.train_steps = parameters["train_steps"]
        self.max_epsilon = parameters["max_epsilon"]
        self.min_epsilon = parameters["min_epsilon"]
        self.epsilon_decay = parameters["epsilon_decay"]
        self.print_interval = parameters["print_interval"]