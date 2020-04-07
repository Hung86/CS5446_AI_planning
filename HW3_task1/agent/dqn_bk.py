import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_grid_driving.envs.grid_driving import LaneSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 5000
batch_size    = 32
max_episodes  = 2000
t_max         = 600
min_buffer    = 1000
target_update = 20 # episode(s)
train_steps   = 10
max_epsilon   = 1.0
min_epsilon   = 0.01
epsilon_decay = 500
print_interval= 20


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.buffer_limit = buffer_limit
        self.buffer = []
        self.current_pos = -1
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
        if len(self.buffer) < self.buffer_limit:
            self.buffer.append(transition)
            self.current_pos += 1
        else:
            self.current_pos = (self.current_pos + 1) % self.buffer_limit
            self.buffer[self.current_pos] = transition
        pass
    
    def sample(self, batch_size):
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
        #transitions = np.random.choice(self.buffer, batch_size)
        #batch = Transition(*zip(*transitions))
        transitions = random.sample(self.buffer, batch_size)
        sample_batch = Transition(*zip(*transitions))


        states = torch.tensor(sample_batch.state).float().to(device)
        actions = torch.tensor(sample_batch.action).to(device)
        rewards = torch.tensor(sample_batch.reward).float().to(device)
        next_states = torch.tensor(sample_batch.next_state).float().to(device)
        dones = torch.tensor(sample_batch.done).float().to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)


class Base(nn.Module):
    '''
    Base neural network model that handles dynamic architecture.
    '''
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.construct()

    def construct(self):
        raise NotImplementedError

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        return x.view(1, -1).size(1)

class BaseAgent(Base):
    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            rand = np.random.random()
            if rand < epsilon:
                return np.random.choice(self.num_actions)
            else:
                actions = self.forward(state)
                action = torch.argmax(actions).item()
                return action

        '''
        FILL ME : This function should return epsilon-greedy action.

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''
        pass

class DQN(BaseAgent):
    def construct(self):
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

class ConvDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
        )
        super().construct()


def compute_loss(model, target, states, actions, rewards, next_states, dones):
    '''
    FILL ME : This function should compute the DQN loss function for a batch of experiences.

    Input:
        * `model`       : model network to optimize
        * `target`      : target network
        * `states`      (`torch.tensor` [batch_size, channel, height, width])
        * `actions`     (`torch.tensor` [batch_size, 1])
        * `rewards`     (`torch.tensor` [batch_size, 1])
        * `next_states` (`torch.tensor` [batch_size, channel, height, width])
        * `dones`       (`torch.tensor` [batch_size, 1])

    Output: scalar representing the loss.

    References:
        * MSE Loss  : https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
        * Huber Loss: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
    '''
    # indecies = np.arange(len(rewards))
    # q_state_values = model(states).gather(1, actions)
    # q_next_state_values = target(next_states).max(1)[0].detach()
    # q_target_value = (q_next_state_values * gamma) + rewards[indecies, [0]]
    #
    # loss = F.smooth_l1_loss(q_state_values, q_target_value.unsqueeze(1))
    indices = np.arange(len(rewards))
    q_state_values = model(states).gather(1, actions)
    q_next_state_values = target(next_states).max(1)[0].detach()
    # Create mask based on terminal state
    mask = dones.clone()
    mask[dones == 0.0] = 1.0
    mask[dones > 0.0] = 0.0

    # If its terminal(done), then set as reward only
    q_target_value = ((q_next_state_values * gamma) * mask[indices, [0]]) + rewards[indices, [0]]

    theloss = F.smooth_l1_loss(q_state_values, q_target_value.unsqueeze(1))

    #q_state_values = model.forward(states).to(device)
    #q_next_state_values = model.forward(next_states).

    return theloss

def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

def train(model_class, env):
    '''
    Train a model of instance `model_class` on environment `env` (`GridDrivingEnv`).
    
    It runs the model for `max_episodes` times to collect experiences (`Transition`)
    and store it in the `ReplayBuffer`. It collects an experience by selecting an action
    using the `model.act` function and apply it to the environment, through `env.step`.
    After every episode, it will train the model for `train_steps` times using the 
    `optimize` function.

    Output: `model`: the trained model.
    '''

    # Initialize model and target network
    model = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action

            action = model.act(state, epsilon)

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))

            state = next_state
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
        
        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            if np.mean(rewards[print_interval:]) < 0.1:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                            episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval*10:]), len(memory), epsilon*100))
    return model

def test(model, env, max_episodes=600):
    '''
    Test the `model` on the environment `env` (GridDrivingEnv) for `max_episodes` (`int`) times.

    Output: `avg_rewards` (`float`): the average rewards
    '''
    rewards = []
    for episode in range(max_episodes):
        state = env.reset()
        episode_rewards = 0.0
        for t in range(t_max):
            action = model.act(state)   
            state, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
    avg_rewards = np.mean(rewards)
    print("{} episodes avg rewards : {:.1f}".format(max_episodes, avg_rewards))
    return avg_rewards

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

def get_env():
    '''
    Get the sample test cases for training and testing.
    '''
    config = {  'observation_type': 'tensor', 'agent_speed_range': [-2, -1], 'stochasticity': 0.0, 'width': 10,
                'lanes': [
                    LaneSpec(cars=3, speed_range=[-2, -1]), 
                    LaneSpec(cars=4, speed_range=[-2, -1]), 
                    LaneSpec(cars=2, speed_range=[-1, -1]), 
                    LaneSpec(cars=2, speed_range=[-3, -1])
                ] }
    return gym.make('GridDriving-v0', **config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    args = parser.parse_args()

    env = get_env()

    if args.train:
        model = train(ConvDQN, env)
        save_model(model)
    else:
        model = get_model()
    test(model, env, max_episodes=600)