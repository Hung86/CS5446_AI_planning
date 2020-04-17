import gym
from gym.utils import seeding
import gym_grid_driving
from gym_grid_driving.envs.grid_driving import LaneSpec, MaskSpec, Point


def construct_task2_env():
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -2]),
                        LaneSpec(cars=7, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -2])]
            }
    return gym.make('GridDriving-v0', **config)

def construct_task2_env_1():
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=1, speed_range=[-2, -1]),
                        LaneSpec(cars=1, speed_range=[-2, -1]),
                        LaneSpec(cars=1, speed_range=[-1, -1]),
                        LaneSpec(cars=1, speed_range=[-3, -1]),
                        LaneSpec(cars=1, speed_range=[-2, -1]),
                        LaneSpec(cars=1, speed_range=[-2, -1]),
                        LaneSpec(cars=1, speed_range=[-3, -2]),
                        LaneSpec(cars=1, speed_range=[-1, -1]),
                        LaneSpec(cars=1, speed_range=[-2, -1]),
                        LaneSpec(cars=1, speed_range=[-2, -2])]
            }
    return gym.make('GridDriving-v0', **config)

def construct_task2_env_2():
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=2, speed_range=[-2, -1]),
                        LaneSpec(cars=2, speed_range=[-2, -1]),
                        LaneSpec(cars=2, speed_range=[-1, -1]),
                        LaneSpec(cars=2, speed_range=[-3, -1]),
                        LaneSpec(cars=2, speed_range=[-2, -1]),
                        LaneSpec(cars=2, speed_range=[-2, -1]),
                        LaneSpec(cars=2, speed_range=[-3, -2]),
                        LaneSpec(cars=2, speed_range=[-1, -1]),
                        LaneSpec(cars=2, speed_range=[-2, -1]),
                        LaneSpec(cars=2, speed_range=[-2, -2])]
            }
    return gym.make('GridDriving-v0', **config)

def construct_task2_env_3():
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=3, speed_range=[-2, -1]),
                        LaneSpec(cars=3, speed_range=[-2, -1]),
                        LaneSpec(cars=3, speed_range=[-1, -1]),
                        LaneSpec(cars=3, speed_range=[-3, -1]),
                        LaneSpec(cars=3, speed_range=[-2, -1]),
                        LaneSpec(cars=3, speed_range=[-2, -1]),
                        LaneSpec(cars=3, speed_range=[-3, -2]),
                        LaneSpec(cars=3, speed_range=[-1, -1]),
                        LaneSpec(cars=3, speed_range=[-2, -1]),
                        LaneSpec(cars=3, speed_range=[-2, -2])]
            }
    return gym.make('GridDriving-v0', **config)

def construct_task2_env_4():
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=4, speed_range=[-2, -1]),
                        LaneSpec(cars=4, speed_range=[-2, -1]),
                        LaneSpec(cars=4, speed_range=[-1, -1]),
                        LaneSpec(cars=4, speed_range=[-3, -1]),
                        LaneSpec(cars=4, speed_range=[-2, -1]),
                        LaneSpec(cars=4, speed_range=[-2, -1]),
                        LaneSpec(cars=4, speed_range=[-3, -2]),
                        LaneSpec(cars=4, speed_range=[-1, -1]),
                        LaneSpec(cars=4, speed_range=[-2, -1]),
                        LaneSpec(cars=4, speed_range=[-2, -2])]
            }
    return gym.make('GridDriving-v0', **config)

def construct_training_env():
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'finish_position':Point(0,0), 'random_seed': 15, 'stochasticity' : 1.,
              'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -2]),
                        LaneSpec(cars=7, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -2])], 'width': 50, 'tensor_state':True, 'flicker_rate':0., 'mask':None}
    return gym.make('GridDriving-v0', **config)
    # LANES = config['lanes']
    # WIDTH = config['width']
    # RANDOM_SEED = config['seed']
    # numiters = config['iters']
    # stochasticity = 1.
    # env = gym.make('GridDriving-v0', lanes=LANES, width=WIDTH,
    #                agent_speed_range=(-3,-1), finish_position=Point(0,0), #agent_ pos_init=Point(4,2),
    #                stochasticity=stochasticity, tensor_state=True, flicker_rate=0., mask=None, random_seed=RANDOM_SEED)
