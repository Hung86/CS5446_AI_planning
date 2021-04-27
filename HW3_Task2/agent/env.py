'''
Name : Tran Khanh Hung (A0212253W)
Name : Lim Jia Xian Clarence (A0212209U)
'''
import gym
from gym_grid_driving.envs.grid_driving import LaneSpec

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