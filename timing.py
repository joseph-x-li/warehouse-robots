#!/home/jet/miniforge3/envs/py36_torch/bin/python3.6

import gym
import time
import numpy as np
from tqdm import trange

# def timeenv(name):
#     env = gym.make(name)
#     env.reset()
#     n_actions = env.action_space.n
#     start = time.time()
#     for action in range(n_actions):
#         env.step(action)

def timeenv_multi(name, timing):
    env = gym.make(name)
    env.reset()
    n_actions = env.action_space.n
    n_agents = env.nagents
    actions = np.random.randint(0, n_actions, size=(n_actions, n_agents), dtype=np.int32)
    for action in actions:
        env.step(action, timing=timing)

def timeenv_mega(name, timing):
    nenv = gym.make(name)
    numenv = 12
    nenv.reset(numenv) #make 12 env
    n_actions = nenv.action_space.n
    n_agents = nenv.nagents
    actions = np.random.randint(0, n_actions, size=(n_actions, n_agents * numenv), dtype=np.int32)
    for action in actions:
        nenv.step(action, timing=timing)

timeenv_multi("warehouse_simple_multi", True)
# for _ in trange(100):
#     # timeenv_multi("warehouse_simple_multi_gpu", False)
#     timeenv_mega("warehouse_simple_mega_gpu", False)
