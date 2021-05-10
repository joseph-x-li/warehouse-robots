#!/home/jet/miniforge3/envs/py36_torch/bin/python3.6

import gym
import time
import numpy as np
from tqdm import trange

# def timeenv(name):
#     env = gym.make(name)
#     n_actions = env.action_space.n
#     start = time.time()
#     for action in range(n_actions):
#         env.step(action)
#     delta = time.time() - start
#     print(f"Env: {name}\nTotal Time: {delta}\nAverge/Step: {delta/n_actions}\n")

def timeenv_multi(name):
    env = gym.make(name)
    n_actions = env.action_space.n
    n_agents = env.nagents
    actions = np.random.randint(0, n_actions, size=(n_actions, n_agents), dtype=np.int32)
    for action in actions:
        env.step(action, timing=False)

for _ in trange(100):
    timeenv_multi("warehouse_simple_multi_gpu")
# timeenv_multi("warehouse_simple_multi")
