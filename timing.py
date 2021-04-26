import gym
import time
import numpy as np

def timeenv(name):
    env = gym.make(name)
    n_actions = env.action_space.n
    start = time.time()
    for action in range(n_actions):
        env.step(action)
    delta = time.time() - start
    print(f"Env: {name}\nTotal Time: {delta}\nAverge/Step: {delta/n_actions}\n")

def timeenv_multi(name):
    env = gym.make(name)
    n_actions = env.action_space.n

    n_agents = env.nagents
    actions = np.random.randint(0, n_actions, size=(n_actions, n_agents), dtype=np.int32)
    start = time.time()
    for action in actions:
        env.step(action)
    delta = time.time() - start
    print(f"Env: {name}\nTotal Time: {delta}\nAverge/Step: {delta/n_actions}\n")
    return delta/n_actions

gpustep = timeenv_multi("warehouse_simple_multi_gpu")
cpustep = timeenv_multi("warehouse_simple_multi")
print(f"Speedup: {cpustep / gpustep}")
# for name in gym.lookup.keys():
    # timeenv(name) if "multi" not in name else timeenv_multi(name)