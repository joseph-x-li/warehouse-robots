#!/home/jet/miniforge3/envs/py36_torch/bin/python3.6
import gym
import numpy as np


# pos, goals, field = gpuenv.state.copy_gpu_data()
# gpustate[0][:-4].astype(np.int32).reshape((11,11))
# cpustate[0][:-4].astype(np.int32).reshape((11,11))

cpuenv = gym.make("warehouse_simple_multi")
gpuenv = gym.make("warehouse_simple_multi_gpu")

cpustate = cpuenv.reset(testing=True)
gpustate = gpuenv.reset(testing=True)
assert np.allclose(cpustate, gpustate)

action1 = np.zeros((11,), dtype=np.int32)
action2 = np.array((
        1, # (0, 0), # (0, 0)
        1, # (1, 0), # (1, 0)
        1, # (1, 1), # (0, 1)
        1, # (2, 1), # (2, 1)
        1, # (0, 2), # (0, 2)
        5, # (2, 2), # (1, 2)
        5, # (2, 3), # (0, 3)
        5, # (2, 4), # (0, 4)
        5, # (1, 5), # (0, 5)
        1, # (0, 6), # (0, 6)
        5, # (2, 6), # (1, 6)
    ), dtype=np.int32
)

actions = [
    action1, action2, action2
]

for action in actions:
    cpustate, cpurewards, _, _ = cpuenv.step(action)
    gpustate, gpurewards, _, _ = gpuenv.step(action)
    assert np.allclose(cpustate, gpustate)
    assert np.allclose(cpurewards, gpurewards)