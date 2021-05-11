#!/home/jet/miniforge3/envs/py36_torch/bin/python3.6
import gym
import numpy as np


# pos, goals, field = gpumenv.state.copy_gpu_data()
# pos, goals, field = gpuenv.state.copy_gpu_data()
# pos, goals, field = cpuenv.state.poss, cpuenv.state.goals, cpuenv.state.field
# gpustate[0][:-4].astype(np.int32).reshape((11,11))
# gpumstate[0][:-4].astype(np.int32).reshape((11,11))
# cpustate[0][:-4].astype(np.int32).reshape((11,11))

cpuenv = gym.make("warehouse_simple_multi")
gpuenv = gym.make("warehouse_simple_multi_gpu")
gpumenv = gym.make("warehouse_simple_mega_gpu")

cpustate = cpuenv.reset(testing=True)
gpustate = gpuenv.reset(testing=True)
gpumstate = gpumenv.reset(11, testing=True)
assert np.allclose(cpustate, gpustate)
assert np.allclose(np.tile(cpustate, (11, 1)), gpumstate)

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
    gpumstate, gpumrewards, _, _ = gpumenv.step(np.tile(action, (11,)))
    assert np.allclose(cpustate, gpustate)
    assert np.allclose(cpurewards, gpurewards)
    assert np.allclose(np.tile(cpustate, (11, 1)), gpumstate)
    assert np.allclose(np.tile(cpurewards, (11,)), gpumrewards)