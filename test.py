import gym
import numpy as np


cpuenv = gym.make("warehouse_simple_multi")
gpuenv = gym.make("warehouse_simple_multi_gpu")

cpustate = cpuenv.reset()
gpustate = gpuenv.reset()

import pdb; pdb.set_trace()

assert np.allclose(cpustate, gpustate)
# pos, goals, field = gpuenv.state.copy_gpu_data()
action1 = np.zeros((11,), dtype=np.int32)
action2 = np.array((
        1, # (0, 0),
        1, # (1, 0),
        1, # (1, 1),
        1, # (2, 1),
        1, # (0, 2),
        5, # (2, 2),
        5, # (2, 3),
        5, # (2, 4),
        5, # (1, 5),
        1, # (0, 6),
        5, # (2, 6),
    ), dtype=np.int32
)

