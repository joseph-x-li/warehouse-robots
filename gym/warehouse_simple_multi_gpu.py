from .gym_mock import GymMock
import random
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from gym import gpu_kernels

# Tweak these to your heart's content
WALL_COLLISION_REWARD = -1.1
ROBOT_COLLISION_REWARD = -3
GOAL_REWARD = 2


class State:
    step_gpu = None
    tensor_gpu = None
    def __init__(self, nagents, rows, cols, view_size, testing):
        assert view_size % 2 == 1
        self.nagents = nagents
        self.view_size = view_size
        self.rows, self.cols = rows, cols
        poss = self._generate_positions(start=testing)
        goals = self._generate_positions(end=testing)
        field = np.zeros((self.rows, self.cols), dtype=np.int32)
        for pos in poss:
            field[tuple(pos)] = 1
        self.poss_gpu = cuda.mem_alloc(poss.nbytes)
        self.goals_gpu = cuda.mem_alloc(goals.nbytes)
        self.field_gpu = cuda.mem_alloc(field.nbytes)
        cuda.memcpy_htod(self.poss_gpu, poss)
        cuda.memcpy_htod(self.goals_gpu, goals)
        cuda.memcpy_htod(self.field_gpu, field)

        if self.step_gpu is None: # cache activity
            self.step_gpu = gpu_kernels.stepkernel(
                self.rows,
                self.cols,
                self.nagents,
                WALL_COLLISION_REWARD,
                ROBOT_COLLISION_REWARD,
                GOAL_REWARD,
            )
        if self.tensor_gpu is None:
            self.tensor_gpu = gpu_kernels.tensorkernel(
                self.rows, 
                self.cols, 
                self.view_size, 
                self.nagents
            )

    def step(self, actions):
        rewards = np.zeros((self.nagents,), dtype=np.float32)
        start = cuda.Event()
        end = cuda.Event()
        hold1 = cuda.Out(rewards)
        hold2 = cuda.In(actions)
        start.record()
        self.step_gpu(
            hold1,
            hold2,
            self.poss_gpu,
            self.goals_gpu,
            self.field_gpu,
            block=(1024, 1, 1),
            grid=(1, 1, 1),
        )
        end.record()
        end.synchronize()
        print("[GPU]\t[STEP]\t[KRNL]\t(ms): {:.4f}".format(start.time_till(end)))
        return rewards

    @property
    def tensor(self):
        states = np.zeros((self.nagents, self.view_size ** 2 + 4), dtype=np.float32)
        start = cuda.Event()
        end = cuda.Event()
        hold1 = cuda.Out(states)
        start.record()
        self.tensor_gpu(
            hold1,
            self.poss_gpu,
            self.goals_gpu,
            self.field_gpu,
            block=(1024, 1, 1),
            grid=(1, 1, 1),
        )
        end.record()
        end.synchronize()
        print("[GPU]\t[TSR]\t[KRNL]\t(ms): {:.4f}".format(start.time_till(end)))
        return states

    def _generate_positions(self, start=False, end=False):
        if start:
            return np.array([
                    (0, 0),
                    (1, 0),
                    (1, 1),
                    (2, 1),
                    (0, 2),
                    (2, 2),
                    (2, 3),
                    (2, 4),
                    (1, 5),
                    (0, 6),
                    (2, 6),
                ], 
                dtype=np.int32
            )
        if end:
            return np.array([
                    (6, 0),
                    (6, 1),
                    (6, 2),
                    (6, 3),
                    (6, 4),
                    (6, 5),
                    (1, 3),
                    (0, 4),
                    (0, 5),
                    (0, 6),
                    (1, 6),
                ], 
                dtype=np.int32
            )
            
        num_set = set()
        while len(num_set) < self.nagents:
            a, b = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if (a, b) not in num_set:
                num_set.add((a, b))
        return np.array(list(num_set), dtype=np.int32)

    def copy_gpu_data(self):
        poss_cpu = np.zeros((self.nagents, 2), dtype=np.int32)
        goals_cpu = np.zeros((self.nagents, 2), dtype=np.int32)
        field_cpu = np.zeros((self.rows, self.cols), dtype=np.int32)
        cuda.memcpy_dtoh(poss_cpu, self.poss_gpu)
        cuda.memcpy_dtoh(goals_cpu, self.goals_gpu)
        cuda.memcpy_dtoh(field_cpu, self.field_gpu)
        return poss_cpu, goals_cpu, field_cpu


class Gym(GymMock):
    testing = True
    rows, cols = (11, 11) if testing else (1000, 1000)
    speed_mod = False
    nagents = 11 if testing else 10000
    view_size = 11

    def __init__(self):
        action_space_size = 9 if self.speed_mod else 5  # S + (NEWS, 2 * NEWS)
        super().__init__(
            action_space_size, (4 + (self.view_size ** 2),)
        )  # mypos, goalpos, receptive field
        self.reset()

    def reset(self):
        self.state = State(self.nagents, self.rows, self.cols, self.view_size, self.testing)
        return self.state.tensor

    def step(self, actions):
        start = time.time()
        rewards = self.state.step(actions)
        print(
            "[GPU]\t[STEP]\t[CALL]\t(ms): {:.4f}".format((time.time() - start) * 1000)
        )
        start = time.time()
        tsr = self.state.tensor
        print("[GPU]\t[TSR]\t[CALL]\t(ms): {:.4f}".format((time.time() - start) * 1000))
        return tsr, rewards, False, None
