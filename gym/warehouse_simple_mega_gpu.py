from .gym_mock import GymMock
import gym.testcases as testcases
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

step_gpu = None
tensor_gpu = None

class State:

    def __init__(self, nenv, nagents, rows, cols, view_size, testing):
        global step_gpu, tensor_gpu
        assert view_size % 2 == 1
        self.nenv = nenv
        self.nagents = nagents
        self.view_size = view_size
        self.rows, self.cols = rows, cols
        poss, goals, field = [], [], []
        for _ in range(self.nenv):
            p = self._generate_positions(start=testing)
            g = self._generate_positions(end=testing)
            f = np.zeros((self.rows, self.cols), dtype=np.int32)
            for pos in p:
                f[tuple(pos)] = 1
            poss.append(p)
            goals.append(g)
            field.append(f)
        poss, goals, field = np.vstack(poss), np.vstack(goals), np.vstack(field)
        self.poss_gpu = cuda.mem_alloc(poss.nbytes)
        self.goals_gpu = cuda.mem_alloc(goals.nbytes)
        self.field_gpu = cuda.mem_alloc(field.nbytes)
        cuda.memcpy_htod(self.poss_gpu, poss)
        cuda.memcpy_htod(self.goals_gpu, goals)
        cuda.memcpy_htod(self.field_gpu, field)

        if step_gpu is None:  # cache activity
            step_gpu = gpu_kernels.stepkernel(
                self.rows,
                self.cols,
                self.nagents,
                self.nenv,
                WALL_COLLISION_REWARD,
                ROBOT_COLLISION_REWARD,
                GOAL_REWARD,
            )
        if tensor_gpu is None:
            tensor_gpu = gpu_kernels.tensorkernel(
                self.rows, self.cols, self.view_size, self.nagents, self.nenv
            )

    def step(self, actions, timing):
        rewards = np.zeros((self.nenv * self.nagents,), dtype=np.float32)
        hold1 = cuda.Out(rewards)
        hold2 = cuda.In(actions)
        if timing:
            start = cuda.Event()
            end = cuda.Event()
            start.record()
        step_gpu(
            hold1,
            hold2,
            self.poss_gpu,
            self.goals_gpu,
            self.field_gpu,
            block=(1024, 1, 1),
            grid=(self.nenv, 1, 1),
        )
        if timing:
            end.record()
            end.synchronize()
            print("[GPUM]\t[STEP]\t[KRNL]\t(ms): {:.4f}".format(start.time_till(end)))
        return rewards

    def tensor(self, timing):
        states = np.zeros((self.nenv * self.nagents, self.view_size ** 2 + 4), dtype=np.float32)
        hold1 = cuda.Out(states)
        if timing:
            start = cuda.Event()
            end = cuda.Event()
            start.record()
        tensor_gpu(
            hold1,
            self.poss_gpu,
            self.goals_gpu,
            # self.field_gpu,
            block=(1024, 1, 1),
            grid=(self.nenv, 1, 1),
        )
        if timing:
            end.record()
            end.synchronize()
            print("[GPUM]\t[TSR]\t[KRNL]\t(ms): {:.4f}".format(start.time_till(end)))
        return states

    def _generate_positions(self, start=False, end=False):
        if start:
            return testcases.genposs()
        if end:
            return testcases.gengoals()

        num_set = set()
        while len(num_set) < self.nagents:
            a, b = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if (a, b) not in num_set:
                num_set.add((a, b))
        return np.array(list(num_set), dtype=np.int32)

    def copy_gpu_data(self):
        poss_cpu = np.zeros((self.nenv, self.nagents, 2), dtype=np.int32)
        goals_cpu = np.zeros((self.nenv, self.nagents, 2), dtype=np.int32)
        field_cpu = np.zeros((self.nenv, self.rows, self.cols), dtype=np.int32)
        cuda.memcpy_dtoh(poss_cpu, self.poss_gpu)
        cuda.memcpy_dtoh(goals_cpu, self.goals_gpu)
        cuda.memcpy_dtoh(field_cpu, self.field_gpu)
        return poss_cpu, goals_cpu, field_cpu


class Gym(GymMock):
    rows, cols = 1000, 1000
    nagents = 5000
    view_size = 11

    def __init__(self):
        action_space_size = 9
        super().__init__(action_space_size, (4 + (self.view_size ** 2),))
        # self.reset()

    def reset(self, nenv, testing=False):
        if testing:
            self.state = State(nenv, 11, 11, 11, 11, testing)
        else:
            self.state = State(
                nenv, self.nagents, self.rows, self.cols, self.view_size, testing
            )
        return self.state.tensor(False)

    def step(self, actions, timing=False):
        if timing:
            start = time.time()
            rewards = self.state.step(actions, timing)
            print(
                "[GPUM]\t[STEP]\t[CALL]\t(ms): {:.4f}".format(
                    (time.time() - start) * 1000
                )
            )
            start = time.time()
            tsr = self.state.tensor(timing)
            print(
                "[GPUM]\t[TSR]\t[CALL]\t(ms): {:.4f}".format(
                    (time.time() - start) * 1000
                )
            )
            return tsr, rewards, False, None
        else:
            rewards = self.state.step(actions, timing)
            tsr = self.state.tensor(timing)
            return tsr, rewards, False, None
