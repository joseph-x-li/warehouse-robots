from .gym_mock import GymMock
import random
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from gym import gpu_kernels

# Tweak these to your heart's content
WALL_COLLISION_REWARD = -1.1 
ROBOT_COLLISION_REWARD = -3
GOAL_REWARD = 2

class State:
    def __init__(self, nagents, rows, cols, view_size=11):
        assert view_size % 2 == 1
        self.nagents = nagents
        self.view_size = view_size
        self.rows, self.cols = rows, cols
        poss = self._generate_positions()
        goals = self._generate_positions()
        field = np.zeros((rows, cols), dtype=np.int32)
        for pos in poss:
            field[tuple(pos)] = 1
        self.poss_gpu = cuda.mem_alloc(poss.nbytes)
        self.goals_gpu = cuda.mem_alloc(goals.nbytes)
        self.field_gpu = cuda.mem_alloc(field.nbytes)
        cuda.memcpy_htod(self.poss_gpu, poss)
        cuda.memcpy_htod(self.goals_gpu, goals)
        cuda.memcpy_htod(self.field_gpu, field)

        self.step_gpu = gpu_kernels.stepkernel(rows, cols, nagents, WALL_COLLISION_REWARD, ROBOT_COLLISION_REWARD, GOAL_REWARD)

    def step(self, actions):
        rewards = np.zeros((self.nagents,), dtype=np.float32)
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        self.step_gpu(
            cuda.Out(rewards), 
            cuda.In(actions), 
            self.poss_gpu, 
            self.goals_gpu, 
            self.field_gpu,
            block=(1024,1,1)
        )
        end.record()
        end.synchronize()
        print(f"GPU time (ms): {start.time_till(end)}")
        return rewards

    @property
    def tensor(self):
        accum = []
        n_vec = np.array([self.rows / 2, self.cols / 2])
        for pos, goal in zip(self.poss, self.goals):
            accum.append(
                np.hstack((
                    self._render_view(pos).flatten(),
                    pos / n_vec - 1,
                    goal / n_vec - 1,
                ))
            )
        return np.array(accum)

    def _collision(self, position):
        if not ((0 <= pos[0] < self.rows) and (0 <= pos[1] < self.cols)):
            return -1

        if self.field[position] != 0:
            return -2

        return 0

    @staticmethod
    def _mdist(a, b):
        return abs(a - b).sum()

    def _render_view(self, pos):
        view = np.full((self.view_size, self.view_size), -1, dtype=np.int32)
        correction = self.view_size // 2
        for r in range(self.view_size):
            for c in range(self.view_size):
                fieldr, fieldc = pos + np.array((r - correction, c - correction), dtype=np.int32)
                if not ((0 <= fieldr < self.rows) and (0 <= fieldc < self.cols)):
                    continue
                view[r, c] = self.field[fieldr, fieldc]

        return view

    def _generate_positions(self):
        num_set = set()
        while len(num_set) < self.nagents:
            a, b = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if (a, b) not in num_set:
                num_set.add((a, b))
        return np.array(list(num_set), dtype=np.int32)


class Gym(GymMock):
    rows, cols = (100, 400) 
    speed_mod = False
    nagents = 1000

    def __init__(self):
        action_space_size = 9 if self.speed_mod else 5 # S + (NEWS, 2 * NEWS)
        super().__init__(action_space_size, (4, )) # mypos, goalpos
        self.reset()

    def reset(self):
        self.state = State(self.nagents, self.rows, self.cols)
        # return self.state.tensor

    def step(self, actions):
        rewards = self.state.step(actions)
        # return self.state.tensor, rewards, False, None
