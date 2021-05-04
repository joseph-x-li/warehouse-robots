from .gym_mock import GymMock
import random
import numpy as np
from numba.experimental import jitclass
from numba import int32, float32

spec = [
    ("WALL_COLLISION_REWARD", float32),
    ("ROBOT_COLLISION_REWARD", float32),
    ("GOAL_REWARD", float32),
    ("nagents", int32),
    ("view_size", int32),
    ("rows", int32),
    ("cols", int32),
    ("poss", int32[:, :]),
    ("goals", int32[:, :]),
    ("movlookup", int32[:, :]),
    ("field", int32[:, :]),
]


@jitclass(spec)
class State:
    def __init__(self, nagents, rows, cols, view_size):
        assert view_size % 2 == 1
        self.WALL_COLLISION_REWARD = -1.1
        self.ROBOT_COLLISION_REWARD = -3
        self.GOAL_REWARD = 2
        self.nagents = nagents
        self.view_size = view_size
        self.rows, self.cols = rows, cols
        self.poss = self._generate_positions()
        self.goals = self._generate_positions()
        self.movlookup = np.array(
            [(-1, 0), (0, 1), (1, 0), (0, -1)],  # north  # east  # south  # west
            dtype=np.int32,
        )
        self.field = np.zeros((rows, cols), dtype=np.int32)
        for pos in self.poss:
            self.field[pos[0], pos[1]] = 1

    def step(self, actions, numba=False):
        rewards = np.zeros((self.nagents,), dtype=float32)
        oldpositions = []
        for idx, (action, pos, goal) in enumerate(zip(actions, self.poss, self.goals)):
            reward = 0
            if action == 0:
                nextpos = pos
            else:
                nextpos = pos + self.movlookup[action - 1]
                collision_status = self._collision(nextpos)
                if collision_status == 0:  # no collision
                    pass
                elif collision_status == -1:
                    reward += self.WALL_COLLISION_REWARD
                    nextpos = pos
                elif collision_status == -2:
                    reward += self.ROBOT_COLLISION_REWARD
                    nextpos = pos

            reward += self._mdist(goal, pos) - self._mdist(goal, nextpos)

            if np.array_equal(nextpos, goal):
                reward += self.GOAL_REWARD

            if not np.array_equal(nextpos, pos):
                oldpositions.append(pos)
                self.poss[idx] = nextpos

            self.field[nextpos[0], nextpos[1]] = 1

            rewards[idx] = reward

        for pos in oldpositions:
            self.field[pos[0], pos[1]] = 0
        return rewards

    @property
    def tensor(self):
        dimshape = 4 + self.view_size ** 2
        accum = np.zeros((self.nagents, dimshape), dtype=np.float32)
        n_vec = np.array([self.rows / 2, self.cols / 2], dtype=np.float32)
        for idx, (pos, goal) in enumerate(zip(self.poss, self.goals)):
            accum[idx] = np.hstack(
                (
                    self._render_view(pos).flatten(),
                    (pos.astype(np.float32) / n_vec) - 1,
                    (goal.astype(np.float32) / n_vec) - 1,
                )
            ).astype(np.float32)
        return accum

    def _collision(self, pos):
        if not ((0 <= pos[0] < self.rows) and (0 <= pos[1] < self.cols)):
            return -1

        if self.field[pos[0], pos[1]] != 0:
            return -2

        return 0

    @staticmethod
    def _mdist(a, b):
        return (np.abs(a - b)).sum()

    def _render_view(self, pos):
        view = np.full((self.view_size, self.view_size), -1, dtype=np.int32)
        correction = self.view_size // 2
        for r in range(self.view_size):
            for c in range(self.view_size):
                fieldr, fieldc = pos + np.array(
                    (r - correction, c - correction), dtype=np.int32
                )
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


import time


class Gym(GymMock):
    rows, cols = (1000, 1000)
    speed_mod = False
    nagents = 10000
    view_size = 11

    def __init__(self):
        action_space_size = 9 if self.speed_mod else 5  # S + (NEWS, 2 * NEWS)
        super().__init__(action_space_size, (4,))  # mypos, goalpos
        self.reset()

    def reset(self):
        self.state = State(self.nagents, self.rows, self.cols, self.view_size)
        return self.state.tensor

    def step(self, actions):
        start = time.time()
        rewards = self.state.step(actions)
        print(
            "[CPU]\t[STEP]\t[CALL]\t(ms): {:.4f}".format((time.time() - start) * 1000)
        )
        start = time.time()
        tsr = self.state.tensor
        print("[CPU]\t[TSR]\t[CALL]\t(ms): {:.4f}".format((time.time() - start) * 1000))
        return tsr, rewards, False, None
