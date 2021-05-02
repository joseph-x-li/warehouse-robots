from .gym_mock import GymMock
import random
import numpy as np


class State:
    def __init__(self, pos, goal, br, bc):
        self.br, self.bc = br, bc
        self.pos = pos
        self.goal = goal
        self.velocity = 0  # 0, 1, 2
        self.dir = 0  # N, E, S, W (0, 1, 2, 3)
        self.movlookup = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W

    @property
    def done(self):
        retval = self.pos == self.goal
        return retval

    @property
    def tensor(self):
        vec = (
            (self.pos[0] * 2 / self.br) - 1,
            (self.pos[1] * 2 / self.bc) - 1,
            (self.goal[0] * 2 / self.br) - 1,
            (self.goal[1] * 2 / self.bc) - 1,
            self.velocity - 1,
            (self.dir / 1.5) - 1,
        )
        return np.array(vec)

    def rotate(self, drot):
        self.dir = (self.dir + drot) % 4

    def getforward(self, patch=None):
        dr, dc = self.movlookup[self.dir]
        if patch is None:
            dr, dc = dr * self.velocity, dc * self.velocity
        else:
            dr, dc = dr * patch, dc * patch
        return (self.pos[0] + dr, self.pos[1] + dc)

    def _goaldist(self, pos):  # manhattan distance
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def reward(self, nextpos):
        rwd = self._goaldist(self.pos) - self._goaldist(nextpos)
        return rwd


class Gym(GymMock):
    rows, cols = BOARDSIZE = (50, 50)  # 100 rows, 200 cols (wide)
    WALL_COLLISION_REWARD = -0.05
    DONE_REWARD = 2

    def __init__(self):
        super().__init__(6, (6,))
        self.reset()

    def reset(self):
        start, end = self._sample_point(), self._sample_point()
        self.state = State(start, end, self.rows, self.cols)
        return self.state.tensor

    def step(self, action):  # Action is 6-vector, S, C, F, R, U, L
        # print(
        #     f"\rAction received: {action}; Dist: {self.state._goaldist(self.state.pos)}", end="")
        if not (0 <= action <= 5):
            print("Unrecognized action... Defaulting to Stop/Stay")
            action = 0

        if self.state.velocity == 0:  # Robot currently at rest
            if action == 0:
                return self._step_vel_0()
            elif action in [1, 2]:
                self.state.velocity = 1
                return self._step_vel_1()
            elif action in [3, 4, 5]:
                self.state.rotate(action - 2)
                reward = self.DONE_REWARD if self.state.done else 0
                return self.state.tensor, reward, False, None

        elif self.state.velocity == 1:  # Robot currently creeping
            if action == 0:
                self.state.velocity = 0
                return self._step_vel_0()
            elif action in [1, 3, 4, 5]:
                return self._step_vel_1()
            elif action == 2:
                self.state.velocity = 2
                return self._step_vel_2()

        elif self.state.velocity == 2:  # Robot currently fast
            if action in [2, 3, 4, 5]:
                return self._step_vel_2()
            else:
                self.state.velocity = 1
                return self._step_vel_1()

    def _is_collision(self, pos):
        return not ((0 <= pos[0] < self.rows) and (0 <= pos[1] < self.cols))

    def _step_vel_0(self):
        reward = self.DONE_REWARD if self.state.done else 0
        return self.state.tensor, reward, False, None

    def _step_vel_1(self):
        nextpos = self.state.getforward()
        if self._is_collision(nextpos):
            self.state.velocity = 0
            reward = self.WALL_COLLISION_REWARD
        else:
            reward = self.state.reward(nextpos)
            self.state.pos = nextpos
        reward += self.DONE_REWARD if self.state.done else 0
        return self.state.tensor, reward, False, None

    def _step_vel_2(self):
        nextpos = self.state.getforward()
        if self._is_collision(nextpos):
            reward = self.WALL_COLLISION_REWARD
            self.state.velocity = 0
            nextpos_v1 = self.state.getforward(patch=1)
            if not self._is_collision(nextpos_v1):  # flatten against wall
                self.state.pos = nextpos_v1
        else:
            reward = self.state.reward(nextpos)
            self.state.pos = nextpos

        reward += self.DONE_REWARD if self.state.done else 0

        return self.state.tensor, reward, False, None

    def _sample_point(self):
        return (
            random.randint(0, self.BOARDSIZE[0] - 1),
            random.randint(0, self.BOARDSIZE[1] - 1),
        )
