import numpy as np
import torch
import torch.nn as nn

ITERATIONS = 100


class SingleEnvironState:
    def __init__(self, pos, goal):
        self.pos = pos
        self.goal = goal
        self.velocity = 0  # 0, 1, 2
        self.dir = 0       # N, E, S, W (0, 1, 2, 3)
        self.movlookup = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W

    @property
    def done(self):
        return self.state.pos == self.state.goal

    def rotate(self, drot):
        self.dir = (self.dir + drot) % 4

    def getforward(self, patch=None):
        dr, dc = self.movlookup[self.state['dir']]
        if patch is None:
            dr, dc = dr * self.velocity, dc * self.velocity
        else:
            dr, dc = dr * patch, dc * patch
        return (self.pos[0] + dr, self.pos[1] + dc)

    def _goaldist(self, pos):  # manhattan distance
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def reward(self, nextpos):
        return self.state._goaldist(self.state.pos) - self.state._goaldist(nextpos)


class SingleEnviron:
    rows, cols = BOARDSIZE = (100, 200)  # 100 rows, 200 cols (wide)
    NAGENTS = 1

    def reset(self):
        self.board = np.zeros(self.BOARDSIZE, dtype=np.int32)
        self.state = SingleEnvironState(
            self._sample_point(), self._sample_point())
        return self.state

    def step(self, action):  # Action is 6-vector, S, C, F, R, U, L
        '''return state, reward, done'''
        action = torch.argmax(action)
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
                return self.state, 0, self.state.done

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
        return self.state, 0, self.state.done

    def _step_vel_1(self):
        nextpos = self.state.getforward()
        if self._is_collision(nextpos):
            self.state.velocity = 0
            reward = -10
        else:
            reward = self.state.reward(nextpos)
            self.state.pos = nextpos
        return self.state, reward, self.state.done

    def _step_vel_2(self):
        nextpos = self.state.getforward()
        if self._is_collision(nextpos):
            reward = -20
            self.state.velocity = 0
            nextpos_v1 = self.state.getforward(patch=1)
            if not self._is_collision(nextpos_v1):  # flatten against wall
                self.state.pos = nextpos_v1
        else:
            reward = self.state.reward(nextpos)
            self.state.pos = nextpos

        return self.state, reward, self.state.done

    def _sample_point(self):
        # gen pos from [0, BOARDSIZE[0])
        initial_pos_r = np.random.randint(BOARDSIZE[0], size=self.NAGENTS)
        initial_pos_c = np.random.randint(BOARDSIZE[1], size=self.NAGENTS)
        return list(zip(initial_pos_r, initial_pos_c))[0]


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6, 24)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(24, 12)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(12, 6)  # S, C, F, R, U, L

    def forward(self, out):
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)


breakpoint()
