from .gym_mock import GymMock
import random
import numpy as np

class State:
    def __init__(self, pos, goal, br, bc):
        self.br, self.bc = br, bc
        self.pos = pos
        self.goal = goal
        self.movlookup = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W

    @property
    def done(self):
        retval = self.pos == self.goal
        if retval:
            print("SEKS SEE")
        return retval

    @property
    def tensor(self):
        vec = (
            (self.pos[0] * 2 / self.br) - 1, 
            (self.pos[1] * 2 / self.bc) - 1,
            (self.goal[0] * 2 / self.br) - 1, 
            (self.goal[1] * 2 / self.bc) - 1,
        )
        return np.array(vec)

    def getforward(self, action):
        dr, dc = self.movlookup[action % 4]
        velocity = 2 if action >= 4 else 1
        dr, dc = dr * velocity, dc * velocity
        return (self.pos[0] + dr, self.pos[1] + dc)

    def _goaldist(self, pos):  # manhattan distance
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def reward(self, nextpos):
        return self._goaldist(self.pos) - self._goaldist(nextpos)

class Gym(GymMock):
    rows, cols = BOARDSIZE = (50, 50)  # 100 rows, 200 cols (wide)
    WALL_COLLISION_REWARD = -0.05
    speed_mod = False

    def __init__(self):
        action_space_size = 9 if self.speed_mod else 5 # S + (NEWS, 2 * NEWS)
        super().__init__(action_space_size, (4, )) # mypos, goalpos
        self.reset()

    def reset(self):
        start, end = self._sample_point(), self._sample_point()
        print(f"Start: {start} \tEnd: {end}")
        self.state = State(start, end, self.rows, self.cols)
        return self.state.tensor
    
    def step(self, action):
        print(
            f"\rAction received: {action}; Dist: {self.state._goaldist(self.state.pos)}  ", end="")
        if action == 0:
            return self.state.tensor, 0, self.state.done, None

        nextpos = self.state.getforward(action - 1)
        if self._is_collision(nextpos):
            reward = self.WALL_COLLISION_REWARD
        else:
            reward = self.state.reward(nextpos)
            reward -= 0.01
            self.state.pos = nextpos
        return self.state.tensor, reward, self.state.done, None

    def _sample_point(self):
        return (random.randint(0, self.BOARDSIZE[0] - 1), random.randint(0, self.BOARDSIZE[1] - 1))

    def _is_collision(self, pos):
        return not ((0 <= pos[0] < self.rows) and (0 <= pos[1] < self.cols))

def make(s):
    return Gym()
