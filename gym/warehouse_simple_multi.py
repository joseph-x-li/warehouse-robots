from .gym_mock import GymMock
import random
import numpy as np


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
        self.poss = self._generate_positions()
        self.goals = self._generate_positions()
        self.movlookup = np.array([
            (-1, 0), # north
            (0, 1),  # east
            (1, 0),  # south
            (0, -1)  # west
        ], dtype=np.int32)
        self.field = np.zeros((rows, cols), dtype=np.int32)
        for pos in self.poss:
            self.field[pos] = 1

    def step(self, actions):
        rewards = []
        oldpositions = []
        for idx, (action, pos, goal) in enumerate(zip(actions, self.poss, self.goals)):
            reward = 0
            if action == 0:
                nextpos = pos
            else:
                nextpos = pos + self.movlookup[action]
                collision_status = self._collision(nextpos)
                if collision_status == 0: # no collision
                    pass
                elif collision_status == -1:
                    reward += WALL_COLLISION_REWARD
                    nextpos = pos
                elif collision_status == -2:
                    reward += ROBOT_COLLISION_REWARD
                    nextpos = pos

            reward += self._mdist(goal, pos) - self._mdist(goal, nextpos)

            if nextpos == goal:
                reward += GOAL_REWARD

            if nextpos != pos:
                oldpositions.append(pos)
                self.poss[idx] = nextpos

            rewards.append(reward)

        for pos in oldpositions:
            self.field[pos] = 0

        return np.array(rewards)

    @property
    def tensor(self):
        accum = []
        for pos, goal in zip(self.poss, self.goals):
            n_vec = np.array([self.rows / 2, self.cols / 2])
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
    rows, cols = (50, 50) 
    speed_mod = False
    nagents = 50

    def __init__(self):
        action_space_size = 9 if self.speed_mod else 5 # S + (NEWS, 2 * NEWS)
        super().__init__(action_space_size, (4, )) # mypos, goalpos
        self.reset()

    def reset(self):
        self.state = State(self.nagents, self.rows, self.cols)
        return self.state.tensor
    
    def step(self, actions):
        rewards = self.state.step(actions)
        return self.state.tensor, rewards, False, None
