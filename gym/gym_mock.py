import random


class ActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randint(0, self.n - 1)


class ObservationSpace:
    def __init__(self, shape):
        self.shape = shape


class GymMock:
    def __init__(self, action_space_size, observation_space_shape):
        """Base Class to help you mock a gym

        Args:
            action_space_size (int): Number of different actions you can take
            observation_space_shape (tuple): Shape of state
        """
        self.action_space = ActionSpace(action_space_size)
        self.observation_space = ObservationSpace(observation_space_shape)
