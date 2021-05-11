import numpy as np


def genposs():
    return np.array(
        [
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
        dtype=np.int32,
    )


def gengoals():
    return np.array(
        [
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
        dtype=np.int32,
    )
