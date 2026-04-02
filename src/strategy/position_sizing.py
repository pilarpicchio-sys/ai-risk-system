import numpy as np


def compute_position_size(probs):
    """
    Position sizing continuo
    """

    sizes = []

    for p in probs:

        p_down = p[0] + p[1]
        p_up = p[4] + p[5]

        size = p_up - p_down

        # niente short
        size = max(0, size)

        sizes.append(size)

    return np.array(sizes)