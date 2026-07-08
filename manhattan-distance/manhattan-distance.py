import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return np.sum(np.abs(x-y))