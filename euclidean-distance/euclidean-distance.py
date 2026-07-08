import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x = np.array(x)
    y = np.array(y)

    d_xy = (x-y)**2
    return np.sqrt(np.sum(d_xy))