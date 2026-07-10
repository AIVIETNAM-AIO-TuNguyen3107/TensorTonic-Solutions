import numpy as np

def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    return np.sqrt(np.sum((np.array(x, dtype=np.float32) - np.array(y, dtype=np.float32))**2))