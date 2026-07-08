import numpy as np

def pairwise_diff(a):
    """Returns: np.ndarray of shape (n, n) where out[i,j] = a[i] - a[j]"""
    a = np.array(a)
    return a[:, None] - a[None,:]