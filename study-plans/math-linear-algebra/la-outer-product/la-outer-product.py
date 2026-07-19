import numpy as np

def outer_product(u, v):
    """
    Returns: float64 matrix of shape (m, n), the outer product u v^T.
    """
    return np.array(u)[:, np.newaxis] @ np.array(v)[np.newaxis, :]