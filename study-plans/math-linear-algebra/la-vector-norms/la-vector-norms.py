import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    v = np.array(v, dtype=np.float64)

    return np.array(
        [
            np.sum(np.abs(v)),
            np.sqrt(np.sum(v**2)),
            np.max(np.abs(v))
        ]
    )
