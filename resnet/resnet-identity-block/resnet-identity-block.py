import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    x1 = np.maximum(0, x@W1.T)
    x2 = np.maximum(0, x1@W2.T+x)
    return x2.tolist()
    