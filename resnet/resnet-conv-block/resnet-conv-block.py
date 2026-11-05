import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    Ws = np.asarray(Ws, dtype=np.float64)
    return np.maximum(0, np.maximum(0, x@W1)@W2+x@Ws)