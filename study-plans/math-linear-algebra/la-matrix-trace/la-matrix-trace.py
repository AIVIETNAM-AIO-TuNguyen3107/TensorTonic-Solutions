import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a Python float.
    """
    A = np.array(A, dtype=np.float64)
    total = 0.0
    for r in range(A.shape[0]):
        total += float(A[r, r])
    return total