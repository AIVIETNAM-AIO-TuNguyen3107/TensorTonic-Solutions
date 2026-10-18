import numpy as np
from numpy import linalg as LA

def matrix_rank(A: list) -> int:
    """
    Returns the numerical rank as a Python integer.
    """
    A = np.array(A, dtype=np.float32)

    S = LA.svd(A, compute_uv=False)
    tolerance = max(A.shape) * S[0] * np.finfo(np.float64).eps
    return int(np.count_nonzero(S > tolerance))