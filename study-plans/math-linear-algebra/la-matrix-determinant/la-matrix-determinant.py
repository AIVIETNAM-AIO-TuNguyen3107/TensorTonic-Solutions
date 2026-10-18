import numpy as np

def matrix_determinant(A: list) -> float:
    """
    Returns the determinant as a Python float.
    """
    A = np.array(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    n = A.shape[0]

    if n == 0:
        raise ValueError("A must not be empty")

    if n == 1:
        return float(A[0, 0])

    if n == 2:
        return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])

    result = 0.0
    for j in range(A.shape[1]):
        cofactor = A[0,j]*((-1)**j)
        minor = np.hstack([A[1:, :j], A[1:, j+1:]])
        result += float(cofactor*matrix_determinant(minor))
    return result