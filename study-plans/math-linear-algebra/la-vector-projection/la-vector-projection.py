import numpy as np

def vector_projection(u: list, v: list) -> np.ndarray:
    """
    Returns the float64 projection of u onto v.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    return (np.dot(u, v) / np.dot(v, v) )*v