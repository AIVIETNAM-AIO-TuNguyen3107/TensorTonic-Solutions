import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.array(a)
    b = np.array(b)
    norm_product = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
    return 0 if norm_product ==0 else a @ b / norm_product