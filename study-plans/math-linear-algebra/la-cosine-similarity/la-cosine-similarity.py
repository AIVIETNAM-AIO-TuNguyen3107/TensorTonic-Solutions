import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.array(a)
    b = np.array(b)
    length_a = np.sqrt(np.sum(a**2))
    length_b = np.sqrt(np.sum(b**2))
    if length_a==0 or length_b==0:
        return 0
    return (a@b)/ (length_a*length_b)