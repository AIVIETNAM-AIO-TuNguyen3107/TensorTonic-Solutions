import numpy as np

def normalize(data: list) -> np.ndarray:
    """
    Returns a float64 matrix standardized independently by column.
    """
    data = np.array(data, dtype=np.float64)
    n, d = data.shape
    m = np.mean(data, axis=0, keepdims=True)
    sigma = np.sqrt((np.sum((data-m)**2, axis=0, keepdims=True)) / n)
    return (data - m) / sigma
