import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    arr = np.array(data, dtype=np.float64)
    mask = arr > threshold
    return np.stack([
        mask,
        np.where(mask.any(axis=1, keepdims=True), arr, 0),
        np.where(mask.all(axis=1, keepdims=True), arr, 0)
    ])